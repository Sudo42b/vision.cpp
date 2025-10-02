import torch
import torch.nn as nn
import math
from typing import List, Tuple, Optional, Dict, Any, Union
import re
import numpy as np
from torch.nn import functional as F
from copy import copy

def autopad(k, p=None, d=1):
    """자동 패딩 계산"""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):
    """표준 컨볼루션 + 배치정규화 + 활성화 함수"""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2,  eps=0.001, momentum=0.03, affine=True)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())


    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class AConv(nn.Module):
    """Average pooling convolution for downsampling"""
    def __init__(self, c1, c2):
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)  # 단순한 3x3 stride=2 컨볼루션
        self.avgpool = nn.AvgPool2d(2, 1, 0, False, True)

    def forward(self, x):
        return self.cv1(self.avgpool(x))

class RepConv(nn.Module):
    """
    RepConv module with training and deploy modes.

    This module is used in RT-DETR and can fuse convolutions during inference for efficiency.

    Attributes:
        conv1 (Conv): 3x3 convolution.
        conv2 (Conv): 1x1 convolution.
        bn (nn.BatchNorm2d, optional): Batch normalization for identity branch.
        act (Module): Activation function.
        default_act (Module): Default activation function (SiLU).

    References:
        https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """
        Initialize RepConv module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
            bn (bool): Use batch normalization for identity branch.
            deploy (bool): Deploy mode for inference.
        """
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """
        Forward pass for deploy mode.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))

    def forward(self, x):
        """
        Forward pass for training mode.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """
        Calculate equivalent kernel and bias by fusing convolutions.

        Returns:
            (torch.Tensor): Equivalent kernel
            (torch.Tensor): Equivalent bias
        """
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        """
        Pad a 1x1 kernel to 3x3 size.

        Args:
            kernel1x1 (torch.Tensor): 1x1 convolution kernel.

        Returns:
            (torch.Tensor): Padded 3x3 kernel.
        """
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """
        Fuse batch normalization with convolution weights.

        Args:
            branch (Conv | nn.BatchNorm2d | None): Branch to fuse.

        Returns:
            kernel (torch.Tensor): Fused kernel.
            bias (torch.Tensor): Fused bias.
        """
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = nn.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Fuse convolutions for inference by creating a single equivalent convolution."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class RepNCSPELAN4(nn.Module):
    """RepNCSPELAN4 block"""
    def __init__(self, c1: int, c2: int, c3: int, c4: int, n: int = 1):
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)
        # self.out = Conv(c3 * 4, c2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))
    
class ELAN1(RepNCSPELAN4):
    """ELAN-1 block"""
    def __init__(self, c1: int, c2: int, c3: int, c4: int):
        super().__init__(c1, c2, c3, c4)
        # 32 32 32 16
        self.c = c3 // 2
        # 16
        self.cv1 = Conv(c1, c3, 1, 1) # 32, 32, 1, 1
        self.cv2 = Conv(c3 // 2, c4, 3, 1) # 16, 16, 3, 1
        self.cv3 = Conv(c4, c4, 3, 1) # 16, 16, 3, 1
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1) # 64, 32, 1, 1


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: Tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """
        Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bottleneck with optional shortcut connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """
        Initialize the CSP Bottleneck with 3 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CSP bottleneck with 3 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: Tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """
        Initialize RepBottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)

class RepCSP(C3):
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """
        Initialize RepCSP layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of RepBottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))




##### GELAN #####
class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1: int, c2: int, c3: int, k: int = 5):
        """
        Initialize SPP-ELAN block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            k (int): Kernel size for max pooling.
        """
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))
        
# class Concat(nn.Module):
#     """Concatenate layers along specified dimension"""
#     def __init__(self, dimension=1):
#         super().__init__()
#         self.d = dimension

#     def forward(self, x):
#         return torch.cat(x, self.d)



class Concat(nn.Module):
    """
    Concatenate a list of tensors along specified dimension.

    Attributes:
        d (int): Dimension along which to concatenate tensors.
    """

    def __init__(self, dimension=1):
        """
        Initialize Concat module.

        Args:
            dimension (int): Dimension along which to concatenate tensors.
        """
        super().__init__()
        self.d = dimension

    def forward(self, x: List[torch.Tensor]):
        """
        Concatenate input tensors along specified dimension.

        Args:
            x (List[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Concatenated tensor.
        """
        return torch.cat(x, self.d)



class DWConv(Conv):
    """Depth-wise convolution module."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """
        Initialize depth-wise convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)



class Detect(nn.Module):
    """YOLOv9 Detect head"""
    def __init__(self, nc=80, ch=(64, 96, 128)):
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))
        
        self.stride = torch.tensor([8., 16., 32.])  # P3, P4, P5의 stride
        
        # Bbox regression head
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3),
                Conv(c2, c2, 3),
                nn.Conv2d(c2, 4 * self.reg_max, 1)
            ) for x in ch
        )
        
        # Classification head
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c3, 3),
                Conv(c3, c3, 3),
                nn.Conv2d(c3, self.nc, 1)
            ) for x in ch
        )
        
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
    
    def forward(self, x):
        """Concatenate and return predicted bounding boxes and class probabilities."""
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        
        if self.training:
            return x
        
        # Inference
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = self.dfl(box)
        
        return torch.cat((dbox, cls.sigmoid()), 1), x


class DFL(nn.Module):
    """Distribution Focal Loss (DFL)"""
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)

class YOLOv9t(nn.Module):
    """YOLOv9-t model implementation"""
    def __init__(self, ch=3, nc=80):
        super().__init__()
        self.nc = nc
        self.model = nn.Sequential()
        # GELAN backbone
        # Layer 0: Conv [16, 3, 2] - P1/2
        self.conv0 = Conv(ch, 16, 3, 2)
        
        # Layer 1: Conv [32, 3, 2] - P2/4
        self.conv1 = Conv(16, 32, 3, 2)
        
        # Layer 2: ELAN1 [32, 32, 16]
        self.elan1 = ELAN1(32, 32, 32, 16)
        
        # Layer 3: AConv [64] - P3/8
        self.aconv3 = AConv(32, 64)  # 32 -> 64
        
        # Layer 4: RepNCSPELAN4 [64, 64, 32, 3]
        self.rep_elan4 = RepNCSPELAN4(64, 64, 64, 32, 3)
        
        # Layer 5: AConv [96] - P4/16
        self.aconv5 = AConv(64, 96)  # 64 -> 96
        
        # Layer 6: RepNCSPELAN4 [96, 96, 48, 3]
        self.rep_elan6 = RepNCSPELAN4(96, 96, 96, 48, 3)
        
        # Layer 7: AConv [128] - P5/32
        self.aconv7 = AConv(96, 128)  # 96 -> 128
        
        # Layer 8: RepNCSPELAN4 [128, 128, 64, 3]
        self.rep_elan8 = RepNCSPELAN4(128, 128, 128, 64, 3)
        
        # Layer 9: SPPELAN [128, 64]
        self.sppelan = SPPELAN(128, 128, 64)
        
        # Head part
        # Layer 10: Upsample
        self.up1 = nn.Upsample(None, 2, 'nearest')
        
        # Layer 11: Concat with backbone P4 (layer 6 output)
        self.concat1 = Concat(1)
        
        # Layer 12: RepNCSPELAN4 [96, 96, 48, 3]
        self.rep_elan12 = RepNCSPELAN4(224, 96, 96, 48, 3)  # 128+96=224 input channels
        
        # Layer 13: Upsample
        self.up2 = nn.Upsample(None, 2, 'nearest')
        
        # Layer 14: Concat with backbone P3 (layer 4 output)
        self.concat2 = Concat(1)
        
        # Layer 15: RepNCSPELAN4 [64, 64, 32, 3]
        self.rep_elan15 = RepNCSPELAN4(160, 64, 64, 32, 3)  # 96+64=160 input channels

        # Layer 16: AConv [48]
        self.aconv16 = AConv(64, 48)  # 64 -> 48
        
        # Layer 17: Concat with head P4 (layer 12 output)
        self.concat3 = Concat(1)
        # Layer 18: RepNCSPELAN4 [96, 96, 48, 3] - P4/16
        self.rep_elan18 = RepNCSPELAN4(144, 96, 96, 48, 3)  # 48+96=144 input channels
        
        # Layer 19: AConv [64]
        self.aconv19 = AConv(96, 64)  # 96 -> 64
        # Layer 20: Concat with head P5 (layer 9 output)
        self.concat4 = Concat(1)
        # Layer 21: RepNCSPELAN4 [128, 128, 64, 3] - P5/32
        self.rep_elan21 = RepNCSPELAN4(192, 128, 128, 64, 3)  # 64+128=192 input channels
        
        # Layer 22: Detect head
        self.detect = Detect(nc, (64, 96, 128))
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass"""
        # Backbone
        x = self.conv0(x)  # 0-P1/2 
        x = self.conv1(x)  # 1-P2/4
        x = self.elan1(x)  # 2

        P3 = self.aconv3(x)  # 3-P3/8
        x = self.rep_elan4(P3)  # 4
        P4 = self.aconv5(x) # 5-P4/16
        x = self.rep_elan6(P4)  # 6
        P5 = self.aconv7(x) # 7-P5/32
        x = self.rep_elan8(P5) # 8
        p5 = self.sppelan(x) # 9
        
        # Head - Bottom-up path
        x = self.up1(p5)  # 10-Upsample P5
        x = self.concat1([x, P4])  # 11-Concat with P4
        x = self.rep_elan12(x)  # 12-N4 feature
        x = self.up2(x)  # 13-Upsample N4
        x = self.concat2([x, P3])  # 14-Concat with P3
        n3 = self.rep_elan15(x)  # 15-N3 feature (P3 output)
        x = self.aconv16(n3) # 16
        x = self.concat3([x, P4])  # 17-Concat with N4
        n4_out = self.rep_elan18(x)  # 18-N4 output (P4 output)
        x = self.aconv19(n4_out) # 19
        x = self.concat4([x, P5])  # 20-Concat with P5
        n5_out = self.rep_elan21(x)  # 21-N5 output (P5 output)
        #22 Detection head
        outputs = [n3, n4_out, n5_out]  # P3, P4, P5

        if self.training:
            return self.detect(outputs)
        else:
            # 22
            # Return both formats like ultralytics
            inference_out, raw_outputs = self.detect(outputs)
            return inference_out, raw_outputs

    def fuse(self):
        """Fuse Conv2d + BatchNorm2d layers for inference optimization"""
        for m in self.modules():
            if isinstance(m, Conv) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, 'bn')
                m.forward = m.forward_fuse
            elif isinstance(m, RepConv):
                m.fuse_repvgg_block()
        return self

def fuse_conv_and_bn(conv, bn):
    """Fuse convolution and batchnorm layers"""
    fusedconv = nn.Conv2d(conv.in_channels,
                         conv.out_channels,
                         kernel_size=conv.kernel_size,
                         stride=conv.stride,
                         padding=conv.padding,
                         groups=conv.groups,
                         bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv

def create_yolov9t(nc=80, ch=3):
    """Create YOLOv9-t model"""
    model = YOLOv9t(ch=ch, nc=nc)
    return model


# def transfer_yolov9t_weights(target_model, source_yolo_path='yolov9t.pt'):
#     """
#     Ultralytics YOLOv9t에서 커스텀 YOLOv9t 모델로 weight 전송
    
#     Args:
#         target_model: 커스텀 YOLOv9t 모델 인스턴스
#         source_yolo_path: Ultralytics YOLO 모델 경로
    
#     Returns:
#         target_model: weight가 복사된 모델
#     """
#     print(f"\nLoading source model from {source_yolo_path}...")
#     source_model = YOLO(source_yolo_path)
#     source = source_model.model
    
#     # Get model layers as list
#     source_layers = list(source.model)
    
#     # 레이어 매핑
#     layer_mapping = [
#         ('conv0', 0, 'Conv', copy_conv_weights),
#         ('conv1', 1, 'Conv', copy_conv_weights),
#         ('elan1', 2, 'ELAN1', copy_elan1_weights),
#         ('aconv3', 3, 'AConv', copy_aconv_weights),
#         ('rep_elan4', 4, 'RepNCSPELAN4', copy_repncsp_elan4_weights),
#         ('aconv5', 5, 'AConv', copy_aconv_weights),
#         ('rep_elan6', 6, 'RepNCSPELAN4', copy_repncsp_elan4_weights),
#         ('aconv7', 7, 'AConv', copy_aconv_weights),
#         ('rep_elan8', 8, 'RepNCSPELAN4', copy_repncsp_elan4_weights),
#         ('sppelan', 9, 'SPPELAN', copy_sppelan_weights),
#         ('up1', 10, 'Upsample', None),
#         ('concat1', 11, 'Concat', None),
#         ('rep_elan12', 12, 'RepNCSPELAN4', copy_repncsp_elan4_weights),
#         ('up2', 13, 'Upsample', None),
#         ('concat2', 14, 'Concat', None),
#         ('rep_elan15', 15, 'RepNCSPELAN4', copy_repncsp_elan4_weights),
#         ('aconv16', 16, 'AConv', copy_aconv_weights),
#         ('concat3', 17, 'Concat', None),
#         ('rep_elan18', 18, 'RepNCSPELAN4', copy_repncsp_elan4_weights),
#         ('aconv19', 19, 'AConv', copy_aconv_weights),
#         ('concat4', 20, 'Concat', None),
#         ('rep_elan21', 21, 'RepNCSPELAN4', copy_repncsp_elan4_weights),
#         ('detect', 22, 'Detect', copy_detect_weights),
#     ]
    
#     print("\n" + "="*60)
#     print("Starting Weight Transfer")
#     print("="*60 + "\n")
    
#     success_count = 0
#     total_count = 0
    
#     # 레이어별 가중치 복사
#     for target_attr, source_idx, layer_type, copy_func in layer_mapping:
#         print(f"Layer {source_idx:2d}: {target_attr:15s} ({layer_type})")
        
#         if copy_func is None:
#             print("  → No weights to copy\n")
#             continue
        
#         total_count += 1
        
#         try:
#             target_layer = getattr(target_model, target_attr)
#             source_layer = source_layers[source_idx]
            
#             # 가중치 복사
#             if copy_func(target_layer, source_layer):
#                 print("  ✓ Success\n")
#                 success_count += 1
#             else:
#                 print("  ✗ Failed\n")
            
#         except Exception as e:
#             print(f"  ✗ Error: {e}\n")
#             continue
    
#     print("="*60)
#     print(f"Weight Transfer Complete: {success_count}/{total_count} layers")
#     print("="*60 + "\n")
    
#     return target_model


