import torch
import torch.nn as nn
import math
from typing import List, Tuple, Optional, Dict, Any, Union
import re
import numpy as np
from torch.nn import functional as F
from copy import copy
from torch import Tensor
import ggml
from torch import nn
from ggml.utils import to_numpy, from_numpy


import weakref
import atexit


class GGMLConv2d(nn.Conv2d):
    """PyTorch nn.Conv2d와 유사한 인터페이스의 GGML Conv2d 래퍼

    - 입력 x: [W, H, C, N] (F32)  # GGML conv2d 요구사항
    - 가중치 w: [KW, KH, IC, OC] (F32)
    - 출력 y: [OW, OH, OC, N] (F32)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias=bias,
            padding_mode=padding_mode,
        )

        # 메모리 크기 설정 (더 큰 메모리 할당)
        params = ggml.ggml_init_params(mem_size=64 * 1024 * 1024, mem_buffer=None)  # 64MB로 증가
        self.ctx = ggml.ggml_init(params)
        if self.ctx is None:
            raise RuntimeError("GGML context initialization failed")

    @staticmethod
    def _make_cleanup(ctx, ggml_free_func):
        """weakref callback을 생성하는 팩토리"""

        def cleanup(weak_self):
            try:
                if ctx is not None and ggml_free_func is not None:
                    ggml_free_func(ctx)
            except:
                pass

        return cleanup

    def _reset_context_if_needed(self):
        """컨텍스트가 가득 차면 리셋"""
        try:
            # 컨텍스트 메모리 체크 (선택적)
            pass
        except:
            # 문제가 있으면 컨텍스트 재생성
            if self.ctx is not None:
                ggml.ggml_free(self.ctx)
            params = ggml.ggml_init_params(mem_size=128 * 1024 * 1024, mem_buffer=None)
            self.ctx = ggml.ggml_init(params)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        입력: PyTorch [N, C, H, W]
        출력: PyTorch [N, OC, OH, OW]
        """
        # 파라미터 추출
        N, C, H, W = x.shape
        kw, kh = (
            self.kernel_size
            if isinstance(self.kernel_size, tuple)
            else (self.kernel_size, self.kernel_size)
        )
        ic, oc = self.in_channels, self.out_channels
        sx, sy = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)
        px, py = self.padding if isinstance(self.padding, tuple) else (self.padding, self.padding)
        dx, dy = (
            self.dilation if isinstance(self.dilation, tuple) else (self.dilation, self.dilation)
        )

        # 출력 크기 계산 및 검증
        ow = (W + 2 * px - dx * (kw - 1) - 1) // sx + 1
        oh = (H + 2 * py - dy * (kh - 1) - 1) // sy + 1
        if ow <= 0 or oh <= 0:
            raise ValueError(f"Invalid output dimensions")

        x_np = x.detach().cpu().float().numpy().astype(np.float32)
        x_np = np.ascontiguousarray(x_np)
        I = from_numpy(x_np, self.ctx)

        w_np = self.weight.detach().cpu().float().numpy().astype(np.float32)
        w_np = np.ascontiguousarray(w_np)
        WEG = from_numpy(w_np, self.ctx)

        y = ggml.ggml_conv_2d(self.ctx, WEG, I, sx, sy, px, py, dx, dy)
        # 바이어스 추가 (수정됨)
        if self.bias is not None:
            b_np = self.bias.detach().cpu().float().numpy().astype(np.float32)
            b_np = np.ascontiguousarray(b_np)

            # 바이어스를 4D로 reshape: [OC] → [1, OC, 1, 1]
            # from_numpy가 reverse하므로 GGML에서는 [1, 1, OC, 1]이 됨
            b_np_4d = b_np.reshape(1, oc, 1, 1)
            B = from_numpy(b_np_4d, self.ctx)

            # Repeat: [1, 1, OC, 1] → [OW, OH, OC, N]
            B_expanded = ggml.ggml_repeat(self.ctx, B, y)
            y = ggml.ggml_add(self.ctx, y, B_expanded)

        gf = ggml.ggml_new_graph(self.ctx)
        ggml.ggml_build_forward_expand(gf, y)
        ggml.ggml_graph_compute_with_ctx(self.ctx, gf, 1)

        y_np = to_numpy(y)

        if y_np.shape == (oc, oh, ow):
            y_np = y_np.reshape(N, oc, oh, ow)
        elif y_np.shape == (ow, oh, oc, N):
            y_np = np.transpose(y_np, (3, 2, 1, 0))

        return torch.from_numpy(np.ascontiguousarray(y_np)).float().to(x.device)

    def __del__(self):
        """완전히 안전한 cleanup - 속성 설정 없음"""
        try:
            ctx = object.__getattribute__(self, "ctx")
            if ctx is not None:
                # ggml 모듈이 아직 살아있는지 확인
                import sys

                ggml_module = sys.modules.get("ggml")
                if ggml_module is not None and hasattr(ggml_module, "ggml_free"):
                    ggml_module.ggml_free(ctx)
        except:
            pass


def test_conv2d():
    """GGML conv2d 테스트 및 PyTorch 비교"""
    print("=== GGML Conv2D vs PyTorch 비교 테스트 ===")
    in_channels = 5
    out_channels = 16
    kernel_size = (3, 3)
    stride = (1, 1)
    padding = 1
    dilation = (1, 1)

    # 더 작은 입력으로 테스트 시작
    torch.manual_seed(42)
    INPUT = torch.randn(1, in_channels, 64, 64)  # 5x5 -> 4x4로 줄임
    print(f"Input tensor shape: {INPUT.shape}")

    print("\n--- PyTorch Conv2d 먼저 생성 (참조용) ---")
    pytorch_conv = torch.nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=False,
    )
    print("\n--- GGML Conv2d 생성 및 가중치 동기화 ---")
    ggml_conv = GGMLConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=False,
    )

    # PyTorch 가중치를 GGML에 복사
    with torch.no_grad():
        ggml_conv.weight.copy_(pytorch_conv.weight)
        if pytorch_conv.bias is not None and ggml_conv.bias is not None:
            ggml_conv.bias.copy_(pytorch_conv.bias)

    print("\n--- Pytorch Forward Pass 실행 ---")
    pytorch_output = pytorch_conv(INPUT)
    print(f"PyTorch: {INPUT.shape}->{pytorch_output.shape}")
    print(f"PyTorch 출력 dtype: {pytorch_output.dtype}\n")

    ggml_output = ggml_conv(INPUT)

    if ggml_output is not None:
        print(f"GGML: {INPUT.shape}->{ggml_output.shape}")
        print(f"GGML 출력 dtype: {ggml_output.dtype}")

        print("\nGGML 출력 값:")
        print(ggml_output)
        print("\nPyTorch 출력 값:")
        print(pytorch_output)

        # PyTorch 출력은 [N, OC, OH, OW]
        # GGML 출력은 [N, OC, OW, OH] 일 수 있으므로 비교를 위해 차원 순서 조정
        if ggml_output.shape != pytorch_output.shape:
            # [N, OC, OW, OH] -> [N, OC, OH, OW]
            ggml_output = ggml_output.permute(0, 1, 3, 2)
            print(f"GGML output permuted to: {ggml_output.shape}")

    else:
        print("GGML 출력이 None입니다.")

    print("\n=== 비교 테스트 완료 ===")


# Conv2d = GGMLConv2d
Conv2d = nn.Conv2d


def dist2bbox(distance: Tensor, anchor_points: Tensor, xywh: bool = True, dim: int = -1) -> Tensor:
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = torch.split(distance, 2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.concatenate((c_xy, wh), dim)  # xywh bbox
    return torch.concatenate((x1y1, x2y2), dim)  # xyxy bbox


def make_anchors(
    feats: Tensor, strides: Tensor, grid_cell_offset: float = 0.5
) -> Tuple[Tensor, Tensor]:
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).reshape(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), fill_value=stride, dtype=dtype, device=device))
    return torch.concatenate(anchor_points), torch.concatenate(stride_tensor)


def autopad(k, p=None, d=1):
    """자동 패딩 계산"""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """in_channels, out_channels, kernel_size, stride, padding, groups, dilation, activation"""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03, affine=True)
        self.act = (
            nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        )

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        return x


class AConv(nn.Module):
    """Average pooling convolution for downsampling"""

    def __init__(self, c1, c2):
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)  # 단순한 3x3 stride=2 컨볼루션
        self.avgpool = nn.AvgPool2d(2, 1, 0, False, True)

    def forward(self, x):
        # print(f"AConv In: {x.shape}")
        pool = self.avgpool(x)
        # print(f"AConv Pool: {pool.shape}, k={self.avgpool.kernel_size}, s={self.avgpool.stride}, p={self.avgpool.padding}")
        # print(f"AConv cv1: c1={self.cv1.conv.in_channels}, c2={self.cv1.conv.out_channels}, k={self.cv1.conv.kernel_size}, s={self.cv1.conv.stride}, p={self.cv1.conv.padding}, g={self.cv1.conv.groups}, d={self.cv1.conv.dilation}, act={self.cv1.act}")
        x = self.cv1(pool)
        # print(f"AConv Out: {x.shape}")
        return x


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
        # version 1

        # # 채널 차원에서 분할
        # cv1_out = self.cv1(x)

        # # 채널 차원에서 분할
        # y1 = cv1_out[:, :self.c, :, :]
        # y2_input = cv1_out[:, self.c:, :, :]

        # # 순차적으로 cv2, cv3 적용
        # y2 = self.cv2(y2_input)
        # y3 = self.cv3(y2)

        # # 4개 모두 연결 (y2_input도 포함!)
        # y = torch.cat([y1, y2_input, y2, y3], dim=1)
        # print(y.shape)
        # exit()
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
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ELAN1 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        # print(f"ELAN1 cv1: c1={self.cv1.conv.in_channels}, c2={self.cv1.conv.out_channels}, k={self.cv1.conv.kernel_size}, s={self.cv1.conv.stride}, p={self.cv1.conv.padding}, g={self.cv1.conv.groups}, d={self.cv1.conv.dilation}, act={self.cv1.act}")
        # print(f"ELAN1 cv2: c2={self.cv2.conv.in_channels}, c2={self.cv2.conv.out_channels}, k={self.cv2.conv.kernel_size}, s={self.cv2.conv.stride}, p={self.cv2.conv.padding}, g={self.cv2.conv.groups}, d={self.cv2.conv.dilation}, act={self.cv2.act}")
        # print(f"ELAN1 cv3: c3={self.cv3.conv.in_channels}, c3={self.cv3.conv.out_channels}, k={self.cv3.conv.kernel_size}, s={self.cv3.conv.stride}, p={self.cv3.conv.padding}, g={self.cv3.conv.groups}, d={self.cv3.conv.dilation}, act={self.cv3.act}")
        # print(f"ELAN1 cv4: c4={self.cv4.conv.in_channels}, c4={self.cv4.conv.out_channels}, k={self.cv4.conv.kernel_size}, s={self.cv4.conv.stride}, p={self.cv4.conv.padding}, g={self.cv4.conv.groups}, d={self.cv4.conv.dilation}, act={self.cv4.act}")
        x = self.cv4(torch.cat(y, 1))
        # print(f"ELAN1 Out: {x.shape}")
        return x


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        k: Tuple[int, int] = (3, 3),
        e: float = 0.5,
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


class RepConv(nn.Module):
    """
    RepConv module with training and deploy modes.

    This module is used in RT-DETR and can fuse convolutions during inference for efficiency.

    Attributes:
        conv1 (Conv): 3x3 convolution.
        conv2 (Conv): 1x1 convolution.
        bn (BatchNorm2d, optional): Batch normalization for identity branch.
        act (nn.Module): Activation function.
        default_act (nn.Module): Default activation function (SiLU).

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
        self.act = (
            self.default_act
            if act is True
            else act
            if isinstance(act, nn.Module)
            else nn.Identity()
        )

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

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

    def forward_fuse(self, x):
        """
        Forward pass for deploy mode.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))

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
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

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
        self.conv = Conv2d(
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


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        k: Tuple[int, int] = (3, 3),
        e: float = 0.5,
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


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(
        self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5
    ):
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
        self.m = nn.Sequential(
            *(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CSP bottleneck with 3 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class RepCSP(C3):
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""

    def __init__(
        self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5
    ):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CSP bottleneck with 3 convolutions."""
        x = self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
        # print(f"RepCSP Out: {x.shape}")
        return x


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
        # y = [self.cv1(x), self.cv2(self.cv1(x)), self.cv3(self.cv2(self.cv1(x))), self.cv4(self.cv3(self.cv2(self.cv1(x))))]
        # y = torch.cat(y, 1)
        # print(y.shape)
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


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


from torch import Tensor
import copy


class Detect(nn.Module):
    """YOLOv9 Detect head"""

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    format = None  # export format
    end2end = False  # end2end
    max_det = 300  # max_det
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init
    legacy = True  # backward compatibility for v3/v5/v8/v9 models
    xyxy = False  # xyxy or xywh output

    def __init__(self, nc=80, ch=(64, 96, 128)):
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build

        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))

        self.stride = torch.tensor([8.0, 16.0, 32.0])  # P3, P4, P5의 stride

        # Bbox regression head
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), Conv2d(c2, 4 * self.reg_max, 1, 1))
            for x in ch
        )

        # Classification head
        # self.cv3 = nn.ModuleList(
        #     nn.Sequential(
        #         Conv(x, c3, 3),
        #         Conv(c3, c3, 3),
        #         Conv2d(c3, self.nc, 1)
        #     ) for x in ch
        # )
        self.cv3 = (
            nn.ModuleList(
                nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), Conv2d(c3, self.nc, 1)) for x in ch
            )
            # if self.legacy
            # else nn.ModuleList(
            #     nn.Sequential(
            #         nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
            #         nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
            #         Conv2d(c3, self.nc, 1),
            #     )
            #     for x in ch
            # )
        )

        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        if self.end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    def forward(self, x: List[torch.Tensor]) -> Union[List[torch.Tensor], Tuple]:
        """Concatenate and return predicted bounding boxes and class probabilities."""

        # if self.end2end:
        #     return self.forward_end2end(x)
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        # if self.training:  # Training path
        #     return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def forward_end2end(self, x: List[torch.Tensor]) -> Union[dict, Tuple]:
        """
        Perform forward pass of the v10Detect module.

        Args:
            x (List[torch.Tensor]): Input feature maps from different levels.

        Returns:
            outputs (dict | tuple): Training mode returns dict with one2many and one2one outputs.
                Inference mode returns processed detections or tuple with detections and raw outputs.
        """
        x_detach = [xi.detach() for xi in x]
        one2one = [
            torch.cat((self.one2one_cv2[i](x_detach[i]), self.one2one_cv3[i](x_detach[i])), 1)
            for i in range(self.nl)
        ]
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return {"one2many": x, "one2one": one2one}

        y = self._inference(one2one)
        y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)
        return y if self.export else (y, {"one2many": x, "one2one": one2one})

    def _inference(self, x: List[torch.Tensor]) -> torch.Tensor:
        """
        Decode predicted bounding boxes and class probabilities based on multiple-level feature maps.

        Args:
            x (List[torch.Tensor]): List of feature maps from different detection layers.

        Returns:
            (torch.Tensor): Concatenated tensor of decoded bounding boxes and class probabilities.
        """
        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
        self.shape = shape
        # if self.format != "imx" and (self.dynamic or self.shape != shape):
        #     self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
        #     self.shape = shape

        if self.export and self.format in {
            "saved_model",
            "pb",
            "tflite",
            "edgetpu",
            "tfjs",
        }:  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]

        else:
            # torch.Size([1, 144, 8400]) -> torch.Size([1, 64, 8400]) torch.Size([1, 80, 8400])
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(
                1, 4, 1
            )
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:

            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides
            # torch.Size([1, 64, 8400]) -> torch.Size([1, 4, 8400])

        if self.export and self.format == "imx":
            return dbox.transpose(1, 2), cls.sigmoid().permute(0, 2, 1)
        # 
        return torch.cat((dbox, cls.sigmoid()), 1)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(
                5 / m.nc / (640 / s) ** 2
            )  # cls (.01 objects, 80 classes, 640 img)
        if self.end2end:
            for a, b, s in zip(m.one2one_cv2, m.one2one_cv3, m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                b[-1].bias.data[: m.nc] = math.log(
                    5 / m.nc / (640 / s) ** 2
                )  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes: Tensor, anchors: Tensor, xywh: bool = True) -> Tensor:
        """Decode bounding boxes from predictions."""
        return dist2bbox(bboxes, anchors, xywh=xywh and not (self.end2end or self.xyxy), dim=1)

    @staticmethod
    def postprocess(preds: Tensor, max_det: int, nc: int = 80) -> Tensor:
        """
        Post-process YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc) with last dimension
                format [x, y, w, h, class_probs].
            max_det (int): Maximum detections per image.
            nc (int, optional): Number of classes.

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6) and last
                dimension format [x, y, w, h, max_class_prob, class_index].
        """
        batch_size, anchors, _ = preds.shape  # i.e. shape(16,8400,84)
        boxes, scores = preds.split([4, nc], dim=-1)
        index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)
        boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
        scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(min(max_det, anchors))
        i = torch.arange(batch_size)[..., None]  # batch indices
        return torch.cat(
            [boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()], dim=-1
        )


class DFL(nn.Module):
    """Distribution Focal Loss (DFL)"""

    def __init__(self, c1=16):
        super().__init__()
        self.conv = Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
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
        self.up1 = nn.Upsample(None, 2, "nearest")

        # Layer 11: Concat with backbone P4 (layer 6 output)
        self.concat1 = Concat(1)

        # Layer 12: RepNCSPELAN4 [96, 96, 48, 3]
        self.rep_elan12 = RepNCSPELAN4(224, 96, 96, 48, 3)  # 128+96=224 input channels

        # Layer 13: Upsample
        self.up2 = nn.Upsample(None, 2, "nearest")

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
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
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
        P4 = self.aconv5(x)  # 5-P4/16
        x = self.rep_elan6(P4)  # 6
        P5 = self.aconv7(x)  # 7-P5/32
        x = self.rep_elan8(P5)  # 8
        p5 = self.sppelan(x)  # 9

        # Head - Bottom-up path
        x = self.up1(p5)  # 10-Upsample P5
        x = self.concat1([x, P4])  # 11-Concat with P4
        x = self.rep_elan12(x)  # 12-N4 feature
        x = self.up2(x)  # 13-Upsample N4
        x = self.concat2([x, P3])  # 14-Concat with P3
        n3 = self.rep_elan15(x)  # 15-N3 feature (P3 output)
        x = self.aconv16(n3)  # 16
        x = self.concat3([x, P4])  # 17-Concat with N4
        n4_out = self.rep_elan18(x)  # 18-N4 output (P4 output)
        x = self.aconv19(n4_out)  # 19
        x = self.concat4([x, P5])  # 20-Concat with P5
        n5_out = self.rep_elan21(x)  # 21-N5 output (P5 output)
        # 22 Detection head
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
            if isinstance(m, Conv) and hasattr(m, "bn"):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, "bn")
                m.forward = m.forward_fuse
            elif isinstance(m, RepConv):
                m.fuse_repvgg_block()
        return self


def fuse_conv_and_bn(conv, bn):
    """Fuse convolution and batchnorm layers"""
    fusedconv = (
        Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            groups=conv.groups,
            bias=True,
        )
        .requires_grad_(False)
        .to(conv.weight.device)
    )

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    b_conv = (
        torch.zeros(conv.weight.size(0), device=conv.weight.device)
        if conv.bias is None
        else conv.bias
    )
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def create_yolov9t(nc=80, ch=3):
    """Create YOLOv9-t model"""
    model = YOLOv9t(ch=ch, nc=nc)
    return model


# 사용 예제
def parse_yolo_model(model=YOLOv9t(ch=3, nc=80), source_yolo_path="yolov9t.pt"):

    # 모델 생성
    custom_model = model
    custom_model.eval()

    # 테스트
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 640, 640)
        output = custom_model(dummy_input)
        if isinstance(output, tuple):
            print(f"\nInference output shape: {output[0].shape}")
        else:
            print(f"\nOutput shape: {output.shape}")


if __name__ == "__main__":
    # load_yolov9t()
    a = torch.randn((1, 3, 640, 640))
    from yolov9t import YOLOv9t_Seq

    target_model = YOLOv9t_Seq(nc=80, ch=3).eval()

    t_model = torch.load("yolov9t_converted.pth", weights_only=False)
    target_model.load_state_dict(t_model)
    from ultralytics import YOLO

    ultra_model = YOLO("yolov9t.pt").model

    # 2. 각 레이어의 weight 비교
    print("=== Layer-by-layer Weight Comparison ===\n")

    # Ultralytics 레이어 접근
    ultra_layers = list(ultra_model.model)

    # 3. State dict 키 비교
    print("=== State Dict Keys ===")
    custom_keys = set(target_model.state_dict().keys())
    ultra_keys = set(ultra_model.state_dict().keys())

    print(f"Custom model keys: {len(custom_keys)}")
    print(f"Ultra model keys: {len(ultra_keys)}")

    # 누락된 키 확인
    missing_in_custom = ultra_keys - custom_keys
    missing_in_ultra = custom_keys - ultra_keys

    if missing_in_custom:
        print(f"\nMissing in custom (first 10): {list(missing_in_custom)[:10]}")
    if missing_in_ultra:
        print(f"\nMissing in ultra (first 10): {list(missing_in_ultra)[:10]}")

    # 4. 중간 출력 확인
    print("\n=== Intermediate Outputs ===")
    x = torch.randn(1, 3, 640, 640)

    with torch.no_grad():
        # Custom 모델 중간 출력
        custom_x = target_model.conv0(x)
        print(
            f"Custom after conv0: mean={custom_x.mean().item():.6f}, std={custom_x.std().item():.6f}"
        )

        # Ultra 모델 중간 출력
        ultra_x = ultra_layers[0](x)
        print(
            f"Ultra after layer[0]: mean={ultra_x.mean().item():.6f}, std={ultra_x.std().item():.6f}"
        )

        print(f"Difference: {torch.abs(custom_x - ultra_x).max().item()}")

    # 5. 최종 출력 형태 확인
    print("\n=== Final Output Shapes ===")
    with torch.no_grad():
        custom_out = target_model(x)
        ultra_out = ultra_model(x)

        print(f"Custom output type: {type(custom_out)}")
        if isinstance(custom_out, tuple):
            print(f"Custom output[0] shape: {custom_out[0].shape}")
            print(f"Custom output[1] type: {type(custom_out[1])}")

        print(f"\nUltra output type: {type(ultra_out)}")
        if isinstance(ultra_out, tuple):
            print(f"Ultra output[0] shape: {ultra_out[0].shape}")
            print(f"Ultra output[1] type: {type(ultra_out[1])}")
