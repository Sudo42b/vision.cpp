"""
cd /mnt/c/Users/x2199/Desktop/vision.cpp_
pytest -q tests/test_yolov9t.py::test_dfl_layer -q
"""

import torch
import numpy as np
import sys
import os
from torch.nn import Conv2d
from torch import nn
import math
from typing import List, Tuple
from torch import Tensor
from . import workbench
# workbench helpers
from .workbench import to_nchw, to_nhwc, input_tensor
import torch.nn.functional as F
from pathlib import Path

# layout parameter used by the workbench C++ harness
nhwc_layout = dict(memory_layout="nhwc")
# Add the scripts directory to the path to import workbench
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


import pytest

class DFL(nn.Module):
    """Distribution Focal Loss (DFL)"""
    def __init__(self, c1=16):
        super().__init__()
        self.conv = Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)

def test_dfl_layer(c1=16, reg_max=16, debug=False):
    """
    parameter:
        c1(channels 1D): 16
        reg_max(int): 16
    Input/Output:
        input: torch.Size([1, 64, 8400])
        output: torch.Size([1, 4, 8400])
    """
    # Load the DFL conv weight from the provided checkpoint
    # Expected key: 'detect.dfl.conv.weight'
    ckpt_paths = [
        Path(__file__).parent.parent / 'scripts' / 'yolov9t_converted.pth',
        Path.cwd() / 'scripts' / 'yolov9t_converted.pth',
        Path(__file__).parent / '..' / 'scripts' / 'yolov9t_converted.pth',
    ]
    ckpt_file = None
    for p in ckpt_paths:
        if p.exists():
            ckpt_file = p
            break
    if ckpt_file is None:
        raise FileNotFoundError(
            'Checkpoint ./scripts/yolov9t_converted.pth not found. Place the file at repo scripts/ directory.'
        )

    ckpt = torch.load(str(ckpt_file), map_location='cpu')
    key = 'detect.dfl.conv.weight'
    if isinstance(ckpt, dict) and key in ckpt:
        weight = ckpt[key]
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt and key in ckpt['state_dict']:
        weight = ckpt['state_dict'][key]
    else:
        # try to find a matching key
        found = None
        if isinstance(ckpt, dict):
            for k in ckpt.keys():
                if k.endswith('detect.dfl.conv.weight') or k == key:
                    found = ckpt[k]
                    break
        if found is None:
            raise KeyError(f"Key '{key}' not found in checkpoint {ckpt_file}")
        weight = found

    # Ensure weight is a torch tensor
    if not isinstance(weight, torch.Tensor):
        weight = torch.tensor(weight)

    # Map into state for workbench
    state = {key: weight}

    # Use fixed input shape: (1, 64, 8400)
    box3 = input_tensor(1, 64, 8400)
    # bool debug = p.get("debug", 0) != 0;
    # int reg_max = p.get("reg_max", 16);
    # tensor weight = m.weight("detect.dfl.conv.weight");
    params = dict(debug=int(bool(debug)), reg_max=reg_max)
    # pass the state dict (mapping of weight name -> tensor) as the third arg
    # The workbench harness expects a 4D NHWC tensor, so expand a spatial dim
    box_4d = box3.unsqueeze(2)  # shape (N, C, 1, anchors)
    box_nhwc = to_nhwc(box_4d)
    result = workbench.invoke_test("dfl_layer", box_nhwc, state, params)

    # Compute expected using the DFL computation with the loaded weight
    # Following the original DFL implementation:
    # x.view(b, 4, c1, a).transpose(2,1) -> (b, c1, 4, a)
    # softmax over c1 dim, then apply 1x1 conv with loaded weight (shape (1, c1, 1, 1))
    b, C, A = box3.shape
    assert C == 4 * c1, f"expected C==4*c1 ({4*c1}), got {C}"
    x_view = box3.view(b, 4, c1, A).transpose(2, 1)  # (b, c1, 4, a)
    soft = torch.softmax(x_view, dim=1)
    # Ensure weight shape is (out_channels=1, in_channels=c1, 1, 1)
    w = weight
    if w.dim() == 1:
        w = w.view(1, -1, 1, 1)
    elif w.dim() == 4 and w.shape[0] != 1 and w.shape[1] == 1:
        # some conversions store transposed shapes; try to reshape
        w = w.permute(1, 0, 2, 3)
    w = w.to(dtype=soft.dtype)
    expected_conv = F.conv2d(soft, w)
    expected = expected_conv.view(b, 4, A)

    # Normalize possible return types from workbench.invoke_test to shape (N,4,anchors)
    if result is None:
        raise AssertionError("workbench.invoke_test returned no output")

    if not isinstance(result, torch.Tensor):
        raise AssertionError(f"Unexpected result type: {type(result)}")

    # If the C++ side returned a 4D tensor (NHWC), convert it back to NCHW and collapse spatial dims
    if result.dim() == 4:
        # convert NHWC -> NCHW
        result_nchw = to_nchw(result)
        N = result_nchw.size(0)
        C = result_nchw.size(1)
        H = result_nchw.size(2)
        W = result_nchw.size(3)
        # Common possibilities from the C++ side:
        # 1) (N, 4, A1, A2) -> channels already in dim1 (C==4)
        # 2) (N, 1, 4, anchors) -> channels collapsed into dim2
        # 3) (N, 1, H, W) where H*W == 4*anchors -> need to split
        if C == 4:
            result = result_nchw.view(N, 4, -1)
        elif C == 1 and H == 4:
            # (N,1,4,anchors) -> squeeze channel dim
            result = result_nchw.squeeze(1)
        elif C == 1 and (H * W) % 4 == 0:
            anchors = (H * W) // 4
            result = result_nchw.view(N, 4, anchors)
        else:
            # fallback: collapse spatial dims into one dimension
            result = result_nchw.view(N, C, -1)
    elif result.dim() == 3:
        # already (N, C, anchors)
        pass
    else:
        raise AssertionError(f"Unexpected result dimensionality: {result.dim()}")

    assert result.shape == expected.shape, f"shape mismatch result={result.shape} expected={expected.shape}"
    assert torch.allclose(result, expected)




if __name__ == "__main__":
    test_dfl_layer()