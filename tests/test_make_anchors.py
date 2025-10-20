import torch
import numpy as np
from scripts.modules import make_anchors
from . import workbench
from .workbench import to_nhwc, to_nchw

def test_make_anchors_basic():
    # Create synthetic feature maps: two scales
    # torch.Size([1, 144, 80, 80])
    # torch.Size([1, 144, 40, 40])
    # torch.Size([1, 144, 20, 20])
    # feats[0] -> shape (N, C, H=2, W=3)
    # feats[1] -> shape (N, C, H=1, W=2)
    feats = [torch.rand(1, 144, 80, 80, dtype=torch.float32), 
             torch.rand(1, 144, 40, 40, dtype=torch.float32),
             torch.rand(1, 144, 20, 20, dtype=torch.float32)]
    strides = torch.Tensor([8.0, 16.0, 32.0])
    params = dict()
    params['grid_cell_offset'] = 0.5
    anchor_p = dict()
    anchor_p['strides'] = strides
    
    
    for idx, feat in enumerate(feats):
        feats[idx] = to_nhwc(feat)

    workbench.invoke_test("make_anchors", {}, anchor_p, params)
    
    anchor_tensor, stride_tensor = anchor_p['anchors_tensor'], anchor_p['stride_tensor']
    anchor_tensor = to_nchw(anchor_tensor)
    stride_tensor = to_nchw(stride_tensor)

    assert torch.allclose(anchor_tensor, expected_anchors)
    assert torch.allclose(stride_tensor, expected_stride_tensor)
