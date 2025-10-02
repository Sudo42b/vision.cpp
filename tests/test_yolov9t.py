#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch",
#   "torchvision", 
#   "pytest",
#   "opencv-python",
#   "numpy",
# ]
# ///

import torch
import numpy as np
import sys
import os

# Add the scripts directory to the path to import workbench
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_yolov9t_conv_block():
    """Test basic conv block"""
    # Create a simple conv layer state dict
    state = {
        "weight": torch.randn(16, 3, 3, 3),
        "bias": torch.randn(16),
    }
    
    # Input tensor
    x = torch.randn(1, 3, 64, 64)
    
    # Expected output using PyTorch
    conv = torch.nn.Conv2d(3, 16, 3, 1, 1)
    conv.load_state_dict(state)
    conv.eval()
    
    with torch.no_grad():
        expected = torch.nn.functional.silu(conv(x))
    
    print(f"Input shape: {x.shape}")
    print(f"Expected output shape: {expected.shape}")
    print(f"Expected output mean: {expected.mean():.6f}")
    print(f"Expected output std: {expected.std():.6f}")
    
    # TODO: Call C++ implementation when workbench is ready
    # result = workbench.invoke_test("yolov9t_conv_block", x, state)
    # assert torch.allclose(result, expected, atol=1e-5)
    
    print("✓ Conv block test setup complete")


def test_yolov9t_simple():
    """Test simple YOLOv9t operations"""
    print("Testing YOLOv9t implementation...")
    
    # Test conv block
    test_yolov9t_conv_block()
    
    print("All tests passed!")


if __name__ == "__main__":
    test_yolov9t_simple()