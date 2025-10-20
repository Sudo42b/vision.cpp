import torch
import numpy as np
import os

pth_path = "/home/sw.lee/supergate/GTX_PY/yolov9/yolov9t_fused.pth"
save_dir = "/home/sw.lee/supergate/vision.cpp/tests"
os.makedirs(save_dir, exist_ok=True)

# load pth
state = torch.load(pth_path, map_location="cpu", weights_only=False)
layer = "model.0.conv.weight"

# numpy → tensor 자동 변환
W = state[layer]
if isinstance(W, np.ndarray):
    W = torch.from_numpy(W)
W = W.float()

# bias도 numpy인 경우 변환
B = state.get(layer.replace("weight", "bias"), np.zeros(W.shape[0], dtype=np.float32))
if isinstance(B, np.ndarray):
    B = torch.from_numpy(B)
B = B.float()

# dummy input
X = torch.randn(1, W.shape[1], 32, 32)

# fused PyTorch Conv 결과
Y = torch.nn.functional.conv2d(X, W, B, stride=1, padding=1)

# 저장
np.savez(f"{save_dir}/conv_test_ref.npz",
         input=X.numpy().astype(np.float32),
         output_ref=Y.numpy().astype(np.float32))
print(f"✅ Saved: {save_dir}/conv_test_ref.npz")
