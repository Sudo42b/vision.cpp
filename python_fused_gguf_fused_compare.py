import torch
import numpy as np
from gguf import GGUFReader
import matplotlib.pyplot as plt
import os
import itertools

pth_path = "/home/sw.lee/supergate/GTX_PY/yolov9/yolov9t_fused.pth"
gguf_path = "/home/sw.lee/supergate/vision.cpp/models/yolov9t_converted.gguf"
save_dir = "/home/sw.lee/supergate/vision.cpp/compare_plots"

os.makedirs(save_dir, exist_ok=True)

def rmse(a, b): return np.sqrt(np.mean((a - b) ** 2))
def mae(a, b):  return np.mean(np.abs(a - b))
def cosine_similarity(a, b):
    return np.dot(a.flatten(), b.flatten()) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)

def find_best_permutation(p, g):
    """모든 4D permutation 테스트 → 최소 RMSE 조합 탐색"""
    if len(g.shape) != 4:
        return g, None, None

    perms = list(itertools.permutations([0,1,2,3]))
    best_rmse, best_perm, best_g = float("inf"), None, None

    for perm in perms:
        try:
            g_perm = np.transpose(g, perm)
            if g_perm.shape == p.shape:
                err = rmse(p, g_perm)
                if err < best_rmse:
                    best_rmse, best_perm, best_g = err, perm, g_perm
        except Exception:
            continue

    return best_g, best_perm, best_rmse

def normalize_scale(p, g):
    """스케일 정규화 (norm 기준으로 보정)"""
    scale = np.linalg.norm(p) / (np.linalg.norm(g) + 1e-12)
    g_scaled = g * scale
    return g_scaled, scale

# 1️⃣ PyTorch 가중치 로드
pth_state = torch.load(pth_path, map_location="cpu", weights_only=False)

# 2️⃣ GGUF 텐서 로드
reader = GGUFReader(gguf_path)
gguf_tensors = {t.name: t for t in reader.tensors}

# 3️⃣ 비교
results = []
for name, pth_tensor in pth_state.items():
    if name not in gguf_tensors:
        continue

    g = gguf_tensors[name]
    p = pth_tensor
    g_arr = g.data.astype(np.float32)

    # layout 변환: HWIO → OIHW (초기 추정)
    if len(g_arr.shape) == 4:
        g_arr = np.transpose(g_arr, (3, 2, 0, 1))

    # shape 불일치 시 permutation 자동 탐색
    if len(p.shape) == 4 and p.shape != g_arr.shape:
        g_best, perm, perm_rmse = find_best_permutation(p, g_arr)
        if g_best is not None:
            print(f"🔄 {name} perm fix candidate → {perm}, RMSE={perm_rmse:.6f}")
            g_arr = g_best

    # 스케일 보정 적용
    g_scaled, scale_factor = normalize_scale(p, g_arr)
    scaled_rmse = rmse(p, g_scaled)

    # flatten 후 최소 길이 비교
    min_len = min(p.size, g_scaled.size)
    p = p.flatten()[:min_len]
    g_scaled = g_scaled.flatten()[:min_len]

    err_rmse = rmse(p, g_scaled)
    err_mae = mae(p, g_scaled)
    cos = cosine_similarity(p, g_scaled)

    results.append((name, err_rmse, err_mae, cos, scale_factor))
    print(f"{name:40s} RMSE={err_rmse:.6f}  MAE={err_mae:.6f}  Cosine={cos:.6f}  Scale={scale_factor:.6f}")

# 4️⃣ 전체 평균 요약
if results:
    mean_rmse = np.mean([r[1] for r in results])
    mean_mae = np.mean([r[2] for r in results])
    mean_cos = np.mean([r[3] for r in results])
    mean_scale = np.mean([r[4] for r in results])
    print("\n==== Overall Stats ====")
    print(f"Avg RMSE : {mean_rmse:.6f}")
    print(f"Avg MAE  : {mean_mae:.6f}")
    print(f"Avg Cosine: {mean_cos:.6f}")
    print(f"Avg Scale : {mean_scale:.6f}")

    plt.figure(figsize=(6,3))
    plt.bar(["RMSE", "MAE", "Cosine"], [mean_rmse, mean_mae, mean_cos],
            color=["#6baed6","#fd8d3c","#31a354"])
    plt.title("Average Error Statistics (Scaled)")
    plt.savefig(os.path.join(save_dir, "overall_stats_scaled.png"), dpi=200)
    plt.close()
    print(f"📊 Summary plot saved: {os.path.join(save_dir, 'overall_stats_scaled.png')}")


