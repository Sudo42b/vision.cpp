#!/usr/bin/env python3
"""배포 경로 스윕의 레퍼런스를 만든다 — 원본 체크포인트를 torch 로 돌린 값.

    python3 deploy_reference.py --kind seg --config <cfg> --checkpoint <ckpt> \
        --image <jpg> --size 1024x512 --out <prefix>

g2c 산출물을 쓰지 않는다. 변환이 틀렸을 때 레퍼런스까지 같이 틀리면 검증이 성립하지
않기 때문이다. 전처리는 vision-cli 와 같은 규약으로 맞춘다 — 단순 bilinear 리사이즈 뒤
config 의 mean/std 로 정규화한다(letterbox 없음).

출력: <prefix>.<i>.bin (fp32) + <prefix>.shapes.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

V = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def build(kind, cfg, ckpt, size):
    sys.path.insert(0, V + "/tools/frontend/" + ("mmseg" if kind == "seg" else "mmpose"))
    sys.path.insert(0, V + "/tools/frontend/mmdet")
    if kind == "seg":
        import mmseg_wrap as wrap
    else:
        import mmpose_wrap as wrap
    h, w = size[1], size[0]
    model, _ = wrap.build(cfg, ckpt, (h, w))
    return model, wrap


def preprocess(image_path, size, mean, std):
    """vision-cli 와 같은 규약 — 단순 bilinear 리사이즈 + mean/std 정규화."""
    w, h = size
    img = Image.open(image_path).convert("RGB").resize((w, h), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32)                 # HWC, 0~255
    x = (x - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    x = np.ascontiguousarray(x.transpose(2, 0, 1))[None]  # NCHW
    return torch.from_numpy(x)


def norm_values(cfg, kind):
    """config 의 data_preprocessor 에서 mean/std 를 읽는다. 없으면 ImageNet 값."""
    try:
        from mmengine.config import Config
        c = Config.fromfile(cfg)
        dp = c.get("model", {}).get("data_preprocessor", {}) or {}
        mean = dp.get("mean") or [123.675, 116.28, 103.53]
        std = dp.get("std") or [58.395, 57.12, 57.375]
        return list(mean), list(std)
    except Exception:
        return [123.675, 116.28, 103.53], [58.395, 57.12, 57.375]


def main():
    ap = argparse.ArgumentParser(prog="deploy_reference")
    ap.add_argument("--kind", choices=["seg", "pose"], required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--size", required=True, help="WxH")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    w, h = (int(v) for v in a.size.lower().split("x"))
    model, _ = build(a.kind, a.config, a.checkpoint, (w, h))
    model.eval()
    mean, std = norm_values(a.config, a.kind)
    x = preprocess(a.image, (w, h), mean, std)

    with torch.no_grad():
        outs = model(x)
    if isinstance(outs, torch.Tensor):
        outs = (outs,)
    outs = tuple(outs)

    shapes = []
    for i, t in enumerate(outs):
        arr = t[0].detach().cpu().numpy().astype("float32")
        # ⚠️ **torch 는 CHW 로 내고 `vision-cli` 는 HWC 로 덤프한다.** 돌려놓지 않으면
        #    원소 수가 같아 비교가 그냥 통과하고 값만 무상관이 된다. 무상관인 두 텐서의
        #    상대 L2 는 √2 라 계열마다 1.41 근처가 나오고, 모델별 결함처럼 보인다.
        #    실측(ann): 그대로 대면 relL1 1.3817, HWC 로 돌리면 3.41e-02 로 통과한다.
        if arr.ndim == 3:
            arr = arr.transpose(1, 2, 0)
        shapes.append(list(arr.shape))
        np.ascontiguousarray(arr).tofile(f"{a.out}.{i}.bin")
    json.dump({"shapes": shapes, "layout": "HWC", "mean": mean, "std": std,
               "size": [w, h]},
              open(a.out + ".shapes.json", "w"))
    print(f"REF_OK outputs={len(outs)} size={w}x{h} mean={mean}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
