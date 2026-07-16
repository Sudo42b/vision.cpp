#!/usr/bin/env python3
"""mmdet_to_pt.py — mmdet config → backbone/neck wrapped nn.Module → .pt 저장 (CLI).

전처리 단계(= "mmdet 을 파일로 뺀다"). g2c 와 무관하게 mmdet 만 여기(vision.cpp/tools/mmdet)서
다룬다. 클래스는 mmdet_wrap.MMDetBackbone → torch.load 가 동일 클래스로 복원(pickle 모듈명 =
'mmdet_wrap'). g2c 로 이 .pt 를 컴파일할 때도 이 폴더가 PYTHONPATH 에 있어야 mmdet_wrap 를 import 함.

사용:
  # (이 폴더 = vision.cpp/tools/mmdet 을 PYTHONPATH 에)
  python mmdet_to_pt.py --config retinanet_r18_fpn_1x_coco.py --out retinanet_bb.pt --size 512
  # 이후 g2c(코어 무수정, mmdet 코드 없음)로 컴파일:
  PYTHONPATH=<GTX_Compiler>:<이 폴더> g2c --model retinanet_bb.pt --name Retina --output output/retina
"""
import argparse
import json
import os
import sys
import torch

# self-contained import: 이 파일 폴더(tools/mmdet)를 경로에 넣고 top-level 모듈로 import
# → torch.save 피클 모듈경로 = 'mmdet_wrap' (mmdet 라이브러리와 이름 충돌 없음).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mmdet_wrap import MMDetBackbone, build   # noqa: E402,F401 (MMDetBackbone: 피클 등록)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mmdet_to_pt")
    ap.add_argument("--config", required=True, help="mmdet config .py")
    ap.add_argument("--checkpoint", default=None, help="가중치(.pth) — 없으면 config init")
    ap.add_argument("--out", required=True, help="출력 .pt")
    ap.add_argument("--size", type=int, default=512)
    a = ap.parse_args(argv)

    m, shapes, cfg = build(a.config, a.checkpoint, a.size)
    print(f"  backbone/neck features: {shapes}")
    torch.save(m, a.out)   # MMDetBackbone.__module__ == 'mmdet_wrap' → 어디서든 로드 가능
    n_head = sum(1 for k in m.state_dict() if k.startswith("bbox_head"))
    print(f"  → saved: {a.out}  (state_dict {len(m.state_dict())} tensors, head {n_head} 포함)")

    # decode/anchor + head-conv config 사이드카 (vision.cpp mmdet 부품이 읽음)
    sidecar = os.path.splitext(a.out)[0] + ".postproc.json"
    with open(sidecar, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  → sidecar: {sidecar}  (head_type={cfg.get('head_type')})")


if __name__ == "__main__":
    main()
