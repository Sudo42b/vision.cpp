#!/usr/bin/env python3
"""frcnn_to_pt.py — Faster R-CNN config → SubA.pt + SubB.pt + frcnn.json (CLI, 전처리 단계).

two-stage 를 g2c 컴파일 가능한 두 subgraph 로 분해해 저장한다(mmdet 지식은 여기만).
이후 g2c 로 각각 컴파일 → run_frcnn 러너가 host op(rpn_proposals/roi_align/detect_roi)과 조립.

사용 (이 폴더 = vision.cpp/tools/frontend/mmdet 을 PYTHONPATH 에):
  python frcnn_to_pt.py --config faster-rcnn_r50_fpn_1x_coco.py --checkpoint frcnn.pth --out /tmp/frcnn
  # → /tmp/frcnn/FRCNN_SubA.pt, FRCNN_SubB.pt, frcnn.json
  # 이후:
  PYTHONPATH=<G2C>:<이 폴더> g2c --model /tmp/frcnn/FRCNN_SubA.pt --name FRCNN_SubA --input-shape 1,3,800,800 --output output/FRCNN_SubA
  PYTHONPATH=<G2C>:<이 폴더> g2c --model /tmp/frcnn/FRCNN_SubB.pt --name FRCNN_SubB --input-shape 4,256,7,7 --output output/FRCNN_SubB
  bash build_frcnn_cpp.sh output/FRCNN_SubA output/FRCNN_SubB
"""
import argparse
import json
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frcnn_wrap import FRCNN_SubA, FRCNN_SubB, MaskRCNN_SubC, frcnn_cfg  # noqa: E402,F401 (피클: frcnn_wrap)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="frcnn_to_pt")
    ap.add_argument("--config", required=True, help="Faster R-CNN config .py")
    ap.add_argument("--checkpoint", default=None, help="가중치(.pth)")
    ap.add_argument("--out", required=True, help="출력 디렉토리")
    ap.add_argument("--size", type=int, default=800)
    a = ap.parse_args(argv)
    from mmdet.apis import init_detector
    # `.eval()` 체이닝 금지 — train() 을 오버라이드한 계열에서 None 이 돌아온다.
    det = init_detector(a.config, a.checkpoint, device="cpu")
    det.eval()
    os.makedirs(a.out, exist_ok=True)

    torch.save(FRCNN_SubA(det).eval(), f"{a.out}/FRCNN_SubA.pt")   # 이미지 → 14 출력
    torch.save(FRCNN_SubB(det).eval(), f"{a.out}/FRCNN_SubB.pt")   # roi_feat → cls/bbox
    cfg = frcnn_cfg(det, a.size)
    if cfg.get("has_mask"):
        torch.save(MaskRCNN_SubC(det).eval(), f"{a.out}/MaskRCNN_SubC.pt")   # mask_feat → mask_logits
    # feat_hw (P2-P6) — 러너 CWHN flat 해석용. dummy forward 로 크기 취득.
    with torch.no_grad():
        feats = det.neck(det.backbone(torch.zeros(1, 3, a.size, a.size)))
    cfg["feat_hw"] = [[int(f.shape[2]), int(f.shape[3])] for f in feats]
    json.dump(cfg, open(f"{a.out}/frcnn.json", "w"), indent=2)
    print(f"  → {a.out}/FRCNN_SubA.pt, FRCNN_SubB.pt, frcnn.json (feat_hw={cfg['feat_hw']})")


if __name__ == "__main__":
    main()
