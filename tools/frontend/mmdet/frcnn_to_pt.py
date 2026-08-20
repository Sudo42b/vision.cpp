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
import mmdet_compat  # noqa: E402
from frcnn_wrap import (FRCNN_SubA, FRCNN_SubB, MaskRCNN_SubC, MSRCNN_SubD,  # noqa: E402,F401
                        GridRCNN_SubE, frcnn_cfg)                           # (피클: frcnn_wrap)


def _desync_norm(cfg_path):
    """config 의 `SyncBN`/`MMSyncBN` 을 `BN` 으로 바꾼 Config 를 돌려준다.

    ⚠️ **모델을 만든 뒤에 되돌리면 늦다.** `mmcv.ops.SyncBatchNorm.__init__` 이 생성 시점에
    `dist.get_world_size()` 를 부르므로, 단일 프로세스에서는 `init_process_group` 을
    요구하며 **build 중에** 죽는다(`mmengine.revert_sync_batchnorm` 은 이미 만들어진 모듈만
    바꾼다). simple_copy_paste 가 그렇게 "export 실패" 로 남았다 — 못 하는 것이 아니라
    **안 재본 것**이었다.

    추론 수치는 BN 과 같다(둘 다 저장된 running stats 를 쓴다). 바꾸는 것은 학습 시 통계를
    프로세스 간에 합치느냐뿐이다.
    """
    from mmengine.config import Config
    cfg = Config.fromfile(cfg_path) if isinstance(cfg_path, str) else cfg_path

    def walk(o):
        if isinstance(o, dict):
            if o.get("type") in ("SyncBN", "MMSyncBN"):
                o["type"] = "BN"
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)

    walk(cfg._cfg_dict)
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser(prog="frcnn_to_pt")
    ap.add_argument("--config", required=True, help="Faster R-CNN config .py")
    ap.add_argument("--checkpoint", default=None, help="가중치(.pth)")
    ap.add_argument("--out", required=True, help="출력 디렉토리")
    ap.add_argument("--size", type=int, default=800)
    a = ap.parse_args(argv)
    from mmdet.apis import init_detector
    # `.eval()` 체이닝 금지 — train() 을 오버라이드한 계열에서 None 이 돌아온다.
    det = init_detector(_desync_norm(a.config), a.checkpoint, device="cpu")
    det.eval()
    os.makedirs(a.out, exist_ok=True)

    # 클래스가 `frcnn_wrap` 이름으로 절여진다 → 로더가 import 할 수 있게 같이 둔다.
    for f in mmdet_compat.install_loader_modules(a.out, "frcnn_wrap"):
        print(f"  → loader module: {f}")
    # ⚠️ **저장 전에 dummy forward 를 한 번 돈다.** eager forward 가 있어야 만들어지는
    #    상수 버퍼가 있다 — swin 의 shifted-window 어텐션 마스크(`_visp_attn_mask`,
    #    mmdet_compat._patch_swin_mask)가 그렇다. 저장을 먼저 하면 버퍼가 .pt 에 안 실리고,
    #    trace 가 마스크를 0 으로 만들어 **크래시 없이 값만 틀린다**(swin 실측 L1 0.82).
    #    feat_hw 도 같은 forward 에서 얻는다(neck 없는 C4 계열은 건너뛴다 — 아래 참고).
    with torch.no_grad():
        feats = det.backbone(torch.zeros(1, 3, a.size, a.size))
        if getattr(det, "neck", None) is not None:
            feats = det.neck(feats)
    torch.save(FRCNN_SubA(det).eval(), f"{a.out}/FRCNN_SubA.pt")   # 이미지 → 14 출력
    from frcnn_wrap import num_bbox_stages, SPARSE_SubB
    ns = num_bbox_stages(det)
    # SparseR-CNN/QueryInst 은 단계마다 query 를 함께 나르므로 전용 래퍼를 쓴다.
    # 파일 이름은 캐스케이드와 같게 둔다 — 러너·하네스가 같은 경로를 탄다.
    if type(det.roi_head).__name__ == "SparseRoIHead":
        for i in range(ns):
            torch.save(SPARSE_SubB(det, i).eval(), f"{a.out}/FRCNN_SubB{i}.pt")
    elif ns == 1:
        torch.save(FRCNN_SubB(det).eval(), f"{a.out}/FRCNN_SubB.pt")   # roi_feat → cls/bbox
    else:
        # 캐스케이드: 단계마다 따로 낸다. 러너가 사이에 호스트 RoIAlign 을 끼워 돈다.
        for i in range(ns):
            torch.save(FRCNN_SubB(det, i).eval(), f"{a.out}/FRCNN_SubB{i}.pt")
    cfg = frcnn_cfg(det, a.size)
    if cfg.get("has_mask"):
        torch.save(MaskRCNN_SubC(det).eval(), f"{a.out}/MaskRCNN_SubC.pt")   # mask_feat → mask_logits
    if cfg.get("has_mask_iou"):
        torch.save(MSRCNN_SubD(det).eval(), f"{a.out}/MSRCNN_SubD.pt")       # (feat|mask) → mask_iou
    if cfg.get("has_grid"):
        torch.save(GridRCNN_SubE(det).eval(), f"{a.out}/GridRCNN_SubE.pt")   # roi_feat → 격자 히트맵
    # feat_hw (P2-P6) — 러너 CWHN flat 해석용. 위 dummy forward 에서 얻었다.
    cfg["feat_hw"] = [[int(f.shape[2]), int(f.shape[3])] for f in feats]
    json.dump(cfg, open(f"{a.out}/frcnn.json", "w"), indent=2)
    print(f"  → {a.out}/FRCNN_SubA.pt, FRCNN_SubB.pt, frcnn.json (feat_hw={cfg['feat_hw']})")


if __name__ == "__main__":
    main()
