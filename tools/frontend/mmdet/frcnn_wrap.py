"""frcnn_wrap.py — Faster R-CNN two-stage 를 g2c 로 컴파일 가능한 두 subgraph 로 분해.

  SubA = backbone + FPN + RPN head  (이미지 → P2-P5 + rpn_cls×5 + rpn_bbox×5 = 14 출력)
  SubB = RoI bbox_head (Shared2FC)  (RoIAlign feat (N,256,7,7) → cls(N,81), bbox(N,320))

그 사이(RPN proposals, RoIAlign, 최종 decode+NMS)는 host C++ 부품(postproc: rpn_proposals/
roi_align/detect_roi). 두 wrapper 모두 flat tuple 반환 → g2c 가 out_0.. 다출력으로 컴파일.
피클 모듈명 = frcnn_wrap (self-contained). g2c 코어 무관 — mmdet 만 import.
"""
import torch
import torch.nn as nn


class FRCNN_SubA(nn.Module):
    """이미지 → (P2,P3,P4,P5, rpn_cls×5, rpn_bbox×5) = 14 출력. P2-P5=RoIAlign 입력, rpn=5레벨(P2-P6)."""
    def __init__(self, det):
        super().__init__()
        self.backbone = det.backbone
        self.neck = det.neck
        self.rpn_head = det.rpn_head

    def forward(self, x):
        feats = self.neck(self.backbone(x))          # tuple len 5: P2..P6
        rpn_cls, rpn_bbox = self.rpn_head(feats)     # (list5, list5)
        return tuple(feats[:4]) + tuple(rpn_cls) + tuple(rpn_bbox)


class FRCNN_SubB(nn.Module):
    """RoIAlign feat (N,256,7,7) → (cls_score (N,81), bbox_pred (N,320))."""
    def __init__(self, det):
        super().__init__()
        self.bbox_head = det.roi_head.bbox_head

    def forward(self, roi_feat):
        return self.bbox_head(roi_feat)


class MaskRCNN_SubC(nn.Module):
    """mask RoIAlign feat (M,256,14,14) → mask_logits (M, num_classes, 28, 28). (Mask R-CNN)."""
    def __init__(self, det):
        super().__init__()
        self.mask_head = det.roi_head.mask_head

    def forward(self, mask_feat):
        return self.mask_head(mask_feat)


def frcnn_cfg(det, size=800):
    """host 부품(rpn_proposals/roi_align/detect_roi)용 config 추출 → .frcnn.json."""
    rh = det.rpn_head
    pg = rh.prior_generator
    bc = rh.bbox_coder
    ext = det.roi_head.bbox_roi_extractor
    bh = det.roi_head.bbox_head
    rpn_c, rcnn_c = det.test_cfg.rpn, det.test_cfg.rcnn
    strides = [s[0] if isinstance(s, (tuple, list)) else int(s) for s in pg.strides]
    scales = pg.scales.tolist() if hasattr(pg.scales, "tolist") else list(pg.scales)
    mask = {}
    if getattr(det.roi_head, "with_mask", False):
        mext = det.roi_head.mask_roi_extractor
        mask = {
            "has_mask": True,
            "mask_roi_out": int(mext.roi_layers[0].output_size[0]),   # 14
            "mask_strides": [int(s) for s in mext.featmap_strides],
            "mask_finest_scale": int(mext.finest_scale),
            "mask_thr_binary": float(rcnn_c.mask_thr_binary),
        }
    return {**mask,
        "img_size": int(size),
        # RPN
        "rpn_strides": [float(s) for s in strides],
        "rpn_scale": float(scales[0]),
        "rpn_ratios": [float(r) for r in pg.ratios.tolist()],
        "rpn_means": [float(v) for v in bc.means], "rpn_stds": [float(v) for v in bc.stds],
        "rpn_nms_pre": int(rpn_c.nms_pre), "rpn_nms_thr": float(rpn_c.nms.iou_threshold),
        "rpn_max": int(rpn_c.max_per_img),
        # RoIAlign
        "roi_out": int(ext.roi_layers[0].output_size[0]),
        "roi_strides": [int(s) for s in ext.featmap_strides],
        "roi_finest_scale": int(ext.finest_scale),
        "roi_sampling_ratio": int(ext.roi_layers[0].sampling_ratio),
        "roi_aligned": bool(ext.roi_layers[0].aligned),
        # RCNN decode (detect_roi)
        "num_classes": int(bh.num_classes),
        "rcnn_means": [float(v) for v in bh.bbox_coder.means],
        "rcnn_stds": [float(v) for v in bh.bbox_coder.stds],
        "class_agnostic": bool(getattr(bh, "reg_class_agnostic", False)),
        "rcnn_score_thr": float(rcnn_c.score_thr), "rcnn_nms_thr": float(rcnn_c.nms.iou_threshold),
        "rcnn_max": int(rcnn_c.max_per_img),
    }
