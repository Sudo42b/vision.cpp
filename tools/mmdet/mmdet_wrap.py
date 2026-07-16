"""mmdet_wrap.py — mmdet 검출기를 backbone/neck '정적 forward' nn.Module 로 감싸는 부품.

**클래스 전용 import 모듈**(스크립트로 직접 실행 금지). torch.save 는 클래스를 __module__ 로
피클하므로, 이 모듈에 클래스를 두고 CLI(mmdet_to_pt.py)와 로더(serialize.py)가 **import** 해야
동일 클래스 객체로 로드된다(__main__ 피클 함정 회피).

g2c 무관 — mmdet 만 import. forward = backbone→neck FPN features 까지(head decode/NMS 제외).
bbox_head 는 attribute 로 유지 → state_dict(→gguf)에 head 가중치가 실려 vision.cpp head 부품이 사용.
"""
import torch
import torch.nn as nn


class MMDetBackbone(nn.Module):
    """backbone(+neck) features 만 노출. head 는 가중치 유지용 attribute(forward 미사용)."""
    def __init__(self, det):
        super().__init__()
        self.backbone = det.backbone
        self.neck = det.neck if getattr(det, "with_neck", False) else None
        self.bbox_head = getattr(det, "bbox_head", None)   # 가중치 보존 (state_dict→gguf)

    def forward(self, x):
        f = self.backbone(x)
        if self.neck is not None:
            f = self.neck(f)
        return tuple(f)   # FPN features (레벨별 텐서)


def _tolist(v):
    try:
        return list(v)
    except TypeError:
        return [v]


def postproc_cfg(det):
    """검출 head 의 decode/anchor **config** + head-conv **구조**를 dict 로 추출(→ .postproc.json).

    vision.cpp 의 host decode(detect_anchor)와 C++ head 부품(anchor_head_forward)이 이걸 읽는다.
    g2c pipeline.py 의 _extract_cfg 로직을 여기로 이식(= mmdet 지식을 g2c 밖으로). anchor head 만
    (Delta coder) 지원, 그 외엔 head_type=raw 반환."""
    import torch.nn as nn
    bh = getattr(det, "bbox_head", None)
    if bh is None:
        return {"head_type": "raw"}
    pg = getattr(bh, "prior_generator", None)
    bc = getattr(bh, "bbox_coder", None)
    if pg is None or bc is None or "Delta" not in type(bc).__name__:
        return {"head_type": "raw"}   # anchor(Delta) 만 이번 PoC 지원

    ncls = int(getattr(bh, "cls_out_channels", getattr(bh, "num_classes", 80)))
    strides = [s[0] if isinstance(s, (tuple, list)) else int(s) for s in pg.strides]
    obs = float(getattr(pg, "octave_base_scale", 1.0) or 1.0)
    scales = [float(x) for x in _tolist(getattr(pg, "scales", [obs]))]
    ratios = [float(x) for x in _tolist(getattr(pg, "ratios", [1.0]))]
    num_base = int(_tolist(getattr(pg, "num_base_priors", [len(scales) * len(ratios)]))[0])

    # head-conv 구조 (C++ anchor_head_forward 가 조립) — 최종 cls/reg conv 이름 자동 탐지.
    stacked = len(bh.cls_convs) if hasattr(bh, "cls_convs") else 0
    feat_ch = int(getattr(bh, "feat_channels", 256))
    cls_head = reg_head = None
    for name, mod in bh.named_children():          # retina_cls/retina_reg, atss_cls/atss_reg 등
        if isinstance(mod, nn.Conv2d):
            if mod.out_channels == num_base * ncls:
                cls_head = name
            elif mod.out_channels == num_base * 4:
                reg_head = name
    # head 타워에 norm(GN 등) 있으면 conv 이름이 .conv, 없으면도 .conv (ConvModule) — 기록만.
    has_norm = bool(getattr(bh, "norm_cfg", None))

    # ── 전처리(pre) 메타: mmdet data_preprocessor(모델 안 서브모듈)에서 추출 ──
    #   normalize mean/std(픽셀스케일 0-255) + 채널변환. vision.cpp preprocess() 가 소비.
    #   mmdet channel_conversion=True → mean/std 는 RGB 순(bgr→rgb 후). vision.cpp image_load 는
    #   RGB 로 로드하므로 preprocess to_rgb(=BGR→RGB swap)는 그 반대: to_rgb = not channel_conversion.
    dp = getattr(det, "data_preprocessor", None)
    img_mean, img_std, to_rgb = [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], False
    if dp is not None and getattr(dp, "_enable_normalize", False):
        if getattr(dp, "mean", None) is not None:
            img_mean = [float(v) for v in dp.mean.flatten().tolist()]
        if getattr(dp, "std", None) is not None:
            img_std = [float(v) for v in dp.std.flatten().tolist()]
        to_rgb = not bool(getattr(dp, "_channel_conversion", False))

    return {
        "head_type": "anchor",
        # ── 전처리(pre) — 이미지→텐서 (vision.cpp preprocess) ──
        "img_mean": img_mean,
        "img_std": img_std,
        "to_rgb": to_rgb,
        "use_sigmoid": bool(getattr(bh, "use_sigmoid_cls", True)),
        "num_classes": ncls,
        "strides": [float(s) for s in strides],
        "octave_base_scale": obs,
        "octave_scales": [s / obs for s in scales],   # =2^(i/n)
        "ratios": ratios,
        "center_offset": float(getattr(pg, "center_offset", 0.0) or 0.0),
        "means": [float(v) for v in getattr(bc, "means", [0.0] * 4)],
        "stds": [float(v) for v in getattr(bc, "stds", [1.0] * 4)],
        # ── C++ head 부품(anchor_head_forward)용 구조 ──
        "num_base": num_base,
        "stacked_convs": stacked,
        "feat_channels": feat_ch,
        "cls_convs_prefix": "bbox_head.cls_convs",
        "reg_convs_prefix": "bbox_head.reg_convs",
        "cls_head": "bbox_head." + (cls_head or "retina_cls"),
        "reg_head": "bbox_head." + (reg_head or "retina_reg"),
        "head_has_norm": has_norm,
    }


def build(config, checkpoint=None, size=512):
    """mmdet config(.py) → (MMDetBackbone(eval), feature shapes, postproc cfg)."""
    from mmdet.apis import init_detector       # mmdet 만 import (g2c 무관)
    det = init_detector(config, checkpoint, device="cpu").eval()
    m = MMDetBackbone(det).eval()
    cfg = postproc_cfg(det)
    cfg["img_size"] = int(size)   # 정방 resize 크기 (pre)
    with torch.no_grad():
        outs = m(torch.randn(1, 3, size, size))
    return m, [tuple(o.shape) for o in outs], cfg
