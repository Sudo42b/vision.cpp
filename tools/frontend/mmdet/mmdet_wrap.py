"""mmdet_wrap.py — mmdet 검출기를 backbone/neck '정적 forward' nn.Module 로 감싸는 부품.

**클래스 전용 import 모듈**(스크립트로 직접 실행 금지). torch.save 는 클래스를 __module__ 로
피클하므로, 이 모듈에 클래스를 두고 CLI(mmdet_to_pt.py)와 로더(serialize.py)가 **import** 해야
동일 클래스 객체로 로드된다(__main__ 피클 함정 회피).

g2c 무관 — mmdet 만 import. forward = backbone→neck FPN features 까지(head decode/NMS 제외).
bbox_head 는 attribute 로 유지 → state_dict(→gguf)에 head 가중치가 실려 vision.cpp head 부품이 사용.
"""
import torch
import torch.nn as nn


def trace_friendly_ops():
    """trace 가 통째로 삼키는 mmdet 커스텀 op 을 **등가 수식**으로 바꾼다.

    `torch.autograd.Function` 은 forward/backward 를 직접 정의한 불투명 단위라
    `torch.jit.trace` 가 내부를 안 편다 — 그래프에 원자 노드 하나로 남고, 컴파일러는
    `unhandled op '<클래스명>'` 을 낸다. 렌더러를 새로 쓸 일이 아니라 **여기서 풀 일**이다.
    이 클래스들은 학습(역전파 수치안정) 때문에 존재하고 순전파는 등가이기 때문이다.

    호출은 멱등. mmdet 이 없거나 구조가 바뀌었으면 조용히 넘어간다.
    """
    try:
        import mmdet.models.dense_heads.tood_head as _tood
    except Exception:
        return
    # TOOD. mmdet docstring 이 직접 밝힌다 — "substitutes the autograd function of
    # (x.sigmoid() * y.sigmoid()).sqrt()". 학습용 해석적 gradient 라 추론 값은 같다.
    if getattr(_tood.sigmoid_geometric_mean, "__module__", "") != __name__:
        def sigmoid_geometric_mean(x, y):
            return (x.sigmoid() * y.sigmoid()).sqrt()
        _tood.sigmoid_geometric_mean = sigmoid_geometric_mean


def allow_mmengine_checkpoint_globals():
    """mmengine 이 체크포인트에 같이 넣는 클래스를 torch 의 안전 목록에 올린다.

    PyTorch 2.6 부터 `torch.load` 의 `weights_only` 기본값이 True 다. mmdet v3 체크포인트는
    학습 메타(`HistoryBuffer` — 손실 이력)를 함께 담고 있어 그대로는 로드가 거부된다.

    **`weights_only=False` 로 되돌리지 않는다.** 그건 파일 전체에 임의 코드 실행을 허용하는
    것이라 검증 도구가 열어줄 문이 아니다. 필요한 클래스만 이름으로 허용한다.
    """
    import numpy as np
    allow = []
    try:
        from mmengine.logging.history_buffer import HistoryBuffer
        allow.append(HistoryBuffer)            # 손실 이력 (mmdet v3 체크포인트)
    except Exception:
        pass
    # 그 이력이 numpy 배열을 담고 있어 배열 재구성 함수도 함께 필요하다.
    # **데이터 생성자만** 올린다 — 임의 호출 가능한 것을 올리는 게 아니다.
    #
    # ⚠️ **피클에 적힌 이름으로** 등록해야 한다. numpy 2 는 내부를 `numpy._core` 로 옮겼는데
    #    오래된 체크포인트는 `numpy.core.multiarray._reconstruct` 로 적혀 있다. 객체만 넘기면
    #    torch 가 `obj.__module__` 로 이름을 만들어 새 경로로 등록하고, 대조가 안 돼 계속 막힌다.
    #    `(객체, "전체이름")` 튜플로 옛 이름을 같이 준다.
    allow += [
        (np.core.multiarray._reconstruct, "numpy.core.multiarray._reconstruct"),
        (np.core.multiarray.scalar, "numpy.core.multiarray.scalar"),
        (np.ndarray, "numpy.ndarray"),
        (np.dtype, "numpy.dtype"),
    ]
    for name in dir(getattr(np, "dtypes", None)):
        t = getattr(np.dtypes, name, None)
        if isinstance(t, type) and name.endswith("DType"):
            allow.append((t, f"numpy.dtypes.{name}"))
    try:
        import torch.serialization as ts
        ts.add_safe_globals(allow)
    except Exception:
        pass          # 없는 버전이면 그냥 넘어간다 — 로드가 되면 그만이다


def fold_head_bn(bh):
    """head 의 `ConvModule` 안 BatchNorm 을 conv 에 흡수시키고 Identity 로 바꾼다.

    GGUF 에는 **학습된 파라미터만** 실린다 — `running_mean`/`running_var` 는 안 간다.
    백본은 컴파일러가 fold 해서 이 문제가 없는데, head 는 그래프 밖이라 그 pass 를 안 탄다.
    그래서 여기서 접는다(수학적으로 정확한 변환이라 값이 안 바뀐다).

    `RetinaSepBNHead`(efficientnet·nas_fpn)가 유일한 사례다 — mmdet head 는 대개 GN 을 쓴다.
    """
    import torch
    folded = 0
    for mod in bh.modules():
        bn = getattr(mod, "bn", None)
        conv = getattr(mod, "conv", None)
        if not isinstance(bn, nn.modules.batchnorm._BatchNorm) or conv is None:
            continue
        with torch.no_grad():
            std = (bn.running_var + bn.eps).sqrt()
            scale = bn.weight / std
            w = conv.weight * scale.reshape(-1, 1, 1, 1)
            b = bn.bias - bn.running_mean * scale
            if conv.bias is not None:
                b = b + conv.bias * scale
            conv.weight.copy_(w)
            if conv.bias is None:
                conv.bias = nn.Parameter(b)
            else:
                conv.bias.copy_(b)
        mod.bn = nn.Identity()          # ConvModule.forward 의 norm 단이 통과만 한다
        folded += 1
    return folded


class MMDetBackbone(nn.Module):
    """backbone(+neck) features 만 노출. head 는 가중치 유지용 attribute(forward 미사용)."""
    def __init__(self, det):
        super().__init__()
        self.backbone = det.backbone
        self.neck = det.neck if getattr(det, "with_neck", False) else None
        self.bbox_head = getattr(det, "bbox_head", None)
        # head 는 forward 에 참여하지 않는다 — 그 계산은 tools/detect 의 C++ 부품이 한다.
        # 그래도 가중치는 GGUF 에 있어야 그 부품이 이름으로 찾을 수 있다. trace 로는 안
        # 잡히므로 여기서 선언한다.
        self.gguf_extra_weights = ("bbox_head",)

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

    # 어느 조립기를 쓸지. **MRO 를 순서대로 훑는다** — 계열 이름이 아니라 head 클래스가
    # 기준이고, 상속한 계열은 부모의 조립기를 그대로 탄다(GHM·PVT 는 RetinaHead 자체,
    # FSAF·FreeAnchor 는 RetinaHead 상속, LD 는 GFLHead 상속 …). 손실·라벨할당·증류는
    # 학습 시점 얘기라 추론 그래프가 같기 때문이다.
    # MRO 순서가 곧 우선순위다: VFNetHead 는 ATSSHead 를 상속하지만 자기 항목이 먼저 걸린다.
    #
    # ⚠️ 부모로 떨어지는 건 **추정**이다. `_init_layers`/`forward` 를 오버라이드한 계열이면
    #    조립이 조용히 틀린다 — verify_heads.py 로 재기 전에는 지원한다고 말하지 마라.
    HEADS = {
        "VFNetHead": "vfnet", "RepPointsHead": "reppoints", "TOODHead": "tood",
        "GFLHead": "gfl", "FCOSHead": "fcos",
        "ATSSHead": "anchor", "PAAHead": "anchor", "RetinaHead": "anchor",
        "AnchorHead": "anchor", "AnchorFreeHead": "fcos",
        "CenterNetHead": "centernet",
    }
    kind = next((HEADS[c.__name__] for c in type(bh).__mro__ if c.__name__ in HEADS), None)
    if kind is None:
        return {"head_type": "raw"}   # 모르는 계열 — 백본만 내보낸다
    # ⚠️ prior_generator 는 **앵커 계열에만** 있다. CenterNet 은 heatmap 의 최대점을 쓰므로
    #    없다. 이걸 먼저 검사하면 조립 가능한 계열까지 raw 로 떨어진다.
    if pg is None and kind != "centernet":
        return {"head_type": "raw"}

    # 디코드 지원은 **별개 판단**이다. head 조립은 되는데 박스 코더가 다른 계열이 있다
    # (FSAF 는 RetinaHead 인데 TBLRBBoxCoder 를 쓴다). 하나로 묶으면 조립까지 같이 막힌다.
    can_decode = bc is not None and "Delta" in type(bc).__name__

    ncls = int(getattr(bh, "cls_out_channels", getattr(bh, "num_classes", 80)))
    # stride 는 prior_generator 가 없으면 `build()` 가 실제 feature 크기에서 채운다.
    strides = ([s[0] if isinstance(s, (tuple, list)) else int(s) for s in pg.strides]
               if pg is not None else [])
    obs = float(getattr(pg, "octave_base_scale", 1.0) or 1.0) if pg is not None else 1.0
    scales = [float(x) for x in _tolist(getattr(pg, "scales", [obs]))] if pg is not None else [1.0]
    ratios = [float(x) for x in _tolist(getattr(pg, "ratios", [1.0]))] if pg is not None else [1.0]
    num_base = int(_tolist(getattr(pg, "num_base_priors", [len(scales) * len(ratios)]))[0]) \
        if pg is not None else 1

    # head-conv 구조 (C++ 조립기가 소비) — 최종 cls/reg conv 이름 **자동 탐지**.
    # 계열마다 이름이 다르다(retina_cls / atss_cls / gfl_cls / conv_cls …). 이름 표를 두는 대신
    # **출력 채널 수로 알아본다** — 그러면 새 계열이 와도 표를 안 고쳐도 된다.
    # 타워 ModuleList 이름이 계열마다 다르다. TOOD 는 cls/reg 를 따로 두지 않고
    # **inter_convs 하나**를 쓴 뒤 task decomposition 으로 가른다.
    cls_tower = "inter_convs" if hasattr(bh, "inter_convs") else "cls_convs"
    reg_tower = "inter_convs" if hasattr(bh, "inter_convs") else "reg_convs"
    towers = getattr(bh, cls_tower, [])
    # RetinaSepBNHead 는 레벨마다 타워를 따로 둔다 → ModuleList 안에 또 ModuleList 다.
    # 그때 `len(towers)` 는 단 수가 아니라 **레벨 수**이므로 한 겹 들어가야 한다.
    per_level = bool(towers) and isinstance(towers[0], nn.ModuleList)
    stacked = len(towers[0]) if per_level else len(towers)
    feat_ch = int(getattr(bh, "feat_channels", 256))
    reg_max = int(getattr(bh, "reg_max", 0) or 0)
    reg_ch = num_base * 4 * (reg_max + 1)          # GFL 은 방향당 (reg_max+1) 개 빈
    # 출력 conv 가 **레벨별 ModuleList** 인 계열이 있다(RTMDetSepBNHead 의 rtm_cls/rtm_reg).
    # 대표로 0번을 보고 채널을 재고, 조립기에 `.<레벨>` 을 붙이라고 알린다.
    cls_head = reg_head = ctr_head = None
    per_level_heads = False
    for name, mod in bh.named_children():
        if isinstance(mod, nn.ModuleList) and len(mod) and isinstance(mod[0], nn.Conv2d):
            mod, per_level_heads = mod[0], True
        if not isinstance(mod, nn.Conv2d):
            continue
        if mod.out_channels == num_base * ncls:
            cls_head = name
        elif mod.out_channels == reg_ch:
            reg_head = name
        elif mod.out_channels == num_base and any(
                s in name for s in ("centerness", "iou", "objectness")):
            # 세 번째 갈래. **이름이 계열마다 다르다** — atss_centerness / conv_centerness /
            # atss_iou(DDOD) / objectness. 하는 일은 같다(위치당 품질 점수 1채널).
            ctr_head = name

    # ConvModule 의 norm. 있으면 타워가 conv+GN+relu, 없으면 conv+relu 다.
    # BatchNorm 은 conv 로 접는다(GGUF 가 running stats 를 안 실으므로) → 접은 뒤엔 norm 이 없다.
    n_folded = fold_head_bn(bh)
    norm_cfg = getattr(bh, "norm_cfg", None) or {}
    has_norm = bool(norm_cfg) and not n_folded
    gn_groups = int(norm_cfg.get("num_groups", 32)) if isinstance(norm_cfg, dict) else 32

    # FCOS 만 bbox 에 후처리가 붙는다. norm_on_bbox 가 둘 중 어느 쪽인지 가른다.
    # ⚠️ `kind` 가 아니라 **실제 클래스**로 판단한다. FoveaHead 도 anchor-free 라 kind 는
    #    fcos 로 떨어지지만, bbox_pred 에 exp 를 안 건다(exp 는 feature_adaption 에만 쓴다).
    is_fcos = any(c.__name__ == "FCOSHead" for c in type(bh).__mro__)
    norm_on_bbox = bool(getattr(bh, "norm_on_bbox", False))
    bbox_clamp_stride = is_fcos and norm_on_bbox
    bbox_exp = is_fcos and not norm_on_bbox

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
        "head_type": kind,
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
        "center_offset": float(getattr(pg, "center_offset", 0.0) or 0.0) if pg is not None else 0.0,
        "means": [float(v) for v in getattr(bc, "means", [0.0] * 4)],
        "stds": [float(v) for v in getattr(bc, "stds", [1.0] * 4)],
        "can_decode": can_decode,
        # ── C++ head 부품(anchor_head_forward)용 구조 ──
        "num_base": num_base,
        "stacked_convs": stacked,
        "feat_channels": feat_ch,
        "cls_convs_prefix": "bbox_head." + cls_tower,
        "reg_convs_prefix": "bbox_head." + reg_tower,
        "cls_head": "bbox_head." + (cls_head or "retina_cls"),
        "reg_head": "bbox_head." + (reg_head or "retina_reg"),
        "head_has_norm": has_norm,
        "gn_groups": gn_groups,
        "per_level_towers": per_level,
        "per_level_heads": per_level_heads,
        # 곁가지 — 없으면 조립기가 그 단계를 건너뛴다.
        "centerness_head": ("bbox_head." + ctr_head) if ctr_head else "",
        "centerness_on_reg": bool(getattr(bh, "centerness_on_reg", True)),
        "scales_prefix": "bbox_head.scales" if hasattr(bh, "scales") else "",
        "bbox_exp": bbox_exp,
        "bbox_clamp_stride": bbox_clamp_stride,
        "reg_max": reg_max,
        # VFNet 의 레벨별 정규화 범위. stride 에서 유도하면 안 된다 — 마지막 레벨만 두 배다.
        "reg_denoms": [float(v) for v in getattr(bh, "reg_denoms", []) or []],
    }


def build(config, checkpoint=None, size=512):
    """mmdet config(.py) → (MMDetBackbone(eval), feature shapes, postproc cfg)."""
    from mmdet.apis import init_detector       # mmdet 만 import (g2c 무관)
    trace_friendly_ops()                       # trace 가 삼키는 커스텀 op 을 등가 수식으로
    allow_mmengine_checkpoint_globals()        # v3 체크포인트의 학습 메타 허용
    # ⚠️ `.eval()` 을 체이닝하지 마라. `nn.Module.eval()` 은 `self.train(False)` 의 반환값을
    #    그대로 돌려주는데, 증류 계열(`KnowledgeDistillationSingleStageDetector` — ld·lad)이
    #    `train()` 을 오버라이드하며 `return self` 를 빠뜨렸다. 체이닝하면 det 이 None 이 되고,
    #    한참 뒤 속성 접근에서 터져 원인 지점을 잃는다.
    det = init_detector(config, checkpoint, device="cpu")
    det.eval()
    m = MMDetBackbone(det)
    m.eval()
    cfg = postproc_cfg(det)
    cfg["img_size"] = int(size)   # 정방 resize 크기 (pre)
    with torch.no_grad():
        outs = m(torch.randn(1, 3, size, size))
    return m, [tuple(o.shape) for o in outs], cfg
