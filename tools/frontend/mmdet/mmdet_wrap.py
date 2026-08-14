"""mmdet_wrap.py — mmdet 검출기를 backbone/neck '정적 forward' nn.Module 로 감싸는 부품.

**클래스 전용 import 모듈**(스크립트로 직접 실행 금지). torch.save 는 클래스를 __module__ 로
피클하므로, 이 모듈에 클래스를 두고 CLI(mmdet_to_pt.py)와 로더(serialize.py)가 **import** 해야
동일 클래스 객체로 로드된다(__main__ 피클 함정 회피).

g2c 무관 — mmdet 만 import. forward = backbone→neck FPN features 까지(head decode/NMS 제외).
bbox_head 는 attribute 로 유지 → state_dict(→gguf)에 head 가중치가 실려 vision.cpp head 부품이 사용.
"""
import sys
import types

import torch
import torch.nn as nn


import mmdet_compat  # noqa: F401  (import 만으로 호환 패치가 걸린다)


# op 패치는 **공유 모듈**에 있다 — 진입점이 둘(`mmdet_wrap`·`frcnn_wrap`)이라
# 한쪽에만 두면 다른 쪽에서 조용히 안 걸린다(실제로 CARAFE 가 그래서 두 번 막혔다).
trace_friendly_ops = mmdet_compat.patch_ops


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
    # ⚠️ `builtins.getattr` — DDQ DETR 체크포인트가 이걸 피클에 담고 있어 그대로는 못 연다.
    #    이건 **임의 속성 접근 수단**이라 허용하면 `weights_only=True` 의 방어가 크게 약해진다.
    #    (신뢰할 수 없는 .pth 에 대해서는 사실상 `weights_only=False` 에 가깝다.)
    #    그래도 넣은 이유: 이 도구는 **mmdetection 공식 model zoo 에서 받은 체크포인트만**
    #    검증에 쓰고, 그 출처는 config 와 함께 저장소에 적혀 있다. 사용자 판단으로 진행했다.
    #    → 출처가 불분명한 체크포인트에는 이 경로를 쓰지 마라. `DECISIONS.md`
    allow.append(getattr)
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
    import copy
    import torch
    folded = 0
    # ⚠️ **conv 를 레벨끼리 공유하는 head 가 있다**(RTMDetSepBNHead(share_conv=True),
    #    RetinaSepBNHead: conv 는 하나인데 BN 만 레벨별로 따로다). 그대로 접으면 같은
    #    conv 에 BN 을 여러 번 겹쳐 접는다.
    #
    # ⚠️ **사본을 만드는 것만으로는 모자란다.** 두 번째 레벨에서 복사하는 그 conv 는
    #    **이미 0번 BN 이 접힌** 값이다 — 사본에 1번 BN 을 또 접으면 두 번 접힌다.
    #    레벨이 올라갈수록 어긋남이 커지는 것이 그 증상이다(efficientnet 실측:
    #    레벨0 1e-4 → 레벨4 3.8). 그래서 **접기 전 값을 따로 들고** 매번 거기서 접는다.
    pristine = {}                       # id(conv) -> 접기 전 (weight, bias)
    for mod in bh.modules():
        bn = getattr(mod, "bn", None)
        conv = getattr(mod, "conv", None)
        if not isinstance(bn, nn.modules.batchnorm._BatchNorm) or conv is None:
            continue
        key = id(conv)
        if key not in pristine:
            pristine[key] = (conv.weight.detach().clone(),
                             None if conv.bias is None else conv.bias.detach().clone())
        else:
            conv = copy.deepcopy(conv)  # 이 ConvModule 전용 사본(값은 아래에서 덮는다)
            mod.conv = conv
        w0, b0 = pristine[key]
        with torch.no_grad():
            std = (bn.running_var + bn.eps).sqrt()
            scale = bn.weight / std
            w = w0 * scale.reshape(-1, 1, 1, 1)
            b = bn.bias - bn.running_mean * scale
            if b0 is not None:
                b = b + b0 * scale
            conv.weight.copy_(w)
            if conv.bias is None:
                conv.bias = nn.Parameter(b)
            else:
                conv.bias.copy_(b)
        mod.bn = nn.Identity()          # ConvModule.forward 의 norm 단이 통과만 한다
        folded += 1
    return folded


def _unwrap_checkpoint_forward(det):
    """인스턴스에 덮인 `forward` 를 걷어낸다 — **`torch.save` 를 살리려고**.

    mmengine 이 activation checkpointing 을 걸면 모듈 인스턴스에
    `self.forward = functools.partial(...)` 를 심는데, 그 partial 이 모듈 자신에 대한
    **weakref** 를 물고 있어 피클이 안 된다:

        TypeError: cannot pickle 'weakref.ReferenceType' object

    (grounding_dino·mm_grounding_dino 의 `encoder.layers[i]`·`fusion_layers[i]` 가 그렇다.
     BERT 가 범인일 거라 추측했는데 아니었다 — 하나씩 좁혀서 찾았다.)

    checkpointing 은 **학습 때 메모리를 아끼는 래퍼**일 뿐이라, 지우면 클래스의 원래
    `forward` 가 다시 보이고 추론 값은 같다.
    """
    n = 0
    for mod in det.modules():
        if "forward" in vars(mod):
            del mod.forward
            n += 1
    if n:
        print(f"  · checkpointing forward 래퍼 {n}개 제거 (피클 가능하게)")


class MMDetBackbone(nn.Module):
    """backbone(+neck) features 만 노출. head 는 가중치 유지용 attribute(forward 미사용)."""
    def __init__(self, det):
        super().__init__()
        self.backbone = det.backbone
        self.neck = det.neck if getattr(det, "with_neck", False) else None
        self.bbox_head = getattr(det, "bbox_head", None)
        # head 는 forward 에 참여하지 않는다 — 그 계산은 tools/detect 의 C++ 부품이 한다.
        # 그래도 가중치는 GGUF 에 있어야 그 부품이 이름으로 찾을 수 있는데, trace 에는 안
        # 잡힌다. **g2c 를 고치지 않고** `append_head_weights.py` 가 g2c 산출물에 덧붙인다 —
        # mmdet 지식은 mmdet 폴더에 둔다. 여기서는 무엇을 실을지 이름만 선언한다.
        declared = ["bbox_head"]

        # ⚠️ **DETR 계열은 encoder/decoder 가 head 가 아니라 detector 에 달려 있다**
        #    (mmdet v3 `DetectionTransformer`). `bbox_head` 는 cls/reg 분기 둘뿐이고
        #    `forward(hidden_states)` 로 decoder 출력을 받는다. 그래서 "bbox_head 만 선언"
        #    으로는 transformer 가중치가 통째로 빠진다.
        #    이름을 나열하지 않고 **detector 의 자식 중 우리가 이미 아는 것 말고 전부**를
        #    붙인다 — 계열이 늘어도 여기를 안 고친다. 이름은 mmdet 원본 그대로 유지해야
        #    C++ 조립기가 같은 키로 찾는다.
        known = {"backbone", "neck", "bbox_head", "data_preprocessor"}
        for name, child in det.named_children():
            if name in known or child is None:
                continue
            setattr(self, name, child)
            declared.append(name)
        # level_embed 처럼 **Module 이 아니라 Parameter** 로 달린 것도 있다(Deformable DETR).
        for name, prm in det.named_parameters(recurse=False):
            setattr(self, name, prm)
            declared.append(name)
        self.head_weight_prefixes = tuple(declared)

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


def _center_offset(pg):
    """격자 중심 오프셋. 생성기마다 **속성 이름이 다르다.**

    AnchorGenerator 는 `center_offset`(기본 0), MlvlPointGenerator 는 `offset`(기본 0.5)다
    (`point_generator.py:107`). 하나만 보면 point 계열이 전부 0.0 으로 나가 박스가 반 칸씩
    밀린다 — 오차가 stride 에 비례해 커지는 것이 그 증상이다.
    """
    if pg is None:
        return 0.0
    v = getattr(pg, "center_offset", None)
    if v is None:
        v = getattr(pg, "offset", None)
    return float(v or 0.0)


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
        "RPNHead": "rpn",
        "ATSSHead": "anchor", "PAAHead": "anchor", "RetinaHead": "anchor",
        "AnchorHead": "anchor", "AnchorFreeHead": "fcos",
        "CenterNetHead": "centernet", "YOLOFHead": "yolof",
        "YOLOXHead": "yolox", "YOLOV3Head": "yolo",
        "CornerHead": "cornernet", "CentripetalHead": "cornernet",
        "DeformableDETRHead": "deformable_detr", "DABDETRHead": "dab_detr", "ConditionalDETRHead": "conditional_detr",
        "DETRHead": "detr",
    }
    kind = next((HEADS[c.__name__] for c in type(bh).__mro__ if c.__name__ in HEADS), None)
    if kind is None:
        return {"head_type": "raw"}   # 모르는 계열 — 백본만 내보낸다
    # ⚠️ prior_generator 는 **앵커 계열에만** 있다. CenterNet 은 heatmap 최대점, CornerNet 은
    #    코너 짝짓기를 쓰므로 없다. 이걸 먼저 검사하면 조립 가능한 계열까지 raw 로 떨어진다.
    #    없다. 이걸 먼저 검사하면 조립 가능한 계열까지 raw 로 떨어진다.
    # deformable 계열은 MRO 로 `DETRHead` 에 떨어지지만 attention 자체가 다르다 —
    # 여기서 kind 를 바로잡는다. two-stage(DINO·DDQ)는 아직 조립기가 없으므로 raw 로 떨군다.
    # **지원 안 하는 것은 크래시가 아니라 "인식 못 함" 으로 말한다.**
    enc = getattr(det, "encoder", None)
    dec = getattr(det, "decoder", None)
    deformable = (dec is not None
                  and type(dec.layers[0].cross_attn).__name__.startswith("MultiScaleDeform"))
    if deformable:
        # two-stage 여부는 `memory_trans_fc` 로 가른다 — DINO 도 query_embedding 은 갖고 있어서
        # 그걸로 가르면 두 계열이 같은 조립기를 타고 조용히 틀린다.
        two_stage = getattr(det, "memory_trans_fc", None) is not None
        # DDQ 는 층마다 NMS 로 query 를 골라낸다 — `dqs_cfg` 유무가 그 표시다.
        if getattr(det, "dqs_cfg", None) is not None:
            kind = "ddq"
        else:
            kind = "dino" if two_stage else "deformable_detr"

    if pg is None and kind not in ("centernet", "cornernet", "detr", "conditional_detr",
                                   "dab_detr", "deformable_detr", "dino", "ddq"):
        return {"head_type": "raw"}

    # 디코드 지원은 **별개 판단**이다. head 조립은 되는데 박스 코더가 다른 계열이 있다
    # (FSAF 는 RetinaHead 인데 TBLRBBoxCoder 를 쓴다). 하나로 묶으면 조립까지 같이 막힌다.
    can_decode = bc is not None and "Delta" in type(bc).__name__
    # (레벨별 anchor 수가 다르면 뒤에서 취소한다 — num_base 하나로는 못 푼다)

    # cls 출력 채널 수. **의미상 클래스 수와 다를 수 있다** — softmax head 는 배경을 한 칸
    # 더 쓰므로 `cls_out_channels == num_classes + 1` 이다(detr_head.py:114-118).
    # 디코드는 둘을 다 알아야 한다: 채널 폭은 인덱싱에, 클래스 수는 배경 제외에 쓴다.
    ncls = int(getattr(bh, "cls_out_channels", getattr(bh, "num_classes", 80)))
    ncls_semantic = int(getattr(bh, "num_classes", ncls))

    # 활성: 대부분 `use_sigmoid_cls` 를 갖지만 **DETR 계열은 없다** — 그쪽은
    # `loss_cls.use_sigmoid` 가 정본이다(고전 DETR=softmax, Deformable/DINO=sigmoid).
    # 기본값 True 로 두면 고전 DETR 이 조용히 sigmoid 로 디코드된다.
    _use_sig = getattr(bh, "use_sigmoid_cls", None)
    if _use_sig is None:
        _use_sig = getattr(getattr(bh, "loss_cls", None), "use_sigmoid", True)
    # stride 는 prior_generator 가 없으면 `build()` 가 실제 feature 크기에서 채운다.
    strides = ([s[0] if isinstance(s, (tuple, list)) else int(s) for s in pg.strides]
               if pg is not None else [])
    obs = float(getattr(pg, "octave_base_scale", 1.0) or 1.0) if pg is not None else 1.0

    def _floats(v, dflt):
        # ⚠️ SSDAnchorGenerator 는 scales/ratios 를 **레벨별 리스트**로 둔다. 원소가 스칼라가
        #    아니라 `float(x)` 가 터진다("only one element tensors …"). 그때는 디코드를
        #    포기하고(호출자 몫) head 조립만 한다 — 여기서 죽으면 계열이 통째로 막힌다.
        try:
            return [float(x) for x in _tolist(v)]
        except (TypeError, ValueError):
            return dflt
    scales = _floats(getattr(pg, "scales", [obs]), []) if pg is not None else [1.0]
    ratios = _floats(getattr(pg, "ratios", [1.0]), []) if pg is not None else [1.0]
    nbp = _tolist(getattr(pg, "num_base_priors", [len(scales) * len(ratios)])) if pg is not None else [1]
    num_base = int(nbp[0])
    # 레벨마다 anchor 수가 다르면(SSD: 4·6·6·6·4·4) 하나의 num_base 로 디코드할 수 없다.
    uniform_priors = len(set(int(x) for x in nbp)) <= 1

    # head-conv 구조 (C++ 조립기가 소비) — 최종 cls/reg conv 이름 **자동 탐지**.
    # 계열마다 이름이 다르다(retina_cls / atss_cls / gfl_cls / conv_cls …). 이름 표를 두는 대신
    # **출력 채널 수로 알아본다** — 그러면 새 계열이 와도 표를 안 고쳐도 된다.
    # 타워 ModuleList 이름이 계열마다 다르다. TOOD 는 cls/reg 를 따로 두지 않고
    # **inter_convs 하나**를 쓴 뒤 task decomposition 으로 가른다.
    # YOLOF 는 `cls_subnet`/`bbox_subnet` 이고 **깊이도 서로 다르다**(2 / 4).
    # 후보를 순서대로 보고 처음 있는 것을 쓴다 — 새 이름이 와도 여기 한 줄만 는다.
    def _tower(*names):
        return next((n for n in names if getattr(bh, n, None) is not None), names[-1])
    cls_tower = _tower("inter_convs", "head_convs", "cls_convs", "cls_subnet", "multi_level_cls_convs")
    reg_tower = _tower("inter_convs", "head_convs", "reg_convs", "bbox_subnet", "multi_level_reg_convs")
    towers = getattr(bh, cls_tower, [])
    reg_towers = getattr(bh, reg_tower, [])
    # RetinaSepBNHead 는 레벨마다 타워를 따로 둔다 → ModuleList 안에 또 ModuleList 다.
    # 그때 `len(towers)` 는 단 수가 아니라 **레벨 수**이므로 한 겹 들어가야 한다.
    # 레벨별 타워인지 판별. 컨테이너 종류로 보면 안 된다 — RetinaSepBNHead 는 ModuleList,
    # YOLOX 는 nn.Sequential 이다. **원소가 ConvModule 인가**(=`.conv` 를 갖는가)로 가른다.
    per_level = bool(towers) and getattr(towers[0], "conv", None) is None
    stacked = len(towers[0]) if per_level else len(towers)
    # reg 타워 깊이가 다르면 따로 싣는다. 같으면 0(= cls 와 같다)으로 둔다.
    reg_stacked = 0 if reg_tower == cls_tower else len(reg_towers)
    if reg_stacked == stacked:
        reg_stacked = 0
    feat_ch = int(getattr(bh, "feat_channels", 256))
    reg_max = int(getattr(bh, "reg_max", 0) or 0)
    reg_ch = num_base * 4 * (reg_max + 1)          # GFL 은 방향당 (reg_max+1) 개 빈
    # 출력 conv 가 **레벨별 ModuleList** 인 계열이 있다(RTMDetSepBNHead 의 rtm_cls/rtm_reg).
    # 대표로 0번을 보고 채널을 재고, 조립기에 `.<레벨>` 을 붙이라고 알린다.
    cls_head = reg_head = ctr_head = None
    ctr_tanh = False
    pre_conv = ""
    # CornerHead 는 embedding 갈래를 config 로 끌 수 있다(CentripetalNet 은 끈다).
    corner_emb = bool(getattr(bh, "with_corner_emb", False))
    corner_centripetal = getattr(bh, "tl_feat_adaption", None) is not None
    # DETR: encoder/decoder 는 detector 에 있다. 층수·헤드수를 **모듈에서 읽는다**(config 아님).
    enc = getattr(det, "encoder", None)
    dec = getattr(det, "decoder", None)
    # ⚠️ deformable attention 계열(Deformable DETR · DINO · DDQ)은 **조립기가 없다.**
    #    MRO 로는 DETRHead 로 떨어져 detr 조립기를 타는데, 가중치 이름부터 달라 크래시한다.
    #    "지원한다고 말하지 않는" 쪽이 맞다 — raw 로 내보내고 러너가 조용히 넘어가게 한다.
    #    (없는 것: 네트워크가 예측한 **소수점 좌표에서 읽는** bilinear 샘플링.)
    detr_dims = {}
    if dec is not None:
        sa = dec.layers[0].self_attn
        detr_dims = {"embed_dims": int(sa.embed_dims), "n_heads": int(sa.num_heads),
                     "enc_layers": len(enc.layers) if enc is not None else 0,
                     "dec_layers": len(dec.layers),
                     "n_points": int(getattr(dec.layers[0].cross_attn, "num_points", 4)),
                     "num_queries": int(getattr(det, "num_queries", 300)),
                     "ddq_iou_thr": float((getattr(det, "dqs_cfg", None) or {})
                                          .get("iou_threshold", 0.8))}
    per_level_heads = False
    head_tail = ""
    # SSD 는 **타워가 없다** — `cls_convs[l]` 자체가 예측 conv(Sequential)다. 그대로 두면
    # 같은 conv 를 타워로 한 번, head 로 또 한 번 태운다. 구조로 알아본다:
    # 타워 후보의 원소가 Sequential 이고 그 **마지막 Conv2d 출력 채널이 cls 채널 수**면 head 다.
    def _last_conv(seq):
        idx = [i for i, m in enumerate(seq) if isinstance(m, nn.Conv2d)]
        return (idx[-1], seq[idx[-1]]) if idx else (None, None)
    if getattr(bh, "convs_bridge", None) is not None:
        # YOLOv3. 타워 반복이 없고 출력이 한 갈래다 — 채널이 na*(5+nc) 라 채널수 탐지에 안 걸린다.
        cls_tower = reg_tower = "convs_bridge"
        cls_head = reg_head = "convs_pred"
        stacked, per_level_heads = 1, True
        towers = bh.convs_bridge          # 아래 활성화 탐지가 이 타워를 보게 한다
    elif towers and isinstance(towers[0], nn.Sequential):
        i_last, last = _last_conv(towers[0])
        if last is not None and last.out_channels == num_base * ncls:
            stacked, per_level_heads, head_tail = 0, True, f".{i_last}"
            cls_head, reg_head = cls_tower, reg_tower
    # 최종 conv 를 **출력 채널 수**로 알아본다 — 이름 표를 두면 계열이 늘 때마다 고쳐야 한다
    # (retina_cls / atss_cls / gfl_cls / conv_cls …).
    #
    # ⚠️ **채널 수가 겹치는 경우가 있다.** MOT 용 YOLOX 는 클래스가 "사람" 하나라
    #    cls(=num_base·ncls=1) 와 objectness(=num_base=1) 가 **둘 다 1채널**이다.
    #    한 번의 순회로 먼저 만난 것을 집으면 obj 가 cls 자리에 들어간다(실측: bytetrack).
    #    → **두 번 훑는다.** 이름이 확실한 세 번째 갈래를 먼저 걷어내고, 남은 것만 채널로 가른다.
    #    (같은 부류: DCN v1/v2 도 54채널이 양쪽에 걸려 다른 불변량으로 갈랐다 → 위키)
    cands = []
    for name, mod in (() if cls_head else bh.named_children()):
        if isinstance(mod, nn.ModuleList) and len(mod) and isinstance(mod[0], nn.Conv2d):
            mod, per_level_heads = mod[0], True
        if isinstance(mod, nn.Conv2d):
            cands.append((name, mod))

    # ① 이름이 분명한 갈래부터 확정한다(품질 점수 · mask 계수).
    for name, mod in cands:
        if "coeff" in name:
            # YOLACT 의 mask coefficient. 채널이 na*num_protos 라 채널수로는 못 알아본다.
            ctr_head, ctr_tanh = name, True
        elif ctr_head is None and any(s in name for s in ("centerness", "iou", "obj")):
            # 세 번째 갈래. **이름이 계열마다 다르다** — atss_centerness / conv_centerness /
            # atss_iou(DDOD) / object_pred(YOLOF) / conv_obj(YOLOX). 하는 일은 같다.
            ctr_head = name

    # ② 남은 것만 채널 수로 가른다.
    for name, mod in cands:
        if name in (ctr_head,):
            continue
        if cls_head is None and mod.out_channels == num_base * ncls:
            cls_head = name
        elif reg_head is None and mod.out_channels == reg_ch:
            reg_head = name

    # ③ 그러고도 남는 Conv2d 가 있으면 **분기 앞 공유 conv** 다(RPNHead 의 `rpn_conv`).
    #    컨테이너가 아니라 맨 Conv2d 라 타워 탐지에 안 걸린다. 채널이 안 변해
    #    빼먹어도 크래시가 없고 값만 틀린다(rpn 실측 L1 5.56).
    for name, mod in cands:
        if name not in (cls_head, reg_head, ctr_head) and mod.out_channels == feat_ch:
            pre_conv = "bbox_head." + name
            break

    # ConvModule 의 norm. 있으면 타워가 conv+GN+relu, 없으면 conv+relu 다.
    # BatchNorm 은 conv 로 접는다(GGUF 가 running stats 를 안 실으므로) → 접은 뒤엔 norm 이 없다.
    n_folded = fold_head_bn(bh)
    # 활성화도 계열마다 다르다 — mmdet ConvModule 의 기본은 ReLU 인데 RTMDet 은 SiLU 다.
    # **실제 모듈에서 읽는다**(config 의 act_cfg 는 head 가 덮어쓰는 경우가 있다).
    # 활성화가 틀리면 shape 는 그대로고 값만 조금씩 작아진다 → 조용히 틀린다.
    # ⚠️ **이름이 하나가 아니다.** SiLU 와 Swish 는 같은 함수인데 mmcv 는 `Swish` 라는
    #    자체 클래스로 등록한다(YOLOX). 이름 하나만 보면 조용히 ReLU 로 떨어진다 — 실측 L1 0.213.
    _tw = towers[0][0] if per_level else (towers[0] if towers else None)
    _act = type(getattr(_tw, "activate", None)).__name__
    head_leaky = float(getattr(getattr(_tw, "activate", None), "negative_slope", 0.0) or 0.0)
    head_silu = _act in ("SiLU", "Swish", "MemoryEfficientSwish")
    if _tw is not None and _act not in ("SiLU", "Swish", "MemoryEfficientSwish",
                                        "ReLU", "LeakyReLU", "NoneType"):
        # 모르는 활성화를 ReLU 로 떨어뜨리면 shape 는 맞고 값만 틀린다 → 조용히 통과한다.
        print(f"  ⚠️ 모르는 head 활성화 '{_act}' — ReLU 로 조립한다. 값이 틀릴 수 있다.")
    norm_cfg = getattr(bh, "norm_cfg", None) or {}
    has_norm = bool(norm_cfg) and not n_folded
    gn_groups = int(norm_cfg.get("num_groups", 32)) if isinstance(norm_cfg, dict) else 32

    # FCOS 만 bbox 에 후처리가 붙는다. norm_on_bbox 가 둘 중 어느 쪽인지 가른다.
    # ⚠️ `kind` 가 아니라 **실제 클래스**로 판단한다. FoveaHead 도 anchor-free 라 kind 는
    #    fcos 로 떨어지지만, bbox_pred 에 exp 를 안 건다(exp 는 feature_adaption 에만 쓴다).
    is_fcos = any(c.__name__ == "FCOSHead" for c in type(bh).__mro__)
    norm_on_bbox = bool(getattr(bh, "norm_on_bbox", False))
    bbox_clamp_stride = is_fcos and norm_on_bbox
    # RTMDet 계열은 거리를 **stride 단위**로 낸다(`rtm_reg(...)[.exp()] * stride[0]`).
    # `exp_on_reg` 속성의 유무가 그 계열이라는 표시다 — 이름으로 가르지 않는다.
    is_rtmdet = hasattr(bh, "exp_on_reg")
    bbox_exp = (is_fcos and not norm_on_bbox) or bool(getattr(bh, "exp_on_reg", False))
    bbox_mul_stride = is_rtmdet
    # AutoAssign 은 `forward_single` 을 갈아끼워 **norm_on_bbox 와 무관하게** clamp·stride 를
    # 건다(`bbox_pred.clamp(min=0) * stride`). 그리고 centerness 를 항상 reg_feat 에서 뽑는데,
    # `centerness_on_reg` 속성은 상속받은 기본값(False)이라 **속성만 보면 틀린다**.
    # 구조적 표시는 `center_prior` — 이 계열에만 있는 서브모듈이다.
    is_autoassign = hasattr(bh, "center_prior")
    if is_autoassign:
        bbox_exp, bbox_clamp_stride = False, True

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

    # ── 디코드 임계값: mmdet 의 test_cfg 가 정본이다 ────────────────────────────
    # 안 실으면 러너가 라이브러리 기본값(0.05/0.5/1000/100)을 쓴다. 계열마다 다르다 —
    # 실측: retinanet 은 nms 0.5 인데 **ATSS 는 0.6**, RTMDet 은 score 0.001·nms 0.65·
    # max 300 이다. 다르면 살아남는 박스 집합이 달라져 mmdet 과의 비교가 성립하지 않는다.
    tc = getattr(bh, "test_cfg", None) or getattr(det, "test_cfg", None)

    def _tc(key, default):
        return default if tc is None else tc.get(key, default)

    _nms = _tc("nms", {}) or {}
    thresholds = {
        "score_thr": float(_tc("score_thr", 0.05)),
        "nms_thr": float(_nms.get("iou_threshold", 0.5)),
        "nms_pre": int(_tc("nms_pre", 1000)),
        "max_per_img": int(_tc("max_per_img", 100)),
    }

    return {
        "head_type": kind,
        # ── 전처리(pre) — 이미지→텐서 (vision.cpp preprocess) ──
        "img_mean": img_mean,
        "img_std": img_std,
        "to_rgb": to_rgb,
        **thresholds,
        "use_sigmoid": bool(_use_sig),
        # 배경 제외 클래스 수. softmax head 에서 `num_classes`(=채널폭)와 갈린다.
        "num_classes_semantic": ncls_semantic,
        "num_classes": ncls,
        "strides": [float(s) for s in strides],
        "octave_base_scale": obs,
        "octave_scales": [s / obs for s in scales],   # =2^(i/n)
        "ratios": ratios,
        # ⚠️ 속성 이름이 생성기마다 다르다. AnchorGenerator 는 `center_offset`,
        #    MlvlPointGenerator 는 **`offset`** 이다(point_generator.py:107).
        #    하나만 보면 point 계열이 전부 0.0 으로 나가 박스가 반 칸씩 밀린다.
        "center_offset": _center_offset(pg),
        "means": [float(v) for v in getattr(bc, "means", [0.0] * 4)],
        "stds": [float(v) for v in getattr(bc, "stds", [1.0] * 4)],
        "can_decode": can_decode and uniform_priors,
        # ── C++ head 부품(anchor_head_forward)용 구조 ──
        "num_base": num_base,
        "stacked_convs": stacked,
        "reg_stacked_convs": reg_stacked,
        "feat_channels": feat_ch,
        "cls_convs_prefix": "bbox_head." + cls_tower,
        "reg_convs_prefix": "bbox_head." + reg_tower,
        "cls_head": "bbox_head." + (cls_head or "retina_cls"),
        "reg_head": "bbox_head." + (reg_head or "retina_reg"),
        "head_has_norm": has_norm,
        "gn_groups": gn_groups,
        "per_level_towers": per_level,
        "per_level_heads": per_level_heads,
        "per_level_head_tail": head_tail,
        # 곁가지 — 없으면 조립기가 그 단계를 건너뛴다.
        "centerness_head": ("bbox_head." + ctr_head) if ctr_head else "",
        "head_leaky": head_leaky,
        "pre_conv": pre_conv,
        "ctr_tanh": ctr_tanh,
        "corner_emb": corner_emb,
        "corner_centripetal": corner_centripetal,
        **detr_dims,
        "centerness_on_reg": is_autoassign or bool(getattr(bh, "centerness_on_reg", True)),
        "scales_prefix": "bbox_head.scales" if hasattr(bh, "scales") else "",
        "bbox_exp": bbox_exp,
        "bbox_clamp_stride": bbox_clamp_stride,
        "bbox_mul_stride": bbox_mul_stride,
        "head_silu": head_silu,
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
    _unwrap_checkpoint_forward(det)
    m = MMDetBackbone(det)
    m.eval()
    cfg = postproc_cfg(det)
    cfg["img_size"] = int(size)   # 정방 resize 크기 (pre)
    with torch.no_grad():
        outs = m(torch.randn(1, 3, size, size))
    # CenterNet 은 prior_generator 가 없다(heatmap 최대점을 쓰므로 anchor 가 필요 없다).
    # 그래도 러너는 레벨 수·stride 를 알아야 한다 — **실제 feature 크기에서 역산**한다.
    if not cfg.get("strides"):
        cfg["strides"] = [float(size) / float(o.shape[-1]) for o in outs]

    # DETR 계열의 sine positional encoding 은 **입력 크기가 정해지면 상수**다. C++ 에서
    # 공식을 다시 구현하면 틀릴 여지가 생기므로 **mmdet 자신의 모듈로 계산해** GGUF 에 굽는다.
    # 레이아웃도 여기서 맞춘다: mmdet 은 (bs, C, H, W) → flatten(2).permute → (bs, H*W, C).
    # (H*W, C) 로 저장하면 ggml ne = {C, H*W} 라 조립기가 그대로 쓴다.
    pe = getattr(det, "positional_encoding", None)
    if pe is not None and outs:
        with torch.no_grad():
            o = outs[0]
            mask = torch.zeros((1, o.shape[2], o.shape[3]), dtype=torch.bool)
            pos = pe(mask)                                  # (1, C, H, W)
            pos = pos.flatten(2).permute(0, 2, 1)[0].contiguous()
        m.register_buffer("pos_embed", pos)
        m.head_weight_prefixes = tuple(m.head_weight_prefixes) + ("pos_embed",)

    # Conditional DETR 계열의 **참조점**도 상수다. `reference = sigmoid(ref_point_head(query_pos))`
    # 이고 query_pos 는 학습된 embedding(입력과 무관)이라, 거기서 나오는 sine 인코딩도 상수다.
    # C++ 에서 sine 공식을 다시 짜지 않고 **mmdet 자신의 함수로** 계산해 굽는다.
    dec = getattr(det, "decoder", None)
    rph = getattr(dec, "ref_point_head", None)
    # ⚠️ DAB-DETR 도 `ref_point_head` 를 갖지만 **입력이 다르다** — query_embedding 이 4D 앵커라
    #    그 sine 인코딩(2*dim)을 받는다. 입력 폭이 query_pos 와 같을 때만 이 경로다.
    qemb = getattr(det, "query_embedding", None)   # two-stage 계열(DDQ·DINO)은 아예 없다
    if (rph is not None and qemb is not None
            and int(rph.layers[0].weight.shape[1]) == int(qemb.weight.shape[1])):
        from mmdet.models.layers.transformer.utils import coordinate_to_encoding
        from mmdet.models.layers.transformer.utils import inverse_sigmoid
        with torch.no_grad():
            qp = det.query_embedding.weight[None]              # (1, nq, dim)
            ref = dec.ref_point_head(qp).sigmoid()             # (1, nq, 2 또는 4)
            sine = coordinate_to_encoding(ref[..., :2])[0].contiguous()   # (nq, dim)
            inv = inverse_sigmoid(ref[0, :, :2])
            # head 는 박스 앞 2채널에만 이 값을 더한다 → (nq, 4) 로 만들어 그냥 더하게 한다.
            pad = torch.zeros((inv.shape[0], 4))
            pad[:, :2] = inv
        m.register_buffer("ref_sine_embed", sine)
        m.register_buffer("ref_inv_pad", pad.contiguous())
        m.head_weight_prefixes = tuple(m.head_weight_prefixes) + ("ref_sine_embed", "ref_inv_pad")

    # Deformable 계열(Deformable DETR · DINO · DDQ): 다중 레벨 위치 인코딩(level_embed 포함)과
    # **encoder 참조점**은 입력 크기에만 의존하는 상수다. mmdet 자신의 `pre_transformer` 로
    # 계산해 굽는다 — 레벨 오프셋·정규화 규약을 C++ 에서 다시 짜면 틀릴 여지가 생긴다.
    if getattr(det, "level_embed", None) is not None:
        from mmdet.structures import DetDataSample
        ds = DetDataSample()
        ds.set_metainfo({"batch_input_shape": (size, size), "img_shape": (size, size)})
        with torch.no_grad():
            ei = det.pre_transformer(tuple(outs), [ds])[0]
            lvl_pos = ei["feat_pos"][0].contiguous()            # (sum_HW, dim)
            ss = ei["spatial_shapes"]
            vr = ei["valid_ratios"]
            enc_ref = det.encoder.get_encoder_reference_points(ss, vr, device="cpu")
            enc_ref = enc_ref[0].reshape(enc_ref.shape[1], -1).contiguous()  # (sum_HW, 2L)
        m.register_buffer("pos_embed", lvl_pos)
        m.register_buffer("enc_ref", enc_ref)
        m.head_weight_prefixes = tuple(m.head_weight_prefixes) + ("pos_embed", "enc_ref")

        # two-stage(DINO·DDQ): encoder 출력에서 proposal 을 만들어 상위 k 개를 query 로 쓴다.
        # proposal 격자와 유효 마스크는 **입력 크기에만 의존하는 상수**다 — 구워서 넘긴다.
        if getattr(det, "memory_trans_fc", None) is not None:
            with torch.no_grad():
                mem0 = torch.zeros(1, int(lvl_pos.shape[0]), int(lvl_pos.shape[1]))
                _, proposals = det.gen_encoder_output_proposals(mem0, None, ss)
                valid = torch.isfinite(proposals[0]).all(-1, keepdim=True).float()
            m.register_buffer("enc_proposals", proposals[0].contiguous())
            m.register_buffer("enc_valid", valid.contiguous())
            m.head_weight_prefixes = tuple(m.head_weight_prefixes) + ("enc_proposals", "enc_valid")

    # DAB-DETR: 참조점이 층마다 갱신돼 상수로 못 굽는다. 대신 `coordinate_to_encoding` 이
    # 쓰는 **고정 계수**만 굽는다 — 2π/dim_t 와 짝/홀 자리 마스크.
    # (인접한 두 dim_t 가 같은 값이라 결과는 sin·cos 이 번갈아 놓인 꼴이 된다.)
    if dec is not None and getattr(dec, "ref_point_head", None) is not None:
        import math
        nf = int(dec.embed_dims) // 2
        dim_t = 10000.0 ** (2 * (torch.arange(nf, dtype=torch.float32) // 2) / nf)
        m.register_buffer("dab_inv_dim_t", (2 * math.pi / dim_t).contiguous())
        idx = torch.arange(nf)
        m.register_buffer("dab_even", (idx % 2 == 0).float().contiguous())
        m.register_buffer("dab_odd", (idx % 2 == 1).float().contiguous())
        m.head_weight_prefixes = tuple(m.head_weight_prefixes) + (
            "dab_inv_dim_t", "dab_even", "dab_odd")
    return m, [tuple(o.shape) for o in outs], cfg
