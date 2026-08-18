"""frcnn_wrap.py — Faster R-CNN two-stage 를 g2c 로 컴파일 가능한 두 subgraph 로 분해.

  SubA = backbone + FPN + RPN head  (이미지 → P2-P5 + rpn_cls×5 + rpn_bbox×5 = 14 출력)
  SubB = RoI bbox_head (Shared2FC)  (RoIAlign feat (N,256,7,7) → cls(N,81), bbox(N,320))

그 사이(RPN proposals, RoIAlign, 최종 decode+NMS)는 host C++ 부품(postproc: rpn_proposals/
roi_align/detect_roi). 두 wrapper 모두 flat tuple 반환 → g2c 가 out_0.. 다출력으로 컴파일.
피클 모듈명 = frcnn_wrap (self-contained). g2c 코어 무관 — mmdet 만 import.
"""
import mmdet_compat  # noqa: F401  (import 만으로 호환 패치가 걸린다)
import torch
import torch.nn as nn


def _n_roi_levels(det):
    """RoIAlign 이 실제로 쓰는 레벨 수. **4 로 박으면 안 된다.**

    FPN 계열은 P2..P6 중 앞 4개를 쓰지만, C4 계열(TridentNet 등)은 neck 없이
    백본 마지막 단계 **하나**만 쓴다. extractor 의 `featmap_strides` 가 정답이다.
    """
    ext = getattr(getattr(det, "roi_head", None), "bbox_roi_extractor", None)
    if isinstance(ext, nn.ModuleList):
        ext = ext[0]
    strides = getattr(ext, "featmap_strides", None)
    return len(strides) if strides else 4


class FRCNN_SubA(nn.Module):
    """이미지 → (RoI 레벨 feats, rpn_cls×L, rpn_bbox×L).

    FPN 계열은 (P2..P5, rpn×5, rpn×5) = 14 출력. C4 계열(TridentNet)은 neck 이 없어
    (C4, rpn_cls, rpn_bbox) = 3 출력이다.
    """
    def __init__(self, det):
        super().__init__()
        self.backbone = det.backbone
        # ⚠️ **neck 이 항상 있다고 보면 안 된다.** C4 계열(TridentFasterRCNN)은 FPN 없이
        #    백본 C4 를 그대로 쓰고 `shared_head`(ResLayer)가 C5 역할을 한다.
        self.neck = getattr(det, "neck", None)
        self.rpn_head = det.rpn_head
        self.n_roi = _n_roi_levels(det)
        # HTC 계열의 **시맨틱 갈래**. FPN 위에 FCN 을 따로 돌려 나온 융합 feature 를
        # RoIAlign 해서 `bbox_feats` 에 더한다(htc_roi_head.py:99-104).
        # ⚠️ **별도 그래프로 빼지 않는다.** 입력이 FPN 전 레벨이라 다중 입력 그래프가 되는데,
        #    컴파일 경로가 단일 입력만 받는다. 같은 그래프의 **출력 하나**로 붙이면
        #    백본을 두 번 돌 필요도 없다.
        self.semantic_head = getattr(det.roi_head, "semantic_head", None)
        # SCNet 의 **전역 컨텍스트**. 최상위 레벨에 conv 몇 개 + AdaptiveAvgPool(1) 이라
        # 결과가 **이미지당 채널 벡터 하나**이고, 모든 RoI feature 에 채널별로 더해진다
        # (`scnet_roi_head._fuse_glbctx`). 시맨틱과 같은 방식으로 출력에 붙인다.
        self.glbctx_head = getattr(det.roi_head, "glbctx_head", None)

    def forward(self, x):
        feats = self.backbone(x)
        if getattr(self, "neck", None) is not None:
            feats = self.neck(feats)                 # tuple len 5: P2..P6
        rpn_cls, rpn_bbox = self.rpn_head(feats)     # (listL, listL)
        out = tuple(feats[:getattr(self, "n_roi", 4)]) + tuple(rpn_cls) + tuple(rpn_bbox)
        if getattr(self, "semantic_head", None) is not None:
            # `(mask_preds, fused)` 중 **fused** 가 bbox 융합에 쓰이는 쪽이다.
            _, sem = self.semantic_head(feats)
            out = out + (sem,)
        if getattr(self, "glbctx_head", None) is not None:
            # `(mc_pred, x)` 중 **x** 가 융합에 쓰인다. shape 은 (B, C, 1, 1).
            _, glb = self.glbctx_head(feats)
            out = out + (glb,)
        return out


class FRCNN_SubB(nn.Module):
    """RoIAlign feat (N,256,7,7) → (cls_score (N,81), bbox_pred (N,320)).

    캐스케이드 계열(`CascadeRoIHead`·HTC·SCNet)은 `bbox_head` 가 **ModuleList** 다 —
    단계마다 박스를 정제하고 다음 단계에서 그 박스로 RoIAlign 을 다시 한다.
    그래서 단계 하나만 담고, 루프는 러너가 돈다(호스트 RoIAlign 이 사이에 끼므로
    한 그래프로 못 돈다 — two-stage 와 같은 이유).
    """
    def __init__(self, det, stage=None):
        super().__init__()
        bh = det.roi_head.bbox_head
        self.bbox_head = bh if stage is None else bh[stage]
        # ⚠️ C4 계열은 RoIAlign 과 bbox_head 사이에 **`shared_head`(ResLayer=C5)** 가 있다.
        #    빼먹으면 채널이 안 맞아 `mat1 and mat2 shapes cannot be multiplied
        #    (N×1024 and 2048×81)` 로 죽는다(tridentnet 실측). mmdet `_bbox_forward` 와
        #    같은 순서다: extractor → shared_head → bbox_head.
        self.shared_head = (det.roi_head.shared_head
                            if getattr(det.roi_head, "with_shared_head", False) else None)
        # ⚠️ Double-Head R-CNN 은 head 가 **입력 두 개**를 받는다(`forward(x_cls, x_reg)`).
        #    같은 proposal 로 RoIAlign 을 두 번 하는데, 회귀용은 상자를 `reg_roi_scale_factor`
        #    배(1.3) 키워 넓은 맥락을 본다. g2c 그래프는 입력이 하나뿐이라
        #    **배치로 이어붙여** 받고 여기서 가른다(앞 절반=cls, 뒤 절반=reg).
        self.two_in = getattr(det.roi_head, "reg_roi_scale_factor", None) is not None
        # ⚠️ 가르는 지점을 **export 시점 상수**로 박는다. `roi_feat.shape[0] // 2` 로 두면
        #    trace 가 실행 중 값으로 봐서 렌더러가 시작 인덱스를 모르고 **offset 0 으로
        #    폴백**한다 — 두 슬라이스가 같은 자리를 읽어 확대판 RoI 가 무시된다(L1 0.740).
        #    proposal 개수는 `test_cfg.rpn.max_per_img` 로 이미 정해져 있다.
        self.n_half = int(det.test_cfg.rpn.max_per_img) if self.two_in else 0
        _fold_normed_linear(self.bbox_head)

    def forward(self, roi_feat):
        # ⚠️ **`self.<새속성>` 을 그냥 읽지 마라.** 이 클래스는 `torch.save` 로 통째 절여지는데,
        #    저장은 **오래 사는 워커 프로세스**(스윕 시작 때 import 한 옛 코드)가 하고
        #    불러오기는 **새 서브프로세스**(새 코드)가 한다. 코드를 고치는 순간 그 사이가
        #    갈라져 `AttributeError: no attribute 'two_in'` 이 난다(스윕 도중 실측).
        #    새 속성은 항상 `getattr(..., 기본값)` 으로 읽는다.
        if getattr(self, "shared_head", None) is not None:
            roi_feat = self.shared_head(roi_feat)
        if getattr(self, "two_in", False):
            m = getattr(self, "n_half", 0) or roi_feat.shape[0] // 2
            return self.bbox_head(roi_feat[:m], roi_feat[m:])
        return self.bbox_head(roi_feat)


class _NormedCls(nn.Module):
    """`NormedLinear` 를 trace 되는 연산으로 바꾼다 (seesaw_loss 의 분류기).

    mmdet 원식(`normed_predictor.py`):
        w_ = w / (||w||_row^power + eps)          ← **추론에서 상수다**
        x_ = x / (||x||_row^power + eps) * T
        out = x_ @ w_^T + b

    가중치 정규화는 상수이므로 **여기서 미리 접어** 평범한 Linear 로 만든다. 남는 것은
    입력 정규화와 온도뿐인데, `Tensor.norm` 은 trace 로 안 잡히므로 sqrt(sum(x*x)) 로
    풀어 쓴다 — 값은 같고 렌더러가 아는 연산만 남는다.

    ⚠️ 온도를 빼먹으면 **크래시 없이 점수만** 틀린다. 행 전체에 같은 배수가 곱해지지만
       softmax 는 스케일 불변이 아니고(bias 가 뒤에 더해져 상쇄도 안 된다), 온도 20 은
       분포를 크게 날카롭게 만든다.
    """
    def __init__(self, lin):
        super().__init__()
        power = float(getattr(lin, "power", 1.0))
        if power != 1.0:
            raise NotImplementedError(f"NormedLinear power={power} — 1.0 만 접을 수 있다")
        w = lin.weight.detach()
        eps = float(getattr(lin, "eps", 1e-6))
        wn = w / (w.norm(dim=1, keepdim=True) + eps)
        self.fc = nn.Linear(w.shape[1], w.shape[0], bias=lin.bias is not None)
        with torch.no_grad():
            self.fc.weight.copy_(wn)
            if lin.bias is not None:
                self.fc.bias.copy_(lin.bias.detach())
        self.t = float(getattr(lin, "tempearture", 20.0))   # mmdet 의 철자 그대로다
        self.eps = eps

    def forward(self, x):
        n = torch.sqrt((x * x).sum(dim=1, keepdim=True))
        return self.fc(x / (n + self.eps) * self.t)


def _fold_normed_linear(head):
    """head 의 `NormedLinear` 예측기를 `_NormedCls` 로 갈아 끼운다(있을 때만)."""
    for name in ("fc_cls", "fc_reg"):
        m = getattr(head, name, None)
        if m is not None and type(m).__name__ == "NormedLinear":
            setattr(head, name, _NormedCls(m))


def num_bbox_stages(det):
    """RoI bbox head 단계 수. 1 이면 평범한 two-stage."""
    bh = getattr(getattr(det, "roi_head", None), "bbox_head", None)
    return len(bh) if isinstance(bh, nn.ModuleList) else 1


class MaskRCNN_SubC(nn.Module):
    """mask RoIAlign feat (M,256,14,14) → mask_logits (M, num_classes, 28, 28). (Mask R-CNN)."""
    def __init__(self, det):
        super().__init__()
        self.mask_head = det.roi_head.mask_head

    def forward(self, mask_feat):
        return self.mask_head(mask_feat)


class MSRCNN_SubD(nn.Module):
    """mask-IoU head. 이어붙인 (M,257,14,14) → mask_iou (M, num_classes). (Mask Scoring R-CNN).

    ⚠️ **mmdet 의 `MaskIoUHead.forward` 를 통째로 태우지 않는다.** 그 forward 는 앞부분에서
    `mask_preds[range(M), labels]` 로 **행마다 다른 채널**을 고른 뒤 sigmoid·maxpool·concat
    까지 한다. 라벨은 실행할 때까지 모르는 값이라 trace 로는 고정 그래프가 안 나온다
    (`index_put_`/`gather` 가 그래프에 안 실려 **크래시 없이 0번 채널**을 고르게 된다).
    그래서 그 앞부분은 호스트가 하고, 여기는 conv/fc 만 맡는다.
    """
    def __init__(self, det):
        super().__init__()
        h = det.roi_head.mask_iou_head
        self.convs, self.fcs = h.convs, h.fcs
        self.fc_mask_iou, self.relu = h.fc_mask_iou, h.relu

    def forward(self, x):
        for conv in self.convs:
            x = self.relu(conv(x))
        x = x.flatten(1)
        for fc in self.fcs:
            x = self.relu(fc(x))
        return self.fc_mask_iou(x)


class GridRCNN_SubE(nn.Module):
    """grid head. RoI feat (M,256,14,14) → 격자점 히트맵 (M, grid_points, 56, 56).

    ⚠️ `GridHead.forward` 는 dict 를 낸다(`fused`/`unfused`). 추론에서 둘은 **같은
    텐서**이므로 하나만 낸다 — dict 를 그대로 두면 g2c 가 출력을 못 잡는다.
    `test_mode` 를 세워 학습용 분기(unfused 를 따로 계산)를 끈다.
    """
    def __init__(self, det):
        super().__init__()
        self.grid_head = det.roi_head.grid_head
        self.grid_head.test_mode = True

    def forward(self, grid_feat):
        return self.grid_head(grid_feat)["fused"]


def frcnn_cfg(det, size=800):
    """host 부품(rpn_proposals/roi_align/detect_roi)용 config 추출 → .frcnn.json."""
    rh = det.rpn_head
    # ⚠️ RPN 이 표준 `AnchorGenerator` 를 안 쓰는 계열이 있다. 호스트 `rpn_proposals` 는
    #    (strides, scale, ratios) 로 앵커를 깔아 디코드하는 구조라 그대로는 못 쓴다.
    #    **`AttributeError` 로 흘려보내지 말고** 왜 안 되는지 말한다 — 그래야 "미지원" 과
    #    "우리 버그" 가 구분된다.
    if not hasattr(rh, "prior_generator"):
        raise NotImplementedError(
            f"{type(rh).__name__}: 표준 앵커 RPN 이 아니다. "
            "CascadeRPNHead=다단계 정제(stages), GARPNHead=앵커 모양을 예측, "
            "EmbeddingRPNHead=학습된 proposal — 각각 호스트 proposal 생성이 달라진다")
    pg = rh.prior_generator
    bc = rh.bbox_coder
    ext = det.roi_head.bbox_roi_extractor
    # 캐스케이드는 extractor 도 ModuleList 다(단계마다 하나). 설정은 전부 같으므로 0번을 쓴다.
    if isinstance(ext, nn.ModuleList):
        ext = ext[0]
    # ⚠️ bbox extractor 가 `GenericRoIExtractor` 면 **레벨을 고르지 않고 전 레벨을 합친다**
    #    (게다가 groie 는 레벨마다 5x5 conv + GeneralizedAttention 을 건다). 호스트
    #    RoIAlign 은 그걸 표현 못 한다 — 조용히 레벨 선택으로 떨어뜨리면 값만 틀린다.
    if type(ext).__name__ == "GenericRoIExtractor":
        raise NotImplementedError(
            "GenericRoIExtractor(bbox): 전 레벨 집계 + pre/post 모듈이라 호스트 RoIAlign 으로 "
            "표현 못 한다. 레벨별 conv 가 그래프에 들어가야 한다")
    bh = det.roi_head.bbox_head
    # 캐스케이드는 bbox_head 도 ModuleList 다. 클래스 수·coder 종류는 단계 공통이라
    # **마지막 단계**를 쓴다(최종 박스를 내는 단계라 디코드 규약이 거기 맞춰져 있다).
    if isinstance(bh, nn.ModuleList):
        bh = bh[-1]
    rpn_c, rcnn_c = det.test_cfg.rpn, det.test_cfg.rcnn
    strides = [s[0] if isinstance(s, (tuple, list)) else int(s) for s in pg.strides]
    scales = pg.scales.tolist() if hasattr(pg.scales, "tolist") else list(pg.scales)
    mask = {}
    if getattr(det.roi_head, "with_mask", False):
        mext = det.roi_head.mask_roi_extractor
        if isinstance(mext, nn.ModuleList):
            mext = mext[0]
        mask = {
            "has_mask": True,
            "mask_roi_out": int(mext.roi_layers[0].output_size[0]),   # 14
            "mask_strides": [int(s) for s in mext.featmap_strides],
            # ⚠️ `GenericRoIExtractor`(PointRend 의 마스크 경로)에는 `finest_scale` 이 없다.
            #    전 레벨을 합치므로 레벨 선택 개념 자체가 없다. **bbox 검증에는 안 쓰이는
            #    값이라** 여기서 죽으면 안 된다 — 없으면 기본값으로 둔다.
            "mask_finest_scale": int(getattr(mext, "finest_scale", 56)),
            "mask_thr_binary": float(rcnn_c.mask_thr_binary),
        }
    # Mask Scoring R-CNN: 마스크 IoU 로 점수를 다시 매긴다. 없으면 안 싣는다.
    if getattr(det.roi_head, "mask_iou_head", None) is not None:
        mih = det.roi_head.mask_iou_head
        mask["has_mask_iou"] = True
        # ⚠️ 마지막 conv 만 stride 2 다(14→7). 호스트가 이어붙일 채널 수(+1)와 함께
        #    여기서 명시한다 — 런너가 shape 을 추측하면 조용히 어긋난다.
        mask["mask_iou_in_channels"] = int(mih.in_channels)          # 256 (+1 은 런너가 붙인다)
        mask["mask_iou_num_classes"] = int(mih.num_classes)

    # ⚠️ **점수 활성이 softmax 가 아닌 계열이 있다.** SeesawLoss 는 `custom_activation`
    #    을 세우고 채널을 `num_classes + 2` 로 쓴다 — 앞 C 개는 클래스, 뒤 2개는
    #    전경/배경 objectness 다. 활성도 두 단계다(각각 softmax → 클래스 점수에 전경 확률을
    #    곱하고 배경 확률을 뒤에 붙인다). 평범한 softmax 로 읽으면 **채널 수부터 틀리고**
    #    (C+2 를 "C+1 클래스 + 배경" 으로 오해한다) 점수도 통째로 달라진다.
    lc = getattr(bh, "loss_cls", None)
    if getattr(lc, "custom_activation", False):
        #    (러너의 tiny_json 은 숫자·배열만 읽는다 → 문자열 대신 플래그로 싣는다)
        mask["seesaw_activation"] = 1
        mask["cls_num_classes"] = int(bh.num_classes)

    # CrowdDet: proposal 하나가 사람 둘을 낸다고 보고 (cls, box) 쌍을 2벌 낸다.
    # 디코드도 NMS 도 다르다 — **같은 proposal 에서 나온 상자끼리는 서로 안 누른다**(set-NMS).
    ni = int(getattr(bh, "num_instance", 1) or 1)
    if ni > 1:
        mask["num_instance"] = ni
        # 정제(refine) 분기가 있으면 mmdet 은 **정제된 쌍**으로 예측한다
        # (`multi_instance_roi_head.py:102` — cls_score_ref/bbox_pred_ref).
        mask["with_refine"] = bool(getattr(bh, "with_refine", False))

    # Grid R-CNN: bbox head 는 회귀 분기가 없고(`with_reg=False`) 격자점 히트맵으로
    # 박스를 다시 낸다. 디코드에 필요한 상수를 전부 여기서 뽑는다 — 러너가 재계산하면
    # 두 곳이 갈린다(`calc_sub_regions` 는 정수 절단이 섞여 있어 특히 위험하다).
    gh = getattr(det.roi_head, "grid_head", None)
    if gh is not None:
        # ⚠️ **grid head 의 deconv 는 grouped + padded 다**(groups=grid_points=9,
        #    padding=(k-2)//2=1). ggml 이 가진 것은 `ggml_conv_transpose_2d_p0` 하나이고
        #    이름 그대로 padding 0 · groups 1 전용이다. 그대로 태우면 출력이 2px 크고
        #    채널 묶음이 섞여 `add_bias_2d` 에서 `ggml_can_repeat` 로 죽는다.
        #    **크래시로 두면 "우리 버그" 처럼 보인다** — 왜 안 되는지 여기서 말한다.
        d1 = gh.deconv1
        if getattr(d1, "groups", 1) != 1 or any(v != 0 for v in getattr(d1, "padding", (0, 0))):
            raise NotImplementedError(
                f"GridHead.deconv: groups={getattr(d1, 'groups', 1)} · "
                f"padding={tuple(getattr(d1, 'padding', (0, 0)))} — ggml 의 conv_transpose 는 "
                "padding 0 · groups 1 만 한다(ggml_conv_transpose_2d_p0). 그룹별로 쪼개 돌리고 "
                "가장자리를 잘라내는 분해가 필요하다 — 디코드가 아니라 연산 부족이다")
        gext = det.roi_head.grid_roi_extractor
        if isinstance(gext, nn.ModuleList):
            gext = gext[0]
        mask["has_grid"] = True
        mask["grid_roi_out"] = int(gext.roi_layers[0].output_size[0])       # 14
        mask["grid_strides"] = [int(v) for v in gext.featmap_strides]
        mask["grid_finest_scale"] = int(getattr(gext, "finest_scale", 56))
        mask["grid_points"] = int(gh.grid_points)
        mask["grid_size"] = int(gh.grid_size)
        mask["whole_map_size"] = int(gh.whole_map_size)
        # (x1,y1) 만 쓴다 — 히트맵을 전체 좌표로 옮기는 오프셋이다.
        mask["grid_sub_xy"] = [float(v) for r in gh.sub_regions for v in r[:2]]

    # HTC 의 시맨틱 융합 파라미터. 없으면 안 싣는다.
    sem = {}
    if getattr(det.roi_head, "semantic_head", None) is not None:
        sext = det.roi_head.semantic_roi_extractor
        sem = {
            "has_semantic": True,
            "sem_roi_out": int(sext.roi_layers[0].output_size[0]),      # 14
            "sem_stride": float(sext.featmap_strides[0]),               # 8
            "sem_channels": int(sext.out_channels),
            "sem_sampling_ratio": int(sext.roi_layers[0].sampling_ratio),
            "sem_aligned": bool(sext.roi_layers[0].aligned),
        }
    if getattr(det.roi_head, "glbctx_head", None) is not None:
        sem["has_glbctx"] = True
    ns = num_bbox_stages(det)
    heads = det.roi_head.bbox_head
    heads = list(heads) if ns > 1 else [heads]
    return {**mask, **sem,
        # 캐스케이드: 단계 수와 **단계별 bbox 정규화 상수**. 단계마다 다르다
        # (예: [0.1,0.1,0.2,0.2] → [0.05,0.05,0.1,0.1] → [0.033,0.033,0.067,0.067]).
        "num_bbox_stages": ns,
        "stage_stds": [list(map(float, h.bbox_coder.stds)) for h in heads],
        "stage_means": [list(map(float, h.bbox_coder.means)) for h in heads],
        "img_size": int(size),
        # RPN
        "rpn_strides": [float(s) for s in strides],
        "rpn_scale": float(scales[0]),
        "rpn_ratios": [float(r) for r in pg.ratios.tolist()],
        "rpn_means": [float(v) for v in bc.means], "rpn_stds": [float(v) for v in bc.stds],
        "rpn_nms_pre": int(rpn_c.nms_pre), "rpn_nms_thr": float(rpn_c.nms.iou_threshold),
        "rpn_max": int(rpn_c.max_per_img),
        # ⚠️ 아래 셋은 **기본값이 아닌 계열이 있다**(crowddet 이 셋 다 다르다).
        #    전부 shape 를 안 바꾸므로 빼먹으면 크래시 없이 proposal 만 달라진다.
        #    `centers` — 주어지면 그 값이 base anchor 중심이고 center_offset 은 무시된다.
        "rpn_centers": [float(v) for c in (getattr(pg, "centers", None) or []) for v in c],
        #    `clip_border` — false 면 proposal 을 이미지로 안 자른다(800 입력에 1054 가 나온다).
        "rpn_clip_border": 1.0 if getattr(bc, "clip_border", True) else 0.0,
        #    `min_bbox_size` — NMS **앞에서** 작은 상자를 버린다.
        "rpn_min_bbox_size": float(getattr(rpn_c, "min_bbox_size", 0) or 0),
        #    `cls_out_channels` — RPNHead 는 보통 1(sigmoid)이지만 loss_cls 에
        #    use_sigmoid 를 안 적으면 2(softmax, 전경 index 0)가 된다. **모듈에서 읽는다.**
        "rpn_cls_out_channels": int(getattr(rh, "cls_out_channels", 1) or 1),
        # RoIAlign
        # ⚠️ **RoI feature 채널을 256 으로 박으면 안 된다.** FPN 계열은 256 이지만
        #    C4 계열(TridentNet)은 neck 이 없어 백본 C4 채널(1024)이 그대로 온다.
        # Double-Head: 회귀용 RoI 는 상자를 이만큼 키워 다시 자른다(0 이면 안 씀).
        "reg_roi_scale_factor": float(getattr(det.roi_head, "reg_roi_scale_factor", 0.0) or 0.0),
        "roi_channels": int(ext.out_channels),
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
