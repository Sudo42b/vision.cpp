"""mmdet_families.py — 계열 → (대표 config, 체크포인트) 매핑. **검증 도구들이 공유한다.**

왜 공유해야 하나
---------------
config 와 체크포인트는 **짝이 맞아야** 한다. 도구마다 대표 config 를 따로 고르면
`fetch_checkpoints.py` 가 받은 가중치와 `verify_heads.py` 가 컴파일하는 config 가
어긋나고, 그러면 로드는 되는데(strict=False) 일부가 랜덤으로 남아 **조용히 틀린다.**

그리고 매핑을 **손으로 적으면** 목록에 없는 계열이 존재하지 않는 것처럼 보인다 —
`pisa`·`rpn` 이 실제로 그랬다. mmdet 은 계열마다 `metafile.yml` 에
`Config → Weights` 를 적어 둔다. 그걸 읽는다.
"""
import glob
import os

# 모델이 아닌 폴더(데이터셋 변형·학습 레시피 변형·뼈대 조각). 대조 스크립트로 확인했다 —
# 이 안에 다른 폴더에 없는 아키텍처는 하나도 없다. `sweep_integ.py` 와 같은 목록.
SKIP_DIRS = {"_base_", "common", "misc", "legacy_1.x", "strong_baselines", "selfsup_pretrain",
             "scratch", "dsdl", "objects365", "lvis", "openimages", "cityscapes", "wider_face",
             "pascal_voc", "deepfashion", "v3det"}

# 계열별 시험 이미지. **여기 없으면 하네스의 기본 이미지를 쓴다.**
#
# ⚠️ 왜 필요한가 — MOT 트래커는 `num_classes=1`(보행자)로 학습됐다. 고양이 사진을 넣으면
#    mmdet 쪽도 우리 쪽도 **0건**이 나온다. 그러면 하네스가 `EMPTY`("한쪽이 비었다")를
#    내므로 **조용히 통과하지는 않지만**, 그 계열을 아예 못 재게 된다. 못 잰 것을
#    "대상이 못 한다" 로 적지 않으려면 맞는 사진을 줘야 한다.
#
# `bench-image.jpg` 는 이미 저장소에 있고(새 자산·라이선스 불필요) 다섯 계열 전부에서
# 2~5건이 나오는 것을 실측했다(2026-08-19, 800px·score>0.30):
#   deepsort 2 · sort 2 · qdtrack 3 · masktrack_rcnn 5 · bytetrack 2
_IMAGE = {f: "bench-image.jpg" for f in
          ("bytetrack", "deepsort", "ocsort", "qdtrack", "sort", "strongsort",
           "masktrack_rcnn")}
# 이미지는 **저장소 안**에 있다. 사용자가 준 `--image` 의 디렉토리에서 찾으면
# 엉뚱한 곳(`~/pics/bench-image.jpg`)을 가리키고, 상대경로면 하위 프로세스의 cwd 로 풀린다.
# ⚠️ `..` 은 **넷**이다. dense_head → mmdet → verify → tools → vision.cpp.
#    셋이면 `tools/tests/input` 이라 없는 경로다. 그러면 `test_image` 가 default 로
#    조용히 떨어져 **위 매핑이 통째로 무효가 된다** — 트래커를 사람 없는 이미지로 재고
#    양쪽 0건이 나와 `EMPTY` 로 찍힌다(2026-09-01 실측).
_IMG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        "..", "..", "..", "..", "tests", "input"))


def test_image(fam, default):
    """이 계열로 잴 때 쓸 이미지 경로. 지정이 없거나 파일이 없으면 `default` 그대로."""
    name = _IMAGE.get(fam)
    if not name:
        return default
    p = os.path.join(_IMG_DIR, name)
    return p if os.path.exists(p) else default


def pick_cfgs(fam_dir):
    """대표 config 후보(점수순). `_` 로 시작하거나 `_base` 로 끝나는 뼈대는 뺀다.

    후보를 여럿 돌려주는 이유: 1순위가 없는 형제 config 를 상속해 죽는 계열이 있다
    (`sort` 의 mot20 → FileNotFoundError). 차순위까지 시도해야 한다.
    """
    cands = [f for f in glob.glob(os.path.join(fam_dir, "*.py"))
             if not os.path.basename(f).startswith("_")
             and not os.path.splitext(os.path.basename(f))[0].endswith("_base")]

    def score(f):
        n, s = os.path.basename(f), 0
        for kw, w in (("r50", 4), ("fpn", 3), ("1x", 3), ("coco", 4), ("r18", 2)):
            if kw in n:
                s += w
        return -s
    return sorted(cands, key=score)


def weights_map(configs_root, fam):
    """{config 파일명: 체크포인트 URL} — 그 계열 `metafile.yml` 에서."""
    import yaml
    mf = os.path.join(configs_root, fam, "metafile.yml")
    if not os.path.exists(mf):
        return {}
    try:
        d = yaml.safe_load(open(mf)) or {}
    except Exception:
        return {}
    out = {}
    for m in (d.get("Models") or []):
        cfg, w = m.get("Config"), m.get("Weights")
        # ⚠️ `Weights` 가 **리스트인 계열이 있다**(여러 체크포인트 나열). 첫 개를 쓴다.
        if isinstance(w, (list, tuple)):
            w = w[0] if w else None
        if cfg and isinstance(w, str) and w.startswith("http"):
            out[os.path.basename(cfg)] = w
    return out


def resolve(configs_root, fam):
    """(config 상대경로, 체크포인트 파일명, URL). 못 찾으면 해당 항목이 None.

    **metafile 에 가중치가 있는 첫 후보**를 고른다 — 그래야 config 와 체크포인트가 짝이 맞는다.
    가중치가 하나도 없으면 1순위 config 와 `None` 을 돌려준다(그 계열은 CKPT_NONE 이 된다).
    """
    cands = pick_cfgs(os.path.join(configs_root, fam))
    if not cands:
        return None, None, None
    wm = weights_map(configs_root, fam)
    for cfg in cands:
        url = wm.get(os.path.basename(cfg))
        if url:
            return os.path.relpath(cfg, configs_root), os.path.basename(url), url
    return os.path.relpath(cands[0], configs_root), None, None


def families(configs_root):
    """[(계열, config 상대경로, 체크포인트 파일명 or None)] — 아키텍처 계열 전부."""
    out = []
    for name in sorted(os.listdir(configs_root)):
        d = os.path.join(configs_root, name)
        if not os.path.isdir(d) or name in SKIP_DIRS:
            continue
        cfg, ckpt, _ = resolve(configs_root, name)
        if cfg:
            out.append((name, cfg, ckpt))
    return out

# ── 손으로 고른 대표 config·체크포인트 ────────────────────────────────────────
# ⚠️ **두 하네스가 같은 목록을 봐야 한다.** 예전엔 one-stage 는 이 손목록을, two-stage 는
#    metafile 을 봐서 **같은 계열이 다르게 풀렸다.** 그러면 한쪽으로 굽고 다른 쪽으로 재게
#    되고, 남의 가중치를 진 그래프를 대조하면서 수치가 그럴듯하게 틀린다(retinanet 20px ·
#    tood 13.89px 를 그렇게 결함으로 오해했다). 짝은 여기 한 곳에서만 온다.
#
# metafile 이 고르는 것과 **13계열이 갈린다**. 대표적으로 retinanet(r18 vs r50-caffe)·
# tood(anchor-free vs anchor-based)·pvt(PVT-Tiny vs PVTv2-B5, 아예 다른 아키텍처)다.
_OVERRIDE_LIST = [
    ("retinanet", "retinanet/retinanet_r18_fpn_1x_coco.py",
     "retinanet_r18_fpn_1x_coco_20220407_171055-614fd399.pth"),
    ("atss", "atss/atss_r50_fpn_1x_coco.py",
     "atss_r50_fpn_1x_coco_20200209-985f7bd0.pth"),
    ("paa", "paa/paa_r50_fpn_1x_coco.py",
     "paa_r50_fpn_1x_coco_20200821-936edec3.pth"),
    ("fcos", "fcos/fcos_r50-caffe_fpn_gn-head_1x_coco.py",
     "fcos_r50_caffe_fpn_gn-head_1x_coco-821213aa.pth"),
    ("gfl", "gfl/gfl_r50_fpn_1x_coco.py",
     "gfl_r50_fpn_1x_coco_20200629_121244-25944287.pth"),
    ("vfnet", "vfnet/vfnet_r50_fpn_1x_coco.py",
     "vfnet_r50_fpn_1x_coco_20201027-38db6f58.pth"),
    ("reppoints", "reppoints/reppoints-moment_r50_fpn_1x_coco.py",
     "reppoints_moment_r50_fpn_1x_coco_20200330-b73db8d1.pth"),
    ("tood", "tood/tood_r50_fpn_1x_coco.py",
     "tood_r50_fpn_1x_coco_20211210_103425-20e20746.pth"),

    # ⚠️ `reid` 는 **자기 metafile 에 가중치가 없다** — metafile 만 보면 `CKPT_NONE` 으로
    #    멈춘다. 학습 가중치는 없는 게 아니라 **다른 데 있다**: 트래커 config 가
    #    `reid.init_cfg.checkpoint` 로 가리킨다(deepsort·sort 가 같은 것을 쓴다).
    ("reid", "reid/reid_r50_8xb32-6e_mot15train80_test-mot15val20.py",
     "tracktor_reid_r50_iter25245-a452f51f.pth"),

    # ⚠️ `fast_rcnn` 도 자기 가중치가 없다. **못 하는 게 아니라 metafile 이 안 낸다** —
    #    이 계열은 미리 뽑은 proposal 을 입력으로 받는 구조라 단독 배포가 없다.
    #    구조는 `faster_rcnn` 에서 RPN 만 뺀 것과 **같다**(ResNet-50 + FPN +
    #    StandardRoIHead + Shared2FCBBoxHead) — 백본·넥·RoI head 이름이 그대로 맞으므로
    #    그 가중치를 빌린다. 남는 `rpn_head.*` 는 안 붙고 버려진다.
    ("fast_rcnn", "fast_rcnn/fast-rcnn_r50_fpn_1x_coco.py",
     "faster_rcnn_r50_fpn_iou_1x_coco_20200506_095954-938e81f0.pth"),

    # 위 8계열의 head 를 **그대로 상속**한 계열들. 손실이나 백본만 다르므로 조립기는
    # 같은 것을 탄다. 새 함수를 쓰기 전에 이런 게 있는지 먼저 본다.
    ("ghm", "ghm/retinanet_r50_fpn_ghm-1x_coco.py",              # RetinaHead
     "retinanet_ghm_r50_fpn_1x_coco_20200130-a437fda3.pth"),
    ("pvt", "pvt/retinanet_pvt-t_fpn_1x_coco.py",                # RetinaHead
     "retinanet_pvt-t_fpn_1x_coco_20210831_103110-17b566bd.pth"),
    ("free_anchor", "free_anchor/freeanchor_r50_fpn_1x_coco.py", # RetinaHead 상속
     "retinanet_free_anchor_r50_fpn_1x_coco_20200130-0f67375f.pth"),
    ("fsaf", "fsaf/fsaf_r50_fpn_1x_coco.py",                     # RetinaHead 상속
     "fsaf_r50_fpn_1x_coco-94ccc51f.pth"),
    ("dyhead", "dyhead/atss_r50-caffe_fpn_dyhead_1x_coco.py",    # ATSSHead
     "atss_r50_fpn_dyhead_for_reproduction_4x4_1x_coco_20220107_213939-162888e6.pth"),
    ("nas_fcos", "nas_fcos/nas-fcos_r50-caffe_fpn_nashead-gn-head_4xb4-1x_coco.py",  # FCOSHead
     "nas_fcos_nashead_r50_caffe_fpn_gn-head_4x4_1x_coco_20200520-1bdba3ce.pth"),
    ("ld", "ld/ld_r50-gflv1-r101_fpn_1x_coco.py",                # GFLHead 상속
     "ld_r50_gflv1_r101_fpn_coco_1x_20220629_145355-8dc5bad8.pth"),
    ("lad", "lad/lad_r101-paa-r50_fpn_2xb8_coco_1x.py",          # PAAHead 상속
     "lad_r101_paa_r50_fpn_coco_1x_20220708_124357-9407ac54.pth"),


    # ── 텍스트+이미지 계열 ─────────────────────────────────────────────────
    # 언어 모델(BERT)이 함께 들어 있다. head 는 ATSS/DINO 계열을 상속하므로 조립기가
    # 있을 수도 있는데, **재본 적이 없어서** 결과를 말할 수 없었다 → 체크포인트를 받아 등록한다.
    ("glip", "glip/glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365.py",
     "glip_tiny_a_mmdet-b3654169.pth"),
    ("grounding_dino", "grounding_dino/grounding_dino_swin-t_finetune_16xb2_1x_coco.py",
     "groundingdino_swint_ogc_mmdet-822d7e9d.pth"),
    ("mm_grounding_dino", "mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365.py",
     "grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth"),

    # ── 아직 조립기가 없는 계열 ─────────────────────────────────────────────
    # 여기 있다고 지원한다는 뜻이 아니다. **어디서 어떻게 막히는지 재려고** 둔다 —
    # 실패도 기록해야 다음 사람이 같은 걸 다시 조사하지 않는다.
    ("ddod", "ddod/ddod_r50_fpn_1x_coco.py",
     "ddod_r50_fpn_1x_coco_20220523_223737-29b2fc67.pth"),
    ("autoassign", "autoassign/autoassign_r50-caffe_fpn_1x_coco.py",
     "auto_assign_r50_fpn_1x_coco_20210413_115540-5e17991f.pth"),
    ("foveabox", "foveabox/fovea_r50_fpn_4xb4-1x_coco.py",
     "fovea_r50_fpn_4x4_1x_coco_20200219-ee4d5303.pth"),
    ("yolof", "yolof/yolof_r50-c5_8xb8-1x_coco.py",
     "yolof_r50_c5_8x8_1x_coco_20210425_024427-8e864411.pth"),
    ("efficientnet", "efficientnet/retinanet_effb3_fpn_8xb4-crop896-1x_coco.py",
     "retinanet_effb3_fpn_crop896_8x4_1x_coco_20220322_234806-615a0dda.pth"),
    ("nas_fpn", "nas_fpn/retinanet_r50_fpn_crop640-50e_coco.py",
     "retinanet_r50_fpn_crop640_50e_coco-9b953d76.pth"),
    ("ssd", "ssd/ssd300_coco.py",
     "ssd300_coco_20210803_015428-d231a06e.pth"),
    ("yolo", "yolo/yolov3_d53_8xb8-320-273e_coco.py",
     "yolov3_d53_320_273e_coco-421362b6.pth"),
    ("yolox", "yolox/yolox_s_8xb8-300e_coco.py",
     "yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth"),
    ("rtmdet", "rtmdet/rtmdet_tiny_8xb32-300e_coco.py",
     "rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth"),
    ("centernet", "centernet/centernet_r18-dcnv2_8xb16-crop512-140e_coco.py",
     "centernet_resnet18_dcnv2_140e_coco_20210702_155131-c8cd631f.pth"),
    ("cornernet", "cornernet/cornernet_hourglass104_10xb5-crop511-210e-mstest_coco.py",
     "cornernet_hourglass104_mstest_10x5_210e_coco_20200824_185720-5fefbf1c.pth"),
    ("centripetalnet", "centripetalnet/centripetalnet_hourglass104_16xb6-crop511-210e-mstest_coco.py",
     "centripetalnet_hourglass104_mstest_16x6_210e_coco_20200915_204804-3ccc61e5.pth"),
    ("yolact", "yolact/yolact_r50_1xb8-55e_coco.py",
     "yolact_r50_1x8_coco_20200908-f38d58df.pth"),
    ("condinst", "condinst/condinst_r50_fpn_ms-poly-90k_coco_instance.py",
     "condinst_r50_fpn_ms-poly-90k_coco_instance_20221129_125223-4c186406.pth"),
    ("boxinst", "boxinst/boxinst_r50_fpn_ms-90k_coco.py",
     "boxinst_r50_fpn_ms-90k_coco_20221228_163052-6add751a.pth"),

    # DETR 계열 — transformer decoder 라 conv 타워 구조 자체가 없다. 별개 작업이다.
    ("detr", "detr/detr_r50_8xb2-150e_coco.py",
     "detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth"),
    ("conditional_detr", "conditional_detr/conditional-detr_r50_8xb2-50e_coco.py",
     "conditional-detr_r50_8xb2-50e_coco_20221121_180202-c83a1dc0.pth"),
    ("dab_detr", "dab_detr/dab-detr_r50_8xb2-50e_coco.py",
     "dab-detr_r50_8xb2-50e_coco_20221122_120837-c1035c8c.pth"),
    ("deformable_detr", "deformable_detr/deformable-detr_r50_16xb2-50e_coco.py",
     "deformable-detr_r50_16xb2-50e_coco_20221029_210934-6bc7d21b.pth"),
    ("dino", "dino/dino-4scale_r50_8xb2-12e_coco.py",
     "dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth"),
    ("ddq", "ddq/ddq-detr-4scale_r50_8xb2-12e_coco.py",
     "ddq-detr-4scale_r50_8xb2-12e_coco_20230809_170711-42528127.pth"),
]
OVERRIDE = {f: (c, k) for f, c, k in _OVERRIDE_LIST}


def resolve_pair(configs_root, fam):
    """(config 절대경로, 체크포인트 파일명) — **하네스가 실제로 쓰는 짝.**

    손목록이 있으면 그것, 없으면 metafile. 두 하네스 다 이걸 부른다.
    """
    import os as _os
    if fam in OVERRIDE:
        c, k = OVERRIDE[fam]
        return _os.path.join(configs_root, c), k
    c, k, _ = resolve(configs_root, fam)
    return (_os.path.join(configs_root, c) if c else None), k
