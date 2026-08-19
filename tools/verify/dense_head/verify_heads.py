#!/usr/bin/env python3
"""tools/detect/head.cpp 가 mmdet 의 bbox_head 를 재현하는지 계열별로 잰다.

백본만 컴파일하고 head 는 C++ 부품이 조립한다. 계열마다:
mmdet → backbone .pt + 파라미터 헤더 → 컴파일 → run_mmdet 빌드 →
MMDET_DUMP_HEAD 로 **디코드 전** 원시 텐서 덤프 → torch bbox_head 출력과 대조.
점수는 **출력 텐서의 상대 L1/L2 거리**다(cosine 은 스케일 불변이라 안 쓴다).

디코드/NMS 앞에서 끊는 이유: NMS 를 거치면 어느 텐서가 틀렸는지 못 짚는다.

    python verify_heads.py              # 8계열 전부
    python verify_heads.py vfnet tood   # 골라서

체크포인트가 `~/mmbuild/mmdetection/checkpoints/` 에 있어야 한다.
**랜덤 초기화로 재지 마라** — 항등 초기값(γ=1·β=0, scale=1)이 빠진 연산을 덮어
검증을 통과시킨다. 실제로 VFNet 의 scale 하드코딩이 그렇게 숨어 있었다.
"""
import os
import re
import subprocess
import sys

# (계열, config, 학습된 체크포인트). **랜덤 초기화로 재지 않는다** — 항등 초기값이
# 빠진 연산을 덮어 검증을 통과시킨다(group_norm affine 이 실제로 그랬다).
FAMILIES = [
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
# 설정은 **파일**에서 온다(`verify.toml`). 환경변수로 받으면 어떤 값으로 잰 숫자인지
# 로그에 안 남아 재현이 안 된다. 덮어쓰려면 `--set run.workers=2` 처럼 준다 — 그것도 찍힌다.
# ⚠️ **파이프로 보내면 블록 버퍼링**이라 30분간 아무것도 안 보이고, 중간에 죽으면 통째로
#    유실된다(위키: 진행상황이-안보이고-타임아웃때-전부유실 — 여러 세션이 반복해 밟았다).
#    우회(파일 리다이렉트) 대신 원인을 고친다: 줄 단위로 내보낸다.
sys.stdout.reconfigure(line_buffering=True)


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vconfig                                              # noqa: E402
CFG, ARGS = vconfig.load()
CKPT = CFG.ckpt
MM = CFG.configs
UNWRAP = CFG.unwrap

# ── 계열 열거 ─────────────────────────────────────────────────────────────────
# **손으로 적지 않는다.** mmdet 의 `metafile.yml` 이 계열마다 `Config → Weights` 를 갖고
# 있으므로 그걸 읽는다(`mmdet_families`). 손목록은 그 위에 얹는 **예외**로만 남긴다 —
# metafile 의 대표 config 가 우리 조립기에 안 맞는 경우가 있어서다.
#
# 목록을 손으로 적으면 거기 없는 계열이 **존재하지 않는 것처럼** 보인다.
# `pisa`·`rpn` 이 실제로 그렇게 몇 주 동안 안 보였다.
import mmdet_families                                       # noqa: E402


def _all_families():
    override = {f[0]: f for f in FAMILIES}
    out = []
    for name, cfg, ckpt in mmdet_families.families(MM):
        if name in override:
            out.append(override.pop(name))          # 손으로 고른 config/체크포인트가 우선
        else:
            out.append((name, cfg, ckpt))
    out.extend(override.values())                   # configs/ 에 없는 손목록 항목도 남긴다
    return out


ROOT = CFG.workdir
# head.cpp 를 미리 컴파일해 둘 자리(계열 무관). 지우면 다시 만든다.
HEAD_OBJ = os.path.join(ROOT, "head.o")
# 검증 빌드의 최적화 수준. 수치는 libggml 이 내므로 -O1 로 충분하다(컴파일이 빠르다).
OPT = CFG.opt
# 판정 기준: 출력 텐서의 **상대 L1/L2 거리**. 둘 다 임계 이하여야 PASS.
#   rel_L1 = Σ|a−b| / Σ|a|      평균적으로 얼마나 어긋났나
#   rel_L2 = ‖a−b‖ / ‖a‖        큰 오차에 더 민감(제곱)
# cosine 을 안 쓰는 이유: **스케일 불변**이라 크기가 통째로 틀려도 1.0 이 나온다.
# 저장소 관례(verify_pt.py)의 REL_L2_TOL=0.05 를 따른다.
L1_TOL = CFG.l1
L2_TOL = CFG.l2
# g2c(컴파일러)와 vision.cpp 경로. 이 파일은 vision.cpp/tools/verify/dense_head/ 에 있다.
V = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
P = CFG.g2c
PY = sys.executable
FE = V + "/tools/frontend/mmdet"
GGUF_PY = V + "/depend/llama/gguf-py"
SZ = CFG.size

STUB = '''import sys, types
for n in ("mmpretrain.models.multimodal.blip", "mmpretrain.models.multimodal.blip.language_model"):
    m = types.ModuleType(n); m.__path__ = []; sys.modules[n] = m
'''

# torch 기준값. **C++ 이 내는 것과 같은 지점**까지만 계산한다 — 디코드는 양쪽 다 안 한다.
REF = r'''
import _stub, os, sys, torch, numpy as np
sys.path.insert(0, "%(FE)s")
import mmdet_wrap
mmdet_wrap.trace_friendly_ops()

# ⚠️ **내보낸 .pt 를 그대로 로드한다.** config 로 다시 init 하면 랜덤 초기화가 새로 굴러
#    GGUF 와 다른 가중치가 된다 — 백본부터 cos 0 이 나온다(실제로 겪음).
fam, cfg_path, gen = sys.argv[1], sys.argv[2], sys.argv[3]
mod = torch.load("bb.pt", weights_only=False).cpu()
mod.eval()   # 체이닝 금지 — train() 오버라이드 시 None
bh = mod.bbox_head
np.random.seed(0)
x = np.random.randn(1, 3, %(SZ)d, %(SZ)d).astype("float32")
LAYOUT = "chw"          # 덤프 해석 방식. DETR 계열만 "flat" 이다(아래 참조).
with torch.no_grad():
    f = mod.backbone(torch.from_numpy(x))
    if mod.neck is not None:
        f = mod.neck(f)
    if getattr(mod, "decoder", None) is not None:
        # ⚠️ **DETR 계열은 encoder/decoder 가 detector 에 달려 있다** — `bbox_head.forward` 는
        #    decoder 출력을 받는 cls/reg 분기 둘뿐이다. 그래서 FPN feature 로는 못 부른다.
        #    detector 를 config 로 다시 세우되 **가중치는 bb.pt 것을 심는다** — config init 의
        #    랜덤이 굴러 GGUF 와 달라지는 사고를 막는다(백본부터 cos 0 이 나왔던 그 함정).
        from mmdet.apis import init_detector
        from mmdet.structures import DetDataSample
        det = init_detector(cfg_path, None, device="cpu")
        det.eval()
        # `pos_embed` 처럼 **우리가 구우려고 추가한 버퍼**는 detector 에 없다. 그것만 뺀다 —
        # 그 외에 안 붙는 키가 있으면 이름 규약이 어긋난 것이므로 시끄럽게 죽인다.
        ours = {"pos_embed", "ref_sine_embed", "ref_inv_pad",
                "dab_inv_dim_t", "dab_even", "dab_odd", "enc_ref",
                "enc_proposals", "enc_valid"}
        sd = {k: v for k, v in mod.state_dict().items() if k.split(".")[0] not in ours}
        r = det.load_state_dict(sd, strict=False)
        assert not r.unexpected_keys, ("bb.pt 키가 detector 에 안 붙는다", r.unexpected_keys[:5])
        ds = DetDataSample()
        ds.set_metainfo({"batch_input_shape": (%(SZ)d, %(SZ)d), "img_shape": (%(SZ)d, %(SZ)d)})
        head_in = det.forward_transformer(tuple(f), [ds])
        outs = bh(**head_in)
        # ⚠️ two-stage(DINO·DDQ)는 encoder 점수 상위 k 개를 query 로 고른다. 그 **점수가 극도로
        #    촘촘해서**(상위 900 인접 간격 중앙값 4.9e-04, 값 범위 −1~−4.4) fp16 가중치로는
        #    **순서를 재현할 수 없다** — 852/899 인접쌍이 fp16 잡음 안이다.
        #    출력은 원래 순서 없는 집합이므로 행을 정렬해 비교한다(집합이 틀리면 그대로 잡힌다).
        # 출력이 (decoder 층, batch, query, ch) 다 — 공간 격자가 아니다. 층을 "레벨"로 쓴다.
        LAYOUT = "set" if getattr(det, "memory_trans_fc", None) is not None else "flat"
    else:
        outs = bh(tuple(f))

# 계열별로 head 출력의 의미가 다르다. C++ 과 같은 형태로 맞춘다.
cls_l, box_l, ctr_l = [], [], []
extra = {}
if isinstance(outs, tuple) and len(outs) == 8:
    # CentripetalHead: (tl_heat, br_heat, tl_off, br_off, tl_gs, br_gs, tl_cs, br_cs).
    cls_l, box_l, ctr_l = list(outs[0]), list(outs[1]), list(outs[2])
    extra["brof"] = list(outs[3])
    extra["tlgs"], extra["brgs"] = list(outs[4]), list(outs[5])
    extra["tlcs"], extra["brcs"] = list(outs[6]), list(outs[7])
elif isinstance(outs, tuple) and len(outs) == 6:
    # CornerHead: (tl_heat, br_heat, tl_emb, br_emb, tl_off, br_off).
    # C++ 조립기와 **같은 이름**으로 맞춘다 — 안 맞으면 덤프가 안 겹쳐 조용히 통과한다.
    cls_l, box_l, ctr_l = list(outs[0]), list(outs[1]), list(outs[4])
    extra["brof"] = list(outs[5])
    if outs[2][0] is not None:
        extra["tlemb"], extra["bremb"] = list(outs[2]), list(outs[3])
elif isinstance(outs, tuple) and len(outs) == 1:
    # YOLOv3 는 `return tuple(pred_maps),` — 한 갈래를 **또 튜플로 감싸** 돌려준다.
    cls_l = list(outs[0])
elif isinstance(outs, tuple) and len(outs) == 2 and isinstance(outs[1][0], tuple):
    # SABL: `forward_single` 이 `(cls_score, (bbox_cls_pred, bbox_reg_pred))` 를 낸다 —
    # box 갈래가 **중첩 튜플**이라 그대로 쓰면 `'tuple' object has no attribute 'shape'`.
    # C++ 조립기와 **같은 이름**(`bcls`)으로 갈라야 덤프가 겹친다.
    cls_l = list(outs[0])
    box_l = [b[1] for b in outs[1]]            # bbox_reg_pred
    extra["bcls"] = [b[0] for b in outs[1]]    # bbox_cls_pred (버킷 분류)
elif isinstance(outs, tuple) and len(outs) == 3:
    cls_l, box_l, ctr_l = [list(t) for t in outs]
else:
    cls_l, box_l = [list(t) for t in outs[:2]]

# **계열 이름이 아니라 head 속성으로 판단한다** — LD 는 GFLHead 를 상속해 DFL 을 쓴다.
if getattr(bh, "reg_max", 0):
    # C++ 은 DFL 기댓값까지 낸다(디코드를 단순하게 두려고) → 기준값도 같은 지점으로.
    strides = [s[0] for s in bh.prior_generator.strides]
    with torch.no_grad():
        box_l = [bh.integral(b.permute(0, 2, 3, 1).reshape(-1, 4 * (bh.reg_max + 1)))
                 .reshape(1, b.shape[2], b.shape[3], 4).permute(0, 3, 1, 2) * s
                 for b, s in zip(box_l, strides)]

def save(tag, lst):
    for i, t in enumerate(lst):
        np.ascontiguousarray(t[0].detach().numpy()).tofile(f"ref.{tag}.{i}.bin")
        s = list(t.shape[1:])
        while len(s) < 3:
            s.append(1)
        print("SHAPE", tag, i, *s[:3], LAYOUT)

save("cls", cls_l); save("box", box_l); save("ctr", ctr_l)
for _tag, _lst in extra.items():
    save(_tag, _lst)
np.ascontiguousarray(x[0].transpose(1, 2, 0)).tofile("in.bin")
print("LEVELS", len(cls_l), "CTR", len(ctr_l))
'''


# 단계별 소요 시간. 어디가 느린지 **재고 나서** 고치려고 둔다.
# 실측(3계열 96초): g2c 46% · export 21% · ref 21% · g++ 9% · 실행 3%.
PHASE = {}
_PHASE_LOCK = __import__("threading").Lock()
# ⚠️ head.o 빌드용 락은 **따로** 둔다. 계측 락을 재사용하면 `run()` 안에서 같은 락을
#    다시 잡아 자기 자신을 기다린다(비재진입 Lock → 교착). 실제로 한 번 걸렸다.
_HEAD_LOCK = __import__("threading").Lock()

# 계열끼리 공유하는 상태가 없어 병렬로 돌릴 수 있다(전부 독립 프로세스).
# 실측 계열당 최대 RSS 1.16GB → 4개면 ~4.6GB. 6코어 중 4개만 쓴다(2개는 호스트 몫).
WORKERS = CFG.workers
# ⚠️ **메모리 가드.** 위키 `wsl-계속-터짐` — 병렬 torch 스윕이 WSL 을 통째로 죽인 적이 있다.
#    가용 메모리가 이 밑으로 내려가면 새 계열을 안 띄우고 기다린다. 느려질지언정 안 죽는다.
MIN_FREE_MB = CFG.min_free_mb


def _avail_mb():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except Exception:
        pass
    return 1 << 30      # 못 읽으면 가드를 끈다(측정 실패로 막지 않는다)


def _wait_for_memory(fam):
    import time
    waited = 0
    while _avail_mb() < MIN_FREE_MB:
        if waited == 0:
            print(f"  … 메모리 대기 ({fam}): 가용 {_avail_mb()}MB < {MIN_FREE_MB}MB", flush=True)
        time.sleep(5); waited += 5
        if waited > 600:      # 10분을 기다려도 안 풀리면 그냥 간다(교착 방지)
            break


def run(cmd, cwd, env_extra=None, timeout=2400, phase=None):
    import time
    env = dict(os.environ, OMP_NUM_THREADS="1")
    env.update(env_extra or {})
    t0 = time.time()
    r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
    if phase:
        with _PHASE_LOCK:
            PHASE[phase] = PHASE.get(phase, 0.0) + (time.time() - t0)
    return r



REF_FRCNN = r'''
import _stub, sys, numpy as np, torch
sys.path.insert(0, "%(FE)s")
import mmdet_compat   # noqa
import frcnn_wrap     # noqa  (피클 클래스 복원)
suba = torch.load("frcnn/FRCNN_SubA.pt", weights_only=False); suba.eval()
# 캐스케이드는 단계별 파일이다(`FRCNN_SubB0/1/2`). 여기서 재는 것은 **1단계** —
# 단계 사이 박스 정제는 호스트 코드라 별도 축이고, 그걸 섞으면 무엇이 틀렸는지 못 가른다.
import glob as _g, os as _o
_p = "frcnn/FRCNN_SubB.pt"
if not _o.path.exists(_p):
    _p = sorted(_g.glob("frcnn/FRCNN_SubB[0-9].pt"))[0]
subb = torch.load(_p, weights_only=False); subb.eval()
np.random.seed(0)
x = np.random.randn(1, 3, %(SZ)d, %(SZ)d).astype("float32")
np.ascontiguousarray(x[0].transpose(1, 2, 0)).tofile("in.bin")   # 러너 입력(cwhn)
with torch.no_grad():
    outs = suba(torch.from_numpy(x))
L = (len(outs) - 4) // 2
for i in range(L):
    np.ascontiguousarray(outs[4 + i][0].numpy()).tofile(f"ref.rpncls.{i}.bin")
    np.ascontiguousarray(outs[4 + i + L][0].numpy()).tofile(f"ref.rpnbox.{i}.bin")
    print("SHAPE rpncls", i, *outs[4 + i].shape[1:], "chw")
    print("SHAPE rpnbox", i, *outs[4 + i + L].shape[1:], "chw")
print("LEVELS", L)
'''

REF_FRCNN_B = r'''
import _stub, sys, numpy as np, torch
sys.path.insert(0, "%(FE)s")
import mmdet_compat   # noqa
import frcnn_wrap     # noqa
# 캐스케이드는 단계별 파일이다(`FRCNN_SubB0/1/2`). 여기서 재는 것은 **1단계** —
# 단계 사이 박스 정제는 호스트 코드라 별도 축이고, 그걸 섞으면 무엇이 틀렸는지 못 가른다.
import glob as _g, os as _o
_p = "frcnn/FRCNN_SubB.pt"
if not _o.path.exists(_p):
    _p = sorted(_g.glob("frcnn/FRCNN_SubB[0-9].pt"))[0]
subb = torch.load(_p, weights_only=False); subb.eval()
# ⚠️ **러너가 고른 proposal 로 만든 RoI feature 를 그대로 넣는다.** 여기서 재는 것은
#    RoI head 이고, RPN/RoIAlign 은 위 단계에서 따로 재기 때문이다. 다시 뽑으면
#    NMS 순서 차이가 섞여 무엇이 틀렸는지 못 가른다.
roi = np.fromfile("cpp.roi.bin", dtype="float32")
m = roi.size // (%(RC)d * %(O)d * %(O)d)
with torch.no_grad():
    _o = subb(torch.from_numpy(roi).view(m, %(RC)d, %(O)d, %(O)d))
# ⚠️ **(cls, box) 한 쌍이라고 언패킹하지 마라.** CrowdDet `MultiInstanceBBoxHead` 는
#    proposal 하나가 사람 둘을 낸다고 보고 쌍을 2벌 낸다(총 4텐서). 언패킹하면
#    `ValueError: too many values to unpack` 로 죽고, 앞 둘만 쓰면 절반이 검증 안 된다.
_o = list(_o) if isinstance(_o, (list, tuple)) else [_o]
assert len(_o) >= 2 and len(_o) %% 2 == 0, "SubB 출력이 (cls, box) 쌍이 아니다: %%d" %% len(_o)
for _k in range(len(_o) // 2):
    _c, _b = _o[2 * _k], _o[2 * _k + 1]
    np.ascontiguousarray(_c.numpy()).tofile("ref.cls.%%d.bin" %% _k)
    np.ascontiguousarray(_b.numpy()).tofile("ref.box.%%d.bin" %% _k)
    print("SHAPE cls", _k, *_c.shape, 1, "flat")
    print("SHAPE box", _k, *_b.shape, 1, "flat")
'''


def _two_stage(fam, cfg_path, cw, d):
    """two-stage(Faster/Mask R-CNN 계열)를 **2패스**로 검증한다.

    dense head 는 한 그래프로 끝나지만 two-stage 는 안 된다 — RPN 후보에서 NMS 로 proposal 을
    고르고(개수·좌표가 실행 중에 정해진다) 그 좌표에서 feature 를 잘라(RoIAlign) 두 번째 head 에
    넣는다. 그래서 러너가 `run_frcnn.cpp` 로 패스를 나눠 돌린다.

    호스트 부품(`rpn_proposals`·`roi_align`·`detect_roi`)은 `postproc.h` 에 이미 있다.
    """
    import json
    fr = os.path.join(d, "frcnn")
    r = run([PY, FE + "/frcnn_to_pt.py", "--config", cfg_path, "--checkpoint", cw,
             "--out", fr, "--size", str(SZ)], os.path.dirname(MM.rstrip("/")),
            {"PYTHONPATH": f"{d}:{FE}"}, phase="1_export2")
    if not os.path.exists(os.path.join(fr, "frcnn.json")):
        # ⚠️ **왜 실패했는지 말한다.** 전부 "조립기 없음" 으로 뭉뚱그리면 진짜 미지원(SOLO 등)과
        #    고칠 수 있는 실패(export 버그)가 구분되지 않는다 — 5계열을 그렇게 놓칠 뻔했다.
        err = _last_error(r.stderr)
        if "roi_head" in err or "has no attribute" in err:
            return fam, "HEAD_NONE", f"two-stage 아님: {err[:50]}"
        return fam, "EXPORT2_FAIL", err[:70]
    J = json.load(open(os.path.join(fr, "frcnn.json")))
    O, MX = int(J["roi_out"]), int(J["rpn_max"])
    RC = int(J.get("roi_channels", 256))
    # Double-Head 는 (cls용, reg용) 두 벌을 배치로 이어 넣는다 → 배치가 2배다.
    MX = MX * 2 if float(J.get("reg_roi_scale_factor", 0) or 0) > 0 else MX

    # SubA / SubB 를 각각 컴파일한다. ⚠️ SubB 는 **proposal 상한 배치**로 컴파일해야 한다 —
    #    batch=1 로 굽고 1000개를 넣으면 reshape 이 안 맞아 죽는다.
    NS = int(J.get("num_bbox_stages", 1))
    # 캐스케이드는 단계마다 SubB 가 따로 있다(`FRCNN_SubB0/1/2`). **구조가 같고 가중치만
    # 다르므로** 그래프는 0번 것만 컴파일해 재사용하고, gguf 는 단계 수만큼 굽는다.
    subs = ["FRCNN_SubB"] if NS == 1 else ["FRCNN_SubB%d" % i for i in range(NS)]

    # (모델파일, 컴파일이름, 출력폴더, 입력shape)
    jobs = [("FRCNN_SubA", "FRCNN_SubA", "out_FRCNN_SubA", "1,3,%d,%d" % (SZ, SZ))]
    for s in subs:
        # ⚠️ **이름은 subs[0] 로 통일한다.** 생성 코드가 `general.architecture` 를 검사하므로
        #    단계마다 다른 이름으로 구우면 gguf 를 갈아 끼울 수 없다(그래프 재사용이 목적이다).
        jobs.append((s, subs[0], "out_" + s, "%d,%d,%d,%d" % (MX, RC, O, O)))
    for src, name, outdir, shape in jobs:
        r = run([PY, "-c", '''
import _stub, sys
sys.argv = ["g2c","--model","%s.pt","--name","%s","--output","%s","--input-shape","%s"]
from shared.compile.pipeline import main; main()
''' % (src, name, outdir, shape)], fr,
                {"PYTHONPATH": f"{d}:{fr}:{P}:{FE}:{GGUF_PY}"}, phase="2_g2c2")
        if not os.path.exists(os.path.join(fr, outdir, f"{name}.gguf")):
            return fam, "COMPILE_FAIL", _last_error(r.stderr)[:70]

    import shutil
    for name, inc in (("FRCNN_SubA", "incA"), (subs[0], "incB")):
        os.makedirs(os.path.join(fr, inc, "visp", "arch"), exist_ok=True)
        shutil.copy(os.path.join(fr, "out_" + name, name + ".h"),
                    os.path.join(fr, inc, "visp", "arch"))
    b = run(["g++", "-std=c++20", OPT, "-DARCH_A=FRCNN_SubA",
             "-DARCH_B=" + subs[0],
             '-DVISP_ARCH_HEADER_A="visp/arch/FRCNN_SubA.h"',
             '-DVISP_ARCH_HEADER_B="visp/arch/' + subs[0] + '.h"',
             "-IincA", "-IincB",
             "-I" + V + "/include", "-I" + V + "/src",
             "-I" + V + "/depend/llama/ggml/include", "-I" + V + "/depend/llama/vendor",
             V + "/tools/verify/backbone/run_frcnn.cpp",
             "out_FRCNN_SubA/FRCNN_SubA.cpp", "out_" + subs[0] + "/" + subs[0] + ".cpp",
             "-L" + V + "/build/lib", "-lvisioncpp", "-lggml", "-lggml-base", "-lggml-cpu",
             "-Wl,-rpath," + V + "/build/lib", "-o", "run_frcnn"], fr, phase="3_build2")
    if not os.path.exists(os.path.join(fr, "run_frcnn")):
        return fam, "BUILD_FAIL", _last_error(b.stderr)[:70]

    # ① torch 기준값(RPN) + 러너 입력 생성
    open(os.path.join(d, "ref_a.py"), "w").write(REF_FRCNN % {"FE": FE, "SZ": SZ})
    ra = run([PY, "ref_a.py"], d, {"PYTHONPATH": f"{d}:{FE}"}, phase="4_ref2")
    if "LEVELS" not in ra.stdout:
        return fam, "REF_FAIL", _last_error(ra.stderr)[:70]

    # ② 2패스 실행
    rr = run([os.path.join(fr, "run_frcnn"),
              "out_FRCNN_SubA/FRCNN_SubA.gguf",
              ",".join("out_" + s + "/" + subs[0] + ".gguf" for s in subs),
              "frcnn.json", os.path.join(d, "in.bin"), os.path.join(d, "cpp"), str(SZ)],
             fr, {"VISP_BACKEND": "cpu"}, phase="5_run2")
    if not os.path.exists(os.path.join(d, "cpp.roi.bin")):
        return fam, "RUN_FAIL", _last_error(rr.stderr)[:70]

    # ③ RoI head 기준값(러너가 만든 RoI feature 로)
    open(os.path.join(d, "ref_b.py"), "w").write(REF_FRCNN_B % {"FE": FE, "O": O, "RC": RC})
    rb = run([PY, "ref_b.py"], d, {"PYTHONPATH": f"{d}:{FE}"}, phase="4_ref2")
    if "SHAPE" not in rb.stdout:
        return fam, "REF_FAIL", _last_error(rb.stderr)[:70]

    shapes = {}
    for line in (ra.stdout + rb.stdout).splitlines():
        if line.startswith("SHAPE"):
            parts = line.split()
            shapes[(parts[1], int(parts[2]))] = (int(parts[3]), int(parts[4]), int(parts[5]),
                                                 parts[6] if len(parts) > 6 else "chw")
    import numpy as np
    w1 = w2 = 0.0
    for (tag, i), (c, h, w, lay) in sorted(shapes.items()):
        pr, pc = os.path.join(d, f"ref.{tag}.{i}.bin"), os.path.join(d, f"cpp.{tag}.{i}.bin")
        if not os.path.exists(pc):
            return fam, "RUN_FAIL", f"덤프 없음 {tag}.{i}"
        a, bnp = np.fromfile(pr, dtype="float32"), np.fromfile(pc, dtype="float32")
        if a.size != bnp.size:
            return fam, "SHAPE_MISMATCH", f"{tag}{i} {a.size} vs {bnp.size}"
        if lay == "chw":       # 그래프 출력은 cwhn — 되돌려야 한다(안 하면 L1 1.5 로 보인다)
            bnp = bnp.reshape(h, w, c).transpose(2, 0, 1).reshape(-1)
        w1 = max(w1, float(np.abs(a - bnp).sum() / (np.abs(a).sum() + 1e-12)))
        w2 = max(w2, float(np.linalg.norm(a - bnp) / (np.linalg.norm(a) + 1e-12)))
    st = "PASS" if (w1 <= L1_TOL and w2 <= L2_TOL) else "FAIL"
    return fam, st, f"L1 {w1:.2e} · L2 {w2:.2e} · kind two_stage · 텐서 {len(shapes)}"


def one(fam, rel, ckpt):
    d = os.path.join(ROOT, fam)
    os.makedirs(d, exist_ok=True)   # ROOT 도 같이 생긴다(head.o 자리)
    open(os.path.join(d, "_stub.py"), "w").write(STUB)
    cfg_path = os.path.join(MM, rel)
    if not os.path.exists(cfg_path):
        return fam, "CONFIG_NONE", "-"

    # 1) backbone .pt + <name>.postproc.h  (프론트엔드 CLI 그대로)
    if not ckpt:
        # **랜덤 가중치로는 검증이 안 된다** — γ=1·β=0 같은 항등 초기값이 누락 연산을 덮는다.
        # 그래서 체크포인트가 없으면 "안 됨" 이 아니라 "안 해봄" 으로 남긴다.
        return fam, "CKPT_NONE", "체크포인트 미다운로드"
    cw = os.path.join(CKPT, ckpt)
    if not os.path.exists(cw):
        return fam, "CKPT_NONE", ckpt
    # ⚠️ **mmdetection 루트에서 돌린다.** 증류 계열의 config 는 교사 모델을
    #    `teacher_config: 'configs/gfl/...'` 처럼 **CWD 기준 상대경로**로 적는다
    #    (`_base_` 와 달리 config 파일 위치 기준이 아니다). 다른 데서 돌리면 FileNotFound.
    #    입출력 경로는 전부 절대라 cwd 를 옮겨도 안전하다.
    mm_root = os.path.dirname(MM.rstrip("/"))
    # ⚠️ **트래커·반지도 래퍼는 config 를 한 겹 벗겨야 한다.** ByteTrack 등은 검출기를
    #    `model.detector` 안에 넣고 자기는 껍데기만 갖는다 →
    #    `'ConfigDict' object has no attribute 'backbone'` 으로 죽는다.
    #    저장소에 전처리기가 이미 있다(`mmdet_unwrap_config.py`). 풀 필요 없는 config 는
    #    원본을 그대로 돌려주므로 분기 없이 전부 통과시킨다.
    #    ⚠️ **config 만 벗기면 안 된다 — 체크포인트도 같이 벗겨야 한다.** 여기가 오래
    #    반쪽이었다: 벗긴 config 는 `backbone.…` 을 기대하는데 래퍼 체크포인트는
    #    `detector.backbone.…`(트래커) 또는 `teacher.…`(반지도)로 저장돼 있다.
    #    이름이 안 맞으면 **한 텐서도 안 실리는데** `load_checkpoint` 는 조용히 넘어가고,
    #    이 하네스의 기준값은 그 `bb.pt` 자신이라 **랜덤 가중치끼리 일치해 PASS 가 뜬다.**
    #    실제로 soft_teacher 가 L1 1.04e-03 로 통과했는데 체크포인트 일치는 0/348 이었다.
    #    ⚠️ **체크포인트를 먼저, config 를 나중에.** 접두사는 원본 config 의
    #    `semi_test_cfg.predict_on` 이 정하는데 벗긴 config 에는 그 키가 없다.
    #    하나라도 실패하면 **둘 다 원본으로 되돌린다**(짝이 어긋나는 게 더 나쁘다).
    if os.path.exists(UNWRAP):
        base = os.path.dirname(os.path.dirname(os.path.dirname(UNWRAP)))
        cfg0, cw0 = cfg_path, cw

        def _unwrap(out_name, *extra):
            ru = run([PY, UNWRAP, cfg0, "-o", os.path.join(d, out_name), *extra],
                     base, phase="0_unwrap")
            if ru is None or ru.returncode != 0:
                raise RuntimeError((ru.stderr or "").strip().splitlines()[-1:] or "unwrap 실패")
            lines = (ru.stdout or "").strip().splitlines()
            last = lines[-1].strip() if lines else ""
            if not (last and os.path.exists(last)):
                raise RuntimeError(f"경로를 못 받았다: {last!r}")
            return last

        try:
            cw = _unwrap("ckpt.pth", "--checkpoint", cw0)
            cfg_path = _unwrap("cfg.py")
        except Exception as e:
            cfg_path, cw = cfg0, cw0
            print(f"  [unwrap] 실패 — 원본으로 되돌린다: {type(e).__name__}: {e}",
                  file=sys.stderr)
    # ⚠️ **먼저 지운다.** export 가 실패해도 지난 실행의 bb.pt/헤더가 남아 있으면 아래 존재
    #    검사를 통과해 **낡은 산출물로 계속 간다** — 그러면 고친 것이 반영 안 된 채 통과/실패가
    #    나온다(dab_detr 에서 실제로 겪었다: 헤더에 새 플래그가 없는데 조용히 진행됐다).
    for stale in ("bb.pt", "bb.postproc.h"):
        try:
            os.remove(os.path.join(d, stale))
        except FileNotFoundError:
            pass
    r = run([PY, FE + "/mmdet_to_pt.py", "--config", cfg_path, "--checkpoint", cw,
             "--out", os.path.join(d, "bb.pt"), "--size", str(SZ)], mm_root,
            {"PYTHONPATH": f"{d}:{FE}"}, phase="1_export")
    if not os.path.exists(os.path.join(d, "bb.pt")):
        return fam, "EXPORT_FAIL", _last_error(r.stderr)[:70]
    ph = os.path.join(d, "bb.postproc.h")
    if not os.path.exists(ph):
        return fam, "PARAMS_NONE", "postproc.h 미생성 (head_type=raw?)"
    # ⚠️ **어떤 짝으로 구웠는지 남긴다.** 계열마다 변종이 여럿이라(retinanet r18 vs r50-caffe,
    #    tood anchor-free vs anchor-based) 나중에 `verify_postproc.py` 를 손으로 부를 때
    #    다른 짝을 주기 쉽다. 그러면 **남의 가중치를 진 그래프**를 대조하게 되고, 결과가
    #    그럴듯하게 틀려서(20px·13px) 결함으로 오해한다 — 오늘만 두 번 겪었다.
    #    `verify_postproc.py` 가 이 파일을 읽어 어긋나면 알린다.
    try:
        import json as _json
        _json.dump({"family": fam, "config": cfg_path, "checkpoint": cw},
                   open(os.path.join(d, "used.json"), "w"), ensure_ascii=False, indent=1)
    except OSError:
        pass
    kind = next((l.split(":")[-1].strip() for l in open(ph) if l.startswith("// head_type")), "?")
    if kind == "raw":
        # 프론트엔드가 이 head 를 인식하지 못했다 = 조립기가 없다. 여기서 끝낸다 —
        # 계속 가면 러너에서 크래시로 나타나 "버그" 처럼 보인다.
        # dense head 가 아니면 two-stage 경로를 태워 본다 — 거기서도 아니면 HEAD_NONE.
        return _two_stage(fam, cfg_path, cw, d)

    # 2) g2c 컴파일 (backbone+neck 만). **g2c 는 main 원본 그대로 쓴다.**
    r = run([PY, "-c", '''
import _stub, sys
sys.argv = ["g2c","--model","bb.pt","--name","Fam","--output","out","--input-shape","1,3,%d,%d"]
from shared.compile.pipeline import main; main()
''' % (SZ, SZ)], d, {"PYTHONPATH": f"{d}:{P}:{FE}:{GGUF_PY}"}, phase="2_g2c")
    if not os.path.exists(os.path.join(d, "out", "Fam.gguf")):
        return fam, "COMPILE_FAIL", _last_error(r.stderr)[:70]

    # 2b) 그래프 밖 가중치(head · DETR transformer)를 **프론트엔드가** 덧붙인다.
    #     trace 에 안 잡히는 건 g2c 잘못이 아니다 — head 를 C++ 로 뺀 이 경로의 사정이다.
    r = run([PY, FE + "/append_head_weights.py", os.path.join(d, "bb.pt"),
             os.path.join(d, "out", "Fam.gguf")], d,
            {"PYTHONPATH": f"{d}:{FE}"}, phase="2b_weights")
    # ⚠️ **성공 판정을 출력 문구로 하지 마라.** 여기서 `"추가"` 를 찾고 있었는데 스크립트는
    #    영어로 `"appended N weights…"` 를 찍는다. 문구가 바뀐 걸 아무도 못 봐서 **one-stage
    #    계열 전부가 WEIGHTS_FAIL 로 떨어지고 있었다** — 계열 탓처럼 보이지만 하네스 탓이다.
    #    종료코드로 판정하고, 문구는 참고로만 쓴다.
    if r.returncode != 0:
        return fam, "WEIGHTS_FAIL", _last_error(r.stderr)[:70]

    # 3) run_mmdet 빌드 (백본 .cpp + head.cpp 를 함께 컴파일)
    gen = os.path.join(d, "out")
    inc = os.path.join(gen, "inc", "visp", "arch")
    os.makedirs(inc, exist_ok=True)
    import shutil
    shutil.copy(os.path.join(gen, "Fam.h"), inc)
    # head.cpp 는 `ARCH`·파라미터 헤더를 안 쓴다 → **계열마다 다시 컴파일할 이유가 없다.**
    # 한 번 .o 로 만들어 두고 링크만 한다(38계열이면 37번을 아낀다).
    # 여러 계열이 동시에 들어와도 한 번만 만든다.
    with _HEAD_LOCK:
        # ⚠️ **head.cpp 가 바뀌면 캐시를 버린다.** mtime 비교가 없으면 조립기를 고쳐도
        #    옛 오브젝트로 링크돼 **값이 그대로**다 — 고친 줄 알고 한참 헤맨다(실제로 겪었다).
        _src = V + "/tools/detect/head.cpp"
        if (os.path.exists(HEAD_OBJ)
                and os.path.getmtime(HEAD_OBJ) < max(os.path.getmtime(_src),
                                                     os.path.getmtime(V + "/tools/detect/head.h"))):
            os.remove(HEAD_OBJ)
        b = run(["g++", "-std=c++20", OPT, "-c", V + "/tools/detect/head.cpp",
                 "-I" + V + "/include", "-I" + V + "/src", "-I" + V + "/tools/detect",
                 "-I" + V + "/depend/llama/ggml/include", "-I" + V + "/depend/llama/vendor",
                 "-o", HEAD_OBJ], d, phase="3a_head") if not os.path.exists(HEAD_OBJ) else None
    if not os.path.exists(HEAD_OBJ):
        return fam, "BUILD_FAIL", "head.o: " + (_last_error(b.stderr)[:70])

    # 최적화 수준은 **수치와 무관**하다 — 실제 계산은 libggml(사전 빌드)이 한다.
    # 이 코드는 그래프를 짜기만 하므로 -O1 이면 충분하고, 컴파일이 훨씬 빠르다.
    # ⚠️ **먼저 지운다.** 존재 검사만 하면 빌드가 실패해도 **지난 실행의 바이너리**로
    #    계속 가고, 낡은 코드가 낸 숫자가 PASS 로 보고된다(실제로 겪었다: 코너 디코드를
    #    새로 붙였는데 goto 컴파일 에러였고, 하네스는 PASS 를 냈다).
    try:
        os.remove(os.path.join(gen, "run_mmdet"))
    except FileNotFoundError:
        pass
    b = run(["g++", "-std=c++20", OPT, "-DARCH=Fam",
             '-DVISP_ARCH_HEADER="visp/arch/Fam.h"',
             f'-DMMDET_PARAMS_HEADER="{ph}"',
             "-I" + gen + "/inc", "-I" + V + "/include", "-I" + V + "/src",
             "-I" + V + "/tools/detect",
             "-I" + V + "/depend/llama/ggml/include", "-I" + V + "/depend/llama/vendor",
             V + "/tools/verify/backbone/run_mmdet.cpp", HEAD_OBJ,
             gen + "/Fam.cpp",
             "-L" + V + "/build/lib", "-lvisioncpp", "-lggml", "-lggml-base", "-lggml-cpu",
             "-Wl,-rpath," + V + "/build/lib", "-o", gen + "/run_mmdet"], d, phase="3b_build")
    if not os.path.exists(os.path.join(gen, "run_mmdet")):
        return fam, "BUILD_FAIL", _last_error(b.stderr)[:80]

    # 4) torch 기준값
    open(os.path.join(d, "ref.py"), "w").write(REF % {"FE": FE, "SZ": SZ})
    r = run([PY, "ref.py", fam, cfg_path, gen], d, {"PYTHONPATH": f"{d}:{FE}"}, phase="4_ref")
    if "LEVELS" not in r.stdout:
        return fam, "REF_FAIL", _last_error(r.stderr)[:70]
    shapes = {}
    for line in r.stdout.splitlines():
        if line.startswith("SHAPE"):
            parts = line.split()
            _, tag, i, c, h, w = parts[:6]
            lay = parts[6] if len(parts) > 6 else "chw"
            shapes[(tag, int(i))] = (int(c), int(h), int(w), lay)

    # 5) C++ 실행 + 대조
    r = run([gen + "/run_mmdet", gen + "/Fam.gguf", "in.bin", "o.bin", str(SZ)], d,
            {"MMDET_DUMP_HEAD": "cpp", "VISP_BACKEND": "cpu"}, phase="5_run")
    import numpy as np
    worst_l1, worst_l2, nmiss = 0.0, 0.0, 0
    for (tag, i), (c, h, w, lay) in sorted(shapes.items()):
        pc = os.path.join(d, f"cpp.{tag}.{i}.bin")
        pr = os.path.join(d, f"ref.{tag}.{i}.bin")
        if not os.path.exists(pc):
            nmiss += 1
            continue
        a = np.fromfile(pr, dtype="float32")
        bnp = np.fromfile(pc, dtype="float32")
        if a.size != bnp.size:
            return fam, "SHAPE_MISMATCH", f"{tag}{i} ref {a.size} vs cpp {bnp.size}"
        # cwhn → chw. DETR 계열(query×ch)은 공간 격자가 아니라 축을 안 바꾼다 —
        # ref (Q, C) 의 평탄화 순서와 ggml ne={C, Q} 의 평탄화 순서가 이미 같다.
        if lay == "set":
            # 순서 없는 출력(two-stage 의 query 집합)에는 **집합 거리**를 쓴다.
            # 정렬 정렬은 안 된다 — 집합이 한 행만 어긋나도 그 뒤가 통째로 밀린다.
            # 각 ref 행에 가장 가까운 cpp 행을 짝지어 그 거리를 잰다(순서와 무관).
            # SHAPE 는 (c,h,w) = torch shape[1:] → DETR 은 c=query 수, h=채널 수.
            ra, rb = a.reshape(c, -1), bnp.reshape(c, -1)
            d2 = ((ra * ra).sum(1)[:, None] + (rb * rb).sum(1)[None, :]
                  - 2.0 * (ra @ rb.T))
            a, bnp = ra.reshape(-1), rb[d2.argmin(1)].reshape(-1)
        elif lay != "flat":
            bnp = bnp.reshape(h, w, c).transpose(2, 0, 1).reshape(-1)
        d_ = a - bnp
        # **상대** 거리로 잰다. 절대 L1/L2 는 텐서마다 스케일이 달라 비교가 안 된다 —
        # 같은 계열 안에서도 cls 는 |x|~8, box 는 |x|~375 다(vfnet 실측).
        worst_l1 = max(worst_l1, float(np.abs(d_).sum() / (np.abs(a).sum() + 1e-12)))
        worst_l2 = max(worst_l2, float(np.linalg.norm(d_) / (np.linalg.norm(a) + 1e-12)))
    if nmiss:
        err = (r.stderr or r.stdout).strip().splitlines()
        return fam, "RUN_FAIL", (err[0][:80] if err else f"덤프 {nmiss}개 없음")
    # cosine 은 **스케일 불변**이라 크기가 통째로 틀려도 1.0 이 나온다(vfnet scale 하드코딩이
    # 그랬다: 박스가 2.4배 작은데 cos 는 높았다). L1/L2 는 그걸 그대로 드러낸다.
    st = "PASS" if (worst_l1 <= L1_TOL and worst_l2 <= L2_TOL) else "FAIL"
    return fam, st, f"L1 {worst_l1:.2e} · L2 {worst_l2:.2e} · kind {kind} · 텐서 {len(shapes)}"


# 기본은 **전 계열**이다. 인자를 주면 그것만 — 디버깅용.
FAMILIES = _all_families()
print(CFG.banner(), flush=True)      # 무엇으로 쟀는지 로그에 남긴다
if ARGS:
    FAMILIES = [x for x in FAMILIES if x[0] in ARGS]

def _last_error(text):
    """stderr 에서 **진짜 원인 줄**을 고른다.

    그냥 마지막 줄을 쓰면 `ResourceWarning: Implicitly cleaning up …` 같은 경고가
    원인을 가린다 — 세 번이나 헛짚었다. 예외처럼 보이는 줄을 뒤에서부터 찾고,
    없으면 경고가 아닌 마지막 줄을 쓴다.
    """
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    if not lines:
        return "-"
    for l in reversed(lines):
        if re.match(r"^[A-Za-z_.]*(Error|Exception|Warning)?\b", l) and (
                "Error" in l or "Exception" in l) and "Warning" not in l:
            return l
    for l in reversed(lines):
        if "Warning" not in l and not l.startswith(("File ", "  ", "warnings.warn")):
            return l
    return lines[-1]


def _one_guarded(args):
    fam, rel, ckpt = args
    _wait_for_memory(fam)          # 가용 메모리가 회복될 때까지 시작을 미룬다
    try:
        return one(fam, rel, ckpt)
    except subprocess.TimeoutExpired:
        return fam, "TIMEOUT", "-"
    except Exception as e:
        return fam, "ERROR", f"{type(e).__name__}: {e}"[:70]


print(f"{'':>7} {'계열':<18} {'판정':<14} 비고", flush=True)
print("-" * 78, flush=True)

# 각 단계가 별도 프로세스라 GIL 을 잡지 않는다 → 스레드 풀로 충분하다.
# 완료 순서가 아니라 **등록 순서**로 출력한다(실행마다 표가 달라지면 비교를 못 한다).
def _emit(row, i, n):
    f, st, note = row
    mark = "O" if st == "PASS" else ("X" if st == "FAIL" else "-")
    print(f"[{i:3d}/{n}] {f:<18} {mark} {st:<12} {note}", flush=True)
    return row

# ⚠️ `list(ex.map(...))` 로 다 모은 뒤 찍으면 **30분간 화면이 빈다.** map 은 게으른
#    제너레이터이므로 그대로 순회하면 등록 순서를 지키면서 끝나는 대로 나온다.
n = len(FAMILIES)
if WORKERS > 1 and n > 1:
    os.makedirs(ROOT, exist_ok=True)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        results = [_emit(r, i, n) for i, r in enumerate(ex.map(_one_guarded, FAMILIES), 1)]
else:
    results = [_emit(_one_guarded(x), i, n) for i, x in enumerate(FAMILIES, 1)]

if PHASE:
    tot = sum(PHASE.values())
    print("\n단계별 소요 (합계 %.0f초)" % tot)
    for k in sorted(PHASE):
        print(f"  {k:<10} {PHASE[k]:7.1f}초  {PHASE[k]/tot*100:4.1f}%")
