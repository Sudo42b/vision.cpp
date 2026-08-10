#!/usr/bin/env python3
"""tools/detect/head.cpp 가 mmdet 의 bbox_head 를 재현하는지 계열별로 잰다.

백본만 컴파일하고 head 는 C++ 부품이 조립한다. 계열마다:
mmdet → backbone .pt + 파라미터 헤더 → 컴파일 → run_mmdet 빌드 →
MMDET_DUMP_HEAD 로 **디코드 전** 원시 텐서 덤프 → torch bbox_head 출력과 대조.

디코드/NMS 앞에서 끊는 이유: NMS 를 거치면 어느 텐서가 틀렸는지 못 짚는다.

    python verify_heads.py              # 8계열 전부
    python verify_heads.py vfnet tood   # 골라서

체크포인트가 `~/mmbuild/mmdetection/checkpoints/` 에 있어야 한다.
**랜덤 초기화로 재지 마라** — 항등 초기값(γ=1·β=0, scale=1)이 빠진 연산을 덮어
검증을 통과시킨다. 실제로 VFNet 의 scale 하드코딩이 그렇게 숨어 있었다.
"""
import os
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
CKPT = os.environ.get("MMDET_CHECKPOINTS",
                      os.path.expanduser("~/mmbuild/mmdetection/checkpoints"))

MM = os.environ.get("MMDET_CONFIGS", os.path.expanduser("~/mmbuild/mmdetection/configs"))
ROOT = os.environ.get("VERIFY_WORKDIR", "/tmp/visp-verify-heads")
# g2c(컴파일러)와 vision.cpp 경로. 이 파일은 vision.cpp/tools/verify/dense_head/ 에 있다.
V = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
P = os.environ.get("G2C_ROOT", os.path.dirname(V))
PY = os.environ.get("VERIFY_PYTHON", sys.executable)
FE = V + "/tools/frontend/mmdet"
GGUF_PY = V + "/depend/llama/gguf-py"
SZ = 512

STUB = '''import sys, types
for n in ("mmpretrain.models.multimodal.blip", "mmpretrain.models.multimodal.blip.language_model"):
    m = types.ModuleType(n); m.__path__ = []; sys.modules[n] = m
'''

# torch 기준값. **C++ 이 내는 것과 같은 지점**까지만 계산한다 — 디코드는 양쪽 다 안 한다.
REF = r'''
import _stub, sys, torch, numpy as np
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
with torch.no_grad():
    f = mod.backbone(torch.from_numpy(x))
    if mod.neck is not None:
        f = mod.neck(f)
    outs = bh(tuple(f))

# 계열별로 head 출력의 의미가 다르다. C++ 과 같은 형태로 맞춘다.
cls_l, box_l, ctr_l = [], [], []
if isinstance(outs, tuple) and len(outs) == 3:
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
        print("SHAPE", tag, i, *t.shape[1:])

save("cls", cls_l); save("box", box_l); save("ctr", ctr_l)
np.ascontiguousarray(x[0].transpose(1, 2, 0)).tofile("in.bin")
print("LEVELS", len(cls_l), "CTR", len(ctr_l))
'''


def run(cmd, cwd, env_extra=None, timeout=2400):
    env = dict(os.environ, OMP_NUM_THREADS="1")
    env.update(env_extra or {})
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)


def one(fam, rel, ckpt):
    d = os.path.join(ROOT, fam)
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "_stub.py"), "w").write(STUB)
    cfg_path = os.path.join(MM, rel)
    if not os.path.exists(cfg_path):
        return fam, "CONFIG_NONE", "-"

    # 1) backbone .pt + <name>.postproc.h  (프론트엔드 CLI 그대로)
    cw = os.path.join(CKPT, ckpt)
    if not os.path.exists(cw):
        return fam, "CKPT_NONE", ckpt
    # ⚠️ **mmdetection 루트에서 돌린다.** 증류 계열의 config 는 교사 모델을
    #    `teacher_config: 'configs/gfl/...'` 처럼 **CWD 기준 상대경로**로 적는다
    #    (`_base_` 와 달리 config 파일 위치 기준이 아니다). 다른 데서 돌리면 FileNotFound.
    #    입출력 경로는 전부 절대라 cwd 를 옮겨도 안전하다.
    mm_root = os.path.dirname(MM.rstrip("/"))
    r = run([PY, FE + "/mmdet_to_pt.py", "--config", cfg_path, "--checkpoint", cw,
             "--out", os.path.join(d, "bb.pt"), "--size", str(SZ)], mm_root,
            {"PYTHONPATH": f"{d}:{FE}"})
    if not os.path.exists(os.path.join(d, "bb.pt")):
        return fam, "EXPORT_FAIL", (r.stderr.strip().splitlines() or ["-"])[-1][:70]
    ph = os.path.join(d, "bb.postproc.h")
    if not os.path.exists(ph):
        return fam, "PARAMS_NONE", "postproc.h 미생성 (head_type=raw?)"
    kind = next((l.split(":")[-1].strip() for l in open(ph) if l.startswith("// head_type")), "?")

    # 2) g2c 컴파일 (backbone+neck 만; head 가중치는 선언으로 실린다)
    r = run([PY, "-c", '''
import _stub, sys
sys.argv = ["g2c","--model","bb.pt","--name","Fam","--output","out","--input-shape","1,3,%d,%d"]
from shared.compile.pipeline import main; main()
''' % (SZ, SZ)], d, {"PYTHONPATH": f"{d}:{P}:{FE}:{GGUF_PY}"})
    if not os.path.exists(os.path.join(d, "out", "Fam.gguf")):
        return fam, "COMPILE_FAIL", (r.stderr.strip().splitlines() or ["-"])[-1][:70]

    # 3) run_mmdet 빌드 (백본 .cpp + head.cpp 를 함께 컴파일)
    gen = os.path.join(d, "out")
    inc = os.path.join(gen, "inc", "visp", "arch")
    os.makedirs(inc, exist_ok=True)
    import shutil
    shutil.copy(os.path.join(gen, "Fam.h"), inc)
    b = run(["g++", "-std=c++20", "-O2", "-DARCH=Fam",
             '-DVISP_ARCH_HEADER="visp/arch/Fam.h"',
             f'-DMMDET_PARAMS_HEADER="{ph}"',
             "-I" + gen + "/inc", "-I" + V + "/include", "-I" + V + "/src",
             "-I" + V + "/tools/detect",
             "-I" + V + "/depend/llama/ggml/include", "-I" + V + "/depend/llama/vendor",
             V + "/tools/verify/backbone/run_mmdet.cpp", V + "/tools/detect/head.cpp",
             gen + "/Fam.cpp",
             "-L" + V + "/build/lib", "-lvisioncpp", "-lggml", "-lggml-base", "-lggml-cpu",
             "-Wl,-rpath," + V + "/build/lib", "-o", gen + "/run_mmdet"], d)
    if not os.path.exists(os.path.join(gen, "run_mmdet")):
        return fam, "BUILD_FAIL", (b.stderr.strip().split(chr(10)) or ["-"])[-1][:80]

    # 4) torch 기준값
    open(os.path.join(d, "ref.py"), "w").write(REF % {"FE": FE, "SZ": SZ})
    r = run([PY, "ref.py", fam, cfg_path, gen], d, {"PYTHONPATH": f"{d}:{FE}"})
    if "LEVELS" not in r.stdout:
        return fam, "REF_FAIL", (r.stderr.strip().splitlines() or ["-"])[-1][:70]
    shapes = {}
    for line in r.stdout.splitlines():
        if line.startswith("SHAPE"):
            _, tag, i, c, h, w = line.split()
            shapes[(tag, int(i))] = (int(c), int(h), int(w))

    # 5) C++ 실행 + 대조
    r = run([gen + "/run_mmdet", gen + "/Fam.gguf", "in.bin", "o.bin", str(SZ)], d,
            {"MMDET_DUMP_HEAD": "cpp", "VISP_BACKEND": "cpu"})
    import numpy as np
    worst, nmiss = 1.0, 0
    for (tag, i), (c, h, w) in sorted(shapes.items()):
        pc = os.path.join(d, f"cpp.{tag}.{i}.bin")
        pr = os.path.join(d, f"ref.{tag}.{i}.bin")
        if not os.path.exists(pc):
            nmiss += 1
            continue
        a = np.fromfile(pr, dtype="float32")
        bnp = np.fromfile(pc, dtype="float32")
        if a.size != bnp.size:
            return fam, "SHAPE_MISMATCH", f"{tag}{i} ref {a.size} vs cpp {bnp.size}"
        bnp = bnp.reshape(h, w, c).transpose(2, 0, 1).reshape(-1)   # cwhn → chw
        worst = min(worst, float((a * bnp).sum() / (np.linalg.norm(a) * np.linalg.norm(bnp) + 1e-12)))
    if nmiss:
        err = (r.stderr or r.stdout).strip().splitlines()
        return fam, "RUN_FAIL", (err[0][:80] if err else f"덤프 {nmiss}개 없음")
    st = "PASS" if worst >= 0.99 else "FAIL"
    return fam, st, f"cos {worst:.6f} · kind {kind} · 텐서 {len(shapes)}"


if sys.argv[1:]:
    FAMILIES = [x for x in FAMILIES if x[0] in sys.argv[1:]]

print(f"{'계열':<12} {'판정':<14} 비고")
print("-" * 70)
for fam, rel, ckpt in FAMILIES:
    try:
        f, st, note = one(fam, rel, ckpt)
    except subprocess.TimeoutExpired:
        f, st, note = fam, "TIMEOUT", "-"
    except Exception as e:
        f, st, note = fam, "ERROR", f"{type(e).__name__}: {e}"[:70]
    mark = "O" if st == "PASS" else ("X" if st == "FAIL" else "-")
    print(f"{f:<12} {mark} {st:<12} {note}", flush=True)
