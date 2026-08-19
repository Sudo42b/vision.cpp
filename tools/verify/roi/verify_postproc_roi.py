#!/usr/bin/env python3
"""verify_postproc_roi.py — **two-stage 계열**의 최종 박스를 mmdet 과 대조한다.

`dense_head/verify_postproc.py` 의 two-stage 판이다. 재는 지점이 같다:

    mmdet 자신의 `detector.predict`  vs  `run_frcnn` 이 낸 최종 박스

왜 별도 스크립트인가
-------------------
one-stage 는 그래프 하나 + `run_mmdet` 하나로 끝난다. two-stage 는 RPN proposal(NMS)과
RoIAlign 이 **실행 중에 개수·좌표가 정해지는** 동작이라 그래프를 둘로 쪼개고 사이에 호스트
코드를 끼운다(`run_frcnn.cpp`). 그래서 준비 단계(export → g2c ×2 → 빌드)가 통째로 다르다.

`verify_heads.py --two-stage` 는 **디코드 앞**(SubB 원시 텐서)에서 끊는다. 이 스크립트는
그 뒤 — 앵커·proposal NMS·RoIAlign·delta 디코드·클래스별 NMS — 를 다 통과한 결과를 본다.

⚠️ **양쪽에 같은 픽셀을 준다.** 정규화까지 마친 배열 하나를 만들어, 러너에는 `.bin` 으로
   torch 에는 그대로 텐서로 넣는다. 각자 리사이즈·정규화하게 두면 백엔드가 아니라 전처리
   구현을 재게 된다(dense head 쪽에서 실제로 그렇게 0.836 → 0.757 로 어긋났다).

사용:
    python verify_postproc_roi.py <계열...>            # 계열명 = configs/ 폴더명
    python verify_postproc_roi.py --all                # two-stage 로 판정된 계열 전부
    python verify_postproc_roi.py faster_rcnn --keep    # 중간 산출물 남기기

환경: `MMDET`(기본 ~/mmbuild/mmdetection) 아래에 `configs/` 와 `checkpoints/` 가 있어야 한다.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
V = os.path.abspath(os.path.join(HERE, "..", "..", ".."))    # vision.cpp (tools/verify/roi 에서 3단계 위)
FE = os.path.join(V, "tools", "frontend", "mmdet")
DH = os.path.join(V, "tools", "verify", "dense_head")
GGUF_PY = os.path.join(V, "depend", "llama", "gguf-py")
G2C = os.path.abspath(os.path.join(V, ".."))                 # vision.cpp 를 담은 컴파일러 루트
MM = os.path.expanduser(os.environ.get("MMDET", "~/mmbuild/mmdetection"))
CFGS = os.path.join(MM, "configs")
CKPTS = os.path.join(MM, "checkpoints")
BUILD = os.environ.get("VISP_BUILD", os.path.join(V, "build"))
PY = sys.executable

sys.path.insert(0, DH)
import mmdet_families as MF                                   # noqa: E402

# 트래커·반지도 래퍼의 껍데기를 벗기는 전처리기. dense_head 하네스가 쓰는 것과 **같은 파일**이다
# (`verify.toml` 의 `unwrap`). 둘이 서로 다른 것을 쓰면 같은 계열이 다르게 풀린다.
# ⚠️ **서브프로세스로 부르지 않는다.** 계열마다 파이썬을 새로 띄우면 `import mmdet` 재비용
#    (실측 5.95초)이 그대로 붙는다 — 두 번 부르니 계열당 ~5초, 40계열이면 3분이다.
#    이 하네스는 이미 부모에서 `mmengine.config` 를 쓴다(`two_stage_families()`).
sys.path.insert(0, os.path.join(G2C, "test_script", "mmdet"))
try:
    import mmdet_unwrap_config as UW                          # noqa: E402
except ImportError:                                           # 전처리기가 없는 트리
    UW = None

# mmpretrain 의 blip 이 이 조합에서 import 시 죽는다 — 하위 프로세스에도 같은 우회를 심는다.
STUB = '''import sys, types
for n in ("mmpretrain.models.multimodal.blip", "mmpretrain.models.multimodal.blip.language_model"):
    m = types.ModuleType(n); m.__path__ = []; sys.modules[n] = m
'''

# 판정 기준 — dense head 와 **같게 둔다**. 다르면 18계열 숫자와 나란히 못 놓는다.
BOX_TOL, SCORE_TOL, THR = 2.0, 0.05, 0.30


def run(cmd, cwd, env_extra=None, timeout=3600):
    env = dict(os.environ, OMP_NUM_THREADS="1")
    env.update(env_extra or {})
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)


def last_error(text):
    """스택트레이스에서 마지막 예외 줄만. 없으면 마지막 비어있지 않은 줄."""
    lines = [l for l in (text or "").splitlines() if l.strip()]
    for l in reversed(lines):
        if ("Error" in l or "error" in l or "Exception" in l) and not l.startswith(" "):
            return l.strip()
    return lines[-1].strip() if lines else "-"


# ── 기준값: mmdet 자신의 predict. 러너에 준 것과 **같은 배열**을 받는다. ─────────
REF = r'''
import _stub, os, sys, numpy as np, torch
cfg, ckpt, size, npy, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
sys.path.insert(0, "%(FE)s")
try:
    import mmdet_wrap; mmdet_wrap.allow_mmengine_checkpoint_globals()
except Exception as e:
    print("  ⚠️ 체크포인트 허용 목록 설정 실패: %%s: %%s" %% (type(e).__name__, e))
from mmdet.apis import init_detector
# ⚠️ 기준값 쪽도 **같은 config 손질**을 거쳐야 한다. SyncBN 은 단일 프로세스에서 build 중에
#    죽으므로(frcnn_to_pt._desync_norm 참고), 내보내기만 고치면 여기서 다시 막힌다.
from frcnn_to_pt import _desync_norm
from mmdet.structures import DetDataSample
det = init_detector(_desync_norm(cfg), ckpt, device="cpu"); det.eval()
x = np.load(npy)                                  # HWC, 정규화까지 끝난 것
t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).contiguous()
meta = {"img_shape": (size, size), "ori_shape": (size, size),
        "scale_factor": (1.0, 1.0), "batch_input_shape": (size, size)}
ds = DetDataSample(); ds.set_metainfo(meta)
with torch.no_grad():
    try:
        res = det.predict(t, [ds], rescale=False)[0].pred_instances
    except AttributeError:
        # ⚠️ **`predict` 가 박스를 안 내는 계열이 있다.** PanopticFPN 은 파놉틱 융합까지 하고
        #    `pred_panoptic_seg` 만 남겨 `pred_instances` 가 없다. 그렇다고 "측정 불가" 로
        #    두면 안 된다 — 우리가 재려는 것은 **two-stage 박스 경로**이고, 그 정본은
        #    detector 가 아니라 `roi_head.predict` 다. 같은 RPN·같은 RoI head 를 탄다.
        feats = det.extract_feat(t)
        rpn = det.rpn_head.predict(feats, [ds], rescale=False)
        res = det.roi_head.predict(feats, rpn, [ds], rescale=False)[0]
np.save(out, np.concatenate([res.bboxes.numpy(),
                             res.scores.numpy()[:, None],
                             res.labels.numpy()[:, None].astype("float32")], 1))
print("REF_OK", len(res.scores))
'''

# 전처리 상수는 detector 의 `data_preprocessor` 가 정본이다. 러너와 torch 에 **같은 배열**을
# 주기 위해, 여기서 한 번만 만들어 `.npy`(torch용) 와 `.bin`(러너용, cwhn) 로 내보낸다.
PREP = r'''
import _stub, os, sys, numpy as np, torch
from PIL import Image
cfg, ckpt, size, img, npy, binp = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6]
sys.path.insert(0, "%(FE)s")
try:
    import mmdet_wrap; mmdet_wrap.allow_mmengine_checkpoint_globals()
except Exception:
    pass
from mmdet.apis import init_detector
# ⚠️ 기준값 쪽도 **같은 config 손질**을 거쳐야 한다. SyncBN 은 단일 프로세스에서 build 중에
#    죽으므로(frcnn_to_pt._desync_norm 참고), 내보내기만 고치면 여기서 다시 막힌다.
from frcnn_to_pt import _desync_norm
det = init_detector(_desync_norm(cfg), ckpt, device="cpu"); det.eval()
dp = det.data_preprocessor
mean = dp.mean.view(3).numpy() if hasattr(dp, "mean") else np.zeros(3, "float32")
std  = dp.std.view(3).numpy()  if hasattr(dp, "std")  else np.ones(3, "float32")
# ⚠️ `_channel_conversion` 이 True 면 **입력이 BGR 이라고 보고 RGB 로 뒤집는** 설정이다.
#    즉 네트워크가 원하는 것은 RGB. False 면 config 의 mean/std 가 BGR 순서다.
to_rgb = bool(getattr(dp, "_channel_conversion", False))
im = np.asarray(Image.open(img).convert("RGB").resize((size, size), Image.BILINEAR), dtype="float32")
x = im if to_rgb else im[:, :, ::-1]              # 네트워크가 먹는 채널 순서로
x = (np.ascontiguousarray(x) - mean) / std
np.save(npy, x)
np.ascontiguousarray(x).tofile(binp)              # HWC 연속 == 러너의 cwhn
print("PREP_OK mean=%%s std=%%s to_rgb=%%s" %% (list(mean), list(std), to_rgb))
'''


def match(ref, got):
    """`dense_head/verify_postproc.match` 와 **같은 짝짓기**. 두 도구의 숫자를 나란히 놓기 위해서다.

    ref 순서대로 가까운 것을 집으면 앞쪽이 뒤쪽의 짝을 가져가 집합이 통째로 밀린다.
    전역으로 가까운 쌍부터 1:1 로 확정한다. 라벨이 같은 쌍만 후보로 둔다.
    """
    import numpy as np
    if len(ref) == 0 or len(got) == 0:
        return None
    pairs = []
    for i, r in enumerate(ref):
        same = np.flatnonzero(got[:, 5] == r[5])
        pool = same if len(same) else np.arange(len(got))
        d = np.abs(got[pool][:, :4] - r[:4]).max(1)
        pairs.extend((float(dist), i, int(j)) for j, dist in zip(pool, d))
    pairs.sort()
    take_r, take_g = {}, set()
    for dist, i, j in pairs:
        if i in take_r or j in take_g:
            continue
        take_r[i] = (j, dist)
        take_g.add(j)
    return [(ref[i], got[take_r[i][0]], take_r[i][1]) for i in range(len(ref)) if i in take_r]


def one(fam, size, image, workdir, keep, verbose):
    """`_one` 을 돌리고, 벗긴 체크포인트를 **어느 경로로 끝나든** 지운다.

    ⚠️ 본문의 `rmtree(fr)` 은 **성공 경로에서만** 닿는다. EXPORT_FAIL·COMPILE_FAIL 등
    조기 반환이 일곱 군데인데, 래퍼 계열의 벗긴 `.pth` 는 계열당 150~500MB 라
    실패가 몇 개만 나도 스윕이 수 GB 를 남긴다.
    """
    try:
        return _one(fam, size, image, workdir, keep, verbose)
    finally:
        if not keep:
            try:
                os.remove(os.path.join(workdir, fam, "frcnn", f"{fam}.ckpt.pth"))
            except OSError:
                pass


def _one(fam, size, image, workdir, keep, verbose):
    import numpy as np
    t0 = time.time()
    d = os.path.join(workdir, fam)
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "_stub.py"), "w").write(STUB)

    cfg_rel, ckpt_name, _ = MF.resolve(CFGS, fam)
    if not cfg_rel:
        return fam, "CONFIG_NONE", "-", None
    cfg = os.path.join(CFGS, cfg_rel)
    if not ckpt_name:
        return fam, "CKPT_NONE", f"metafile 에 가중치 없음 ({cfg_rel})", None
    ckpt = os.path.join(CKPTS, ckpt_name)
    if not os.path.exists(ckpt):
        return fam, "CKPT_MISSING", ckpt_name, None

    # ⚠️ **트래커·반지도 래퍼는 config 를 한 겹 벗겨야 한다.** `SoftTeacher`·`DeepSORT` 등은
    #    검출기를 `model.detector` 안에 넣고 자기는 껍데기(tracker/reid/semi_train_cfg)만 갖는다
    #    → `'ConfigDict' object has no attribute 'backbone'` 으로 export 에서 죽는다.
    #    dense_head 하네스는 이미 이 단계를 갖고 있었고 여기만 없어서, 안쪽이 이미 통과한
    #    검출기인 8계열이 통째로 못 걸리고 있었다. 풀 필요 없는 config 는 원본을 그대로
    #    돌려주므로 분기 없이 전부 통과시킨다. 체크포인트도 같이 벗긴다(아래 순서 주의).
    fr = os.path.join(d, "frcnn")
    if UW is not None:
        # ⚠️ **체크포인트를 먼저, config 를 나중에.** 접두사는 **원본** config 의
        #    `semi_test_cfg.predict_on` 이 정하는데 벗긴 config 에는 그 키가 없다.
        #    순서를 바꾸면 한 텐서도 안 실리고, 그런데도 로드는 조용히 성공해서
        #    가중치가 랜덤인 채로 박스가 수백 px 어긋난다(soft_teacher 562px 로 겪었다).
        #    산출물은 **`fr` 안에** 둔다 — 정리(`rmtree(fr)`)에 같이 쓸려 나가야 한다.
        #    계열 밖에 두면 벗긴 체크포인트(계열당 150~500MB)가 실행마다 쌓인다.
        #    ⚠️ **둘 다 되거나 둘 다 안 되거나여야 한다.** 체크포인트만 벗기고 config 를
        #    못 벗기면(또는 반대면) 짝이 어긋나 위와 똑같은 사고가 난다. 하나라도 실패하면
        #    **둘 다 원본으로 되돌린다.**
        cfg0, ckpt0 = cfg, ckpt
        try:
            ckpt = UW.unwrap_checkpoint(cfg0, ckpt0, os.path.join(fr, f"{fam}.ckpt.pth"))
            cfg = UW.unwrap_config(cfg0, os.path.join(fr, f"{fam}.cfg.py"))
        except Exception as e:
            cfg, ckpt = cfg0, ckpt0
            print(f"  [unwrap] 실패 — 원본으로 되돌린다: {type(e).__name__}: {e}",
                  file=sys.stderr)

    # ① export — two-stage 를 두 subgraph 로 가른다. 여기서 죽으면 "왜" 를 그대로 옮긴다.
    # ⚠️ **단계마다 산출물을 먼저 지운다.** 존재 검사만 하면 이번 실행이 실패해도 지난
    #    실행의 것이 남아 있어 **그대로 통과한다** — 고친 코드가 반영 안 된 숫자를
    #    "통과" 로 보고하게 된다. 공유 코드를 자주 고치는 날에는 이게 제일 위험하다
    #    (verify_heads.py 가 `bb.pt` 에 대해 같은 이유로 이미 지우고 있다).
    for stale in ("frcnn.json", "run_frcnn"):
        try:
            os.remove(os.path.join(fr, stale))
        except OSError:
            pass
    r = run([PY, os.path.join(FE, "frcnn_to_pt.py"), "--config", cfg, "--checkpoint", ckpt,
             "--out", fr, "--size", str(size)], MM, {"PYTHONPATH": f"{d}:{FE}"})
    if not os.path.exists(os.path.join(fr, "frcnn.json")):
        err = last_error(r.stderr)
        kind = "UNSUPPORTED" if "NotImplementedError" in (r.stderr or "") else "EXPORT_FAIL"
        return fam, kind, err[:110], None
    J = json.load(open(os.path.join(fr, "frcnn.json")))
    O, RC = int(J["roi_out"]), int(J.get("roi_channels", 256))
    MX = int(J["rpn_max"])
    # Double-Head 는 (cls용, reg용) 두 벌을 배치로 이어 넣는다 → 배치가 2배다.
    if float(J.get("reg_roi_scale_factor", 0) or 0) > 0:
        MX *= 2
    NS = int(J.get("num_bbox_stages", 1))
    subs = ["FRCNN_SubB"] if NS == 1 else [f"FRCNN_SubB{i}" for i in range(NS)]

    # ② g2c 컴파일 — SubA 는 이미지 해상도로, SubB 는 **proposal 상한 배치**로.
    #    batch=1 로 구우면 1000개를 넣을 때 reshape 이 안 맞아 죽는다.
    jobs = [("FRCNN_SubA", "FRCNN_SubA", "out_FRCNN_SubA", f"1,3,{size},{size}")]
    #    캐스케이드는 단계마다 가중치만 다르므로 그래프 이름을 subs[0] 으로 통일해 gguf 만 갈아 낀다.
    jobs += [(s, subs[0], "out_" + s, f"{MX},{RC},{O},{O}") for s in subs]
    # Mask Scoring R-CNN 은 점수를 마스크 IoU 로 다시 매긴다 → 그래프가 둘 더 필요하다.
    #   SubC = mask head      (1, 256, 14, 14) → 마스크 로짓 (1, 80, 28, 28)
    #   SubD = mask-IoU head  (1, 257, 14, 14) → 클래스별 IoU (1, 80)
    # ⚠️ **배치 1 로 굽고 러너가 검출 하나씩 돌린다.** ggml 의
    #    `conv_transpose_2d` 가 배치 축을 안 돌아서(ggml-cpu/ops.cpp — src1 을 풀 때
    #    i13 이 없다) 배치로 묶으면 **0번 행만 계산되고 나머지는 bias 만 남는다.**
    #    크래시가 없어 조용히 틀린다. mask head 의 14→28 deconv 가 그 경로다.
    has_miou = bool(J.get("has_mask_iou"))
    MO = int(J.get("mask_roi_out", 14))
    if has_miou:
        jobs += [("MaskRCNN_SubC", "MaskRCNN_SubC", "out_MaskRCNN_SubC",
                  f"1,{RC},{MO},{MO}"),
                 ("MSRCNN_SubD", "MSRCNN_SubD", "out_MSRCNN_SubD",
                  f"1,{RC + 1},{MO},{MO}")]
    # Grid R-CNN: bbox head 에 회귀 분기가 없고 격자점 히트맵이 박스를 낸다.
    # 여기도 배치 1 이다 — grid head 가 deconv 를 두 번 탄다.
    has_grid = bool(J.get("has_grid"))
    GO = int(J.get("grid_roi_out", 14))
    if has_grid:
        jobs += [("GridRCNN_SubE", "GridRCNN_SubE", "out_GridRCNN_SubE",
                  f"1,{RC},{GO},{GO}")]
    for src, name, outdir, shape in jobs:
        try:
            os.remove(os.path.join(fr, outdir, f"{name}.gguf"))   # 위와 같은 이유
        except OSError:
            pass
        r = run([PY, "-c", f'''
import _stub, sys
sys.argv = ["g2c","--model","{src}.pt","--name","{name}","--output","{outdir}","--input-shape","{shape}"]
from shared.compile.pipeline import main; main()
'''], fr, {"PYTHONPATH": f"{d}:{fr}:{G2C}:{FE}:{GGUF_PY}"})
        if not os.path.exists(os.path.join(fr, outdir, f"{name}.gguf")):
            return fam, "COMPILE_FAIL", f"{src}: " + last_error(r.stderr)[:100], None

    # ③ 러너 빌드 — 빌드 라인은 build_frcnn_cpp.sh / verify_heads.py 와 같아야 한다.
    import shutil
    incs = [("FRCNN_SubA", "incA"), (subs[0], "incB")]
    if has_miou:
        incs += [("MaskRCNN_SubC", "incC"), ("MSRCNN_SubD", "incD")]
    if has_grid:
        incs += [("GridRCNN_SubE", "incE")]
    for name, inc in incs:
        os.makedirs(os.path.join(fr, inc, "visp", "arch"), exist_ok=True)
        shutil.copy(os.path.join(fr, "out_" + name, name + ".h"),
                    os.path.join(fr, inc, "visp", "arch"))
    extra = []
    if has_miou:
        extra += ["-DARCH_C=MaskRCNN_SubC", "-DARCH_D=MSRCNN_SubD",
                  '-DVISP_ARCH_HEADER_C="visp/arch/MaskRCNN_SubC.h"',
                  '-DVISP_ARCH_HEADER_D="visp/arch/MSRCNN_SubD.h"',
                  "-IincC", "-IincD"]
    if has_grid:
        extra += ["-DARCH_E=GridRCNN_SubE",
                  '-DVISP_ARCH_HEADER_E="visp/arch/GridRCNN_SubE.h"', "-IincE"]
    b = run(["g++", "-std=c++20", "-O1", "-DARCH_A=FRCNN_SubA", "-DARCH_B=" + subs[0],
             '-DVISP_ARCH_HEADER_A="visp/arch/FRCNN_SubA.h"',
             f'-DVISP_ARCH_HEADER_B="visp/arch/{subs[0]}.h"'] + extra + [
             "-IincA", "-IincB", "-I" + V + "/include", "-I" + V + "/src",
             "-I" + V + "/depend/llama/ggml/include", "-I" + V + "/depend/llama/vendor",
             V + "/tools/verify/backbone/run_frcnn.cpp",
             "out_FRCNN_SubA/FRCNN_SubA.cpp", f"out_{subs[0]}/{subs[0]}.cpp"] + (
             ["out_MaskRCNN_SubC/MaskRCNN_SubC.cpp", "out_MSRCNN_SubD/MSRCNN_SubD.cpp"]
             if has_miou else []) + (
             ["out_GridRCNN_SubE/GridRCNN_SubE.cpp"] if has_grid else []) + [
             "-L" + BUILD + "/lib", "-lvisioncpp", "-lggml", "-lggml-base", "-lggml-cpu",
             "-Wl,-rpath," + BUILD + "/lib", "-o", "run_frcnn"], fr)
    if not os.path.exists(os.path.join(fr, "run_frcnn")):
        return fam, "BUILD_FAIL", last_error(b.stderr)[:110], None

    # ④ 같은 픽셀 만들기 (러너용 .bin + torch 용 .npy)
    open(os.path.join(d, "prep.py"), "w").write(PREP % {"FE": FE})
    npy, binp = os.path.join(d, "x.npy"), os.path.join(d, "in.bin")
    p = run([PY, "prep.py", cfg, ckpt, str(size), image, npy, binp], d, {"PYTHONPATH": f"{d}:{FE}"})
    if "PREP_OK" not in (p.stdout or ""):
        return fam, "PREP_FAIL", last_error(p.stderr)[:110], None
    if verbose:
        print("   ", p.stdout.strip().splitlines()[-1])

    # ⑤ 러너 실행
    pref = os.path.join(d, "cpp")
    argv = [os.path.join(fr, "run_frcnn"), "out_FRCNN_SubA/FRCNN_SubA.gguf",
            ",".join(f"out_{s}/{subs[0]}.gguf" for s in subs),
            "frcnn.json", binp, pref, str(size)]
    if has_miou:
        argv += ["out_MaskRCNN_SubC/MaskRCNN_SubC.gguf", "out_MSRCNN_SubD/MSRCNN_SubD.gguf"]
    if has_grid:
        # 러너 인자는 자리로 읽는다 — SubE 는 9번째다. 마스크가 없으면 자리를 채운다.
        while len(argv) < 9:
            argv.append("")
        argv.append("out_GridRCNN_SubE/GridRCNN_SubE.gguf")
    rr = run(argv, fr, {"VISP_BACKEND": "cpu"})
    if not os.path.exists(pref + ".boxes.bin"):
        return fam, "RUN_FAIL", last_error(rr.stderr)[:110], None
    got = np.fromfile(pref + ".boxes.bin", dtype="float32").reshape(-1, 6)
    got = got[got[:, 4] >= THR]

    # ⑥ mmdet 기준값
    open(os.path.join(d, "ref.py"), "w").write(REF % {"FE": FE})
    refnpy = os.path.join(d, "ref.npy")
    q = run([PY, "ref.py", cfg, ckpt, str(size), npy, refnpy], d, {"PYTHONPATH": f"{d}:{FE}"})
    if "REF_OK" not in (q.stdout or ""):
        return fam, "REF_FAIL", last_error(q.stderr)[:110], None
    ref = np.load(refnpy)
    ref = ref[ref[:, 4] >= THR]

    rows = match(ref, got)
    dt = time.time() - t0
    if rows is None:
        return fam, "EMPTY", f"mmdet {len(ref)}건 · C++ {len(got)}건 — 한쪽이 비었다", dt
    worst_b = max((db for _, _, db in rows), default=0.0)
    worst_s = max((abs(r[4] - g[4]) for r, g, _ in rows), default=0.0)
    bad_label = sum(int(r[5] != g[5]) for r, g, _ in rows)
    n_gap = abs(len(ref) - len(got))
    ok = worst_b < BOX_TOL and worst_s < SCORE_TOL and bad_label == 0 and n_gap == 0
    note = (f"박스 {worst_b:.2f}px · 점수 {worst_s:.4f} · 라벨 {bad_label} · 개수차 {n_gap}"
            f" · {len(ref)}/{len(got)}건")
    if verbose and rows:
        for r, g, db in rows:
            print(f"     {int(r[5]):4d} [{r[0]:6.1f},{r[1]:6.1f},{r[2]:6.1f},{r[3]:6.1f}] {r[4]:.3f}"
                  f"  vs [{g[0]:6.1f},{g[1]:6.1f},{g[2]:6.1f},{g[3]:6.1f}] {g[4]:.3f}   {db:6.2f}px")
    if not keep:
        shutil.rmtree(fr, ignore_errors=True)
    return fam, ("PASS" if ok else "FAIL"), note, dt


def two_stage_families():
    """config 로 판정한다 — 이름으로 짐작하지 않는다. roi_head 가 있으면 two-stage.

    ⚠️ **래퍼는 한 겹 안을 본다.** 트래커·반지도는 `roi_head` 를 `model.detector` 안에
    넣으므로 최상위만 보면 통째로 빠진다 — 그러면 통과한 계열이 회귀 검사를 **안 받는다.**
    안 재는 것이 실패보다 위험하다(안 재면 아무도 모른다).

    안쪽이 one-stage 인 래퍼(YOLOX 기반 bytetrack·ocsort·strongsort)는 여기 안 걸린다.
    그건 dense_head 하네스 몫이다 — 이 함수는 `roi_head` 유무로만 가른다.
    """
    from mmengine.config import Config
    out = []
    for fam, cfg_rel, _ in MF.families(CFGS):
        try:
            m = Config.fromfile(os.path.join(CFGS, cfg_rel)).get("model", {})
        except Exception:
            continue
        # 안쪽 키 목록은 전처리기와 **한 곳**에서 온다. 여기 따로 적으면 한쪽이 늘 때
        # 다른 쪽이 안 따라가고, 같은 계열이 두 하네스에서 다르게 분류된다.
        keys = getattr(UW, "_INNER_KEYS", ("detector",)) if UW else ("detector",)
        if not m.get("roi_head") and "backbone" not in m:
            m = next((m[k] for k in keys if isinstance(m.get(k), dict)), m)
        if m.get("roi_head"):
            out.append(fam)
    return out


def main():
    ap = argparse.ArgumentParser(prog="verify_postproc_roi")
    ap.add_argument("families", nargs="*")
    ap.add_argument("--all", action="store_true", help="two-stage 로 판정된 계열 전부")
    ap.add_argument("--size", type=int, default=800)
    ap.add_argument("--image", default=os.path.join(V, "tests", "input", "cat-and-hat.jpg"))
    ap.add_argument("--workdir", default="/tmp/visp-postproc-roi")
    ap.add_argument("--keep", action="store_true", help="중간 산출물(.pt·gguf·러너)을 남긴다")
    # ⚠️ **이미 통과한 계열을 다시 굽지 마라.** 계열 하나가 40~90초이고 그중 42% 가 g2c
    #    컴파일이다. 40계열 전체 스윕은 35분인데, 대개 고친 것은 두세 계열뿐이라
    #    나머지는 같은 답을 다시 계산하는 데 30분을 쓴다.
    #    수정이 특정 조건에서만 도는 코드면(예: `NS>1`·`RSF>0`) 영향 계열만 골라 재라.
    ap.add_argument("--skip-pass", metavar="results.json",
                    help="이전 결과에서 PASS 였던 계열은 건너뛴다")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    fams = a.families or (two_stage_families() if a.all else [])
    if not fams:
        print(__doc__)
        return 2
    if a.skip_pass:
        prev = {r[0]: r[1] for r in json.load(open(a.skip_pass))}
        done = [f for f in fams if prev.get(f) == "PASS"]
        fams = [f for f in fams if prev.get(f) != "PASS"]
        # 건너뛴 것을 **말한다.** 조용히 줄이면 다음 사람이 "전부 쟀다" 로 읽는다.
        print(f"이전 PASS {len(done)}계열 건너뜀: {' '.join(done)}\n")
    os.makedirs(a.workdir, exist_ok=True)
    print(f"size={a.size} · image={os.path.basename(a.image)} · thr={THR}"
          f" · 판정: 박스<{BOX_TOL}px 점수<{SCORE_TOL} 라벨0 개수차0")
    print(f"{len(fams)}계열: {' '.join(fams)}\n")

    res = []
    for i, fam in enumerate(fams, 1):
        print(f"[{i}/{len(fams)}] {fam} …", flush=True)
        try:
            row = one(fam, a.size, a.image, a.workdir, a.keep, a.verbose)
        except subprocess.TimeoutExpired:
            row = (fam, "TIMEOUT", "-", None)
        except Exception as e:                     # 한 계열이 죽어도 스윕은 계속한다
            row = (fam, "HARNESS_FAIL", f"{type(e).__name__}: {e}"[:110], None)
        res.append(row)
        print(f"    {row[1]:14s} {row[2]}" + (f"   ({row[3]:.0f}s)" if row[3] else ""), flush=True)
        with open(os.path.join(a.workdir, "results.json"), "w") as f:
            json.dump(res, f, indent=1, ensure_ascii=False)

    print("\n" + "=" * 78)
    for fam, st, note, dt in res:
        print(f"  {fam:22s} {st:14s} {note}")
    n_pass = sum(1 for _, s, _, _ in res if s == "PASS")
    print(f"\nPASS {n_pass}/{len(res)}")
    return 0 if n_pass == len(res) else 1


if __name__ == "__main__":
    sys.exit(main())
