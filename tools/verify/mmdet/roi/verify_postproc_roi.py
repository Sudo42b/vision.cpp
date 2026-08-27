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
V = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))    # vision.cpp (tools/verify/mmdet/roi 에서 4단계 위)
FE = os.path.join(V, "tools", "frontend", "mmdet")
DH = os.path.join(V, "tools", "verify", "dense_head")
GGUF_PY = os.path.join(V, "depend", "llama", "gguf-py")
G2C = os.path.abspath(os.path.join(V, ".."))                 # vision.cpp 를 담은 컴파일러 루트
MM = os.path.expanduser(os.environ.get("MMDET", "~/mmbuild/mmdetection"))
CFGS = os.path.join(MM, "configs")
CKPTS = os.path.join(MM, "checkpoints")
BUILD = os.environ.get("VISP_BUILD", os.path.join(V, "build"))
DEFAULT_IMAGE = os.path.join(V, "tests", "input", "cat-and-hat.jpg")
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

# 마스크 축 — `paste_mask` **전** 로짓의 상대 L1. 텐서 축이 이미 쓰는 수라 새 기준을
# 만들지 않는다. ⚠️ **이진 마스크 IoU 는 쓰지 않는다** — 0.5 이진화는 절벽이라 fp16
# 타이 플립이 그대로 점수에 실린다.
MASK_TOL = 5e-2
# ⚠️ **게이트는 이 집합뿐이다.** 다른 마스크 계열은 수치만 찍는다 — 모수가 1인 축을
#    전 계열 문턱으로 올리면, 재본 적 없는 계열을 측정 없이 재분류하게 된다.
#    수치가 모이면 **별도 이슈로** 전체 게이트를 켤지 판단한다.
MASK_GATE = {"groie"}


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
        # ⚠️ **proposal 파일이 있으면 무조건 그것을 쓴다.** rpn_head 유무로 가르면,
        #    비표준 RPN 계열(GA·CascadeRPN)에서 기준값만 자기 RPN 을 다시 돌려
        #    **양쪽이 다른 proposal 로 재게 된다**(실측: guided_anchoring 7.22px).
        #    러너와 기준값은 같은 파일을 읽어야 비교가 성립한다.
        if os.environ.get("FRCNN_PROPOSALS"):
            # ⚠️ **proposal 을 밖에서 받는 계열**(FastRCNN). `predict` 는 proposal 을
            #    데이터에서 기대하므로 못 부른다. 러너와 **같은 파일**을 읽어
            #    `roi_head.predict` 에 직접 넣는다 — 각자 만들면 proposal 생성기를 재게 된다.
            from mmengine.structures import InstanceData
            pb = np.fromfile(os.environ["FRCNN_PROPOSALS"], dtype="float32").reshape(-1, 4)
            pr = InstanceData(metainfo=meta)
            pr.bboxes = torch.from_numpy(pb.copy())
            pr.scores = torch.ones(len(pb))
            pr.labels = torch.zeros(len(pb), dtype=torch.long)
            # ⚠️ `SparseRoIHead` 는 proposal 에 **query(features)** 가 붙어 있길 기대한다
            #    (`res.pop('features')`). 러너와 **같은 파일**을 읽어야 비교가 성립한다.
            _qp = os.environ["FRCNN_PROPOSALS"] + ".q"
            if os.path.exists(_qp):
                _q = np.fromfile(_qp, dtype="float32").reshape(len(pb), -1)
                pr.features = torch.from_numpy(_q.copy())
            res = det.roi_head.predict(det.extract_feat(t), [pr], [ds], rescale=False)[0]
        else:
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

# 마스크 축 기준값 — **러너가 쓴 그 박스**로 mmdet 의 마스크 갈래를 돌린다.
#
# ⚠️ **mmdet 이 스스로 고른 검출로 재면 안 된다.** 박스가 0.05px 라도 다르면 RoIAlign
#    격자가 달라져 로짓이 통째로 움직인다 — 그러면 마스크 head 가 아니라 **박스 차이**를
#    재게 된다. 러너가 낸 좌표를 그대로 먹여야 마스크 갈래만 남는다.
# ⚠️ `_mask_forward` 를 부른다. RoI 추출기 선택·shared_head 까지 mmdet 자신의 순서다 —
#    우리가 다시 조립하면 그 조립을 재게 된다(`groie` 는 추출기가 `SumGenericRoiExtractor`).
MREF = r'''
import _stub, sys, numpy as np, torch
cfg, ckpt, size, npy, bx, out = (sys.argv[1], sys.argv[2], int(sys.argv[3]),
                                 sys.argv[4], sys.argv[5], sys.argv[6])
sys.path.insert(0, "%(FE)s")
try:
    import mmdet_wrap; mmdet_wrap.allow_mmengine_checkpoint_globals()
except Exception:
    pass
from mmdet.apis import init_detector
from mmdet.structures.bbox import bbox2roi
from frcnn_to_pt import _desync_norm
det = init_detector(_desync_norm(cfg), ckpt, device="cpu"); det.eval()
x = np.load(npy)
t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).contiguous()
boxes = np.load(bx)
with torch.no_grad():
    feats = det.extract_feat(t)
    rois = bbox2roi([torch.from_numpy(boxes.astype("float32"))])
    mp = det.roi_head._mask_forward(feats, rois)["mask_preds"]
np.save(out, mp.numpy())
print("MREF_OK", tuple(mp.shape))
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


# 계열 자신의 rpn_head 로 proposal 을 뽑는다. 러너·기준값이 **이 파일 하나**를 읽는다.
PROPS = r'''
import _stub, os, sys, numpy as np, torch
from PIL import Image
cfg, ckpt, size, img, out, want = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], int(sys.argv[6])
sys.path.insert(0, "%(FE)s")
try:
    import mmdet_wrap; mmdet_wrap.allow_mmengine_checkpoint_globals()
    # ⚠️ GARPNHead 의 MaskedConv2d 는 CUDA 커널만 있다 — CPU 등가 대체를 건다.
    mmdet_wrap.trace_friendly_ops()
except Exception:
    pass
from mmdet.apis import init_detector
from frcnn_to_pt import _desync_norm
from mmdet.structures import DetDataSample
det = init_detector(_desync_norm(cfg), ckpt, device="cpu"); det.eval()
dp = det.data_preprocessor
mean = dp.mean.view(3).numpy() if hasattr(dp, "mean") else np.zeros(3, "float32")
std  = dp.std.view(3).numpy()  if hasattr(dp, "std")  else np.ones(3, "float32")
to_rgb = bool(getattr(dp, "_channel_conversion", False))
im = np.asarray(Image.open(img).convert("RGB").resize((size, size), Image.BILINEAR), dtype="float32")
x = im if to_rgb else im[:, :, ::-1]
t = torch.from_numpy(np.ascontiguousarray((np.ascontiguousarray(x) - mean) / std).astype("float32")).permute(2,0,1)[None]
# ⚠️ `CascadeRPNHead` 는 `pad_shape` 를 읽는다 — 다른 RPN 은 안 읽어서 여태 없었다.
# 우리는 정사각 한 번 리사이즈라 패딩이 없으므로 img_shape 과 같다.
ds = DetDataSample(); ds.set_metainfo({"img_shape": (size, size), "ori_shape": (size, size),
    "pad_shape": (size, size), "scale_factor": (1.0, 1.0), "batch_input_shape": (size, size)})
with torch.no_grad():
    feats = det.extract_feat(t)
    pr = det.rpn_head.predict(feats, [ds], rescale=False)[0]
b = pr.bboxes.numpy()[:want]
# ⚠️ `EmbeddingRPNHead` 는 박스와 **query(features)** 를 같이 낸다 — SparseRoIHead 가
#    단계마다 그 query 를 갱신하며 나르므로 초기값을 러너에 넘겨야 한다.
if "features" in pr:
    np.ascontiguousarray(pr.features.numpy()[:want], dtype="float32").tofile(out + ".q")
    print("QUERIES_OK", pr.features.shape)
# ⚠️ **개수를 상한까지 채운다.** SubB 가 그 행 수로 구워져 있어 모자라면 reshape 이 죽는다.
if len(b) < want:
    b = np.vstack([b, np.zeros((want - len(b), 4), "float32")])
np.ascontiguousarray(b, dtype="float32").tofile(out)
print("PROPS_OK", len(b))
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
    # ⚠️ **계열마다 시험 이미지가 다를 수 있다.** MOT 트래커는 보행자 1클래스라 고양이
    #    사진에서는 양쪽 다 0건이 나온다. 그때는 `match()` 가 None 을 돌려 `EMPTY` 로
    #    보고되므로 **조용히 통과하지는 않지만**, 그 계열을 아예 못 재게 된다.
    # `image` 가 None 이면 기본값이고, 그때만 계열별 지정이 끼어든다.
    default_image = DEFAULT_IMAGE
    image = MF.test_image(fam, DEFAULT_IMAGE) if image is None else image
    import numpy as np
    t0 = time.time()
    d = os.path.join(workdir, fam)
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "_stub.py"), "w").write(STUB)

    # ⚠️ **짝은 공용 자리에서 받는다**(`mmdet_families.resolve_pair`). 예전엔 이 하네스만
    #    metafile 을 보고 one-stage 는 손목록을 봐서 **같은 계열이 다르게 풀렸다.**
    cfg, ckpt_name = MF.resolve_pair(CFGS, fam)
    if not cfg:
        return fam, "CONFIG_NONE", "-", None
    if not ckpt_name:
        return fam, "CKPT_NONE", "가중치 없음 (metafile·손목록 둘 다)", None
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
    # ⚠️ **proposal 을 밖에서 받는 계열**(FastRCNN)은 우리가 넣어 줘야 한다. mmdet 은
    #    proposal 파일을 안 배포하고, 이 계열은 RPN 설정 자체가 없다.
    #    **RPN 출력을 쓰면 그 순간 faster_rcnn 을 재는 것**이 되므로(같은 백본·넥·RoI head),
    #    독립적인 증거가 되도록 **고정 격자**를 쓴다. 결정적이라 재현되고,
    #    RoIAlign+RoI head 를 proposal 생성기와 분리해서 본다.
    #    ⚠️ 러너와 기준값이 **같은 파일**을 읽는다 — 각자 만들면 생성기를 재게 된다.
    prop_env = {}
    if J.get("external_proposals"):
        pf = os.path.join(fr, "proposals.bin")
        want = int(J.get("rpn_max") or 64)
        if J.get("own_rpn"):
            # **그 계열 자신의 RPN** 이 낸 proposal 을 쓴다(GARPNHead·CascadeRPNHead).
            # 호스트가 그 RPN 을 못 깔 뿐, proposal 자체는 이 모델의 것이 정본이다.
            open(os.path.join(d, "props.py"), "w").write(PROPS % {"FE": FE})
            rp = run([PY, "props.py", cfg, ckpt, str(size), image, pf, str(want)], d,
                     {"PYTHONPATH": f"{d}:{FE}"})
            if not os.path.exists(pf):
                return fam, "PROPS_FAIL", last_error(rp.stderr)[:110], None
        else:
            # RPN 이 **아예 없는** 계열(FastRCNN). 낼 주체가 없으므로 고정 격자를 쓴다 —
            # 다른 계열의 RPN 을 빌리면 그 계열을 재는 것이 된다.
            import numpy as _np
            n = int(round(want ** 0.5)); step = size / n
            boxes = [[c * step, rr * step, (c + 2) * step, (rr + 2) * step]
                     for rr in range(n) for c in range(n)][:want]
            _np.asarray(boxes, dtype="float32").clip(0, size).tofile(pf)
        prop_env = {"FRCNN_PROPOSALS": pf}
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
    # ⚠️ **groie 는 배치가 레벨 배다.** GenericRoIExtractor 는 전 레벨에 RoIAlign 을 걸어
    #    레벨별 결과를 배치로 이어붙여 넘긴다(SubB 안에서 pre→합산→post). N 으로 구우면
    #    슬라이스가 빈 텐서가 되어 "tensor a (1000) vs b (0)" 로 죽는다.
    GL = int(J.get("groie_levels") or 0)
    # ⚠️ SparseR-CNN 은 RoI feature 와 **query 를 배치로 이어붙여** 받는다(2N).
    #    proposal 수도 rpn_max 가 아니라 `num_proposals`(학습된 query 개수)다.
    SP = int(J.get("sparse_stages") or 0)
    if SP > 0:
        NP = int(J.get("num_proposals") or MX)
        MB = 2 * NP
    elif GL > 0:
        MB = MX * GL
    else:
        MB = MX
    jobs += [(s, subs[0], "out_" + s, f"{MB},{RC},{O},{O}") for s in subs]
    # Mask Scoring R-CNN 은 점수를 마스크 IoU 로 다시 매긴다 → 그래프가 둘 더 필요하다.
    #   SubC = mask head      (1, 256, 14, 14) → 마스크 로짓 (1, 80, 28, 28)
    #   SubD = mask-IoU head  (1, 257, 14, 14) → 클래스별 IoU (1, 80)
    # ⚠️ **배치 1 로 굽고 러너가 검출 하나씩 돌린다.** ggml 의
    #    `conv_transpose_2d` 가 배치 축을 안 돌아서(ggml-cpu/ops.cpp — src1 을 풀 때
    #    i13 이 없다) 배치로 묶으면 **0번 행만 계산되고 나머지는 bias 만 남는다.**
    #    크래시가 없어 조용히 틀린다. mask head 의 14→28 deconv 가 그 경로다.
    has_miou = bool(J.get("has_mask_iou"))
    MO = int(J.get("mask_roi_out", 14))
    # 마스크 축을 재려면 SubC(mask head)를 구워야 한다. head 가 **하나일 때만** 굽는다 —
    # HTC·SCNet 은 `mask_head` 가 `ModuleList` 라 통째로 태울 수 없다(`mask_head_single`).
    # ⚠️ **여기서 실패해도 박스 판정을 바꾸지 않는다.** 아래 컴파일 루프가 SubC 만
    #    따로 처리하는 이유다 — 마스크는 더 재는 축이지 문턱이 아니다.
    want_mask = bool(J.get("has_mask")) and bool(J.get("mask_head_single"))
    mask_jobs = []
    if want_mask:
        # ⚠️ 배치가 **1 이 아닐 수 있다.** 마스크 추출기가 `GenericRoIExtractor` 면 러너가
        #    레벨 L 개를 배치로 쌓아 넣고 SubC 가 그 안에서 합산한다(`mask_groie_levels`).
        MLV = int(J.get("mask_groie_levels") or 0) or 1
        mask_jobs += [("MaskRCNN_SubC", "MaskRCNN_SubC", "out_MaskRCNN_SubC",
                       f"{MLV},{RC},{MO},{MO}")]
    if has_miou:
        mask_jobs += [("MSRCNN_SubD", "MSRCNN_SubD", "out_MSRCNN_SubD",
                       f"1,{RC + 1},{MO},{MO}")]
    # Grid R-CNN: bbox head 에 회귀 분기가 없고 격자점 히트맵이 박스를 낸다.
    # 여기도 배치 1 이다 — grid head 가 deconv 를 두 번 탄다.
    has_grid = bool(J.get("has_grid"))
    GO = int(J.get("grid_roi_out", 14))
    if has_grid:
        jobs += [("GridRCNN_SubE", "GridRCNN_SubE", "out_GridRCNN_SubE",
                  f"1,{RC},{GO},{GO}")]
    def compile_sub(src, name, outdir, shape):
        """서브그래프 하나를 굽는다. 성공하면 None, 실패하면 사유 문자열."""
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
            return f"{src}: " + last_error(r.stderr)[:100]
        return None

    for src, name, outdir, shape in jobs:
        why = compile_sub(src, name, outdir, shape)
        if why:
            return fam, "COMPILE_FAIL", why, None

    # 마스크 갈래는 **박스 판정과 분리한다.** 마스크 축만 끄고 박스는 그대로 잰다 —
    # 마스크가 없다고 이미 맞은 박스를 못 쓰게 만들 이유가 없다.
    # ⚠️ 단, 끈 사실을 **말한다.** 조용히 끄면 다음 사람이 "쟀는데 통과했다" 로 읽는다.
    # ⚠️ **Mask Scoring R-CNN 은 예외다.** 거기서는 마스크가 더 재는 축이 아니라
    #    **점수를 정하는 경로**다(`score *= mask_iou`) — SubC·SubD 중 하나만 없어도
    #    박스 점수가 틀리므로 예전처럼 COMPILE_FAIL 이어야 한다.
    mask_off = None
    for src, name, outdir, shape in mask_jobs:
        why = compile_sub(src, name, outdir, shape)
        if not why:
            continue
        if has_miou:
            return fam, "COMPILE_FAIL", why, None
        mask_off = why
        want_mask = False
        break

    # ③ 러너 빌드 — 빌드 라인은 build_frcnn_cpp.sh / verify_heads.py 와 같아야 한다.
    import shutil
    incs = [("FRCNN_SubA", "incA"), (subs[0], "incB")]
    if want_mask:
        incs += [("MaskRCNN_SubC", "incC")]
    if has_miou:
        incs += [("MSRCNN_SubD", "incD")]
    if has_grid:
        incs += [("GridRCNN_SubE", "incE")]
    for name, inc in incs:
        os.makedirs(os.path.join(fr, inc, "visp", "arch"), exist_ok=True)
        shutil.copy(os.path.join(fr, "out_" + name, name + ".h"),
                    os.path.join(fr, inc, "visp", "arch"))
    extra = []
    if want_mask:
        extra += ["-DARCH_C=MaskRCNN_SubC",
                  '-DVISP_ARCH_HEADER_C="visp/arch/MaskRCNN_SubC.h"', "-IincC"]
    if has_miou:
        extra += ["-DARCH_D=MSRCNN_SubD",
                  '-DVISP_ARCH_HEADER_D="visp/arch/MSRCNN_SubD.h"', "-IincD"]
    if has_grid:
        extra += ["-DARCH_E=GridRCNN_SubE",
                  '-DVISP_ARCH_HEADER_E="visp/arch/GridRCNN_SubE.h"', "-IincE"]
    b = run(["g++", "-std=c++20", "-O1", "-DARCH_A=FRCNN_SubA", "-DARCH_B=" + subs[0],
             '-DVISP_ARCH_HEADER_A="visp/arch/FRCNN_SubA.h"',
             f'-DVISP_ARCH_HEADER_B="visp/arch/{subs[0]}.h"'] + extra + [
             "-IincA", "-IincB", "-I" + V + "/include", "-I" + V + "/src",
             "-I" + V + "/depend/llama/ggml/include", "-I" + V + "/depend/llama/vendor",
             V + "/tools/verify/mmdet/backbone/run_frcnn.cpp",
             "out_FRCNN_SubA/FRCNN_SubA.cpp", f"out_{subs[0]}/{subs[0]}.cpp"] + (
             ["out_MaskRCNN_SubC/MaskRCNN_SubC.cpp"] if want_mask else []) + (
             ["out_MSRCNN_SubD/MSRCNN_SubD.cpp"] if has_miou else []) + (
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
    # 자리 인자다 — 7=SubC · 8=SubD · 9=SubE. 없는 자리는 빈 문자열로 채운다.
    if want_mask or has_miou or has_grid:
        argv.append("out_MaskRCNN_SubC/MaskRCNN_SubC.gguf" if want_mask else "")
    if has_miou or has_grid:
        argv.append("out_MSRCNN_SubD/MSRCNN_SubD.gguf" if has_miou else "")
    if has_grid:
        argv.append("out_GridRCNN_SubE/GridRCNN_SubE.gguf")
    rr = run(argv, fr, {"VISP_BACKEND": "cpu", **prop_env})
    if not os.path.exists(pref + ".boxes.bin"):
        return fam, "RUN_FAIL", last_error(rr.stderr)[:110], None
    got_all = np.fromfile(pref + ".boxes.bin", dtype="float32").reshape(-1, 6)
    # ⚠️ 마스크 로짓은 **거르기 전** 검출 전부에 대해 나온다 — 러너가 그 순서로 돌렸다.
    #    거른 뒤 배열로 짝지으면 행이 밀린다.
    got = got_all[got_all[:, 4] >= THR]

    # ⑥ mmdet 기준값
    open(os.path.join(d, "ref.py"), "w").write(REF % {"FE": FE})
    refnpy = os.path.join(d, "ref.npy")
    q = run([PY, "ref.py", cfg, ckpt, str(size), npy, refnpy], d,
            {"PYTHONPATH": f"{d}:{FE}", **prop_env})
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
    # 계열마다 이미지가 다를 수 있으므로 **무엇으로 쟀는지**를 숫자 옆에 남긴다.
    # 안 적으면 나중에 0건이 "대상이 못 한다" 인지 "안 맞는 사진을 넣었다" 인지 못 가른다.
    if os.path.basename(image) != os.path.basename(default_image):
        note += f" · {os.path.basename(image)}"
    # ── ⑦ 마스크 축 — `paste_mask` 전 로짓의 상대 L1 ─────────────────────────
    #    ⚠️ **박스 판정과 섞지 않는다.** 게이트는 `MASK_GATE` 뿐이고, 나머지 계열은
    #       수치만 남긴다. 못 잰 계열에 「통과」도 「실패」도 적지 않는다.
    if mask_off:
        note += f" · 마스크 못 잼({mask_off[:40]})"
    elif want_mask:
        m_note, m_ok = _mask_axis(fam, d, fr, pref, cfg, ckpt, size, npy, got_all, prop_env)
        note += " · " + m_note
        if fam in MASK_GATE and not m_ok:
            ok = False

    if verbose and rows:
        for r, g, db in rows:
            print(f"     {int(r[5]):4d} [{r[0]:6.1f},{r[1]:6.1f},{r[2]:6.1f},{r[3]:6.1f}] {r[4]:.3f}"
                  f"  vs [{g[0]:6.1f},{g[1]:6.1f},{g[2]:6.1f},{g[3]:6.1f}] {g[4]:.3f}   {db:6.2f}px")
    if not keep:
        shutil.rmtree(fr, ignore_errors=True)
    return fam, ("PASS" if ok else "FAIL"), note, dt


def _mask_axis(fam, d, fr, pref, cfg, ckpt, size, npy, got_all, prop_env):
    """마스크 로짓을 mmdet 과 대조한다. `(비고 문자열, 통과 여부)`.

    ⚠️ **못 잰 것을 실패로 적지 않는다.** 로짓 파일이 없거나 기준값이 안 나오면
       「못 잼」이라고 쓰고 `True` 를 돌려준다 — 게이트 계열이라도 측정 실패를
       대상의 실패로 바꾸지 않는다. 게이트는 **수치가 나왔는데 넘었을 때** 걸린다.
    """
    import numpy as np
    lg, dims_p = pref + ".mlogit.bin", pref + ".mlogit.dims.bin"
    if not (os.path.exists(lg) and os.path.exists(dims_p)):
        return "마스크 못 잼(로짓 없음)", True
    dims = np.fromfile(dims_p, dtype="float32")
    if dims.size < 4:
        return "마스크 못 잼(축 미상)", True
    MD, NCLS, MH, MW = (int(v) for v in dims[:4])
    if MD <= 0:
        return "마스크 0건", True

    got = np.fromfile(lg, dtype="float32")
    want_n = MD * NCLS * MH * MW
    if got.size != want_n:
        return f"마스크 못 잼(원소 {got.size} != {want_n})", True
    # 러너는 행마다 **cwhn** 으로 낸다: ((y*MW + x) * NCLS + k). torch 는 NCHW 다.
    got = got.reshape(MD, MH, MW, NCLS).transpose(0, 3, 1, 2)

    bx = os.path.join(d, "mboxes.npy")
    np.save(bx, got_all[:MD, :4].astype("float32"))
    open(os.path.join(d, "mref.py"), "w").write(MREF % {"FE": FE})
    refp = os.path.join(d, "mref.npy")
    q = run([PY, "mref.py", cfg, ckpt, str(size), npy, bx, refp], d,
            {"PYTHONPATH": f"{d}:{FE}", **prop_env})
    if "MREF_OK" not in (q.stdout or ""):
        return "마스크 못 잼(기준값: " + last_error(q.stderr)[:40] + ")", True
    ref = np.load(refp)
    if ref.shape != got.shape:
        return f"마스크 못 잼(모양 {tuple(ref.shape)} vs {tuple(got.shape)})", True

    den = float(np.abs(ref).sum())
    if den <= 0:
        return "마스크 못 잼(기준값이 0)", True
    rel = float(np.abs(ref - got).sum()) / den
    # 이진 IoU 는 **적되 판정에 쓰지 않는다** — 0.5 이진화는 절벽이라 fp16 타이 플립이
    # 그대로 점수에 실린다. 눈으로 볼 때만 쓴다.
    a, b = ref > 0, got > 0
    inter, uni = float((a & b).sum()), float((a | b).sum())
    iou = inter / uni if uni else 1.0
    tag = "PASS" if rel < MASK_TOL else "FAIL"
    if fam not in MASK_GATE:
        tag = "수치만"          # 게이트가 아닌 계열은 판정하지 않는다
    return f"마스크 rel L1 {rel:.2e} (IoU {iou:.3f}, {MD}건) {tag}", rel < MASK_TOL


# 계열 하나의 피크 RSS 는 **3.26GB** 다(실측 2026-08-25, 60초 샘플링 — 최대 기여는
# `ref.py` 의 `init_detector` + 800px 추론). WSL 은 `.wslconfig` 로 13GB 상한이다.
# 그래서 워커 2가 안전선이고 3이 상한이다 — 그 이상은 swap 으로 밀려 **더 느려진다.**
# ⚠️ **OOM 을 내면 WSL 이 통째로 죽는다.** 이 저장소는 이미 겪었다(2026-08-10:
#    OOM 0건인데 크래시 덤프 626회로 C: 가 찼다). `.wslconfig` 의 `maxCrashDumpCount=0`
#    이 그 대응이니 되돌리지 마라.
WORKER_CAP = 3
PER_FAM_MB = 3400          # 계열당 피크 RSS(실측 3.26GB)에 여유를 붙인 값


def _avail_mb():
    """가용 메모리(MB). 못 읽으면 가드를 끈다 — 측정 실패로 막지 않는다."""
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except Exception:
        pass
    return 1 << 30


def _safe_workers(want):
    """요청한 워커 수를 **가용 메모리로 다시 깎는다.** 플래그를 그대로 믿지 않는다.

    ⚠️ 왜 플래그를 안 믿나 — 이 기계는 세션이 여럿 떠 있고 그때그때 여유가 다르다.
       `--workers 3` 이 어제 됐다고 오늘 되는 게 아니다. **재서 정한다.**
    """
    want = max(1, int(want))
    if want > WORKER_CAP:
        print(f"  ⚠️ --workers {want} 는 상한 {WORKER_CAP} 을 넘는다 — {WORKER_CAP} 로 낮춘다."
              f" 계열당 피크 RSS 가 {PER_FAM_MB}MB 다(실측).", flush=True)
        want = WORKER_CAP
    avail = _avail_mb()
    fit = max(1, avail // PER_FAM_MB)
    if fit < want:
        print(f"  ⚠️ 가용 메모리 {avail}MB → 워커 {want} → **{fit}** 로 낮춘다"
              f" (계열당 {PER_FAM_MB}MB).", flush=True)
        return fit
    return want


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
    # ⚠️ 기본값일 때만 계열별 이미지가 끼어든다. 사용자가 `--image` 를 **명시하면**
    #    그 뜻을 존중해 전 계열에 그대로 쓴다 — 안 그러면 특정 계열에 다른 사진을 넣어
    #    볼 방법이 없다.
    ap.add_argument("--image", default=None,
                    help="기본: tests/input/cat-and-hat.jpg. 명시하면 계열별 지정을 무시한다")
    ap.add_argument("--workdir", default="/tmp/visp-postproc-roi")
    ap.add_argument("--keep", action="store_true", help="중간 산출물(.pt·gguf·러너)을 남긴다")
    # ⚠️ **이미 통과한 계열을 다시 굽지 마라.** 계열 하나가 40~90초이고 그중 42% 가 g2c
    #    컴파일이다. 40계열 전체 스윕은 35분인데, 대개 고친 것은 두세 계열뿐이라
    #    나머지는 같은 답을 다시 계산하는 데 30분을 쓴다.
    #    수정이 특정 조건에서만 도는 코드면(예: `NS>1`·`RSF>0`) 영향 계열만 골라 재라.
    ap.add_argument("--skip-pass", metavar="results.json",
                    help="이전 결과에서 PASS 였던 계열은 건너뛴다")
    ap.add_argument("--workers", type=int, default=1,
                    help=f"동시 실행 계열 수(기본 1, 상한 {WORKER_CAP}). "
                         "가용 메모리를 읽어 더 낮출 수 있다")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    # ⚠️ **러너를 vision.cpp 빌드 산출물에 링크한다**(`-lvisioncpp -lggml …`). 빌드를
    #    안 했거나 아직 도는 중이면 계열마다 `BUILD_FAIL … ld returned 1 exit status` 만
    #    나와서 **저장소가 깨진 것처럼 보인다**(신규 클론에서 실제로 그렇게 읽었다).
    #    링커가 원인을 못 말하므로 여기서 먼저 말한다.
    missing = [n for n in ("libvisioncpp.so", "libggml.so")
               if not os.path.exists(os.path.join(BUILD, "lib", n))]
    if missing:
        print(f"빌드 산출물이 없다: {', '.join(missing)} (찾은 곳: {BUILD}/lib)\n"
              f"먼저 빌드해라 —\n"
              f"    cmake -S {V} -B {BUILD}\n"
              f"    cmake --build {BUILD} -j4\n"
              f"다른 빌드 디렉토리를 쓰면 VISP_BUILD 로 준다.")
        return 2

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
    # 배너가 거짓말하지 않게 — 계열별 지정이 끼어들 수 있으면 그렇다고 적는다.
    img_note = os.path.basename(a.image) if a.image else \
        f"{os.path.basename(DEFAULT_IMAGE)} (계열별 지정 적용)"
    print(f"size={a.size} · image={img_note} · thr={THR}"
          f" · 판정: 박스<{BOX_TOL}px 점수<{SCORE_TOL} 라벨0 개수차0")
    print(f"{len(fams)}계열: {' '.join(fams)}\n")

    nw = _safe_workers(a.workers)
    if nw > 1:
        print(f"워커 {nw}개로 돈다 (가용 {_avail_mb()}MB · 계열당 {PER_FAM_MB}MB 가정)\n",
              flush=True)

    def _one_guarded(fam):
        """한 계열. **판정 로직은 순차와 완전히 같다** — 감싸기만 한다."""
        try:
            return one(fam, a.size, a.image, a.workdir, a.keep, a.verbose)
        except subprocess.TimeoutExpired:
            return (fam, "TIMEOUT", "-", None)
        except Exception as e:                     # 한 계열이 죽어도 스윕은 계속한다
            return (fam, "HARNESS_FAIL", f"{type(e).__name__}: {e}"[:110], None)

    res = []
    # ⚠️ **계열마다 작업 폴더가 따로다**(`workdir/<fam>`) — 겹치는 파일이 없어야
    #    병렬이 안전하다. `one()` 이 이미 그렇게 짜여 있다(정리도 그 폴더 단위).
    #    여기서 `workdir` 하나를 공유하는 파일은 `results.json` 뿐이고, 그건 부모만 쓴다.
    if nw <= 1:
        for i, fam in enumerate(fams, 1):
            print(f"[{i}/{len(fams)}] {fam} …", flush=True)
            row = _one_guarded(fam)
            res.append(row)
            print(f"    {row[1]:14s} {row[2]}" + (f"   ({row[3]:.0f}s)" if row[3] else ""),
                  flush=True)
            with open(os.path.join(a.workdir, "results.json"), "w") as f:
                json.dump(res, f, indent=1, ensure_ascii=False)
    else:
        import concurrent.futures as _fut
        with _fut.ThreadPoolExecutor(max_workers=nw) as ex:
            futs = {ex.submit(_one_guarded, f): f for f in fams}
            for i, fu in enumerate(_fut.as_completed(futs), 1):
                row = fu.result()
                res.append(row)
                print(f"[{i}/{len(fams)}] {row[0]:22s} {row[1]:14s} {row[2]}"
                      + (f"   ({row[3]:.0f}s)" if row[3] else ""), flush=True)
                with open(os.path.join(a.workdir, "results.json"), "w") as f:
                    json.dump(res, f, indent=1, ensure_ascii=False)
        # ⚠️ **완료 순서로 오므로 다시 정렬한다.** 안 하면 실행마다 줄 순서가 달라져
        #    두 실행을 `diff` 로 못 댄다 — 회귀를 눈으로 보는 길이 막힌다.
        res.sort(key=lambda r: fams.index(r[0]))
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
