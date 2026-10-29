#!/usr/bin/env python3
"""head_support_map.py — 100계열의 **head 지원 여부**만 조사한다(체크포인트 불필요).

왜 따로인가
----------
`verify_heads.py` 는 체크포인트가 없으면 `CKPT_NONE` 으로 먼저 끊는다. 그건 옳다 —
랜덤 가중치로 수치를 재면 항등 초기값(γ=1·β=0)이 누락 연산을 덮어 **조용히 통과**한다.

그런데 그 때문에 표에서 두 가지가 뒤섞인다:

    "체크포인트만 받으면 되는 것"   vs   "조립기가 아예 없는 것"

앞은 곧 잴 수 있고 뒤는 새 부품이 필요하다 — 성격이 완전히 다른데 한 칸에 들어간다.

**head 의 구조는 가중치와 무관하다.** config 로만 모델을 세워 `postproc_cfg` 가 어떤
`head_type` 을 내는지 보면, 체크포인트 없이도 둘을 가를 수 있다. 수치는 재지 않는다 —
이 스크립트는 "지원하나" 만 답하고, "맞나" 는 `verify_heads.py` 가 답한다.

사용:
    python head_support_map.py            # 100계열 전부
    python head_support_map.py atss ddq   # 일부만
"""
import glob
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

# 파이프로 보내도 줄 단위로 나가게 한다(진행이 안 보이면 죽었는지 도는지 모른다).
sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vconfig                                              # noqa: E402

V = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
FE = V + "/tools/frontend/mmdet"
CFG, ARGS = vconfig.load()
MM, PY = CFG.configs, sys.executable
WORKERS, WORK = CFG.workers, CFG.probe_workdir
# ⚠️ **트래커·반지도 래퍼는 config 를 한 겹 벗겨야 한다.** ByteTrack 등은 검출기를
#    `model.detector` 안에 넣고 자기는 껍데기만 갖는다 → `init_detector` 가 로드 전에 죽는다
#    (`'ConfigDict' object has no attribute 'backbone'`). 안 쓰면 7계열이 INIT_FAIL 로 나와
#    **모델 탓처럼 보인다** — 실제로는 조사 도구 탓이다.
UNWRAP = CFG.unwrap
DEPLOY = os.path.dirname(os.path.dirname(os.path.dirname(UNWRAP)))

# `verify_heads.py`·`sweep_integ.py` 와 같은 목록. 모델이 아닌 폴더다(데이터셋 변형·레시피·뼈대).
SKIP_DIRS = {"_base_", "common", "misc", "legacy_1.x", "strong_baselines", "selfsup_pretrain",
             "scratch", "dsdl", "objects365", "lvis", "openimages", "cityscapes", "wider_face",
             "pascal_voc", "deepfashion", "v3det"}

PROBE = r'''
import sys, types
for n in ("mmpretrain.models.multimodal.blip", "mmpretrain.models.multimodal.blip.language_model"):
    m = types.ModuleType(n); m.__path__ = []; sys.modules[n] = m
sys.path.insert(0, %(FE)r)
import mmdet_wrap
mmdet_wrap.trace_friendly_ops()
from mmdet.apis import init_detector
det = init_detector(%(CFG)r, None, device="cpu")   # 가중치 없이 구조만 본다
det.eval()
cfg = mmdet_wrap.postproc_cfg(det)
print("HEADTYPE", cfg.get("head_type", "?"), type(getattr(det, "bbox_head", None)).__name__)
'''


def pick_cfg(d):
    """대표 config **후보 목록**(점수순). 1순위가 없는 형제를 상속해 죽는 경우가 있어
    (`sort` 의 mot20 → FileNotFoundError) 차순위까지 시도해야 한다 — sweep_integ 와 같은 규약."""
    cands = [f for f in glob.glob(os.path.join(d, "*.py"))
             if not os.path.basename(f).startswith("_")
             and not os.path.splitext(os.path.basename(f))[0].endswith("_base")]
    if not cands:
        return None

    def score(f):
        n, s = os.path.basename(f), 0
        for kw, w in (("r50", 4), ("fpn", 3), ("1x", 3), ("coco", 4), ("r18", 2)):
            if kw in n:
                s += w
        return -s
    return sorted(cands, key=score)


def _unwrap(cfg, workdir):
    """래퍼 config 를 안쪽 검출기로 푼다. 풀 필요 없으면 원본을 그대로 돌려준다."""
    if not os.path.exists(UNWRAP):
        return cfg
    out = os.path.join(workdir, "cfg.py")
    try:
        r = subprocess.run([PY, UNWRAP, cfg, "-o", out], cwd=DEPLOY, capture_output=True,
                           text=True, timeout=300)
    except Exception:
        return cfg
    if r.returncode == 0:
        lines = (r.stdout or "").strip().splitlines()
        if lines and os.path.exists(lines[-1].strip()):
            return lines[-1].strip()
    return cfg


def probe(name):
    cands = pick_cfg(os.path.join(MM, name))
    if not cands:
        return name, "NO_CONFIG", "-"
    wd = os.path.join(WORK, name)
    os.makedirs(wd, exist_ok=True)
    err = "-"
    for cfg in cands[:4]:
        cfg = _unwrap(cfg, wd)
        # ⚠️ mmdetection 루트에서 돌린다 — 증류 계열 config 가 교사 모델을
        #    **CWD 기준 상대경로**로 적는다(`_base_` 와 달리 config 위치 기준이 아니다).
        r = subprocess.run([PY, "-c", PROBE % {"FE": FE, "CFG": cfg}],
                           cwd=os.path.dirname(MM.rstrip("/")), capture_output=True, text=True,
                           timeout=600, env={**os.environ, "OMP_NUM_THREADS": "1",
                                             "MPLCONFIGDIR": "/tmp"})
        for line in r.stdout.splitlines():
            if line.startswith("HEADTYPE"):
                _, kind, cls = line.split(None, 2)
                return name, ("HEAD_NONE" if kind == "raw" else "SUPPORTED"), f"{kind:16s} {cls}"
        err = (r.stderr.strip().splitlines() or ["-"])[-1]
    return name, "INIT_FAIL", err[:80]


def main():
    print(CFG.banner(), flush=True)
    os.makedirs(WORK, exist_ok=True)
    names = [d for d in sorted(os.listdir(MM))
             if os.path.isdir(os.path.join(MM, d)) and d not in SKIP_DIRS]
    if ARGS:
        names = [n for n in names if n in ARGS]
    rows = []
    with ThreadPoolExecutor(WORKERS) as ex:
        for name, st, info in ex.map(probe, names):
            rows.append((name, st, info))
            print(f"{name:22s} {st:10s} {info}", flush=True)
    print()
    for st in ("SUPPORTED", "HEAD_NONE", "INIT_FAIL", "NO_CONFIG"):
        n = sum(1 for r in rows if r[1] == st)
        if n:
            print(f"{st:10s} {n}")


if __name__ == "__main__":
    main()
