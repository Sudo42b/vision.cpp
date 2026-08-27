#!/usr/bin/env python3
"""mmseg 계열을 g2c 로 컴파일해 C++(ggml) 출력이 PyTorch 와 같은지 계열별로 잰다.

mmdet 하네스(`tools/verify/mmdet/dense_head/verify_heads.py`)와 **재는 방식이 같고 훨씬 짧다** —
세그멘테이션은 앵커도 NMS 도 박스 디코드도 없어서 `backbone → (neck) → decode_head` 를
한 그래프로 컴파일하고 **출력 텐서를 그대로 대조**하면 끝난다.

점수는 **상대 L1/L2** 다. cosine 은 쓰지 않는다 — 스케일 불변이라 크기가 통째로 틀려도
1.0 이 나온다.

    python verify_seg.py                 # metafile 에 있는 계열 전부
    python verify_seg.py fcn pspnet      # 골라서
    python verify_seg.py --list          # 목록만

⚠️ **랜덤 초기화로 재지 마라.** 항등 초기값(BN γ=1·β=0)이 빠진 연산을 덮는다.
   체크포인트가 없으면 그 계열은 CKPT_NONE 으로 남긴다 — **못 잰 것이지 통과가 아니다.**
"""
import argparse
import os
import subprocess
import sys

sys.stdout.reconfigure(line_buffering=True)

HERE = os.path.dirname(os.path.abspath(__file__))
V = os.path.abspath(os.path.join(HERE, "..", "..", ".."))      # vision.cpp
P = os.path.abspath(os.path.join(V, ".."))                      # g2c 루트
FE = V + "/tools/frontend/mmseg"
FE_DET = V + "/tools/frontend/mmdet"
GGUF_PY = V + "/depend/llama/gguf-py"
PY = sys.executable

MM = os.path.expanduser("~/mmbuild/mmsegmentation")
CKPT = MM + "/checkpoints"
WORKDIR = "/tmp/visp-seg-verify"
SZ = 0                     # 0 = config 의 crop 을 쓴다. `--size` 로만 덮어쓴다
OPT = "-O1"
# 판정 기준: 출력 텐서의 상대 L1/L2. 저장소 관례(mmdet 하네스)와 같은 값을 쓴다.
L1_TOL = L2_TOL = 0.05


def families():
    """`configs/*/metafile.yaml` 에서 (계열, config, weights URL) 을 읽는다.

    **손으로 적지 않는다** — 목록을 손으로 적으면 거기 없는 계열이 존재하지 않는 것처럼
    보인다(mmdet 에서 `pisa`·`rpn` 이 실제로 그렇게 몇 주 동안 안 보였다).
    계열마다 metafile 의 **첫 모델**을 대표로 쓴다.
    """
    import glob
    import yaml
    out = []
    for p in sorted(glob.glob(MM + "/configs/*/metafile.y*ml")):
        fam = os.path.basename(os.path.dirname(p))
        try:
            models = yaml.safe_load(open(p)).get("Models") or []
        except Exception:
            continue
        if not models:
            continue
        m = models[0]
        cfg = MM + "/" + m["Config"] if not m["Config"].startswith("/") else m["Config"]
        out.append((fam, cfg, m.get("Weights", "")))
    return out


# ⚠️ **메모리 가드.** 위키 `wsl-계속-터짐` 5번째 원인 — 폭주한 프로세스가 없어도 **합**이
#    넘치면 WSL 이 통째로 죽는다(2026-08-27 실측: 내 스윕 3.6GB + 다른 python 2.25GB +
#    VS Code 3.4GB + claude 세션 5개 1.5GB = 13GB 상한 초과 → OOM 8건).
#    가용 메모리가 이 밑이면 **새 계열을 안 띄우고 기다린다.** 느려질지언정 안 죽는다.
MIN_FREE_MB = int(os.environ.get("VISP_MIN_FREE_MB", "2500"))
# 한 서브프로세스가 이 이상 잡으면 **그놈만** 죽는다(VM 이 아니라). 위키의 인덱서 대책과 같다.
MAX_SUBPROC_GB = int(os.environ.get("VISP_MAX_SUBPROC_GB", "8"))


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
        time.sleep(5)
        waited += 5
        if waited > 600:      # 10분을 기다려도 안 풀리면 그냥 간다(교착 방지)
            print(f"  … 10분 대기 후에도 부족 — 그대로 진행한다 ({fam})", flush=True)
            break


def run(cmd, cwd=None, env=None, timeout=1800):
    e = dict(os.environ)
    e.setdefault("OMP_NUM_THREADS", "1")
    # glibc 이 스레드마다 arena 를 잡아 RSS 가 부푼다. torch 서브프로세스에서 크게 온다.
    e.setdefault("MALLOC_ARENA_MAX", "2")
    if env:
        e.update(env)

    def _limit():
        # ⚠️ **주소공간 상한.** 폭주하면 이 프로세스만 MemoryError 로 죽고 VM 은 산다.
        #    위키 `wsl-계속-터짐` 의 인덱서 대책(`ulimit -v`)과 같은 수법이다.
        import resource
        n = MAX_SUBPROC_GB * (1 << 30)
        try:
            resource.setrlimit(resource.RLIMIT_AS, (n, n))
        except (ValueError, OSError):
            pass

    try:
        return subprocess.run(cmd, cwd=cwd, env=e, capture_output=True,
                              text=True, timeout=timeout, preexec_fn=_limit)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(cmd, 124, "", "timeout")


def _last_error(text):
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    for l in reversed(lines):
        if "Error" in l or "error" in l or "assert" in l.lower():
            return l
    return lines[-1] if lines else ""


REF = r'''
import os, sys, numpy as np, torch
sys.path.insert(0, "%(FE)s"); sys.path.insert(0, "%(FE_DET)s")
import mmseg_wrap
# ⚠️ **입력 크기를 하나로 고정하지 않는다.** config 의 crop 이 곧 그 모델이 보는 크기다
#    (512x1024 · 640x640 · 680x680 · 1024x1024 …). 고정하면 잰 것이 그 모델이 아니고,
#    ViT 계열은 `pos_embed` 토큰 수가 안 맞아 아예 죽는다.
# ⚠️ 기본은 **config 의 crop** 이다. `--size` 는 진단용 덮어쓰기다(크기 탓인지 가를 때).
#    2026-08-27 까지 `--size` 는 설정만 되고 **아무 데도 안 쓰였다** — 헤더에는 512 라
#    찍히는데 실제로는 crop 으로 쟀다. 거짓말하는 플래그였다.
_ov = "%(SIZE)s"
H, W = ([int(_ov), int(_ov)] if _ov else mmseg_wrap.crop_size("%(CFG)s"))
m, shapes = mmseg_wrap.build("%(CFG)s", "%(CK)s", (H, W))
torch.save(m, "seg.pt")
open("size.txt", "w").write(f"{W} {H}")
# ⚠️ **고정 입력**을 쓴다. 시드 없이 뽑으면 실행마다 숫자가 1~5%% 흔들려 회귀 대조가
#    흐려진다(mmdet 의 no-box 계열에서 실제로 겪었다).
g = torch.Generator().manual_seed(0)
x = torch.randn(1, 3, H, W, generator=g)
with torch.no_grad():
    outs = m(x)
if isinstance(outs, torch.Tensor):
    outs = (outs,)
sh = []
for i, t in enumerate(outs):
    a = t[0].detach().numpy().astype("float32")
    np.ascontiguousarray(a).tofile("ref.out.%%d.bin" %% i)
    sh.append(list(a.shape))
open("ref.shapes.txt", "w").write("\n".join(" ".join(map(str, s)) for s in sh))
np.ascontiguousarray(x[0].numpy().transpose(1, 2, 0)).tofile("in.bin")   # 러너 입력(cwhn)
print("REF_OK", len(outs))
'''


def verify(fam, cfg, weights):
    import shutil
    d = os.path.join(WORKDIR, fam)
    os.makedirs(d, exist_ok=True)
    ck = os.path.join(CKPT, os.path.basename(weights)) if weights else ""
    if not ck or not os.path.exists(ck):
        return fam, "CKPT_NONE", "체크포인트 미다운로드 (--fetch 로 받는다)"
    if not os.path.exists(cfg):
        return fam, "CONFIG_NONE", os.path.basename(cfg)

    # ⚠️ **먼저 지운다.** 지난 실행의 산출물이 남아 있으면 export 가 실패해도 아래 존재
    #    검사를 통과해 **낡은 것으로 계속 간다**(mmdet 하네스에서 실제로 겪었다).
    _wait_for_memory(fam)
    for stale in ("seg.pt", "ref.shapes.txt", "size.txt"):
        try:
            os.remove(os.path.join(d, stale))
        except FileNotFoundError:
            pass

    open(os.path.join(d, "ref.py"), "w").write(
        REF % {"SIZE": (str(SZ) if SZ else ""), "FE": FE, "FE_DET": FE_DET, "CFG": cfg, "CK": ck})
    r = run([PY, "ref.py"], d, {"PYTHONPATH": f"{FE}:{FE_DET}"})
    if not os.path.exists(os.path.join(d, "ref.shapes.txt")):
        return fam, "REF_FAIL", _last_error(r.stderr)[:70]
    W, H = [int(v) for v in open(os.path.join(d, "size.txt")).read().split()]

    gen = os.path.join(d, "out")
    r = run([PY, "-c", '''
import sys
sys.argv = ["g2c","--model","seg.pt","--name","Seg","--output","out","--input-shape","1,3,%d,%d"]
from shared.compile.pipeline import main; main()
''' % (H, W)], d, {"PYTHONPATH": f"{FE}:{FE_DET}:{P}:{GGUF_PY}"})
    if not os.path.exists(os.path.join(gen, "Seg.gguf")):
        return fam, "COMPILE_FAIL", _last_error(r.stderr)[:70]
    src = open(os.path.join(gen, "Seg.cpp")).read()
    n_unhandled = src.count("unhandled op")

    inc = os.path.join(gen, "inc", "visp", "arch")
    os.makedirs(inc, exist_ok=True)
    shutil.copy(os.path.join(gen, "Seg.h"), inc)
    b = run(["g++", "-std=c++20", OPT, "-DARCH=Seg",
             '-DVISP_ARCH_HEADER="visp/arch/Seg.h"',
             "-I" + os.path.join(gen, "inc"), "-I" + V + "/src", "-I" + V + "/include",
             "-I" + V + "/depend/llama/ggml/include",
             V + "/tools/verify/common/run_dump.cpp", os.path.join(gen, "Seg.cpp"),
             "-L" + V + "/build/lib", "-lvisioncpp", "-lggml", "-lggml-base", "-lggml-cpu",
             "-Wl,-rpath," + V + "/build/lib", "-o", os.path.join(gen, "run_dump")], d)
    if not os.path.exists(os.path.join(gen, "run_dump")):
        return fam, "BUILD_FAIL", _last_error(b.stderr)[:70]

    for f in os.listdir(d):
        if f.startswith("cpp.out.") and f.endswith(".bin"):
            os.remove(os.path.join(d, f))
    x = run([os.path.join(gen, "run_dump"), os.path.join(gen, "Seg.gguf"),
             os.path.join(d, "in.bin"), os.path.join(d, "cpp"), str(W), str(H)], d)

    import numpy as np
    shapes = [[int(v) for v in ln.split()]
              for ln in open(os.path.join(d, "ref.shapes.txt")).read().splitlines()
              if ln.strip()]
    worst_l1 = worst_l2 = 0.0
    for i, sh in enumerate(shapes):
        pr, pc = os.path.join(d, f"ref.out.{i}.bin"), os.path.join(d, f"cpp.out.{i}.bin")
        if not os.path.exists(pc):
            return fam, "RUN_FAIL", (_last_error(x.stderr) or f"러너가 out_{i} 를 안 냈다")[:70]
        a = np.fromfile(pr, dtype="float32")
        c = np.fromfile(pc, dtype="float32")
        if a.size != c.size:
            return fam, "SHAPE_MISMATCH", f"out_{i}: torch {a.size} vs 러너 {c.size}"
        # ⚠️ **축 순서가 다르다.** torch 는 CHW, 러너(ggml)는 HWC 로 쓴다. 안 맞추면 값이
        #    아니라 배치가 어긋나 rel L1 이 1.4~2.0(= 무관한 두 텐서)으로 나온다.
        if len(sh) == 3:
            ch, h, w = sh
            c = c.reshape(h, w, ch).transpose(2, 0, 1).reshape(-1)
        den = max(float(np.abs(a).sum()), 1e-9)
        worst_l1 = max(worst_l1, float(np.abs(a - c).sum()) / den)
        worst_l2 = max(worst_l2, float(np.linalg.norm(a - c))
                       / max(float(np.linalg.norm(a)), 1e-9))
    ok = worst_l1 < L1_TOL and worst_l2 < L2_TOL
    note = f"L1 {worst_l1:.2e} · L2 {worst_l2:.2e} · {W}x{H} · 출력 {len(shapes)}"
    # ⚠️ unhandled op 이 있으면 **통과해도 통과가 아니다** — 그 연산이 사라진 채 우연히
    #    수치가 맞은 것일 수 있다. 판정 옆에 반드시 남긴다.
    if n_unhandled:
        note += f" · ⚠ unhandled op {n_unhandled}"
    return fam, "PASS" if ok else "FAIL", note


def fetch(sel):
    import urllib.request
    os.makedirs(CKPT, exist_ok=True)
    for fam, _cfg, w in sel:
        if not w:
            continue
        dst = os.path.join(CKPT, os.path.basename(w))
        if os.path.exists(dst):
            continue
        print(f"  받는 중 {fam}: {os.path.basename(w)}", flush=True)
        try:
            urllib.request.urlretrieve(w, dst + ".part")
            os.replace(dst + ".part", dst)
        except Exception as e:
            print(f"  실패 {fam}: {type(e).__name__}: {e}", flush=True)


def main():
    ap = argparse.ArgumentParser(prog="verify_seg")
    ap.add_argument("families", nargs="*")
    ap.add_argument("--list", action="store_true", help="계열 목록만 찍는다")
    ap.add_argument("--fetch", action="store_true", help="없는 체크포인트를 받는다")
    ap.add_argument("--size", type=int, default=None)
    ap.add_argument("--workdir", default=None)
    a = ap.parse_args()

    global SZ, WORKDIR
    SZ = a.size or 0                    # 0 = 계열별 crop (기본)
    if a.workdir:
        WORKDIR = a.workdir

    all_fams = families()
    sel = [f for f in all_fams if not a.families or f[0] in a.families]
    if a.fetch:
        fetch(sel)
    if a.list:
        for fam, cfg, w in all_fams:
            has = os.path.exists(os.path.join(CKPT, os.path.basename(w))) if w else False
            print(f"{'O' if has else '-'} {fam:22s} {os.path.basename(cfg)}")
        print(f"\n계열 {len(all_fams)} · 체크포인트 보유 "
              f"{sum(1 for _f, _c, w in all_fams if w and os.path.exists(os.path.join(CKPT, os.path.basename(w))))}")
        return

    print(f"mmseg={MM}  g2c={P}  "
          f"size={'계열별 crop' if not SZ else f'{SZ}² (--size 덮어쓰기)'}  "
          f"workdir={WORKDIR}  tol=L1{L1_TOL}/L2{L2_TOL}")
    print(f"{'계열':<24}{'판정':<14}비고")
    print("-" * 86)
    rows = []
    for i, (fam, cfg, w) in enumerate(sel, 1):
        fam, verdict, note = verify(fam, cfg, w)
        mark = "O" if verdict == "PASS" else ("X" if verdict == "FAIL" else "-")
        print(f"[{i:3d}/{len(sel)}] {fam:<20} {mark} {verdict:<14} {note}")
        rows.append((fam, verdict))
    n_pass = sum(1 for _f, v in rows if v == "PASS")
    print(f"\nPASS {n_pass}/{len(rows)}")


if __name__ == "__main__":
    main()
