#!/usr/bin/env python3
"""mmpose 계열을 g2c 로 컴파일해 C++(ggml) 출력이 PyTorch 와 같은지 계열별로 잰다.

`tools/verify/seg/verify_seg.py` 와 같은 자로 잰다 — 앵커도 NMS 도 없으니
`backbone → (neck) → head.forward` 를 한 그래프로 굽고 **출력 텐서를 그대로 대조**한다.
디코드(히트맵 argmax·소수점 보정, SimCC 의 1D argmax)는 후처리라 그래프 밖이다.

    python verify_pose.py                # metafile 에 있는 계열 전부
    python verify_pose.py hrnet          # 골라서
    python verify_pose.py --list         # 목록만

⚠️ **입력이 정방이 아니다.** topdown 은 사람 박스를 `codec.input_size`(대개 192x256, WxH)로
   잘라 넣는다. 정방으로 재면 히트맵 크기가 config 와 달라져 **다른 것을 재게 된다.**
"""
import argparse
import glob
import os
import subprocess
import sys

sys.stdout.reconfigure(line_buffering=True)

HERE = os.path.dirname(os.path.abspath(__file__))
V = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
P = os.path.abspath(os.path.join(V, ".."))
FE = V + "/tools/frontend/mmpose"
FE_DET = V + "/tools/frontend/mmdet"
GGUF_PY = V + "/depend/llama/gguf-py"
PY = sys.executable

MM = os.path.expanduser("~/mmbuild/mmpose")
CKPT = MM + "/checkpoints"
WORKDIR = "/tmp/visp-pose-verify"
OPT = "-O1"
L1_TOL = L2_TOL = 0.05


def families():
    """`configs/**/metafile.yml` 에서 (계열, config, weights URL). **손으로 안 적는다.**

    mmpose 의 metafile 은 `configs/<task>/<...>/<algo>/*.yml` 로 깊이가 들쭉날쭉하다 →
    계열 이름은 **metafile 이 있는 디렉터리 이름**으로 잡고, 같은 이름이 여럿이면 첫 것만.
    """
    import yaml
    out, seen = [], set()
    for p in sorted(glob.glob(MM + "/configs/**/*.yml", recursive=True)):
        try:
            d = yaml.safe_load(open(p)) or {}
        except Exception:
            continue
        models = d.get("Models") or []
        if not models:
            continue
        fam = os.path.basename(os.path.dirname(p))
        if fam in seen:
            continue
        m = models[0]
        cfg = m.get("Config") or ""
        if not cfg:
            continue
        cfg = cfg if cfg.startswith("/") else MM + "/" + cfg
        seen.add(fam)
        out.append((fam, cfg, m.get("Weights", "")))
    return out


def run(cmd, cwd=None, env=None, timeout=1800):
    e = dict(os.environ)
    e.setdefault("OMP_NUM_THREADS", "1")
    if env:
        e.update(env)
    try:
        return subprocess.run(cmd, cwd=cwd, env=e, capture_output=True,
                              text=True, timeout=timeout)
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
import mmpose_wrap
H, W = mmpose_wrap.input_size("%(CFG)s")
m, shapes = mmpose_wrap.build("%(CFG)s", "%(CK)s", (H, W))
torch.save(m, "pose.pt")
open("size.txt", "w").write(f"{W} {H}")
# 고정 시드 — 시드 없이 뽑으면 실행마다 숫자가 흔들려 회귀 대조가 흐려진다.
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
print("REF_OK", len(outs), W, H)
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

    for stale in ("pose.pt", "ref.shapes.txt", "size.txt"):
        try:
            os.remove(os.path.join(d, stale))
        except FileNotFoundError:
            pass

    open(os.path.join(d, "ref.py"), "w").write(
        REF % {"FE": FE, "FE_DET": FE_DET, "CFG": cfg, "CK": ck})
    r = run([PY, "ref.py"], d, {"PYTHONPATH": f"{FE}:{FE_DET}"})
    if not os.path.exists(os.path.join(d, "ref.shapes.txt")):
        return fam, "REF_FAIL", _last_error(r.stderr)[:70]
    W, H = [int(v) for v in open(os.path.join(d, "size.txt")).read().split()]

    gen = os.path.join(d, "out")
    r = run([PY, "-c", '''
import sys
sys.argv = ["g2c","--model","pose.pt","--name","Pose","--output","out","--input-shape","1,3,%d,%d"]
from shared.compile.pipeline import main; main()
''' % (H, W)], d, {"PYTHONPATH": f"{FE}:{FE_DET}:{P}:{GGUF_PY}"})
    if not os.path.exists(os.path.join(gen, "Pose.gguf")):
        return fam, "COMPILE_FAIL", _last_error(r.stderr)[:70]
    n_unhandled = open(os.path.join(gen, "Pose.cpp")).read().count("unhandled op")

    inc = os.path.join(gen, "inc", "visp", "arch")
    os.makedirs(inc, exist_ok=True)
    shutil.copy(os.path.join(gen, "Pose.h"), inc)
    b = run(["g++", "-std=c++20", OPT, "-DARCH=Pose",
             '-DVISP_ARCH_HEADER="visp/arch/Pose.h"',
             "-I" + os.path.join(gen, "inc"), "-I" + V + "/src", "-I" + V + "/include",
             "-I" + V + "/depend/llama/ggml/include",
             V + "/tools/verify/backbone/run_dump.cpp", os.path.join(gen, "Pose.cpp"),
             "-L" + V + "/build/lib", "-lvisioncpp", "-lggml", "-lggml-base", "-lggml-cpu",
             "-Wl,-rpath," + V + "/build/lib", "-o", os.path.join(gen, "run_dump")], d)
    if not os.path.exists(os.path.join(gen, "run_dump")):
        return fam, "BUILD_FAIL", _last_error(b.stderr)[:70]

    for f in os.listdir(d):
        if f.startswith("cpp.out.") and f.endswith(".bin"):
            os.remove(os.path.join(d, f))
    x = run([os.path.join(gen, "run_dump"), os.path.join(gen, "Pose.gguf"),
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
        # torch CHW ↔ 러너 HWC. 안 맞추면 값이 아니라 배치가 어긋난다.
        if len(sh) == 3:
            ch, h, w = sh
            c = c.reshape(h, w, ch).transpose(2, 0, 1).reshape(-1)
        den = max(float(np.abs(a).sum()), 1e-9)
        worst_l1 = max(worst_l1, float(np.abs(a - c).sum()) / den)
        worst_l2 = max(worst_l2, float(np.linalg.norm(a - c))
                       / max(float(np.linalg.norm(a)), 1e-9))
    ok = worst_l1 < L1_TOL and worst_l2 < L2_TOL
    note = f"L1 {worst_l1:.2e} · L2 {worst_l2:.2e} · {W}x{H} · 출력 {len(shapes)}"
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
    ap = argparse.ArgumentParser(prog="verify_pose")
    ap.add_argument("families", nargs="*")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--workdir", default=None)
    a = ap.parse_args()
    global WORKDIR
    if a.workdir:
        WORKDIR = a.workdir

    all_fams = families()
    sel = [f for f in all_fams if not a.families or f[0] in a.families]
    if a.list:
        for fam, cfg, w in all_fams:
            has = os.path.exists(os.path.join(CKPT, os.path.basename(w))) if w else False
            print(f"{'O' if has else '-'} {fam:24s} {os.path.basename(cfg)}")
        print(f"\n계열 {len(all_fams)}")
        return
    if a.fetch:
        fetch(sel)

    print(f"mmpose={MM}  g2c={P}  workdir={WORKDIR}  tol=L1{L1_TOL}/L2{L2_TOL}")
    print(f"{'계열':<26}{'판정':<14}비고")
    print("-" * 92)
    rows = []
    for i, (fam, cfg, w) in enumerate(sel, 1):
        fam, verdict, note = verify(fam, cfg, w)
        mark = "O" if verdict == "PASS" else ("X" if verdict == "FAIL" else "-")
        print(f"[{i:3d}/{len(sel)}] {fam:<22} {mark} {verdict:<14} {note}")
        rows.append((fam, verdict))
    print(f"\nPASS {sum(1 for _f, v in rows if v == 'PASS')}/{len(rows)}")


if __name__ == "__main__":
    main()
