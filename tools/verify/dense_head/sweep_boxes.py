#!/usr/bin/env python3
"""one-stage 계열의 **박스**를 전수로 잰다 — `verify_heads.py` 가 만든 gen 디렉토리 위에서.

    python sweep_boxes.py                      # verify_heads 가 남긴 workdir 전부
    python sweep_boxes.py --workdir /tmp/x     # 다른 곳
    python sweep_boxes.py retinanet fcos       # 골라서

왜 필요한가
----------
one-stage 는 축이 둘이다. `verify_heads.py` 는 **디코드 전 텐서**(상대 L1/L2)에서 끊고,
`verify_postproc.py` 가 그 뒤(앵커·디코드·NMS)를 본다. 그런데 후자를 계열마다 손으로
불러야 해서 **아무도 전수로 안 돌렸다** — 그래서 "박스까지 되는 계열이 몇이냐"에
답이 없었고, 문서는 표를 손으로 옮기다 낡았다.

⚠️ **`verify_heads.py` 를 먼저 돌려야 한다.** 이 스크립트는 그것이 남긴
   `<workdir>/<계열>/out/run_mmdet` 과 `bb.postproc.h` 를 쓴다. 없으면 그 계열은
   `NO_GEN` 으로 남는다 — 조용히 건너뛰지 않는다(건너뛰면 "전수"가 거짓말이 된다).

결과는 `results_boxes.json` 으로 남긴다. 표는 `tools/verify/make_status_table.py` 가
그 파일에서 **생성**한다. 손으로 옮기지 마라.
"""
import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mmdet_families as MF                                    # noqa: E402

MM = os.path.expanduser(os.environ.get("MMDET", "~/mmbuild/mmdetection"))
CFGS, CKPTS = os.path.join(MM, "configs"), os.path.join(MM, "checkpoints")
V = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DEFAULT_IMAGE = os.path.join(V, "tests", "input", "cat-and-hat.jpg")
PY = sys.executable

# `verify_postproc.py` 의 마지막 줄에서 뽑는다. 형식이 바뀌면 여기서 티가 난다.
SUMMARY = re.compile(r"최대: 박스 ([\d.]+)px · 점수 ([\d.]+) · 라벨 불일치 (\d+)건 · 개수차 (\d+)건")


def one(fam, workdir, image, size):
    d = os.path.join(workdir, fam)
    gen = os.path.join(d, "out")
    if not os.path.exists(os.path.join(gen, "run_mmdet")):
        return fam, "NO_GEN", "verify_heads.py 를 먼저 돌려야 한다"

    try:
        cfg, ckpt_name = MF.resolve_pair(CFGS, fam)
    except Exception as e:                                     # 짝을 못 고르면 그대로 말한다
        return fam, "PAIR_FAIL", f"{type(e).__name__}: {e}"[:110]
    ckpt = os.path.join(CKPTS, ckpt_name)
    if not os.path.exists(ckpt):
        return fam, "CKPT_NONE", os.path.basename(ckpt)

    p = subprocess.run([PY, os.path.join(HERE, "verify_postproc.py"),
                        gen, cfg, ckpt, image, str(size)],
                       capture_output=True, text=True, cwd=d)
    out = p.stdout or ""
    m = SUMMARY.search(out)
    if not m:
        tail = (p.stderr or out).strip().splitlines()
        return fam, "RUN_FAIL", (tail[-1] if tail else "출력 없음")[:110]

    px, score, label, count = float(m.group(1)), float(m.group(2)), int(m.group(3)), int(m.group(4))
    note = f"박스 {px:.2f}px · 점수 {score:.4f} · 라벨 {label} · 개수차 {count}"
    ok = px < 2.0 and score < 0.05 and label == 0 and count == 0
    return fam, ("PASS" if ok else "FAIL"), note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("families", nargs="*")
    ap.add_argument("--workdir", default="/tmp/visp-verify-heads")
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--size", type=int, default=512)
    a = ap.parse_args()

    fams = a.families or sorted(
        f for f in os.listdir(a.workdir)
        if os.path.isdir(os.path.join(a.workdir, f)))
    if not fams:
        print(f"{a.workdir} 에 계열 폴더가 없다 — verify_heads.py 를 먼저 돌려라.")
        return 2

    print(f"workdir={a.workdir} · size={a.size} · {len(fams)}계열")
    print("판정: 박스<2.0px · 점수<0.05 · 라벨 0 · 개수차 0\n")
    rows = []
    for i, fam in enumerate(fams, 1):
        row = one(fam, a.workdir, a.image, a.size)
        rows.append([*row, 0.0])
        print(f"[{i:3d}/{len(fams)}] {row[0]:<20} {row[1]:<12} {row[2]}", flush=True)

    out = os.path.join(a.workdir, "results_boxes.json")
    with open(out, "w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    n_pass = sum(1 for r in rows if r[1] == "PASS")
    print(f"\nPASS {n_pass}/{len(rows)}  →  {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
