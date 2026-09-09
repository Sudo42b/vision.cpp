#!/usr/bin/env python3
"""vision-cli 출력과 torch 레퍼런스를 상대 L1·L2 로 댄다.

    python3 deploy_compare.py --cli <prefix> --ref <prefix> --tol 0.05

cosine 을 쓰지 않는다. 스케일 불변이라 값이 배로 틀려도 1.0 이 나온다.
종료 코드 0 이면 통과, 1 이면 값 불일치, 2 면 비교 자체가 불가능한 상태다.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np


def load(prefix):
    out = []
    for p in sorted(glob.glob(prefix + ".*.bin")):
        out.append((p, np.fromfile(p, dtype=np.float32)))
    return out


def main():
    ap = argparse.ArgumentParser(prog="deploy_compare")
    ap.add_argument("--cli", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--tol", type=float, default=0.05)
    a = ap.parse_args()

    cli, ref = load(a.cli), load(a.ref)
    if not cli:
        print("vision-cli 출력이 없다")
        return 2
    if not ref:
        print("레퍼런스가 없다")
        return 2
    if len(cli) != len(ref):
        print(f"출력 개수가 다르다: cli {len(cli)} vs ref {len(ref)}")
        return 2

    worst_l1 = worst_l2 = 0.0
    lines = []
    for i, ((cp, c), (rp, r)) in enumerate(zip(cli, ref)):
        if c.size != r.size:
            print(f"out_{i} 원소 수가 다르다: cli {c.size} vs ref {r.size}")
            return 2
        c64, r64 = c.astype(np.float64), r.astype(np.float64)
        denom_l1 = np.abs(r64).sum()
        denom_l2 = np.sqrt((r64 ** 2).sum())
        l1 = float(np.abs(c64 - r64).sum() / denom_l1) if denom_l1 else float("inf")
        l2 = float(np.sqrt(((c64 - r64) ** 2).sum()) / denom_l2) if denom_l2 else float("inf")
        worst_l1, worst_l2 = max(worst_l1, l1), max(worst_l2, l2)
        lines.append(f"out_{i} n={c.size} relL1={l1:.4e} relL2={l2:.4e}")

    ok = worst_l1 <= a.tol and worst_l2 <= a.tol
    for l in lines:
        print(l)
    print(f"worst relL1={worst_l1:.4e} relL2={worst_l2:.4e} tol={a.tol} "
          f"{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
