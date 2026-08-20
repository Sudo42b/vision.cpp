#!/usr/bin/env python3
"""fetch_checkpoints.py — 계열별 대표 config 의 학습 체크포인트를 받는다.

왜 이게 필요한가
---------------
검증은 **학습된 가중치로만** 의미가 있다. 랜덤 초기화는 γ=1·β=0 같은 항등 초기값이
빠진 연산을 덮어버려, 오늘 잡은 부류의 버그(GroupNorm affine·mmcv.Scale 누락)를
**정의상** 못 잡는다. 그래서 체크포인트가 없으면 `verify_heads.py` 는 아예 안 돌린다.

그런데 어떤 체크포인트가 어느 config 짝인지를 **손으로 적으면** 목록에 없는 계열이
"존재하지 않는 것"처럼 보인다(실제로 `pisa`·`rpn` 이 그렇게 몇 주 동안 안 보였다).
mmdet 은 계열마다 `metafile.yml` 에 `Config → Weights` 를 적어 둔다 — 그걸 읽는다.

사용:
    python fetch_checkpoints.py              # 100계열 전부(이미 있으면 건너뜀)
    python fetch_checkpoints.py --dry-run    # 무엇을 받을지만 출력
    python fetch_checkpoints.py atss dino    # 일부만
"""
import glob
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(line_buffering=True)
import vconfig                                              # noqa: E402

CFG, ARGS = vconfig.load()
DRY = "--dry-run" in ARGS
ARGS = [a for a in ARGS if not a.startswith("-")]
MM, CKPT = CFG.configs, CFG.ckpt

import mmdet_families                                       # noqa: E402

SKIP_DIRS = mmdet_families.SKIP_DIRS


def one(fam):
    # ⚠️ `verify_heads.py` 와 **같은 함수**로 고른다 — 대표 config 선택이 어긋나면
    #    받은 가중치와 컴파일하는 config 가 달라져, 로드는 되는데 일부가 랜덤으로 남는다.
    _, name, url = mmdet_families.resolve(MM, fam)
    if not url:
        return fam, "NO_METAFILE", "-"
    dst = os.path.join(CKPT, name)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return fam, "HAVE", name
    if DRY:
        return fam, "WOULD_GET", name
    try:
        tmp = dst + ".part"
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, dst)
        return fam, "GOT", f"{name} ({os.path.getsize(dst)/1e6:.0f}MB)"
    except Exception as e:
        return fam, "FAIL", f"{type(e).__name__}: {e}"[:70]


def main():
    print(CFG.banner())
    os.makedirs(CKPT, exist_ok=True)
    fams = [d for d in sorted(os.listdir(MM))
            if os.path.isdir(os.path.join(MM, d)) and d not in SKIP_DIRS]
    if ARGS:
        fams = [f for f in fams if f in ARGS]
    rows = []
    # 네트워크 대기라 스레드로 충분하다. 서버에 무리 주지 않게 4개만.
    with ThreadPoolExecutor(4) as ex:
        for i, (fam, st, info) in enumerate(ex.map(one, fams), 1):
            rows.append((fam, st, info))
            print(f"[{i:3d}/{len(fams)}] {fam:22s} {st:12s} {info}")
    print()
    for st in ("HAVE", "GOT", "WOULD_GET", "NO_WEIGHTS", "NO_METAFILE", "FAIL"):
        n = sum(1 for r in rows if r[1] == st)
        if n:
            print(f"{st:12s} {n}")


if __name__ == "__main__":
    main()
