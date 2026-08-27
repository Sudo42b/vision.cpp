#!/usr/bin/env python3
"""box_all.py — one-stage **박스 축**을 계열별로 전수로 돌린다.

    python box_all.py                      # 빌드된 계열 전부
    python box_all.py yolact rpn           # 골라서
    python box_all.py --gen-root ~/work/box-full
    python box_all.py --results /tmp/box-onestage.results.json

`verify_postproc.py` 한 계열을 부르는 얇은 드라이버다. **판정 규약도 짝 결정도 여기서
새로 만들지 않는다** — 전부 하네스 것을 그대로 쓴다.

왜 있나
------
이 드라이버가 **없어서 one-stage 전수를 한 번 통째로 재현 못 했다.** 2026-08-25 에
같은 이름의 스크립트로 44계열을 돌린 기록이 위키에 있는데, 그게 세션 로컬(`/tmp`)에만
있었고 같이 사라졌다. 산출물(`<계열>/out/run_mmdet`)은 멀쩡히 남아 있었는데도
**부를 방법이 없었다.**

→ 위키 규칙 그대로다: **파일에 안 적힌 것은 다음 세션에 존재하지 않는다.**
   그래서 `/tmp` 가 아니라 저장소 안에 둔다.

⚠️ **짝은 `used.json` 에서 받는다 — 손으로 고르지 마라.**
   `verify_heads.py` 가 구울 때 실제로 쓴 config·체크포인트를 그 파일에 남긴다.
   래퍼 계열(KD·트래커)은 언랩한 config 를 `cfg.py` 로 덤프해 굽기 때문에, mmdet 원본
   config 를 주면 **「짝이 다르다」 경고가 항상 뜨고** 기준값도 다른 모델에서 나온다.
   `ld`·`strongsort` 가 그래서 「값 못 냄」으로 잘못 적혔다(둘 다 멀쩡했다).
   → wiki `pitfall/짝을-손으로-고르면-남의-가중치를-잰다.md`

⚠️ **빌드는 여기서 안 한다.** `<계열>/out/run_mmdet` 이 없으면 `BUILD_NONE` 으로 적고
   넘어간다 — 굽는 것은 `verify_heads.py` 몫이고, 그쪽이 export→g2c→가중치→빌드까지
   한다. 두 도구가 각자 구우면 **어느 쪽 산출물을 쟀는지** 알 수 없게 된다.

⚠️ **못 잰 것을 실패로 적지 마라.** `BUILD_NONE`·`PAIR_NONE`·`EMPTY` 는 실패가 아니라
   측정 실패다. 표에서 따로 세고, PASS 수와 나란히 놓지 않는다.
"""
import argparse
import concurrent.futures as _fut
import json
import os
import re
import subprocess
import sys

sys.stdout.reconfigure(line_buffering=True)      # 파이프로 보내도 진행이 보이게

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mmdet_families as MF                                      # noqa: E402
import vconfig                                                   # noqa: E402

CFG, ARGS = vconfig.load()
V = os.path.normpath(os.path.join(HERE, "..", "..", ".."))             # vision.cpp/tools
VISP = os.path.normpath(os.path.join(V, ".."))                   # vision.cpp
POSTPROC = os.path.join(HERE, "verify_postproc.py")
DEFAULT_IMAGE = os.path.join(VISP, "tests", "input", "cat-and-hat.jpg")

# 판정 요약 줄. `verify_postproc.py` 가 찍는 형식과 **한 군데서만** 묶여 있다.
RE_WORST = re.compile(
    r"최대: 박스 ([\d.]+)px · 점수 ([\d.]+) · 라벨 불일치 (\d+)건 · 개수차 (\d+)건")
RE_DIST = re.compile(r"분포: 중앙값 ([\d.]+)px · 95%tile ([\d.]+)px · 1px 이내 (\d+)/(\d+)건")
RE_BAND = re.compile(r"임계값 [\d.]+~[\d.]+ 구간: mmdet (\d+)건 · C\+\+ (\d+)건")
RE_COUNT = re.compile(r"mmdet (\d+)건 · run_mmdet (\d+)건")


def pair_for(fam, gen_root):
    """(config, checkpoint) — **구울 때 쓴 짝**이 정본이다.

    `used.json` 이 있으면 그것. 없으면 `mmdet_families.resolve_pair` 로 떨어진다
    (그 경우 래퍼 계열은 경고가 뜰 수 있다 — 산출물이 옛 것이라는 뜻이다).
    """
    u = os.path.join(gen_root, fam, "used.json")
    if os.path.exists(u):
        try:
            d = json.load(open(u, encoding="utf-8"))
            c, k = d.get("config"), d.get("checkpoint")
            if c and k and os.path.exists(c) and os.path.exists(k):
                return c, k
        except Exception:
            pass
    cfg, ckpt = MF.resolve_pair(os.path.join(CFG.mmdet, "configs"), fam)
    if not cfg or not ckpt:
        return None, None
    ck = ckpt if os.path.isabs(ckpt) else os.path.join(CFG.mmdet, "checkpoints", ckpt)
    return (cfg, ck) if os.path.exists(ck) else (cfg, None)


def one(fam, gen_root, size, image, verbose):
    gen = os.path.join(gen_root, fam, "out")
    if not os.path.exists(os.path.join(gen, "run_mmdet")):
        return dict(family=fam, status="BUILD_NONE",
                    note=f"{gen}/run_mmdet 없음 — verify_heads.py 로 먼저 굽는다")
    cfg, ckpt = pair_for(fam, gen_root)
    if not cfg or not ckpt:
        return dict(family=fam, status="PAIR_NONE", note="config·체크포인트를 못 찾았다")

    img = MF.test_image(fam, image)
    r = subprocess.run([sys.executable, POSTPROC, gen, cfg, ckpt, img, str(size)],
                       capture_output=True, text=True,
                       env={**os.environ, "OMP_NUM_THREADS": "1"})
    out = (r.stdout or "") + (r.stderr or "")
    m = RE_WORST.search(out)
    if not m:
        # 판정 줄이 없다 = 비교까지 못 갔다. **왜** 인지를 그대로 옮긴다.
        why = "한쪽이 비어 비교 불가" if "한쪽이 비어" in out else \
              next((l.strip() for l in reversed(out.splitlines())
                    if l.strip() and not l.startswith(" ")), "출력 없음")
        return dict(family=fam, status="EMPTY" if "한쪽이 비어" in out else "RUN_FAIL",
                    note=why[:110], stdout=out if verbose else None)

    box, score, bad_label, gap = float(m[1]), float(m[2]), int(m[3]), int(m[4])
    ok = box < 2.0 and score < 0.05 and bad_label == 0 and gap == 0
    res = dict(family=fam, status="PASS" if ok else "FAIL",
               box=box, score=score, bad_label=bad_label, gap=gap,
               image=os.path.basename(img),
               note=f"박스 {box:.2f}px · 점수 {score:.3f} · 라벨 {bad_label} · 개수차 {gap}")
    if (c := RE_COUNT.search(out)):
        res["n_ref"], res["n_got"] = int(c[1]), int(c[2])
    # ⚠️ **박스가 수백 개인 계열은 최대값만 보면 오해한다**(rpn: 최대 133px 인데 중앙값
    #    0.09px · 182/185건이 1px 이내). 분포가 있으면 반드시 같이 싣는다.
    if (d := RE_DIST.search(out)):
        res["median"], res["p95"] = float(d[1]), float(d[2])
        res["within_1px"], res["n_rows"] = int(d[3]), int(d[4])
        res["note"] += f" · 중앙값 {float(d[1]):.2f}px · 1px 이내 {d[3]}/{d[4]}"
    # 개수차가 컷 근처에서 났는지 — 「임계 경계」와 「진짜 결함」을 가르는 첫 단서다.
    if (b := RE_BAND.search(out)):
        res["band_ref"], res["band_got"] = int(b[1]), int(b[2])
    if "짝이 다르다" in out:
        # 래퍼 계열에서 **거짓으로도** 뜬다(basename 비교라 `cfg.py` 와 늘 어긋난다).
        # 판정을 바꾸지 않고 표시만 한다 — 수치가 크게 틀릴 때만 의심하면 된다.
        res["pair_warn"] = True
        res["note"] += " · ⚠️짝 경고"
    if verbose:
        res["stdout"] = out
    return res


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("families", nargs="*", help="비우면 빌드된 계열 전부")
    ap.add_argument("--gen-root", default=CFG.workdir,
                    help=f"계열별 산출물 루트 (<루트>/<계열>/out). 기본 {CFG.workdir}")
    ap.add_argument("--size", type=int, default=CFG.size)
    ap.add_argument("--image", default=DEFAULT_IMAGE,
                    help="계열별 권장 이미지가 있으면 그쪽이 우선한다")
    ap.add_argument("--results", default=None, help="results.json 을 쓸 경로")
    ap.add_argument("--workers", type=int, default=CFG.workers)
    ap.add_argument("-v", "--verbose", action="store_true", help="계열별 원본 출력을 남긴다")
    a = ap.parse_args(ARGS)

    root = os.path.expanduser(a.gen_root)
    fams = a.families or sorted(
        d for d in os.listdir(root)
        if os.path.exists(os.path.join(root, d, "out", "run_mmdet")))
    if not fams:
        print(f"돌릴 계열이 없다 — {root}/<계열>/out/run_mmdet 이 하나도 없다.\n"
              f"  verify_heads.py 로 먼저 굽는다 (--set paths.workdir={root}).")
        return 2

    print(CFG.banner() if hasattr(CFG, "banner") else "")
    print(f"gen-root={root} · size={a.size} · 판정: 박스<2.0px 점수<0.05 라벨0 개수차0")
    print(f"{len(fams)}계열: {' '.join(fams)}\n")

    rows = []
    with _fut.ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        futs = {ex.submit(one, f, root, a.size, a.image, a.verbose): f for f in fams}
        for i, fu in enumerate(_fut.as_completed(futs), 1):
            r = fu.result()
            rows.append(r)
            print(f"[{i:3d}/{len(fams)}] {r['family']:<22} {r['status']:<12} {r.get('note','')}")

    rows.sort(key=lambda r: r["family"])
    print("\n" + "=" * 78)
    for r in rows:
        print(f"  {r['family']:<22} {r['status']:<12} {r.get('note','')}")

    # ⚠️ **PASS 수만 적지 마라.** 못 잰 것과 실패한 것을 한 칸에 넣으면 다음 사람이
    #    "이 계열이 못 한다" 로 읽는다. 성격별로 따로 센다.
    n = {}
    for r in rows:
        n[r["status"]] = n.get(r["status"], 0) + 1
    measured = sum(v for k, v in n.items() if k in ("PASS", "FAIL"))
    print(f"\nPASS {n.get('PASS', 0)}/{measured} (잰 것 기준)")
    unmeasured = {k: v for k, v in n.items() if k not in ("PASS", "FAIL")}
    if unmeasured:
        print("  못 잼: " + " · ".join(f"{k} {v}" for k, v in sorted(unmeasured.items()))
              + "  ← 실패가 아니다. PASS 수와 나란히 놓지 마라")

    if a.results:
        with open(a.results, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)
        print(f"  → {a.results}")
    return 0 if n.get("FAIL", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
