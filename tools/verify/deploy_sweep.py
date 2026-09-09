#!/usr/bin/env python3
"""배포 경로 스윕 — 사용자가 치는 명령 그대로 계열을 돌려 값을 판정한다.

    python3 tools/verify/deploy_sweep.py seg --limit 5
    python3 tools/verify/deploy_sweep.py pose --limit 5 --skip 5

`run_dump` 하네스와 다른 점이 세 가지다.

1. 입력이 난수 텐서가 아니라 **샘플 이미지**다. 난수는 실제 이미지가 만드는 활성값
   분포를 재현하지 않는다.
2. 실행기가 `vision-cli` 다. 이미지 로드·리사이즈·정규화·등록 경로가 검사 범위에 들어온다.
3. 레퍼런스를 g2c 산출물에서 만들지 않는다. 원본 체크포인트를 torch 로 돌린 값과 댄다.

판정은 상대 L1 으로 한다. cosine 은 스케일 불변이라 값이 배로 틀려도 1.0 이 나온다.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))       # tools/verify
V = os.path.abspath(os.path.join(HERE, "..", ".."))     # vision.cpp
G2C = os.path.abspath(os.path.join(V, ".."))            # g2c 루트
PY = G2C + "/.venv/bin/python"
BUILD = os.environ.get("VISP_BUILD", V + "/build")
GGUF_PY = V + "/depend/llama/gguf-py"
IMAGE = V + "/tests/input/cat-and-hat.jpg"
WORK = os.environ.get("SWEEP_WORK", "/tmp/visp-deploy-sweep")
TOL = 0.05          # 상대 L1 통과선. mmdet 하네스와 같은 값을 쓴다
TIMEOUT = 900

MM = {
    "seg": os.path.expanduser("~/mmbuild/mmsegmentation"),
    "pose": os.path.expanduser("~/mmbuild/mmpose"),
}


def run(cmd, cwd=None, env=None, timeout=TIMEOUT):
    e = dict(os.environ)
    e.setdefault("PYTHONPATH", GGUF_PY)
    if env:
        e.update(env)
    p = subprocess.run(cmd, cwd=cwd, env=e, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def families(kind):
    """metafile 에서 (계열, config, 체크포인트) 를 읽는다. 손목록을 쓰지 않는다."""
    import glob
    import yaml
    root = MM[kind]
    ck_dir = root + "/checkpoints"
    out = []
    for p in sorted(glob.glob(root + "/configs/*/metafile.y*ml")):
        fam = os.path.basename(os.path.dirname(p))
        try:
            models = yaml.safe_load(open(p)).get("Models") or []
        except Exception:
            continue
        if not models:
            continue
        m = models[0]
        cfg = m["Config"]
        cfg = cfg if cfg.startswith("/") else root + "/" + cfg
        w = m.get("Weights", "")
        ck = os.path.join(ck_dir, os.path.basename(w)) if w else ""
        out.append((fam, cfg, ck))
    return out


def one(kind, fam, cfg, ck):
    """계열 하나를 배포 경로로 돌린다. (판정, 상세) 를 돌려준다."""
    d = os.path.join(WORK, kind, fam)
    os.makedirs(d, exist_ok=True)
    name = "Fam" + fam.replace("-", "").replace("_", "").title()[:20]
    arch = name.lower()

    if not ck or not os.path.exists(ck):
        return "CKPT_NONE", "체크포인트 없음"

    # 1) mmpt — config 가 요구하는 크기로 trace 한 .pt 를 만든다
    rc, log = run([PY, "-m", "shared.frontend.mmpt", cfg, "--checkpoint", ck,
                   "--name", name, "--out", d], cwd=G2C)
    if rc != 0:
        return "EXPORT_FAIL", tail(log)
    pt = os.path.join(d, name + ".pt")
    size_txt = os.path.join(d, name + ".size.txt")
    W, H = read_size(size_txt, log)
    if not W:
        return "SIZE_UNKNOWN", tail(log)

    # 2) g2c 변환
    rc, log = run([PY, "-m", "shared.compile.pipeline", "--model", pt, "--name", name,
                   "--input-shape", f"1,3,{H},{W}", "--output", os.path.join(d, "out")],
                  cwd=G2C)
    if rc != 0 or "→ gguf:" not in log:
        return "COMPILE_FAIL", tail(log)
    gguf = os.path.join(d, "out", name + ".gguf")

    # 3) 등록 + 빌드
    #    정규화 값을 넘기지 않으면 install_arch 가 0-1 스케일로 등록한다. mmlab 모델은
    #    ImageNet 통계를 전제하므로 config 의 data_preprocessor 값을 그대로 박는다.
    mean, std = norm_values(cfg)
    rc, log = run([PY, V + "/tools/install_arch.py", os.path.join(d, "out"), "--name", name,
                   "--mean", ",".join(str(v) for v in mean),
                   "--std", ",".join(str(v) for v in std)], cwd=G2C)
    if rc != 0:
        return "INSTALL_FAIL", tail(log)
    rc, log = run(["cmake", "--build", BUILD, "-j", str(os.cpu_count() or 4)])
    if rc != 0:
        return "BUILD_FAIL", tail(log)

    # 4) vision-cli — 사용자가 치는 그 명령
    out_prefix = os.path.join(d, "cli")
    rc, log = run([BUILD + "/bin/vision-cli", arch, "-m", gguf, "-i", IMAGE,
                   "-o", out_prefix, "-b", "cpu"])
    if rc != 0:
        return "CLI_FAIL", tail(log)

    # 5) torch 레퍼런스 — 같은 이미지, config 의 크기와 정규화
    rc, log = run([PY, HERE + "/deploy_reference.py", "--kind", kind, "--config", cfg,
                   "--checkpoint", ck, "--image", IMAGE, "--size", f"{W}x{H}",
                   "--out", os.path.join(d, "ref")], cwd=G2C)
    if rc != 0:
        return "REF_FAIL", tail(log)

    # 6) 판정
    rc, log = run([PY, HERE + "/deploy_compare.py", "--cli", out_prefix,
                   "--ref", os.path.join(d, "ref"), "--tol", str(TOL)])
    verdict = "PASS" if rc == 0 else "VALUE_FAIL"
    return verdict, log.strip().splitlines()[-1] if log.strip() else ""


def norm_values(cfg):
    """config 의 data_preprocessor 에서 mean/std 를 읽는다. 없으면 ImageNet 값."""
    try:
        from mmengine.config import Config
        dp = (Config.fromfile(cfg).get("model") or {}).get("data_preprocessor", {}) or {}
        return list(dp.get("mean") or [123.675, 116.28, 103.53]), \
               list(dp.get("std") or [58.395, 57.12, 57.375])
    except Exception:
        return [123.675, 116.28, 103.53], [58.395, 57.12, 57.375]


def read_size(path, log):
    if os.path.exists(path):
        try:
            w, h = open(path).read().split()
            return int(w), int(h)
        except Exception:
            pass
    import re
    m = re.search(r"Input:\s*1,3,(\d+),(\d+)", log)
    if m:
        return int(m.group(2)), int(m.group(1))
    return None, None


def tail(log, n=3):
    lines = [l for l in log.strip().splitlines() if l.strip()]
    return " / ".join(lines[-n:])[:300]


def main():
    ap = argparse.ArgumentParser(prog="deploy_sweep")
    ap.add_argument("kind", choices=["seg", "pose"])
    ap.add_argument("--limit", type=int, default=5, help="이번에 돌릴 계열 수")
    ap.add_argument("--skip", type=int, default=0, help="앞에서 건너뛸 계열 수")
    ap.add_argument("--only", nargs="*", help="계열 이름을 직접 지정한다")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    fams = families(a.kind)
    if a.list:
        for i, (f, _, ck) in enumerate(fams):
            print(f"{i:3d} {f:24s} {'ckpt' if os.path.exists(ck) else 'NO-CKPT'}")
        return 0
    if a.only:
        fams = [f for f in fams if f[0] in a.only]
    else:
        fams = fams[a.skip:a.skip + a.limit]

    os.makedirs(WORK, exist_ok=True)
    results = []
    for fam, cfg, ck in fams:
        t0 = time.time()
        try:
            verdict, detail = one(a.kind, fam, cfg, ck)
        except subprocess.TimeoutExpired:
            verdict, detail = "TIMEOUT", f"{TIMEOUT}s 초과"
        except Exception as e:
            verdict, detail = "ERROR", str(e)[:200]
        dt = time.time() - t0
        print(f"[{verdict:12s}] {fam:24s} {dt:6.1f}s  {detail}", flush=True)
        results.append({"family": fam, "verdict": verdict, "detail": detail, "sec": round(dt, 1)})

    path = os.path.join(WORK, f"{a.kind}_result.json")
    prev = json.load(open(path)) if os.path.exists(path) else []
    json.dump(prev + results, open(path, "w"), ensure_ascii=False, indent=1)
    ok = sum(1 for r in results if r["verdict"] == "PASS")
    print(f"\n{ok}/{len(results)} PASS  →  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
