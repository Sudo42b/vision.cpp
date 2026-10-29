#!/usr/bin/env python3
"""mmseg 한 계열을 **층별로 갈라** 컴파일한다 — 값이 어디서 벌어지는지 찾는 도구.

`verify_seg.py` 는 최종 출력 하나만 대조해 PASS/FAIL 을 낸다. FAIL 이 났을 때
**어느 층에서 벌어졌는지**는 안 알려준다. 이 도구는 백본 4단계와 head 내부 단계를
**한 그래프의 출력으로 같이 내보내** 단계마다 torch 와 댄다.

    python probe_seg.py fcn --stage backbone   # 백본 4단계
    python probe_seg.py fcn --stage head       # head 입력 + 내부 + cls_seg
    python probe_seg.py fcn --stage final      # 최종 seg_logits (verify_seg 와 같다)

계열·체크포인트·입력 크기는 `verify_seg.py` 와 **같은 곳에서** 온다(metafile · crop_size).
따로 적어 두면 갈린다.

⚠️ **head 를 두 번 부르지 마라.** 중간값을 꺼내려고 `h.convs(x)` 로 펼쳐 부른 뒤
   `h(f)` 도 부르면 같은 서브모듈이 그래프에 **두 번** 들어가 이름이 갈리고
   (`Sequential[decode_head]` vs `FCNHead[decode_head]`) 가중치 바인딩이 어긋난다.
   2026-08-27 에 그것 때문에 rel L1 1.94 라는 **가짜 실패**를 만들었다.
   아래 `_head_stages` 는 head.forward 를 그대로 펼친 것이고, 끝에서 다시 부르지 않는다.

⚠️ **랜덤 초기화로 재지 마라** — `verify_seg.py` 와 같은 이유다. 항등 초기값이
   빠진 연산을 덮는다. 체크포인트가 없으면 그 계열은 돌리지 않는다.

status: 2026-08-27 `~/work/seg/seg_probe.py`(추적 안 되던 중복본)에서 저장소로 옮기며
        512² 고정 → 계열별 crop, 손목록 3계열 → metafile 열거로 바꿨다.
        **그 두 변경 뒤 아직 안 돌려봤다 — 첫 사용자는 결과를 의심하고 보라.**
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
V = os.path.abspath(os.path.join(HERE, "..", "..", ".."))       # vision.cpp
P = os.path.abspath(os.path.join(V, ".."))                      # g2c 루트
sys.path.insert(0, HERE)

import verify_seg as VS                                          # noqa: E402


def _head_stages(h, feats):
    """decode_head 를 펼쳐 단계별 텐서를 모은다. head 는 **한 번만** 탄다."""
    import torch
    xin = feats[h.in_index] if isinstance(h.in_index, int) else \
        torch.cat([feats[i] for i in h.in_index], dim=1)
    outs = [xin]                                   # head 입력 (백본 마지막)
    if hasattr(h, "psp_modules"):                  # PSPHead / PSPNet 계열
        # ⚠️ `PPM.forward` 가 pool → conv → **resize(입력 크기로 업샘플)** 까지 한다.
        #    가지를 따로 부르면 (1,1)·(2,2)… 인 채로 나와 concat 이 깨진다.
        ppm = list(h.psp_modules(xin))
        outs.extend(ppm)
        y = h.bottleneck(torch.cat([xin] + ppm, dim=1))
        outs.append(y)
    elif hasattr(h, "convs"):                      # FCNHead 계열
        y = h.convs(xin)
        outs.append(y)
        if getattr(h, "concat_input", False):
            y = h.conv_cat(torch.cat([xin, y], dim=1))
            outs.append(y)
    else:
        return None                                # 펼치는 법을 모르는 head
    outs.append(h.conv_seg(y))                     # dropout 은 eval 에서 항등
    return outs


REF = r'''
import os, sys, numpy as np, torch
sys.path.insert(0, "%(FE)s"); sys.path.insert(0, "%(FE_DET)s"); sys.path.insert(0, "%(HERE)s")
import mmseg_wrap
from probe_seg import _head_stages
from mmseg.apis import init_model
import torch.nn as nn

H, W = mmseg_wrap.crop_size("%(CFG)s")
mmseg_wrap.trace_friendly_ops()
import mmdet_wrap; mmdet_wrap.allow_mmengine_checkpoint_globals()
seg = init_model("%(CFG)s", "%(CK)s", device="cpu"); seg.eval()


class Probe(nn.Module):
    def __init__(s, seg, stage):
        super().__init__()
        s.backbone = seg.backbone
        s.neck = seg.neck if getattr(seg, "with_neck", False) else None
        s.decode_head = seg.decode_head
        s.stage = stage

    def forward(s, x):
        f = s.backbone(x)
        if s.neck is not None:
            f = s.neck(f)
        if s.stage == "backbone":
            return tuple(f)
        if s.stage == "final":
            o = s.decode_head(f)
            return o if isinstance(o, tuple) else (o,)
        st = _head_stages(s.decode_head, f)
        if st is None:
            raise SystemExit("HEAD_UNKNOWN: 이 head 는 펼치는 법을 모른다 — --stage final 로 재라")
        return tuple(st)


m = Probe(seg, "%(STAGE)s").eval()
torch.save(m, "seg.pt")
open("size.txt", "w").write(f"{W} {H}")
g = torch.Generator().manual_seed(0)             # verify_seg 와 같은 고정 입력
x = torch.randn(1, 3, H, W, generator=g)
with torch.no_grad():
    outs = m(x)
sh = []
for i, t in enumerate(outs):
    a = t[0].detach().numpy()
    np.ascontiguousarray(a).tofile(f"ref.out.{i}.bin")
    sh.append(list(a.shape))
np.ascontiguousarray(x[0].numpy().transpose(1, 2, 0)).tofile("in.bin")
open("ref.shapes.txt", "w").write("\n".join(" ".join(map(str, s)) for s in sh))
print("[ref]", sh, flush=True)
'''


def main():
    ap = argparse.ArgumentParser(prog="probe_seg")
    ap.add_argument("family")
    ap.add_argument("--stage", default="backbone",
                    choices=["backbone", "head", "final"])
    ap.add_argument("--workdir", default="/tmp/visp-seg-probe")
    a = ap.parse_args()

    fams = {f[0]: f for f in VS.families()}
    if a.family not in fams:
        sys.exit(f"모르는 계열: {a.family} (verify_seg.py --list 로 확인)")
    fam, cfg, w = fams[a.family]
    ck = os.path.join(VS.CKPT, os.path.basename(w)) if w else ""
    if not ck or not os.path.exists(ck):
        sys.exit(f"{fam}: 체크포인트가 없다 — 랜덤으로 재면 안 된다 "
                 f"(verify_seg.py --fetch {fam})")

    d = os.path.join(a.workdir, f"{fam}-{a.stage}")
    os.makedirs(d, exist_ok=True)
    src = os.path.join(d, "_ref.py")
    open(src, "w").write(REF % {"FE": VS.FE, "FE_DET": VS.FE_DET, "HERE": HERE,
                                "CFG": cfg, "CK": ck, "STAGE": a.stage})
    print(f"{fam} · stage={a.stage} · workdir={d}")
    import subprocess
    r = subprocess.run([VS.PY, src], cwd=d)
    if r.returncode:
        sys.exit(r.returncode)
    print("\n기준값을 냈다. 컴파일·대조는 verify_seg.py 의 경로를 그대로 쓴다 —\n"
          f"  {d}/seg.pt · ref.out.*.bin · size.txt")


if __name__ == "__main__":
    main()
