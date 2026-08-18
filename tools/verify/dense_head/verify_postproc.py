"""verify_postproc.py — **디코드 이후**까지 계열별로 잰다.

`verify_heads.py` 는 디코드 직전에서 끊는다(NMS 를 거치면 어느 텐서가 틀렸는지 못 짚기
때문). 그래서 그 숫자에는 앵커 생성 · delta→box · 임계값 · NMS 가 들어가지 않는다.
이 스크립트가 그 뒷단만 본다:

    mmdet 자신의 `predict_by_feat`  vs  `run_mmdet` 이 낸 최종 박스

**후처리는 계열이 아니라 head 종류별로 공유된다** — `postproc.cpp` 의 한 함수를 여러
계열이 함께 탄다. 그래서 종류를 덮는 것이 계열을 덮는 것보다 의미가 크다.

⚠️ **양쪽에 같은 픽셀을 준다.** 각자 리사이즈하게 두면 백엔드가 아니라 리사이즈 구현을
재게 된다. 원본이 정사각이 아니면 한 번만 줄여 무손실로 저장해서 둘 다에 준다.

⚠️ **채널 순서를 맞춘다.** 파라미터 헤더의 `to_rgb` 가 러너의 규약이고, mmdet 쪽
`data_preprocessor` 와 반대일 수 있다. 처음 재면서 이걸 뒤집어 점수가 0.836 → 0.757 로
어긋났다 — 백엔드 문제로 보이지만 하네스 버그였다.

사용:
    python verify_postproc.py <gen_dir> <config.py> <ckpt.pth> <image> [size]

`gen_dir` 은 `build_mmdet_cpp.sh` 가 `run_mmdet` 을 만들어 둔 디렉토리다.
"""
import os
import subprocess
import sys

import numpy as np

# mmpretrain 의 blip 이 이 조합에서 import 시 죽는다 — mmdet 로드에 필요 없으므로 막는다.
import types
for _n in ("mmpretrain.models.multimodal.blip",
           "mmpretrain.models.multimodal.blip.language_model"):
    _m = types.ModuleType(_n)
    _m.__path__ = []
    sys.modules[_n] = _m

import torch                                              # noqa: E402
from PIL import Image                                     # noqa: E402


def cpp_boxes(gen_dir, image, size, thr):
    """run_mmdet 을 돌려 디코드된 박스를 받는다(.bin = 박스당 6값)."""
    out = os.path.join(gen_dir, "_postproc_check.bin")
    exe = os.path.join(gen_dir, "run_mmdet")
    if not os.path.exists(exe):
        sys.exit(f"run_mmdet 이 없다: {exe}  (build_mmdet_cpp.sh 로 먼저 만든다)")
    r = subprocess.run([exe, _one_gguf(gen_dir), image, out, str(size)],
                       capture_output=True, text=True)
    if not os.path.exists(out):
        sys.exit("run_mmdet 이 박스를 안 냈다:\n" + (r.stderr or r.stdout)[-400:])
    d = np.fromfile(out, dtype=np.float32).reshape(-1, 6)
    return d[d[:, 4] >= thr]


def _one_gguf(gen_dir):
    g = [f for f in os.listdir(gen_dir) if f.endswith(".gguf")]
    if len(g) != 1:
        sys.exit(f"gguf 를 하나로 못 좁혔다: {g}")
    return os.path.join(gen_dir, g[0])


def mmdet_boxes(cfg, ckpt, image, size, thr, to_rgb):
    """mmdet 자신의 predict_by_feat — 앵커·디코드·NMS 의 정본."""
    from mmdet.apis import init_detector

    # mmdet v3 체크포인트는 학습 메타(HistoryBuffer)를 함께 담고 있어 torch 2.6 의
    # weights_only=True 기본값에서 로드가 거부된다(rtmdet 이 그랬다). 프론트엔드가
    # 이미 쓰는 우회를 그대로 쓴다 — 필요한 클래스만 이름으로 허용한다.
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "..", "frontend", "mmdet"))
    try:
        import mmdet_wrap
        mmdet_wrap.allow_mmengine_checkpoint_globals()
    except Exception as e:                       # 실패는 조용히 넘기지 않는다
        print(f"  ⚠️ 체크포인트 허용 목록 설정 실패: {type(e).__name__}: {e}")

    det = init_detector(cfg, ckpt, device="cpu")
    det.eval()
    dp = det.data_preprocessor
    # ⚠️ **정규화를 안 하는 계열이 있다** — YOLOX 는 `mean`/`std` 자체를 안 갖는다.
    #    그때 파라미터 헤더도 mean 0 / std 1 을 싣는다(둘이 같아야 같은 픽셀이 된다).
    mean = dp.mean.view(3).numpy() if hasattr(dp, "mean") else np.zeros(3, np.float32)
    std = dp.std.view(3).numpy() if hasattr(dp, "std") else np.ones(3, np.float32)

    im = np.asarray(Image.open(image).convert("RGB").resize((size, size), Image.BILINEAR),
                    dtype=np.float32)
    x = im[:, :, ::-1] if to_rgb else im       # 러너 규약에 맞춘다
    # ⚠️ **float32 로 못 박는다.** mean/std 가 float64 로 오는 계열이 있어(YOLOv3)
    #    나눗셈에서 배열이 double 로 승격되고, conv2d 가
    #    `expected scalar type Double but found Float` 로 죽는다. 계열 탓처럼 보이지만
    #    전처리 dtype 문제다.
    x = ((np.ascontiguousarray(x) - mean) / std).astype(np.float32)
    t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)

    meta = {"img_shape": (size, size), "ori_shape": (size, size),
            "scale_factor": (1.0, 1.0), "batch_input_shape": (size, size)}

    with torch.no_grad():
        try:
            outs = det.bbox_head(det.extract_feat(t))
            res = det.bbox_head.predict_by_feat(*outs, batch_img_metas=[meta],
                                                rescale=False)[0]
        except TypeError:
            # DETR 계열은 transformer 가 **head 가 아니라 detector** 에 달려 있어
            # `bbox_head(feats)` 로는 못 부른다(mmdet_wrap.py:151-157). 그런 계열은
            # detector 의 predict 를 그대로 탄다 — 그게 이 계열의 디코드 정본이다.
            from mmdet.structures import DetDataSample
            ds = DetDataSample()
            ds.set_metainfo(meta)
            res = det.predict(t, [ds], rescale=False)[0].pred_instances

    keep = res.scores.numpy() >= thr
    return np.concatenate([res.bboxes.numpy()[keep],
                           res.scores.numpy()[keep, None],
                           res.labels.numpy()[keep, None].astype(np.float32)], 1)


def match(ref, got):
    """ref 각 박스에 가장 가까운 got 박스를 짝지어 최대 오차를 낸다.

    정렬해서 맞추면 안 된다 — 한 건만 어긋나도 뒤가 통째로 밀린다.

    ⚠️ **개수는 따로 본다.** 이 함수는 `ref → got` 만 걸으므로 C++ 이 **더 낸** 박스나
    **못 낸** 박스를 못 본다. 그것만 보고 통과시키면, 임계값이 어긋나 집합이 달라진
    경우가 "짝지은 것들은 잘 맞음" 으로 조용히 통과한다 — 호출부가 개수를 먼저 비교한다.
    """
    if len(ref) == 0 or len(got) == 0:
        return None

    # **라벨이 같은 쌍만** 후보로 둔다. 박스 거리만 보면, 한 물체에 두 클래스가 겹쳐 나온
    # 경우(DETR 이 흔하다) 두 ref 가 같은 got 에 붙어 "라벨 불일치" 로 잘못 보고된다 —
    # 실제로 DINO 에서 그렇게 오탐이 났다.
    #
    # ⚠️ **ref 순서대로 가까운 것을 집으면 안 된다.** 앞쪽 ref 가 뒤쪽 ref 의 짝을 먼저
    #    가져가면 그 뒤가 통째로 밀려, 사실은 0.1px 로 맞는 집합이 수백 px 로 보고된다
    #    (RPN 제안 1000개처럼 라벨이 하나뿐이고 개수가 한 개 어긋날 때 그랬다).
    #    전역으로 **가까운 쌍부터** 1:1 로 확정한다.
    pairs = []
    for i, r in enumerate(ref):
        same = np.flatnonzero(got[:, 5] == r[5])
        pool = same if len(same) else np.arange(len(got))
        d = np.abs(got[pool][:, :4] - r[:4]).max(1)
        for j, dist in zip(pool, d):
            pairs.append((float(dist), i, int(j)))
    pairs.sort()

    take_r, take_g = {}, set()
    for dist, i, j in pairs:
        if i in take_r or j in take_g:
            continue
        take_r[i] = (j, dist)
        take_g.add(j)

    rows = []
    for i, r in enumerate(ref):
        if i not in take_r:              # 짝이 없는 ref(개수차) — 호출부가 개수로 잡는다
            continue
        j, dist = take_r[i]
        rows.append((r, got[j], dist))
    return rows


def main():
    if len(sys.argv) < 5:
        print(__doc__)
        return 2
    gen, cfg, ckpt, image = sys.argv[1:5]
    size = int(sys.argv[5]) if len(sys.argv) > 5 else 512
    thr = 0.30

    to_rgb = False
    hdr = [f for f in os.listdir(os.path.dirname(cfg) or ".") if f.endswith(".postproc.h")]
    for d in (gen, os.path.dirname(gen)):
        for f in os.listdir(d):
            if f.endswith(".postproc.h"):
                txt = open(os.path.join(d, f), encoding="utf-8").read()
                to_rgb = "c.to_rgb = true" in txt
    print(f"to_rgb={to_rgb} (파라미터 헤더 기준) · thr={thr} · size={size}")

    got = cpp_boxes(gen, image, size, thr)
    ref = mmdet_boxes(cfg, ckpt, image, size, thr, to_rgb)
    print(f"\nmmdet {len(ref)}건 · run_mmdet {len(got)}건")

    rows = match(ref, got)
    if rows is None:
        print("  한쪽이 비어 비교 불가")
        return 1
    print("  라벨   mmdet 박스/점수                    C++ 박스/점수                 박스Δ  점수Δ")
    worst_b = worst_s = 0.0
    bad_label = 0
    for r, g, db in rows:
        ds = abs(r[4] - g[4])
        worst_b, worst_s = max(worst_b, db), max(worst_s, ds)
        bad_label += int(r[5] != g[5])
        print(f"  {int(r[5]):4d}   [{r[0]:6.1f},{r[1]:6.1f},{r[2]:6.1f},{r[3]:6.1f}] {r[4]:.3f}"
              f"   [{g[0]:6.1f},{g[1]:6.1f},{g[2]:6.1f},{g[3]:6.1f}] {g[4]:.3f}"
              f"  {db:6.2f}px {ds:6.3f}")
    n_gap = abs(len(ref) - len(got))
    print(f"\n최대: 박스 {worst_b:.2f}px · 점수 {worst_s:.3f} · 라벨 불일치 {bad_label}건"
          f" · 개수차 {n_gap}건")
    # 박스가 수백 개인 계열(RPN 제안 186개)은 **최대값만 보면 오해한다** — 임계값 언저리에
    # 몰린 후보 한둘이 집합에서 밀리면 그 자리만 크게 벌어지고, 나머지는 0.1px 로 맞는다.
    # 분포를 같이 낸다: 어디가 어긋났는지가 아니라 **몇 개가** 어긋났는지가 보인다.
    if len(rows) >= 20:
        db = np.array([d for _, _, d in rows])
        print(f"  분포: 중앙값 {np.median(db):.2f}px · 95%tile {np.percentile(db, 95):.2f}px"
              f" · 1px 이내 {int((db < 1.0).sum())}/{len(db)}건")
    if n_gap:
        print("  ⚠️ 개수가 다르다 — 임계값(score_thr/nms_thr/max_per_img)이 mmdet 과 어긋났을 때"
              " 나는 증상이다. 짝지은 것만 보면 조용히 통과한다.")
        # ⚠️ **그런데 이 임계값은 우리가 건 것이다.** 양쪽 다 mmdet 의 score_thr(보통 0.05)로
        #    많은 박스를 내고, 여기서 보기 좋으라고 `thr`(0.30)로 한 번 더 자른다. 그래서
        #    점수가 그 언저리인 박스 하나가 **한쪽에서만 살아남아** 개수차로 잡힌다.
        #    fp16 가중치의 점수 오차(실측 0.006~0.025)면 충분히 넘나든다 —
        #    free_anchor 가 그랬다(mmdet 0.2954 vs C++ 0.301). 디코드 버그가 아니다.
        #    그래서 **경계에 몰린 게 몇 개인지 같이 낸다** — 안 내면 다음 사람이 여기를 판다.
        band = 0.05
        near_ref = int(((ref[:, 4] >= thr) & (ref[:, 4] < thr + band)).sum())
        near_got = int(((got[:, 4] >= thr) & (got[:, 4] < thr + band)).sum())
        print(f"     임계값 {thr:.2f}~{thr + band:.2f} 구간: mmdet {near_ref}건 · C++ {near_got}건"
              f"{'  ← 경계 아티팩트일 수 있다(디코드 아님)' if (near_ref or near_got) else ''}")
    return 0 if (worst_b < 2.0 and worst_s < 0.05 and bad_label == 0 and n_gap == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
