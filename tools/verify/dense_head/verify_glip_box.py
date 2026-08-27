#!/usr/bin/env python3
"""GLIP 의 **박스**를 검증한다 — 텐서가 아니라 최종 검출 결과로.

두 단계로 나눠 잰다. **한 번에 재면 틀렸을 때 디코드 탓인지 그래프 탓인지 못 가른다.**

    ① 디코드만    : torch 15텐서 → 우리 디코드   vs   torch 15텐서 → mmdet 디코드
    ② 전체        : C++  15텐서 → 우리 디코드   vs   torch 15텐서 → mmdet 디코드

①이 0 이면 디코드 규칙(`glip_decode.py`)이 mmdet 과 같다는 뜻이고, 그때 ②의 오차는
전부 그래프(fp16·커널) 몫이다.

    python verify_glip_box.py [--caption "person. bicycle. car. dog."] [--size 512]

⚠️ **positive_map 은 호스트가 만든다.** 토크나이저가 문구를 토큰 인덱스로 나눈 결과라
   C++ 토크나이저가 없어도 여기까지 온다 — 그건 CLI 를 만들 때 할 일이다.
"""
import argparse
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
P = os.path.abspath(os.path.join(V, ".."))
FE = V + "/tools/frontend/mmdet"
sys.path.insert(0, HERE)
sys.path.insert(0, FE)

MM = os.path.expanduser("~/mmbuild/mmdetection")
CFG = MM + "/configs/glip/glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365.py"
CK = MM + "/checkpoints/glip_tiny_a_mmdet-b3654169.pth"

# 판정 기준. 박스는 픽셀, 점수는 절대차.  저장소 관례(박스 하네스)를 따른다.
BOX_TOL_PX = 2.0
SCORE_TOL = 0.05


def _match(a_boxes, a_scores, a_labels, b_boxes, b_scores, b_labels):
    """두 검출 집합을 점수 내림차순으로 짝지어 최대 오차를 낸다.

    NMS 를 거친 뒤라 **순서가 곧 짝**이다(같은 입력·같은 임계값이면 같은 순서가 나온다).
    개수가 다르면 그 자체가 결함이므로 따로 보고한다.
    """
    n = min(len(a_boxes), len(b_boxes))
    if n == 0:
        return dict(count_a=len(a_boxes), count_b=len(b_boxes),
                    box_px=float("nan"), score=float("nan"), label_bad=0)
    box_px = float(np.abs(a_boxes[:n] - b_boxes[:n]).max())
    score = float(np.abs(a_scores[:n] - b_scores[:n]).max())
    label_bad = int((a_labels[:n] != b_labels[:n]).sum())
    return dict(count_a=len(a_boxes), count_b=len(b_boxes),
                box_px=box_px, score=score, label_bad=label_bad)


def main():
    ap = argparse.ArgumentParser(prog="verify_glip_box")
    ap.add_argument("--caption", default="person. bicycle. car. dog.")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--workdir", default="/tmp/visp-glip-box")
    ap.add_argument("--skip-cpp", action="store_true", help="①(디코드만) 만 잰다")
    a = ap.parse_args()

    import torch
    import mmdet_wrap
    import glip_decode

    mmdet_wrap.trace_friendly_ops()
    mmdet_wrap.allow_mmengine_checkpoint_globals()
    from mmdet.apis import init_detector
    det = init_detector(CFG, CK, device="cpu")
    det.eval()

    tok = det.language_model.tokenizer
    if not hasattr(tok, "batch_encode_plus"):
        tok.batch_encode_plus = tok.__call__     # transformers 5.x 에서 삭제된 4.x API

    # 캡션 → 토큰 + positive_map. **mmdet 자신의 함수로 만든다** — 규칙을 다시 짜면 틀린다.
    tokenized, caption_string, tokens_positive, entities = \
        det.get_tokens_and_prompts(a.caption, True)
    # ⚠️ `get_positive_map` 은 **튜플**을 돌려준다 —
    #    `(label→token dict, positive_map 텐서)`. 앞엣것이 이미 우리가 쓸 dict 다
    #    (`plus=1` 이 안에서 걸려 라벨이 1부터다). 다시 변환하면 안 된다.
    pos_map, _positive_map = det.get_positive_map(tokenized, tokens_positive)
    n_phrase = len(entities)
    print(f"[caption] {caption_string!r}")
    print(f"[phrases] {entities}  → positive_map {len(pos_map)}문구")

    with torch.no_grad():
        text_dict = det.language_model([caption_string])
    emb = text_dict["embedded"].contiguous()

    SZ = a.size
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, 3, SZ, SZ, generator=g)
    with torch.no_grad():
        feats = det.extract_feat(x)
        cls_logits, bbox_preds, centerness = det.bbox_head(feats, {"embedded": emb})

    # ── ① 디코드만 ────────────────────────────────────────────────────────────
    ours = glip_decode.decode(
        [t[0].numpy() for t in cls_logits],
        [t[0].numpy() for t in bbox_preds],
        [t[0].numpy() for t in centerness],
        {int(k): list(v) for k, v in pos_map.items()}, n_phrase, (SZ, SZ))

    from mmengine.structures import InstanceData          # noqa: F401 (mmdet 내부에서 쓴다)
    ref = det.bbox_head.predict_by_feat(
        cls_logits, bbox_preds, centerness,
        batch_img_metas=[{"img_shape": (SZ, SZ), "scale_factor": (1.0, 1.0),
                          "ori_shape": (SZ, SZ)}],
        batch_token_positive_maps=[pos_map], rescale=False)[0]
    ref_b = ref.bboxes.numpy()
    ref_s = ref.scores.numpy()
    ref_l = ref.labels.numpy()

    m1 = _match(ours[0], ours[1], ours[2], ref_b, ref_s, ref_l)
    print(f"\n① 디코드만  개수 {m1['count_a']} vs {m1['count_b']} · "
          f"박스 {m1['box_px']:.4f}px · 점수 {m1['score']:.5f} · 라벨불일치 {m1['label_bad']}")

    if a.skip_cpp:
        return

    # ── ② 전체 (C++ 텐서 + 우리 디코드) ───────────────────────────────────────
    d = a.workdir
    gen = os.path.join(d, "out")
    run_dump = os.path.join(gen, "run_dump")
    gguf = os.path.join(gen, "Glip.gguf")
    if not (os.path.exists(run_dump) and os.path.exists(gguf)):
        print(f"\n② 건너뜀 — 컴파일 산출물이 없다: {gen}\n"
              f"   (`verify_heads.py glip` 로 먼저 굽거나 --skip-cpp 를 준다)")
        return
    np.ascontiguousarray(x[0].numpy().transpose(1, 2, 0)).tofile(os.path.join(d, "in.bin"))
    subprocess.run([run_dump, gguf, os.path.join(d, "in.bin"), os.path.join(d, "cpp"),
                    str(SZ)], cwd=d, capture_output=True, text=True)

    # 러너 덤프 순서 = 그래프 출력 순서 = cls×5, bbox×5, centerness×5
    cpp_cls, cpp_box, cpp_ctr = [], [], []
    for i in range(5):
        s = cls_logits[i][0].shape                       # (HW, n_tok)
        cpp_cls.append(np.fromfile(os.path.join(d, f"cpp.out.{i}.bin"),
                                   dtype="float32").reshape(s))
    for i in range(5):
        c, h, w = bbox_preds[i][0].shape
        # 러너는 cwhn(HWC)로 쓴다 — torch CHW 로 되돌린다.
        cpp_box.append(np.fromfile(os.path.join(d, f"cpp.out.{5 + i}.bin"),
                                   dtype="float32").reshape(h, w, c).transpose(2, 0, 1))
    for i in range(5):
        c, h, w = centerness[i][0].shape
        cpp_ctr.append(np.fromfile(os.path.join(d, f"cpp.out.{10 + i}.bin"),
                                   dtype="float32").reshape(h, w, c).transpose(2, 0, 1))

    full = glip_decode.decode(cpp_cls, cpp_box, cpp_ctr,
                              {int(k): list(v) for k, v in pos_map.items()},
                              n_phrase, (SZ, SZ))
    m2 = _match(full[0], full[1], full[2], ref_b, ref_s, ref_l)
    print(f"② 전체      개수 {m2['count_a']} vs {m2['count_b']} · "
          f"박스 {m2['box_px']:.4f}px · 점수 {m2['score']:.5f} · 라벨불일치 {m2['label_bad']}")

    ok = (m2["count_a"] == m2["count_b"] and m2["label_bad"] == 0
          and m2["box_px"] < BOX_TOL_PX and m2["score"] < SCORE_TOL)
    print(f"\n{'PASS' if ok else 'FAIL'}  (박스 {BOX_TOL_PX}px · 점수 {SCORE_TOL} · 라벨 0 · 개수차 0)")


if __name__ == "__main__":
    main()
