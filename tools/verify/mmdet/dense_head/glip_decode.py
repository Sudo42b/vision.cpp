#!/usr/bin/env python3
"""GLIP 의 15텐서 → 박스·점수·라벨. **호스트(numpy) 참조 구현.**

컴파일된 그래프가 내는 것은 `cls_logits · bbox_pred · centerness` 15개뿐이고, 거기서
박스를 만드는 규칙은 `mmdet/models/dense_heads/atss_vlfusion_head.py` 의
`_predict_by_feat_single` 이다. 그걸 **여기서 먼저 정확히 재현해** C++ 로 옮길 때
비교 기준으로 쓴다 — C++ 를 바로 쓰면 「값이 틀렸는데 디코드 탓인지 그래프 탓인지」를
못 가른다.

일반 검출기와 두 곳이 다르다.

1. **점수가 클래스가 아니라 토큰이다.** `cls_logits` 의 마지막 축 256 은 **텍스트 토큰
   슬롯**이라 argmax 가 뜻이 없다. `positive_map`(문구 → 토큰 인덱스들)으로 **평균**내
   문구 점수를 만든다. 그 map 은 토크나이저가 만들므로 **호스트가 넘긴다.**
2. **박스 코더가 `DeltaXYWHBBoxCoderForGLIP`** 이다. 표준 Delta 디코드에서 `-1`/`+1` 이
   네 군데 붙는다(아래 `delta2bbox_glip` 주석).

⚠️ **점수는 곱이 아니라 `sqrt(곱)`** 이다. 곱으로 두면 점수가 통째로 낮아져 score_thr 에서
   개수가 줄고, 「박스는 맞는데 개수차」로 보인다.
"""
import numpy as np


def convert_grounding_to_cls_scores(logits, positive_map, num_phrases):
    """토큰 로짓 → 문구 점수. `atss_vlfusion_head.py:32` 와 같은 식.

    logits: (N, n_tokens) — **이미 sigmoid 를 거친 값**
    positive_map: {문구번호 j(1부터): [토큰 인덱스…]}
    반환: (N, num_phrases) — 열 j-1 이 문구 j 의 점수
    """
    out = np.zeros((logits.shape[0], num_phrases), dtype=np.float32)
    for j, toks in positive_map.items():
        if not toks:
            continue
        out[:, int(j) - 1] = logits[:, np.asarray(toks, dtype=np.int64)].mean(-1)
    return out


def anchors(feat_hw, stride, octave_base_scale=8.0, center_offset=0.5):
    """`AnchorGenerator(ratios=[1.0], scales_per_octave=1)` 의 격자 앵커 (N, 4).

    ratio 1·octave 1 이라 앵커는 정사각 하나뿐이다 — 변 = stride × octave_base_scale.
    `center_offset=0.5` 라 격자 중심이 칸 가운데다.
    """
    h, w = feat_hw
    size = stride * octave_base_scale
    cx = (np.arange(w, dtype=np.float32) + center_offset) * stride
    cy = (np.arange(h, dtype=np.float32) + center_offset) * stride
    cx, cy = np.meshgrid(cx, cy)                       # (h, w) — 행 우선
    cx, cy = cx.reshape(-1), cy.reshape(-1)
    half = size / 2.0
    return np.stack([cx - half, cy - half, cx + half, cy + half], axis=1)


def delta2bbox_glip(rois, deltas, means=(0., 0., 0., 0.), stds=(0.1, 0.1, 0.2, 0.2),
                    max_shape=None, wh_ratio_clip=16 / 1000):
    """`DeltaXYWHBBoxCoderForGLIP.decode`. 표준 Delta 와 **네 군데** 다르다.

    | | 표준 | GLIP |
    |---|---|---|
    | 앵커 중심 | `(x1+x2)*0.5` | `(x1+x2-1)*0.5` |
    | 박스 복원 | `gxy ∓ gwh*0.5` | `gxy ∓ (gwh-1)*0.5` |
    | 클립 | `max_shape` | `max_shape - 1` |
    | 마지막 | — | `x2,y2 += 1` (**후처리 뒤**, 여기가 아니다) |

    mmdet 주석에도 "very strange bbox decoder logic" 이라 적혀 있다 — 공식 구현의 mAP 에
    맞추려고 남긴 것이라 **한 칸이라도 빼면 박스가 1px 씩 밀린다.**
    """
    d = deltas.reshape(-1, 4) * np.asarray(stds, np.float32) + np.asarray(means, np.float32)
    dxy, dwh = d[:, :2], d[:, 2:]
    pxy = (rois[:, :2] + rois[:, 2:] - 1) * 0.5        # ⚠️ -1
    pwh = rois[:, 2:] - rois[:, :2]

    max_ratio = abs(np.log(wh_ratio_clip))
    dwh = np.clip(dwh, -max_ratio, max_ratio)
    gxy = pxy + pwh * dxy
    gwh = pwh * np.exp(dwh)

    x1y1 = gxy - (gwh - 1) * 0.5                       # ⚠️ -1
    x2y2 = gxy + (gwh - 1) * 0.5                       # ⚠️ -1
    boxes = np.concatenate([x1y1, x2y2], axis=-1)
    if max_shape is not None:
        h, w = max_shape
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, w - 1)   # ⚠️ -1
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, h - 1)
    return boxes


def _nms(boxes, scores, iou_thr):
    """클래스 무관 NMS. 라벨별로 부르는 쪽에서 갈라 준다."""
    order = scores.argsort()[::-1]
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
        iou = inter / np.maximum(area[i] + area[rest] - inter, 1e-9)
        order = rest[iou <= iou_thr]
    return np.asarray(keep, dtype=np.int64)


def decode(cls_logits, bbox_preds, centernesses, positive_map, num_phrases,
           img_shape, strides=(8, 16, 32, 64, 128),
           score_thr=0.05, nms_pre=1000, nms_thr=0.6, max_per_img=100):
    """15텐서 → (boxes (M,4), scores (M,), labels (M,)).

    cls_logits[i]  : (HW, n_tokens)   — 레벨 i
    bbox_preds[i]  : (4, H, W)
    centernesses[i]: (1, H, W)
    """
    all_boxes, all_scores, all_labels = [], [], []
    for lvl, stride in enumerate(strides):
        bp = bbox_preds[lvl]
        _, h, w = bp.shape
        # (4,H,W) → (HW,4). torch 의 `permute(1,2,0).reshape(-1,4)` 와 같은 순서다.
        deltas = bp.transpose(1, 2, 0).reshape(-1, 4)
        ctr = 1.0 / (1.0 + np.exp(-centernesses[lvl].reshape(-1)))     # sigmoid
        logit = cls_logits[lvl]
        scores = convert_grounding_to_cls_scores(
            1.0 / (1.0 + np.exp(-logit)), positive_map, num_phrases)   # (HW, P)

        flat = scores.reshape(-1)
        idx = np.nonzero(flat > score_thr)[0]
        if idx.size == 0:
            continue
        if idx.size > nms_pre:                        # 상위 nms_pre 개만
            idx = idx[np.argsort(flat[idx])[::-1][:nms_pre]]
        keep_loc, labels = idx // num_phrases, idx % num_phrases
        # ⚠️ 곱이 아니라 **sqrt(곱)** 이다.
        sc = np.sqrt(flat[idx] * ctr[keep_loc])
        pri = anchors((h, w), float(stride))[keep_loc]
        all_boxes.append(delta2bbox_glip(pri, deltas[keep_loc], max_shape=img_shape))
        all_scores.append(sc)
        all_labels.append(labels)

    if not all_boxes:
        return (np.zeros((0, 4), np.float32), np.zeros((0,), np.float32),
                np.zeros((0,), np.int64))
    boxes = np.concatenate(all_boxes).astype(np.float32)
    scores = np.concatenate(all_scores).astype(np.float32)
    labels = np.concatenate(all_labels).astype(np.int64)

    # 라벨별 NMS (mmdet `batched_nms` 와 같은 뜻).
    keep = []
    for lab in np.unique(labels):
        sel = np.nonzero(labels == lab)[0]
        keep.append(sel[_nms(boxes[sel], scores[sel], nms_thr)])
    keep = np.concatenate(keep) if keep else np.zeros((0,), np.int64)
    keep = keep[np.argsort(scores[keep])[::-1][:max_per_img]]

    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    # ⚠️ **후처리 뒤에 +1.** mmdet 이 "1 을 안 더하면 공식 mAP 와 안 맞는다" 고 적어 둔 곳이다.
    boxes[:, 2:] = boxes[:, 2:] + 1
    return boxes, scores, labels
