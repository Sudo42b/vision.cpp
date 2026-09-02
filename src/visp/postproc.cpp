#include "visp/postproc.h"
#include <algorithm>
#include <cmath>

namespace visp {

// ── anchor 생성 (mmdet AnchorGenerator.gen_single_level_base_anchors + grid) ──
std::vector<float> gen_anchors(int feat_h, int feat_w, float stride, float base_size,
                               std::vector<float> const& scales,
                               std::vector<float> const& ratios, float center_offset,
                               float const* center_xy) {
    // base anchors: ratio-major, scale-minor (mmdet: w_ratios[:,None]*scales[None,:]).view(-1)
    // ⚠️ **중심을 재현식으로 만들지 마라.** mmdet `AnchorGenerator` 는 `centers` 가 주어지면
    //    그 값을 그대로 쓰고 `center_offset` 은 무시한다(anchor_generator.py:`if self.centers
    //    is None`). crowddet 이 레벨마다 (8,8) 을 박아 두는데, 재현식으로 만들면 stride 64
    //    레벨에서 중심이 수십 픽셀 어긋나고 **shape 는 그대로**라 아무 검사에 안 걸린다.
    float xc = center_offset * base_size, yc = center_offset * base_size;
    if (center_xy) { xc = center_xy[0]; yc = center_xy[1]; }
    std::vector<float> base;  // [num_base*4]
    for (float r : ratios) {
        float h_ratio = std::sqrt(r), w_ratio = 1.0f / h_ratio;
        for (float s : scales) {
            float ws = base_size * w_ratio * s, hs = base_size * h_ratio * s;
            base.push_back(xc - 0.5f * ws);
            base.push_back(yc - 0.5f * hs);
            base.push_back(xc + 0.5f * ws);
            base.push_back(yc + 0.5f * hs);
        }
    }
    int num_base = (int)(base.size() / 4);
    std::vector<float> anchors;
    anchors.reserve((size_t)feat_h * feat_w * num_base * 4);
    // grid: y-major (meshgrid row_major) → index (h*feat_w+w)*num_base + b
    for (int h = 0; h < feat_h; ++h) {
        for (int w = 0; w < feat_w; ++w) {
            float sx = w * stride, sy = h * stride;
            for (int b = 0; b < num_base; ++b) {
                anchors.push_back(base[b * 4 + 0] + sx);
                anchors.push_back(base[b * 4 + 1] + sy);
                anchors.push_back(base[b * 4 + 2] + sx);
                anchors.push_back(base[b * 4 + 3] + sy);
            }
        }
    }
    return anchors;
}

// ── delta2bbox (mmdet DeltaXYWHBBoxCoder) ────────────────────────────────────
void delta2bbox(float const* anchors, float const* deltas, int n, float* out,
                float const means[4], float const stds[4], int max_w, int max_h,
                float ctr_clamp) {
    float max_ratio = std::fabs(std::log(16.0f / 1000.0f));
    for (int i = 0; i < n; ++i) {
        float dx = deltas[i * 4 + 0] * stds[0] + means[0];
        float dy = deltas[i * 4 + 1] * stds[1] + means[1];
        float dw = deltas[i * 4 + 2] * stds[2] + means[2];
        float dh = deltas[i * 4 + 3] * stds[3] + means[3];
        float ax1 = anchors[i * 4 + 0], ay1 = anchors[i * 4 + 1];
        float ax2 = anchors[i * 4 + 2], ay2 = anchors[i * 4 + 3];
        float pxc = (ax1 + ax2) * 0.5f, pyc = (ay1 + ay2) * 0.5f;
        float pw = ax2 - ax1, ph = ay2 - ay1;
        float mx = pw * dx, my = ph * dy;
        if (ctr_clamp > 0.0f) {
            // add_ctr_clamp: 중심 이동은 **픽셀 단위**로 자르고 dw/dh 는 상한만 자른다.
            mx = std::min(std::max(mx, -ctr_clamp), ctr_clamp);
            my = std::min(std::max(my, -ctr_clamp), ctr_clamp);
            dw = std::min(dw, max_ratio);
            dh = std::min(dh, max_ratio);
        } else {
            dw = std::min(std::max(dw, -max_ratio), max_ratio);
            dh = std::min(std::max(dh, -max_ratio), max_ratio);
        }
        float gxc = pxc + mx, gyc = pyc + my;
        float gw = pw * std::exp(dw), gh = ph * std::exp(dh);
        float x1 = gxc - gw * 0.5f, y1 = gyc - gh * 0.5f;
        float x2 = gxc + gw * 0.5f, y2 = gyc + gh * 0.5f;
        if (max_w > 0) {  // clip_border
            x1 = std::min(std::max(x1, 0.0f), (float)max_w);
            x2 = std::min(std::max(x2, 0.0f), (float)max_w);
            y1 = std::min(std::max(y1, 0.0f), (float)max_h);
            y2 = std::min(std::max(y2, 0.0f), (float)max_h);
        }
        out[i * 4 + 0] = x1; out[i * 4 + 1] = y1;
        out[i * 4 + 2] = x2; out[i * 4 + 3] = y2;
    }
}

// ── NMS (IoU, greedy) — mmcv nms 동일 ────────────────────────────────────────
std::vector<int> nms(std::vector<detection> const& d, float iou_thr) {
    std::vector<int> idx(d.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int)i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return d[a].score > d[b].score; });
    std::vector<char> removed(d.size(), 0);
    std::vector<int> keep;
    for (size_t m = 0; m < idx.size(); ++m) {
        int i = idx[m];
        if (removed[i]) continue;
        keep.push_back(i);
        float ai = std::max(0.0f, d[i].x2 - d[i].x1) * std::max(0.0f, d[i].y2 - d[i].y1);
        for (size_t k = m + 1; k < idx.size(); ++k) {
            int j = idx[k];
            if (removed[j]) continue;
            float xx1 = std::max(d[i].x1, d[j].x1), yy1 = std::max(d[i].y1, d[j].y1);
            float xx2 = std::min(d[i].x2, d[j].x2), yy2 = std::min(d[i].y2, d[j].y2);
            float iw = std::max(0.0f, xx2 - xx1), ih = std::max(0.0f, yy2 - yy1);
            float inter = iw * ih;
            float aj = std::max(0.0f, d[j].x2 - d[j].x1) * std::max(0.0f, d[j].y2 - d[j].y1);
            float iou = inter / (ai + aj - inter + 1e-9f);
            if (iou > iou_thr) removed[j] = 1;
        }
    }
    return keep;
}

static inline float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// 한 레벨의 앵커. `base_anchor_boxes` 가 실려 있으면 **그 값이 정본**이고 격자만 민다
// (SSD: 레벨마다 개수가 다르다 · YOLACT: base_size·중심을 stride 와 따로 준다).
// 없으면 재현식(base = stride·octave_base_scale, 중심 = center_offset·base)으로 만든다.
static std::vector<float> level_anchors(det_params const& p, int l, int fh, int fw,
                                        int& num_base) {
    const float stride = p.strides[l];
    if (!p.base_anchor_boxes.empty()) {
        std::vector<float> const& ba = p.base_anchor_boxes[l];
        num_base = (int)(ba.size() / 4);
        std::vector<float> a;
        a.reserve((size_t)fh * fw * ba.size());
        for (int h = 0; h < fh; ++h)
            for (int w = 0; w < fw; ++w) {
                const float sx = w * stride, sy = h * stride;
                for (int b = 0; b < num_base; ++b) {
                    a.push_back(ba[b * 4 + 0] + sx);
                    a.push_back(ba[b * 4 + 1] + sy);
                    a.push_back(ba[b * 4 + 2] + sx);
                    a.push_back(ba[b * 4 + 3] + sy);
                }
            }
        return a;
    }
    num_base = (int)(p.octave_scales.size() * p.ratios.size());
    // ⚠️ **`base_size` 는 stride 다 — `stride * octave_base_scale` 이 아니다.**
    //    mmdet `AnchorGenerator` 는 `base_sizes = [min(stride) …]` 로 두고
    //    `octave_base_scale` 은 **`scales` 쪽**에 넣는다(`scales = obs * octave_scales`).
    //    앵커 **크기**는 어느 쪽으로 접든 같지만 **중심**이 달라진다:
    //      mmdet    center = center_offset * stride
    //      접은 식  center = center_offset * stride * octave_base_scale
    //    `center_offset` 이 0 인 계열(retinanet·atss·gfl…)은 양쪽 다 0 이라 안 드러나고,
    //    0.5 인 계열에서만 나온다 — stride 32 · obs 8 이면 **정확히 112px** 어긋난다.
    //    노출된 계열은 `dyhead`·`glip` 둘뿐이다(2026-08-19 전 계열 확인).
    std::vector<float> scales;
    scales.reserve(p.octave_scales.size());
    for (float s : p.octave_scales) scales.push_back(s * p.octave_base_scale);
    return gen_anchors(fh, fw, stride, stride, scales, p.ratios, p.center_offset);
}

// ── 앵커-기반 검출 후처리 (mmdet _predict_by_feat_single) ────────────────────
std::vector<detection> detect_anchor(
    std::vector<std::vector<float>> const& cls_scores,
    std::vector<std::vector<float>> const& bbox_preds,
    std::vector<std::pair<int, int>> const& feat_hw, det_params const& p,
    std::vector<std::vector<float>> const* score_factors) {

    int nc = p.num_classes;
    // softmax head(SSD·YOLACT·고전 계열)는 채널이 하나 더다 — 마지막이 배경이고,
    // mmdet 은 `softmax(-1)[:, :-1]` 로 배경만 버린다(anchor_head.py). 시그모이드
    // 계열은 채널 수 = 클래스 수 그대로다.
    const int ncch = nc + (p.use_sigmoid ? 0 : 1);
    // 후보 수집: (score, label, box) — 레벨별 nms_pre topk + score_thr
    std::vector<detection> cand;
    int nlev = (int)feat_hw.size();
    for (int l = 0; l < nlev; ++l) {
        int fh = feat_hw[l].first, fw = feat_hw[l].second;
        float stride = p.strides[l];
        int num_base = 0;
        std::vector<float> anchors = level_anchors(p, l, fh, fw, num_base);
        int C_cls = num_base * ncch, C_box = num_base * 4;
        float const* cls = cls_scores[l].data();  // HWC flat: (h*fw+w)*C_cls + b*ncch + j
        float const* box = bbox_preds[l].data();  // HWC flat: (h*fw+w)*C_box + b*4 + k
        int npos = fh * fw;
        // (score,label,anchor_idx) 후보 → score_thr 넘는 것만, 레벨당 nms_pre topk
        std::vector<std::tuple<float, int, int>> lvl;  // score, label, anchor_idx
        for (int pos = 0; pos < npos; ++pos) {
            for (int b = 0; b < num_base; ++b) {
                int aidx = pos * num_base + b;
                float const* cs = cls + (size_t)pos * C_cls + (size_t)b * ncch;
                if (p.use_sigmoid) {
                    for (int j = 0; j < nc; ++j) {
                        float sc = sigmoidf(cs[j]);
                        if (sc > p.score_thr) lvl.emplace_back(sc, j, aidx);
                    }
                } else {
                    // softmax — 배경(마지막 채널)은 정규화에만 넣고 후보에서는 버린다.
                    float mx = cs[0];
                    for (int j = 1; j < ncch; ++j) mx = std::max(mx, cs[j]);
                    float sum = 0.0f;
                    for (int j = 0; j < ncch; ++j) sum += std::exp(cs[j] - mx);
                    for (int j = 0; j < nc; ++j) {
                        float sc = std::exp(cs[j] - mx) / sum;
                        if (sc > p.score_thr) lvl.emplace_back(sc, j, aidx);
                    }
                }
            }
        }
        // nms_pre: 레벨당 topk
        if (p.nms_pre > 0 && (int)lvl.size() > p.nms_pre) {
            std::nth_element(lvl.begin(), lvl.begin() + p.nms_pre, lvl.end(),
                             [](auto const& a, auto const& c) { return std::get<0>(a) > std::get<0>(c); });
            lvl.resize(p.nms_pre);
        }
        // score factor (ATSS/PAA 의 centerness). **여기서 곱한다 — top-k 뒤다.**
        //   mmdet 은 `filter_scores_and_topk` 로 cls 점수만 보고 자른 뒤
        //   `_bbox_post_process` 에서 `scores *= score_factors` 를 한다
        //   (base_dense_head.py). 앞에서 곱하면 **살아남는 후보가 달라진다** —
        //   점수는 맞는데 경계선 박스 몇 개가 사라지고, 짝지어 비교하면 안 보인다.
        float const* sf = (score_factors && l < (int)score_factors->size() &&
                           !(*score_factors)[l].empty())
                              ? (*score_factors)[l].data()
                              : nullptr;
        for (auto const& t : lvl) {
            int aidx = std::get<2>(t), b = aidx % num_base, pos = aidx / num_base;
            float delta[4];
            float const* bp = box + (size_t)pos * C_box + (size_t)b * 4;
            for (int k = 0; k < 4; ++k) delta[k] = bp[k];
            float outb[4];
            if (p.tblr_normalizer > 0.0f) {
                // FSAF(TBLRBBoxCoder). 채널이 (t,b,l,r) 순이고 t·b 에는 앵커 **높이**,
                // l·r 에는 **너비**를 곱한다(tblr_bbox_coder.py `tblr2bboxes`).
                float const* a = anchors.data() + (size_t)aidx * 4;
                float cx = (a[0] + a[2]) * 0.5f, cy = (a[1] + a[3]) * 0.5f;
                float aw = a[2] - a[0], ah = a[3] - a[1];
                float tt = delta[0] * p.tblr_normalizer * ah;
                float bb = delta[1] * p.tblr_normalizer * ah;
                float ll = delta[2] * p.tblr_normalizer * aw;
                float rr = delta[3] * p.tblr_normalizer * aw;
                outb[0] = cx - ll; outb[1] = cy - tt;
                outb[2] = cx + rr; outb[3] = cy + bb;
                if (p.input_w > 0) {
                    outb[0] = std::min(std::max(outb[0], 0.0f), (float)p.input_w);
                    outb[2] = std::min(std::max(outb[2], 0.0f), (float)p.input_w);
                    outb[1] = std::min(std::max(outb[1], 0.0f), (float)p.input_h);
                    outb[3] = std::min(std::max(outb[3], 0.0f), (float)p.input_h);
                }
            } else {
                delta2bbox(anchors.data() + (size_t)aidx * 4, delta, 1, outb,
                           p.means, p.stds, p.input_w, p.input_h, p.ctr_clamp);
            }
            // min_bbox_size(≥0): 경계로 클램프돼 변이 0 이 된 박스를 버린다 — IoU 0 이라
            // NMS 가 못 지우고, mmdet 은 NMS 전에 걸러낸다(base_dense_head.py:476-478).
            if (p.min_bbox_size >= 0.0f &&
                (outb[2] - outb[0] <= p.min_bbox_size || outb[3] - outb[1] <= p.min_bbox_size))
                continue;
            float sc = std::get<0>(t);
            if (sf) sc *= sigmoidf(sf[(size_t)pos * num_base + b]);
            cand.push_back({outb[0], outb[1], outb[2], outb[3], sc, std::get<1>(t)});
        }
    }
    // multiclass NMS (label별) — mmcv batched_nms 동등
    std::vector<detection> out;
    int maxlabel = 0;
    for (auto const& c : cand) maxlabel = std::max(maxlabel, c.label);
    for (int lab = 0; lab <= maxlabel; ++lab) {
        std::vector<detection> per;
        for (auto const& c : cand) if (c.label == lab) per.push_back(c);
        if (per.empty()) continue;
        std::vector<int> keep = nms(per, p.nms_thr);
        for (int k : keep) out.push_back(per[k]);
    }
    // topk max_per_img
    std::sort(out.begin(), out.end(), [](detection const& a, detection const& b) { return a.score > b.score; });
    if (p.max_per_img > 0 && (int)out.size() > p.max_per_img) out.resize(p.max_per_img);
    return out;
}

// ── anchor-free (FCOS) ───────────────────────────────────────────────────────
std::vector<float> gen_points(int feat_h, int feat_w, float stride, float offset) {
    std::vector<float> pts;
    pts.reserve((size_t)feat_h * feat_w * 2);
    for (int h = 0; h < feat_h; ++h)
        for (int w = 0; w < feat_w; ++w) {
            pts.push_back((w + offset) * stride);
            pts.push_back((h + offset) * stride);
        }
    return pts;
}

void distance2bbox(float const* points, float const* distance, int n, float* out,
                   int max_w, int max_h) {
    for (int i = 0; i < n; ++i) {
        float px = points[i * 2 + 0], py = points[i * 2 + 1];
        float x1 = px - distance[i * 4 + 0], y1 = py - distance[i * 4 + 1];
        float x2 = px + distance[i * 4 + 2], y2 = py + distance[i * 4 + 3];
        if (max_w > 0) {
            x1 = std::min(std::max(x1, 0.0f), (float)max_w);
            x2 = std::min(std::max(x2, 0.0f), (float)max_w);
            y1 = std::min(std::max(y1, 0.0f), (float)max_h);
            y2 = std::min(std::max(y2, 0.0f), (float)max_h);
        }
        out[i * 4 + 0] = x1; out[i * 4 + 1] = y1;
        out[i * 4 + 2] = x2; out[i * 4 + 3] = y2;
    }
}

std::vector<detection> detect_fcos(
    std::vector<std::vector<float>> const& cls_scores,
    std::vector<std::vector<float>> const& bbox_preds,
    std::vector<std::vector<float>> const& centerness,
    std::vector<std::pair<int, int>> const& feat_hw, fcos_params const& p) {
    int nc = p.num_classes, nlev = (int)feat_hw.size();
    std::vector<detection> cand;
    for (int l = 0; l < nlev; ++l) {
        int fh = feat_hw[l].first, fw = feat_hw[l].second, npos = fh * fw;
        float stride = p.strides[l];
        // 격자 중심 오프셋은 **계열마다 다르다.** FCOS 의 MlvlPointGenerator 는 0.5,
        // GFL·VFNet·RTMDet 은 AnchorGenerator(center_offset=0) 라 0 이다. 0.5 를 잘못
        // 쓰면 전 레벨에서 박스가 반 칸씩 밀린다 — 오차가 stride 에 비례해 커지는 것이
        // 그 증상이다(실측 사례: TOOD 에서 어느 레벨이나 +0.47).
        auto pts = gen_points(fh, fw, stride, p.point_offset);
        float const* cls = cls_scores[l].data();  // HWC: pos*nc + j
        float const* box = bbox_preds[l].data();   // HWC: pos*4 + k
        // centerness 갈래가 없는 계열이 있다(GFL 은 cls 가 품질을 겸하고, VFNet 은 IoU-aware).
        // 0 벡터를 넘기면 sigmoid(0)=0.5 라 점수가 **정확히 절반**이 되므로, 없으면 곱하지 않는다.
        float const* ctr = (l < (int)centerness.size() && !centerness[l].empty())
                               ? centerness[l].data()
                               : nullptr;
        std::vector<std::tuple<float, int, int>> lvl;
        for (int pos = 0; pos < npos; ++pos) {
            float const* cs = cls + (size_t)pos * nc;
            for (int j = 0; j < nc; ++j) {
                // 임계값과 top-k 는 **cls 점수만** 보고 건다 — mmdet 의
                // `filter_scores_and_topk` 가 그렇고, centerness 는 그 뒤에 곱한다.
                float sc = sigmoidf(cs[j]);
                if (sc > p.score_thr) lvl.emplace_back(sc, j, pos);
            }
        }
        if (p.nms_pre > 0 && (int)lvl.size() > p.nms_pre) {
            std::nth_element(lvl.begin(), lvl.begin() + p.nms_pre, lvl.end(),
                             [](auto const& a, auto const& c) { return std::get<0>(a) > std::get<0>(c); });
            lvl.resize(p.nms_pre);
        }
        for (auto const& t : lvl) {
            int pos = std::get<2>(t);
            float dist[4];
            // bbox_pred 는 조립기(head.cpp)가 이미 scale·relu·exp·×stride 를 적용한
            // **최종 픽셀 거리**다 → 여기서 다시 곱하지 않는다. 곱하면 레벨마다
            // 8~128배 커진다.
            //
            // 예외는 FoveaBox 뿐이다. `FoveaHead.forward` 는 exp 를 안 걸고
            // `_bbox_decode` 에서 `base_len·exp(pred)` 로 만든다. 조립기가 그걸 하면
            // torch 의 원시 head 출력과 안 맞아 텐서 검증이 깨지므로 **여기서** 한다.
            const bool fovea = !p.base_edge.empty();
            const float be = fovea ? (l < (int)p.base_edge.size() ? p.base_edge[l] : 1.0f) : 1.0f;
            for (int k = 0; k < 4; ++k) {
                float v = box[(size_t)pos * 4 + k];
                dist[k] = fovea ? be * std::exp(v) : v;
            }
            float outb[4];
            if (p.box_xyxy_offset) {
                // RepPoints: 네 값 모두 중심에 **더한다**(격자 단위 → ×stride).
                const float px = pts[(size_t)pos * 2], py = pts[(size_t)pos * 2 + 1];
                const float st = l < (int)p.strides.size() ? p.strides[l] : 1.0f;
                outb[0] = px + dist[0] * st; outb[1] = py + dist[1] * st;
                outb[2] = px + dist[2] * st; outb[3] = py + dist[3] * st;
                if (p.input_w > 0) {
                    outb[0] = std::min(std::max(outb[0], 0.0f), (float)p.input_w);
                    outb[2] = std::min(std::max(outb[2], 0.0f), (float)p.input_w);
                    outb[1] = std::min(std::max(outb[1], 0.0f), (float)p.input_h);
                    outb[3] = std::min(std::max(outb[3], 0.0f), (float)p.input_h);
                }
            } else {
                distance2bbox(pts.data() + (size_t)pos * 2, dist, 1, outb, p.input_w, p.input_h);
            }
            float sc = std::get<0>(t);
            if (ctr) sc *= sigmoidf(ctr[pos]);
            cand.push_back({outb[0], outb[1], outb[2], outb[3], sc, std::get<1>(t)});
        }
    }
    std::vector<detection> out;
    int maxlabel = 0;
    for (auto const& c : cand) maxlabel = std::max(maxlabel, c.label);
    for (int lab = 0; lab <= maxlabel; ++lab) {
        std::vector<detection> per;
        for (auto const& c : cand) if (c.label == lab) per.push_back(c);
        if (per.empty()) continue;
        for (int k : nms(per, p.nms_thr)) out.push_back(per[k]);
    }
    std::sort(out.begin(), out.end(), [](detection const& a, detection const& b) { return a.score > b.score; });
    if (p.max_per_img > 0 && (int)out.size() > p.max_per_img) out.resize(p.max_per_img);
    return out;
}

// ── YOLOX (anchor-free grid) ─────────────────────────────────────────────────
std::vector<detection> detect_yolox(
    std::vector<std::vector<float>> const& cls, std::vector<std::vector<float>> const& box,
    std::vector<std::vector<float>> const& obj,
    std::vector<std::pair<int, int>> const& feat_hw, yolox_params const& p) {
    int nc = p.num_classes, nlev = (int)feat_hw.size();
    std::vector<detection> cand;
    for (int l = 0; l < nlev; ++l) {
        int fh = feat_hw[l].first, fw = feat_hw[l].second, npos = fh * fw;
        float stride = p.strides[l];
        float const* cs = cls[l].data(); float const* bx = box[l].data(); float const* ob = obj[l].data();
        for (int pos = 0; pos < npos; ++pos) {
            int gy = pos / fw, gx = pos % fw;
            float objc = sigmoidf(ob[pos]);
            float const* c = cs + (size_t)pos * nc;
            float const* b = bx + (size_t)pos * 4;
            // decode: prior center = (gx+offset)*stride
            float pcx = (gx + p.prior_offset) * stride, pcy = (gy + p.prior_offset) * stride;
            float xc = b[0] * stride + pcx, yc = b[1] * stride + pcy;
            float w = std::exp(b[2]) * stride, h = std::exp(b[3]) * stride;
            float x1 = xc - w * 0.5f, y1 = yc - h * 0.5f, x2 = xc + w * 0.5f, y2 = yc + h * 0.5f;
            // YOLOX: 위치별 max class 1개 (per-class 후보 아님)
            int best = 0; float bs = -1;
            for (int j = 0; j < nc; ++j) { float s = sigmoidf(c[j]); if (s > bs) { bs = s; best = j; } }
            float sc = bs * objc;
            if (sc > p.score_thr) cand.push_back({x1, y1, x2, y2, sc, best});
        }
    }
    std::vector<detection> out; int ml = 0;
    for (auto const& c : cand) ml = std::max(ml, c.label);
    for (int lab = 0; lab <= ml; ++lab) {
        std::vector<detection> per;
        for (auto const& c : cand) if (c.label == lab) per.push_back(c);
        if (per.empty()) continue;
        for (int k : nms(per, p.nms_thr)) out.push_back(per[k]);
    }
    std::sort(out.begin(), out.end(), [](detection const& a, detection const& b) { return a.score > b.score; });
    if (p.max_per_img > 0 && (int)out.size() > p.max_per_img) out.resize(p.max_per_img);
    return out;
}

// ── DETR (set prediction, NMS 없음) ─────────────────────────────────────────
std::vector<detection> detect_detr(float const* cls, float const* bbox, detr_params const& p) {
    int Q = p.num_queries, nc = p.num_classes;
    std::vector<detection> out;
    if (p.use_sigmoid) {
        // Deformable-DETR: (query,class) flatten → topk max_per_img (NMS 없음)
        std::vector<std::tuple<float, int, int>> all;  // score, query, class
        for (int q = 0; q < Q; ++q)
            for (int j = 0; j < nc; ++j) all.emplace_back(sigmoidf(cls[(size_t)q * nc + j]), q, j);
        int k = std::min(p.max_per_img, (int)all.size());
        std::partial_sort(all.begin(), all.begin() + k, all.end(),
                          [](auto const& a, auto const& c) { return std::get<0>(a) > std::get<0>(c); });
        for (int i = 0; i < k; ++i) {
            int q = std::get<1>(all[i]);
            float const* b = bbox + (size_t)q * 4;  // cxcywh 정규화
            float cx = b[0] * p.input_w, cy = b[1] * p.input_h, w = b[2] * p.input_w, h = b[3] * p.input_h;
            out.push_back({cx - w*0.5f, cy - h*0.5f, cx + w*0.5f, cy + h*0.5f, std::get<0>(all[i]), std::get<2>(all[i])});
        }
    } else {
        // 고전 DETR: softmax(nc+1), 배경 제외 max class → query별 → topk
        std::vector<detection> q_best;
        for (int q = 0; q < Q; ++q) {
            float const* c = cls + (size_t)q * (nc + 1);
            float mx = -1e30f; for (int j = 0; j <= nc; ++j) mx = std::max(mx, c[j]);
            float sum = 0; for (int j = 0; j <= nc; ++j) sum += std::exp(c[j] - mx);
            int best = 0; float bs = -1;
            for (int j = 0; j < nc; ++j) { float s = std::exp(c[j] - mx) / sum; if (s > bs) { bs = s; best = j; } }
            float const* b = bbox + (size_t)q * 4;
            float cx = b[0]*p.input_w, cy = b[1]*p.input_h, w = b[2]*p.input_w, h = b[3]*p.input_h;
            q_best.push_back({cx - w*0.5f, cy - h*0.5f, cx + w*0.5f, cy + h*0.5f, bs, best});
        }
        std::sort(q_best.begin(), q_best.end(), [](detection const& a, detection const& b) { return a.score > b.score; });
        int k = std::min(p.max_per_img, (int)q_best.size());
        out.assign(q_best.begin(), q_best.begin() + k);
    }
    // mmdet 도 이미지 밖을 잘라낸다(`detr_head._predict_by_feat_single`). 안 자르면
    // 가장자리 물체의 박스가 화면 밖으로 나가 비교가 어긋난다.
    if (p.input_w > 0 && p.input_h > 0) {
        for (detection& d : out) {
            d.x1 = std::min(std::max(d.x1, 0.0f), (float)p.input_w);
            d.x2 = std::min(std::max(d.x2, 0.0f), (float)p.input_w);
            d.y1 = std::min(std::max(d.y1, 0.0f), (float)p.input_h);
            d.y2 = std::min(std::max(d.y2, 0.0f), (float)p.input_h);
        }
    }
    return out;
}

// ── mask (Mask RCNN): mask logit → 박스 영역 이진 mask ───────────────────────
std::vector<uint8_t> paste_mask(float const* mask_logit, int mh, int mw,
                                detection const& box, float thr, int* out_h, int* out_w) {
    // mmdet _do_paste_mask: 박스 정수 bounding [floor(x1),ceil(x2)) 의 각 이미지 픽셀(center=+0.5)에서
    // 박스의 float 좌표로 mask 를 grid_sample(align_corners=False, zero-pad) → sigmoid → threshold.
    // 반환은 그 영역(bw×bh); 배치 원점은 (floor(x1),floor(y1)) (caller 가 캔버스에 놓는다).
    int x0 = (int)std::floor(box.x1), y0 = (int)std::floor(box.y1);
    int bw = std::max(1, (int)std::ceil(box.x2) - x0), bh = std::max(1, (int)std::ceil(box.y2) - y0);
    if (out_w) *out_w = bw;
    if (out_h) *out_h = bh;
    float rw = box.x2 - box.x1, rh = box.y2 - box.y1;
    auto get = [&](int yy, int xx) { return (yy >= 0 && yy < mh && xx >= 0 && xx < mw) ? mask_logit[yy * mw + xx] : 0.0f; };
    std::vector<uint8_t> out((size_t)bh * bw, 0);
    for (int y = 0; y < bh; ++y) {
        float sy = ((y0 + y + 0.5f) - box.y1) / rh * mh - 0.5f;
        int yf = (int)std::floor(sy); float wy = sy - yf;
        for (int x = 0; x < bw; ++x) {
            float sx = ((x0 + x + 0.5f) - box.x1) / rw * mw - 0.5f;
            int xf = (int)std::floor(sx); float wx = sx - xf;
            float v = get(yf, xf) * (1 - wy) * (1 - wx) + get(yf, xf + 1) * (1 - wy) * wx +
                      get(yf + 1, xf) * wy * (1 - wx) + get(yf + 1, xf + 1) * wy * wx;
            out[(size_t)y * bw + x] = sigmoidf(v) > thr ? 1 : 0;
        }
    }
    return out;
}

// ── keypoint (heatmap argmax) ───────────────────────────────────────────────
std::vector<float> decode_keypoints(float const* heatmap, int k, int hm_h, int hm_w, float stride) {
    std::vector<float> out((size_t)k * 3);
    for (int i = 0; i < k; ++i) {
        float const* hm = heatmap + (size_t)i * hm_h * hm_w;
        int best = 0; float bs = -1e30f;
        for (int p = 0; p < hm_h * hm_w; ++p) if (hm[p] > bs) { bs = hm[p]; best = p; }
        out[i*3+0] = (best % hm_w) * stride;
        out[i*3+1] = (best / hm_w) * stride;
        out[i*3+2] = bs;
    }
    return out;
}

// ── two-stage RoI 후처리 (RCNN 최종). delta2bbox+nms 재사용 ───────────────────
std::vector<detection> detect_roi(float const* scores, float const* bbox_deltas,
                                  float const* proposals, int n, roi_params const& p) {
    int nc = p.num_classes;
    int nd = p.class_agnostic ? 4 : nc * 4;  // bbox_pred 폭
    std::vector<detection> cand;
    for (int i = 0; i < n; ++i) {
        float const* sc = scores + (size_t)i * (nc + 1);  // softmax 완료, [nc]+배경
        for (int j = 0; j < nc; ++j) {
            if (sc[j] <= p.score_thr) continue;
            // 클래스별(또는 agnostic) delta 로 proposal decode
            float const* dl = bbox_deltas + (size_t)i * nd + (p.class_agnostic ? 0 : (size_t)j * 4);
            float outb[4];
            delta2bbox(proposals + (size_t)i * 4, dl, 1, outb, p.means, p.stds, p.input_w, p.input_h);
            cand.push_back({outb[0], outb[1], outb[2], outb[3], sc[j], j});
        }
    }
    std::vector<detection> out;
    int maxlabel = 0;
    for (auto const& c : cand) maxlabel = std::max(maxlabel, c.label);
    for (int lab = 0; lab <= maxlabel; ++lab) {
        std::vector<detection> per;
        for (auto const& c : cand) if (c.label == lab) per.push_back(c);
        if (per.empty()) continue;
        for (int k : nms(per, p.nms_thr)) out.push_back(per[k]);
    }
    std::sort(out.begin(), out.end(), [](detection const& a, detection const& b) { return a.score > b.score; });
    if (p.max_per_img > 0 && (int)out.size() > p.max_per_img) out.resize(p.max_per_img);
    return out;
}

// ── RPN proposal 생성 (mmdet RPNHead.predict_by_feat) ────────────────────────
std::vector<detection> detect_rpn(
    std::vector<std::vector<float>> const& rpn_cls,
    std::vector<std::vector<float>> const& rpn_bbox,
    std::vector<std::pair<int, int>> const& feat_hw, rpn_params const& p) {

    int num_base = (int)(p.octave_scales.size() * p.ratios.size());
    std::vector<detection> cand;  // label = level id (mmdet batched_nms level_ids → 레벨별 NMS)
    int nlev = (int)feat_hw.size();
    for (int l = 0; l < nlev; ++l) {
        int fh = feat_hw[l].first, fw = feat_hw[l].second;
        float stride = p.strides[l];
        float base_size = stride * p.octave_base_scale;
        float const* cxy = ((int)p.centers.size() >= 2 * (l + 1)) ? &p.centers[2 * l] : nullptr;
        std::vector<float> anchors = gen_anchors(fh, fw, stride, base_size, p.octave_scales,
                                                 p.ratios, 0.0f, cxy);
        // 채널 수는 **파라미터가 정본**이되, 실제 텐서와 다르면 텐서를 믿는다 —
        // 설정을 못 읽은 계열에서 조용히 틀리는 것보다 낫다.
        int cls_ch = std::max(1, p.cls_out_channels);
        if ((int)rpn_cls[l].size() == fh * fw * num_base * 2) cls_ch = 2;
        else if ((int)rpn_cls[l].size() == fh * fw * num_base) cls_ch = 1;
        int C_cls = num_base * cls_ch, C_box = num_base * 4;
        float const* cls = rpn_cls[l].data();   // CWHN: (h*fw+w)*num_base + b
        float const* box = rpn_bbox[l].data();  // CWHN: (h*fw+w)*C_box + b*4 + k
        int npos = fh * fw;
        std::vector<std::pair<float, int>> lvl;  // (objectness, anchor_idx)
        lvl.reserve((size_t)npos * num_base);
        for (int pos = 0; pos < npos; ++pos)
            for (int b = 0; b < num_base; ++b) {
                float sc;
                if (cls_ch == 1) {
                    sc = sigmoidf(cls[(size_t)pos * C_cls + b]);
                } else {
                    // 채널은 앵커-major · 클래스-minor 다(conv 출력 = num_base*cls_ch).
                    // 전경은 index 0 — `softmax(-1)[:, :-1]` 이 그 뜻이다.
                    float const* c2 = cls + (size_t)pos * C_cls + (size_t)b * cls_ch;
                    const float mx = std::max(c2[0], c2[1]);
                    const float e0 = std::exp(c2[0] - mx), e1 = std::exp(c2[1] - mx);
                    sc = e0 / (e0 + e1);
                }
                lvl.emplace_back(sc, pos * num_base + b);
            }
        // 레벨별 nms_pre topk (RPN 은 pre-NMS score_thr 없음)
        if (p.nms_pre > 0 && (int)lvl.size() > p.nms_pre) {
            std::nth_element(lvl.begin(), lvl.begin() + p.nms_pre, lvl.end(),
                             [](auto const& a, auto const& c) { return a.first > c.first; });
            lvl.resize(p.nms_pre);
        }
        for (auto const& t : lvl) {
            int aidx = t.second, b = aidx % num_base, pos = aidx / num_base;
            float const* bp = box + (size_t)pos * C_box + (size_t)b * 4;
            float delta[4] = {bp[0], bp[1], bp[2], bp[3]};
            float outb[4];
            // clip_border=false 인 계열은 이미지로 자르지 않는다(0 을 주면 클리핑이 꺼진다).
            delta2bbox(anchors.data() + (size_t)aidx * 4, delta, 1, outb, p.means, p.stds,
                       p.clip_border ? p.input_w : 0, p.clip_border ? p.input_h : 0);
            // min_bbox_size 는 **NMS 앞에서** 건다(mmdet `_bbox_post_process`, 강부등호).
            if (p.min_bbox_size > 0.0f &&
                !(outb[2] - outb[0] > p.min_bbox_size && outb[3] - outb[1] > p.min_bbox_size)) {
                continue;
            }
            cand.push_back({outb[0], outb[1], outb[2], outb[3], t.first, l});
        }
    }
    // 레벨별 NMS (batched_nms level_ids) → 전체 topk max_per_img
    std::vector<detection> kept;
    for (int l = 0; l < nlev; ++l) {
        std::vector<detection> per;
        for (auto const& c : cand) if (c.label == l) per.push_back(c);
        if (per.empty()) continue;
        for (int k : nms(per, p.nms_thr)) kept.push_back(per[k]);
    }
    std::sort(kept.begin(), kept.end(), [](detection const& a, detection const& b) { return a.score > b.score; });
    if (p.max_per_img > 0 && (int)kept.size() > p.max_per_img) kept.resize(p.max_per_img);
    return kept;
}

std::vector<float> rpn_proposals(
    std::vector<std::vector<float>> const& rpn_cls,
    std::vector<std::vector<float>> const& rpn_bbox,
    std::vector<std::pair<int, int>> const& feat_hw, rpn_params const& p) {
    std::vector<detection> kept = detect_rpn(rpn_cls, rpn_bbox, feat_hw, p);
    std::vector<float> out;
    out.reserve(kept.size() * 4);
    for (auto const& d : kept) { out.push_back(d.x1); out.push_back(d.y1); out.push_back(d.x2); out.push_back(d.y2); }
    return out;
}

// ── RoIAlign (mmcv RoIAlign, aligned=True) ──────────────────────────────────
// CWHN-flat 단일채널 bilinear 샘플 (mmcv bilinear_interpolate: 범위밖=0, 경계 clamp).
static float bilinear_cwhn(float const* feat, int C, int W, int H, int c, float y, float x) {
    if (y < -1.0f || y > (float)H || x < -1.0f || x > (float)W) return 0.0f;
    if (y <= 0) y = 0;
    if (x <= 0) x = 0;
    int y0 = (int)y, x0 = (int)x, y1, x1;
    if (y0 >= H - 1) { y1 = y0 = H - 1; y = (float)y0; } else y1 = y0 + 1;
    if (x0 >= W - 1) { x1 = x0 = W - 1; x = (float)x0; } else x1 = x0 + 1;
    float ly = y - y0, lx = x - x0, hy = 1.0f - ly, hx = 1.0f - lx;
    auto at = [&](int yy, int xx) { return feat[((size_t)yy * W + xx) * C + c]; };
    return hy * (hx * at(y0, x0) + lx * at(y0, x1)) + ly * (hx * at(y1, x0) + lx * at(y1, x1));
}

std::vector<float> roi_align(
    std::vector<std::vector<float>> const& feats,
    std::vector<std::pair<int, int>> const& feat_hw,
    float const* rois, int m, roi_align_params const& p,
    float const* level_rois) {

    int C = p.channels, out = p.output_size, L = (int)feats.size();
    std::vector<float> res((size_t)m * C * out * out, 0.0f);
    for (int i = 0; i < m; ++i) {
        float const* roi = rois + (size_t)i * 4;
        // ⚠️ 레벨은 **키우기 전 박스**로 고른다(mmdet 은 map_roi_levels 를 roi_rescale 보다
        //    먼저 부른다). 안 그러면 경계 상자가 다른 레벨에서 읽혀 조용히 몇 px 밀린다.
        float const* lroi = level_rois ? level_rois + (size_t)i * 4 : roi;
        float rw = lroi[2] - lroi[0], rh = lroi[3] - lroi[1];
        float scale = std::sqrt(std::max(rw, 0.0f) * std::max(rh, 0.0f));
        int lvl = (int)std::floor(std::log2(scale / p.finest_scale + 1e-6f));
        if (lvl < 0) lvl = 0;
        if (lvl > L - 1) lvl = L - 1;
        if (p.force_level >= 0) lvl = std::min(p.force_level, L - 1);
        float ss = 1.0f / p.strides[lvl];
        int H = feat_hw[lvl].first, W = feat_hw[lvl].second;
        float const* feat = feats[lvl].data();
        float off = p.aligned ? 0.5f : 0.0f;
        float rsw = roi[0] * ss - off, rsh = roi[1] * ss - off;
        float roi_w = (roi[2] * ss - off) - rsw, roi_h = (roi[3] * ss - off) - rsh;
        if (!p.aligned) { roi_w = std::max(roi_w, 1.0f); roi_h = std::max(roi_h, 1.0f); }
        float bin_w = roi_w / out, bin_h = roi_h / out;
        int gh = p.sampling_ratio > 0 ? p.sampling_ratio : (int)std::ceil(roi_h / out);
        int gw = p.sampling_ratio > 0 ? p.sampling_ratio : (int)std::ceil(roi_w / out);
        float count = (float)std::max(gh * gw, 1);
        for (int c = 0; c < C; ++c)
            for (int ph = 0; ph < out; ++ph)
                for (int pw = 0; pw < out; ++pw) {
                    float acc = 0.0f;
                    for (int iy = 0; iy < gh; ++iy) {
                        float yy = rsh + ph * bin_h + (iy + 0.5f) * bin_h / gh;
                        for (int ix = 0; ix < gw; ++ix) {
                            float xx = rsw + pw * bin_w + (ix + 0.5f) * bin_w / gw;
                            acc += bilinear_cwhn(feat, C, W, H, c, yy, xx);
                        }
                    }
                    res[(((size_t)i * C + c) * out + ph) * out + pw] = acc / count;
                }
    }
    return res;
}

// ── 전처리: 이미지(HWC u8) → 모델 입력(CWHN f32) ─────────────────────────────
std::vector<float> preprocess_letterbox(uint8_t const* img, int img_h, int img_w, int img_c,
                                        int dst_w, int dst_h, float const mean[3],
                                        float const std[3], bool to_rgb, letterbox_info* info,
                                        float pad_value) {
    // ultralytics LetterBox(auto=false, scaleFill=false): 긴 변에 맞춘 **단일 배율**로 줄이고
    // 남는 자리를 양쪽에 반씩 패딩한다. 배율이 가로·세로 따로면 종횡비가 뭉개져
    // 컷 근처 물체의 점수가 흔들린다.
    const float scale = std::min((float)dst_w / img_w, (float)dst_h / img_h);
    const int rw = std::max(1, (int)std::round(img_w * scale));
    const int rh = std::max(1, (int)std::round(img_h * scale));
    const float pad_x = (dst_w - rw) / 2.0f;
    const float pad_y = (dst_h - rh) / 2.0f;
    if (info) {
        info->scale = scale;
        info->pad_x = pad_x;
        info->pad_y = pad_y;
    }

    std::vector<float> out((size_t)dst_w * dst_h * 3);
    // 패딩 자리를 먼저 채운다 — 정규화도 같이 먹인다(패딩은 원시 114 가 아니라 정규화된 값이다).
    for (int i = 0; i < dst_w * dst_h; ++i) {
        for (int c = 0; c < 3; ++c) {
            out[(size_t)i * 3 + c] = (pad_value - mean[c]) / std[c];
        }
    }

    const int x_off = (int)std::round(pad_x), y_off = (int)std::round(pad_y);
    const float sh = (float)img_h / rh, sw = (float)img_w / rw;
    for (int oh = 0; oh < rh; ++oh) {
        float fy = (oh + 0.5f) * sh - 0.5f;
        int y0 = (int)std::floor(fy); float wy = fy - y0;
        int y0c = std::min(std::max(y0, 0), img_h - 1), y1c = std::min(y0 + 1, img_h - 1);
        for (int ow = 0; ow < rw; ++ow) {
            float fx = (ow + 0.5f) * sw - 0.5f;
            int x0 = (int)std::floor(fx); float wx = fx - x0;
            int x0c = std::min(std::max(x0, 0), img_w - 1), x1c = std::min(x0 + 1, img_w - 1);
            for (int c = 0; c < 3; ++c) {
                int sc = to_rgb ? (2 - c) : c;
                auto at = [&](int yy, int xx) {
                    return (float)img[((size_t)yy * img_w + xx) * img_c + sc];
                };
                float v = at(y0c, x0c) * (1 - wy) * (1 - wx) + at(y0c, x1c) * (1 - wy) * wx +
                          at(y1c, x0c) * wy * (1 - wx) + at(y1c, x1c) * wy * wx;
                // ultralytics 는 리사이즈 결과를 uint8 로 되돌린다. 안 맞추면 픽셀의 12% 가
                // 1~3 레벨 달라지고 컷 근처 점수가 흔들린다(2026-09-02 실측).
                v = std::round(std::min(std::max(v, 0.0f), 255.0f));
                out[((size_t)(oh + y_off) * dst_w + (ow + x_off)) * 3 + c] =
                    (v - mean[c]) / std[c];
            }
        }
    }
    return out;
}

std::vector<float> preprocess(uint8_t const* img, int img_h, int img_w, int img_c,
                              int dst_w, int dst_h, float const mean[3], float const std[3],
                              bool to_rgb, int* out_w, int* out_h) {
    if (out_w) *out_w = dst_w;
    if (out_h) *out_h = dst_h;
    // bilinear resize → dst_w×dst_h, normalize, CWHN(index=(h*W+w)*C+c)
    std::vector<float> out((size_t)dst_w * dst_h * 3);
    float sh = (float)img_h / dst_h, sw = (float)img_w / dst_w;
    for (int oh = 0; oh < dst_h; ++oh) {
        float fy = (oh + 0.5f) * sh - 0.5f;
        int y0 = (int)std::floor(fy); float wy = fy - y0;
        int y0c = std::min(std::max(y0, 0), img_h - 1), y1c = std::min(y0 + 1, img_h - 1);
        for (int ow = 0; ow < dst_w; ++ow) {
            float fx = (ow + 0.5f) * sw - 0.5f;
            int x0 = (int)std::floor(fx); float wx = fx - x0;
            int x0c = std::min(std::max(x0, 0), img_w - 1), x1c = std::min(x0 + 1, img_w - 1);
            for (int c = 0; c < 3; ++c) {
                int sc = to_rgb ? (2 - c) : c;  // BGR→RGB
                auto at = [&](int yy, int xx) {
                    return (float)img[((size_t)yy * img_w + xx) * img_c + sc];
                };
                float v = at(y0c, x0c) * (1 - wy) * (1 - wx) + at(y0c, x1c) * (1 - wy) * wx +
                          at(y1c, x0c) * wy * (1 - wx) + at(y1c, x1c) * wy * wx;
                out[((size_t)oh * dst_w + ow) * 3 + c] = (v - mean[c]) / std[c];
            }
        }
    }
    return out;
}


// ── YOLO dense (v8/v10/v26) ─────────────────────────────────────────────────
std::vector<detection> detect_yolo_dense(
    float const* box, float const* score,
    std::vector<std::pair<int, int>> const& feat_hw, yolo_dense_params const& p) {

    // 레벨별 격자점을 이어 붙인다. 순서는 그래프가 concat 한 순서와 **같아야** 한다
    // (레벨 0 = 가장 큰 feature map). 어긋나면 박스가 통째로 엉뚱한 데 찍힌다.
    std::vector<float> points;
    std::vector<float> pt_stride;
    size_t n_total = 0;
    for (size_t l = 0; l < feat_hw.size(); ++l) {
        const int H = feat_hw[l].first, W = feat_hw[l].second;
        const float s = l < p.strides.size() ? p.strides[l] : p.strides.back();
        std::vector<float> pts = gen_points(H, W, s, p.point_offset);
        points.insert(points.end(), pts.begin(), pts.end());
        pt_stride.insert(pt_stride.end(), (size_t)H * W, s);
        n_total += (size_t)H * W;
    }
    const int N = (int)n_total;
    if (N == 0) {
        return {};
    }

    // ltrb 는 **격자 단위**다 — stride 를 곱해야 픽셀이 된다.
    // 그래프 덤프는 채널 우선(flat[c*N+a])이고 `distance2bbox` 는 앵커 우선([i*4+k])이라
    // 여기서 전치도 같이 한다.
    std::vector<float> dist((size_t)N * 4);
    for (int i = 0; i < N; ++i) {
        const float s = pt_stride[i];
        for (int k = 0; k < 4; ++k) {
            dist[(size_t)i * 4 + k] = box[(size_t)k * N + i] * s;
        }
    }
    std::vector<float> xyxy((size_t)N * 4);
    distance2bbox(points.data(), dist.data(), N, xyxy.data(), p.input_w, p.input_h);

    // 클래스 로짓 → 시그모이드. 앵커마다 최고 클래스만 남긴다(YOLO 규약).
    std::vector<detection> dets;
    dets.reserve(256);
    for (int i = 0; i < N; ++i) {
        int best = 0;
        float best_logit = score[(size_t)0 * N + i];
        for (int c = 1; c < p.num_classes; ++c) {
            const float v = score[(size_t)c * N + i];
            if (v > best_logit) {
                best_logit = v;
                best = c;
            }
        }
        // 시그모이드는 **최댓값 하나만** 계산한다 — 단조라 argmax 가 안 바뀐다.
        const float sc = 1.0f / (1.0f + std::exp(-best_logit));
        if (sc < p.score_thr) {
            continue;
        }
        dets.push_back(detection{xyxy[(size_t)i * 4 + 0], xyxy[(size_t)i * 4 + 1],
                                 xyxy[(size_t)i * 4 + 2], xyxy[(size_t)i * 4 + 3], sc, best});
    }

    std::sort(dets.begin(), dets.end(),
              [](detection const& a, detection const& b) { return a.score > b.score; });

    // one2one(YOLOv10/26) 은 중복을 모델이 이미 없앴다 — NMS 를 또 걸면 겹친 물체를 지운다.
    if (!p.nms_free) {
        std::vector<int> keep = nms(dets, p.nms_thr);
        std::vector<detection> out;
        out.reserve(keep.size());
        for (int idx : keep) {
            out.push_back(dets[idx]);
        }
        dets.swap(out);
    }
    if ((int)dets.size() > p.max_det) {
        dets.resize(p.max_det);
    }
    return dets;
}

// ── YOLOv3 ───────────────────────────────────────────────────────────────────
// mmdet `YOLOV3Head.predict_by_feat` + `YOLOBBoxCoder.decode`.
//
// 다른 계열과 다른 점 셋:
//  ① 레벨당 출력이 **한 갈래**다 — na×(5+nc) 채널에 tx,ty,tw,th,obj,cls 가 붙어 있다.
//  ② 앵커가 scales×ratios 가 아니라 **(w,h) 쌍 목록**이다(base_sizes).
//  ③ objectness 로 **먼저 거른다**(conf_thr). 안 걸면 후보가 수만 개로 불어난다.
std::vector<detection> detect_yolov3(
    std::vector<std::vector<float>> const& pred,
    std::vector<std::pair<int, int>> const& feat_hw, yolov3_params const& p) {

    const int nc = p.num_classes, attrib = 5 + nc;
    const int nlev = (int)feat_hw.size();
    std::vector<detection> cand;
    for (int l = 0; l < nlev; ++l) {
        if (l >= (int)p.base_sizes.size() || l >= (int)p.strides.size()) break;
        const int fh = feat_hw[l].first, fw = feat_hw[l].second;
        const float stride = p.strides[l];
        std::vector<float> const& bs = p.base_sizes[l];
        const int na = (int)(bs.size() / 2);
        const int C = na * attrib;
        float const* pm = pred[l].data();

        std::vector<std::tuple<float, int, int, int>> lvl;   // score, label, pos, anchor
        for (int pos = 0; pos < fh * fw; ++pos) {
            for (int a = 0; a < na; ++a) {
                float const* q = pm + (size_t)pos * C + (size_t)a * attrib;
                const float conf = sigmoidf(q[4]);
                if (conf < p.conf_thr) continue;      // ③ objectness 선필터
                for (int j = 0; j < nc; ++j) {
                    const float sc = conf * sigmoidf(q[5 + j]);
                    if (sc > p.score_thr) lvl.emplace_back(sc, j, pos, a);
                }
            }
        }
        if (p.nms_pre > 0 && (int)lvl.size() > p.nms_pre) {
            std::nth_element(lvl.begin(), lvl.begin() + p.nms_pre, lvl.end(),
                             [](auto const& x, auto const& y) {
                                 return std::get<0>(x) > std::get<0>(y);
                             });
            lvl.resize(p.nms_pre);
        }
        for (auto const& t : lvl) {
            const int pos = std::get<2>(t), a = std::get<3>(t);
            const int gy = pos / fw, gx = pos % fw;
            float const* q = pm + (size_t)pos * C + (size_t)a * attrib;
            // 앵커 중심은 격자 **중심**이다(YOLOAnchorGenerator 의 centers = stride/2).
            const float acx = (gx + 0.5f) * stride, acy = (gy + 0.5f) * stride;
            const float hw = bs[a * 2] * 0.5f, hh = bs[a * 2 + 1] * 0.5f;
            const float cx = acx + (sigmoidf(q[0]) - 0.5f) * stride;
            const float cy = acy + (sigmoidf(q[1]) - 0.5f) * stride;
            const float w2 = hw * std::exp(q[2]), h2 = hh * std::exp(q[3]);
            float x1 = cx - w2, y1 = cy - h2, x2 = cx + w2, y2 = cy + h2;
            if (p.input_w > 0) {
                x1 = std::min(std::max(x1, 0.0f), (float)p.input_w);
                x2 = std::min(std::max(x2, 0.0f), (float)p.input_w);
                y1 = std::min(std::max(y1, 0.0f), (float)p.input_h);
                y2 = std::min(std::max(y2, 0.0f), (float)p.input_h);
            }
            cand.push_back({x1, y1, x2, y2, std::get<0>(t), std::get<1>(t)});
        }
    }
    // 클래스별 NMS
    std::vector<detection> kept;
    for (int j = 0; j < nc; ++j) {
        std::vector<detection> per;
        for (auto const& d : cand) if (d.label == j) per.push_back(d);
        if (per.empty()) continue;
        for (int k : nms(per, p.nms_thr)) kept.push_back(per[k]);
    }
    std::sort(kept.begin(), kept.end(),
              [](detection const& a, detection const& b) { return a.score > b.score; });
    if (p.max_per_img > 0 && (int)kept.size() > p.max_per_img) kept.resize(p.max_per_img);
    return kept;
}


// ── SABL ─────────────────────────────────────────────────────────────────────
// mmdet `SABLRetinaHead._predict_by_feat_single` + `BucketingBBoxCoder.bucket2bbox`.
//
// 델타 회귀가 아니다. 변(l,r,t,d)마다 앵커를 `num_buckets` 칸으로 쪼개
//   ① 어느 칸인지 softmax 분류 → argmax 칸의 경계를 잡고
//   ② 그 칸 안에서 오프셋을 빼서 최종 변 위치를 만든다.
// 그리고 칸 분류의 확신도(loc_confidence)로 점수를 다시 매긴다.
std::vector<detection> detect_sabl(
    std::vector<std::vector<float>> const& cls,
    std::vector<std::vector<float>> const& bcls,
    std::vector<std::vector<float>> const& breg,
    std::vector<std::pair<int, int>> const& feat_hw, sabl_params const& p) {

    const int nc = p.num_classes;
    const int side = (p.num_buckets + 1) / 2;      // ceil(num_buckets/2)
    const int nlev = (int)feat_hw.size();
    std::vector<detection> cand;

    for (int l = 0; l < nlev; ++l) {
        const int fh = feat_hw[l].first, fw = feat_hw[l].second;
        const float stride = l < (int)p.strides.size() ? p.strides[l] : 1.0f;
        const float sz = stride * p.anchor_scale;   // 정사각 앵커 한 변
        float const* cs = cls[l].data();
        float const* bc = bcls[l].data();
        float const* br = breg[l].data();
        const int Cb = side * 4;

        // 후보 수집 — cls 만 보고 자른다. loc_confidence 는 **뒤에** 곱한다.
        std::vector<std::tuple<float, int, int>> lvl;   // score, label, pos
        for (int pos = 0; pos < fh * fw; ++pos)
            for (int j = 0; j < nc; ++j) {
                const float sc = sigmoidf(cs[(size_t)pos * nc + j]);
                if (sc > p.score_thr) lvl.emplace_back(sc, j, pos);
            }
        if (p.nms_pre > 0 && (int)lvl.size() > p.nms_pre) {
            std::nth_element(lvl.begin(), lvl.begin() + p.nms_pre, lvl.end(),
                             [](auto const& a, auto const& b) {
                                 return std::get<0>(a) > std::get<0>(b);
                             });
            lvl.resize(p.nms_pre);
        }

        for (auto const& t : lvl) {
            const int pos = std::get<2>(t);
            const int gy = pos / fw, gx = pos % fw;
            // AnchorGenerator(center_offset=0): 중심이 격자점 자체다.
            const float cx = gx * stride, cy = gy * stride;
            // 디코드 전에 앵커를 `bucket_scale` 배로 키운다(mmdet bbox_rescale).
            const float hw = sz * 0.5f * p.bucket_scale, hh = sz * 0.5f * p.bucket_scale;
            const float px1 = cx - hw, py1 = cy - hh, px2 = cx + hw, py2 = cy + hh;
            const float bw = (px2 - px1) / p.num_buckets;
            const float bh = (py2 - py1) / p.num_buckets;

            float const* qc = bc + (size_t)pos * Cb;
            float const* qr = br + (size_t)pos * Cb;
            int   top1[4];
            float conf = 0.0f;
            float edge[4];
            for (int k = 0; k < 4; ++k) {          // 0=l 1=r 2=t 3=d
                float const* row = qc + (size_t)k * side;
                // softmax → 상위 2개. 상위 2개가 **이웃 칸**이면 두 번째도 확신도에 더한다.
                float mx = row[0];
                for (int i = 1; i < side; ++i) mx = std::max(mx, row[i]);
                float sum = 0.0f;
                for (int i = 0; i < side; ++i) sum += std::exp(row[i] - mx);
                int i1 = 0, i2 = -1;
                for (int i = 1; i < side; ++i) if (row[i] > row[i1]) i1 = i;
                for (int i = 0; i < side; ++i)
                    if (i != i1 && (i2 < 0 || row[i] > row[i2])) i2 = i;
                const float s1 = std::exp(row[i1] - mx) / sum;
                const float s2 = i2 >= 0 ? std::exp(row[i2] - mx) / sum : 0.0f;
                top1[k] = i1;
                conf += s1 + (std::abs(i1 - i2) == 1 ? s2 : 0.0f);
                const float off = qr[(size_t)k * side + i1];
                // l·t 는 좌/상 경계에서 **더해** 들어가고, r·d 는 우/하에서 **빼서** 들어간다.
                if (k == 0)      edge[0] = (px1 + (0.5f + i1) * bw) - off * bw;
                else if (k == 1) edge[1] = (px2 - (0.5f + i1) * bw) - off * bw;
                else if (k == 2) edge[2] = (py1 + (0.5f + i1) * bh) - off * bh;
                else             edge[3] = (py2 - (0.5f + i1) * bh) - off * bh;
            }
            conf *= 0.25f;                          // 네 변 평균

            float x1 = edge[0], x2 = edge[1], y1 = edge[2], y2 = edge[3];
            if (p.input_w > 0) {                    // clip_border: max_shape-1 이다
                x1 = std::min(std::max(x1, 0.0f), (float)p.input_w - 1.0f);
                x2 = std::min(std::max(x2, 0.0f), (float)p.input_w - 1.0f);
                y1 = std::min(std::max(y1, 0.0f), (float)p.input_h - 1.0f);
                y2 = std::min(std::max(y2, 0.0f), (float)p.input_h - 1.0f);
            }
            // score_factors 규약 — top-k 뒤에 곱한다.
            cand.push_back({x1, y1, x2, y2, std::get<0>(t) * conf, std::get<1>(t)});
        }
    }

    std::vector<detection> kept;
    for (int j = 0; j < nc; ++j) {
        std::vector<detection> per;
        for (auto const& d : cand) if (d.label == j) per.push_back(d);
        if (per.empty()) continue;
        for (int k : nms(per, p.nms_thr)) kept.push_back(per[k]);
    }
    std::sort(kept.begin(), kept.end(),
              [](detection const& a, detection const& b) { return a.score > b.score; });
    if (p.max_per_img > 0 && (int)kept.size() > p.max_per_img) kept.resize(p.max_per_img);
    return kept;
}


// (x1,y1,x2,y2) 두 상자의 IoU — mmdet `bbox_overlaps` 와 같은 식(+1 보정 없음).
static inline float box_iou4(float const* a, float const* b) {
    float xx1 = std::max(a[0], b[0]), yy1 = std::max(a[1], b[1]);
    float xx2 = std::min(a[2], b[2]), yy2 = std::min(a[3], b[3]);
    float inter = std::max(0.0f, xx2 - xx1) * std::max(0.0f, yy2 - yy1);
    float aa = std::max(0.0f, a[2] - a[0]) * std::max(0.0f, a[3] - a[1]);
    float ab = std::max(0.0f, b[2] - b[0]) * std::max(0.0f, b[3] - b[1]);
    return inter / (aa + ab - inter + 1e-9f);
}

// ── PAA / LAD ────────────────────────────────────────────────────────────────
// mmdet `paa_head.py _predict_by_feat_single`. detect_anchor 와 셋이 다르다:
//  ① 레벨별 top-k 가 **앵커 단위**다 — 기준은 max_j sqrt(cls_j×iou) (paa_head.py:582-585)
//  ② 임계값·NMS 점수가 전부 sqrt(cls×iou) 다 (:647)
//  ③ NMS 뒤 score voting — IoU>0.01 인 같은 클래스 후보들의
//     exp(−(1−iou)²/0.025)·score 가중 평균으로 박스를 다시 놓는다. 점수는 그대로다.
std::vector<detection> detect_paa(
    std::vector<std::vector<float>> const& cls_scores,
    std::vector<std::vector<float>> const& bbox_preds,
    std::vector<std::vector<float>> const& iou_preds,
    std::vector<std::pair<int, int>> const& feat_hw, det_params const& p) {

    const int nc = p.num_classes;
    std::vector<detection> cand;          // score_thr 를 넘은 (박스, sqrt점수, 라벨)
    const int nlev = (int)feat_hw.size();
    for (int l = 0; l < nlev; ++l) {
        const int fh = feat_hw[l].first, fw = feat_hw[l].second;
        int num_base = 0;
        std::vector<float> anchors = level_anchors(p, l, fh, fw, num_base);
        const int C_cls = num_base * nc, C_box = num_base * 4;
        float const* cls = cls_scores[l].data();
        float const* box = bbox_preds[l].data();
        float const* iou = iou_preds[l].data();
        const int na = fh * fw * num_base;
        // 앵커 단위 순위 — (max_j sqrt(cls×iou), anchor_idx)
        std::vector<std::pair<float, int>> rank;
        rank.reserve(na);
        for (int pos = 0; pos < fh * fw; ++pos)
            for (int b = 0; b < num_base; ++b) {
                float sf = sigmoidf(iou[(size_t)pos * num_base + b]);
                float const* cs = cls + (size_t)pos * C_cls + (size_t)b * nc;
                float best = 0.0f;
                for (int j = 0; j < nc; ++j) best = std::max(best, sigmoidf(cs[j]) * sf);
                rank.emplace_back(std::sqrt(best), pos * num_base + b);
            }
        if (p.nms_pre > 0 && (int)rank.size() > p.nms_pre) {
            std::nth_element(rank.begin(), rank.begin() + p.nms_pre, rank.end(),
                             [](auto const& a, auto const& b) { return a.first > b.first; });
            rank.resize(p.nms_pre);
        }
        for (auto const& r : rank) {
            const int aidx = r.second, b = aidx % num_base, pos = aidx / num_base;
            float delta[4];
            float const* bp = box + (size_t)pos * C_box + (size_t)b * 4;
            for (int k = 0; k < 4; ++k) delta[k] = bp[k];
            float outb[4];
            delta2bbox(anchors.data() + (size_t)aidx * 4, delta, 1, outb,
                       p.means, p.stds, p.input_w, p.input_h, p.ctr_clamp);
            if (p.min_bbox_size >= 0.0f &&
                (outb[2] - outb[0] <= p.min_bbox_size || outb[3] - outb[1] <= p.min_bbox_size))
                continue;
            const float sf = sigmoidf(iou[(size_t)pos * num_base + b]);
            float const* cs = cls + (size_t)pos * C_cls + (size_t)b * nc;
            for (int j = 0; j < nc; ++j) {
                const float sc = std::sqrt(sigmoidf(cs[j]) * sf);
                if (sc > p.score_thr)
                    cand.push_back({outb[0], outb[1], outb[2], outb[3], sc, j});
            }
        }
    }

    // 클래스별 NMS → 점수순 → max_per_img (multiclass_nms 동등)
    std::vector<detection> out;
    int maxlabel = 0;
    for (auto const& c : cand) maxlabel = std::max(maxlabel, c.label);
    for (int lab = 0; lab <= maxlabel; ++lab) {
        std::vector<detection> per;
        for (auto const& c : cand) if (c.label == lab) per.push_back(c);
        if (per.empty()) continue;
        for (int k : nms(per, p.nms_thr)) out.push_back(per[k]);
    }
    std::sort(out.begin(), out.end(),
              [](detection const& a, detection const& b) { return a.score > b.score; });
    if (p.max_per_img > 0 && (int)out.size() > p.max_per_img) out.resize(p.max_per_img);

    // score voting — 후보 풀은 **NMS 전, 임계값 넘은 전체**다(cand 그대로).
    if (p.score_voting) {
        for (auto& d : out) {
            float bx[4] = {d.x1, d.y1, d.x2, d.y2};
            double acc[4] = {0, 0, 0, 0}, wsum = 0.0;
            for (auto const& c : cand) {
                if (c.label != d.label) continue;
                float cb[4] = {c.x1, c.y1, c.x2, c.y2};
                const float ov = box_iou4(bx, cb);
                if (ov <= 0.01f) continue;
                const double w = std::exp(-(1.0 - ov) * (1.0 - ov) / 0.025) * c.score;
                for (int k = 0; k < 4; ++k) acc[k] += w * cb[k];
                wsum += w;
            }
            if (wsum > 0.0) {
                d.x1 = (float)(acc[0] / wsum); d.y1 = (float)(acc[1] / wsum);
                d.x2 = (float)(acc[2] / wsum); d.y2 = (float)(acc[3] / wsum);
            }
        }
    }
    return out;
}

// ── YOLACT (박스 갈래만 — mask/coeff 는 이 하네스 밖) ────────────────────────
// mmdet `yolact_head.py`. detect_anchor 와 셋이 다르다:
//  ① softmax 이고 배경이 마지막 채널  ② 레벨별 top-k 가 **앵커 단위**(배경 뺀 최대 점수)
//  ③ NMS 가 fast NMS — 클래스별 정렬 top_k 안에서 상삼각 IoU 최댓값이 임계 **이하**인
//     것만 살린다(bbox_nms.py `fast_nms`: 이미 제거된 박스도 남을 박스를 누른다).
std::vector<detection> detect_yolact(
    std::vector<std::vector<float>> const& cls_scores,
    std::vector<std::vector<float>> const& bbox_preds,
    std::vector<std::pair<int, int>> const& feat_hw, det_params const& p) {

    const int nc = p.num_classes;
    const int ncch = nc + 1;              // softmax — 마지막이 배경
    std::vector<float> cboxes;            // [n*4]
    std::vector<float> cscores;           // [n*nc] (softmax, 배경 제외)
    const int nlev = (int)feat_hw.size();
    for (int l = 0; l < nlev; ++l) {
        const int fh = feat_hw[l].first, fw = feat_hw[l].second;
        int num_base = 0;
        std::vector<float> anchors = level_anchors(p, l, fh, fw, num_base);
        const int C_cls = num_base * ncch, C_box = num_base * 4;
        float const* cls = cls_scores[l].data();
        float const* box = bbox_preds[l].data();
        // 앵커 단위 top-k — 기준은 배경 뺀 softmax 최댓값
        std::vector<std::pair<float, int>> rank;
        rank.reserve((size_t)fh * fw * num_base);
        std::vector<float> sm((size_t)fh * fw * num_base * nc);
        for (int pos = 0; pos < fh * fw; ++pos)
            for (int b = 0; b < num_base; ++b) {
                const int aidx = pos * num_base + b;
                float const* cs = cls + (size_t)pos * C_cls + (size_t)b * ncch;
                float mx = cs[0];
                for (int j = 1; j < ncch; ++j) mx = std::max(mx, cs[j]);
                float sum = 0.0f;
                for (int j = 0; j < ncch; ++j) sum += std::exp(cs[j] - mx);
                float best = 0.0f;
                for (int j = 0; j < nc; ++j) {
                    const float s = std::exp(cs[j] - mx) / sum;
                    sm[(size_t)aidx * nc + j] = s;
                    best = std::max(best, s);
                }
                rank.emplace_back(best, aidx);
            }
        if (p.nms_pre > 0 && (int)rank.size() > p.nms_pre) {
            std::nth_element(rank.begin(), rank.begin() + p.nms_pre, rank.end(),
                             [](auto const& a, auto const& b) { return a.first > b.first; });
            rank.resize(p.nms_pre);
        }
        for (auto const& r : rank) {
            const int aidx = r.second, b = aidx % num_base, pos = aidx / num_base;
            float delta[4];
            float const* bp = box + (size_t)pos * C_box + (size_t)b * 4;
            for (int k = 0; k < 4; ++k) delta[k] = bp[k];
            float outb[4];
            delta2bbox(anchors.data() + (size_t)aidx * 4, delta, 1, outb,
                       p.means, p.stds, p.input_w, p.input_h, p.ctr_clamp);
            for (int k = 0; k < 4; ++k) cboxes.push_back(outb[k]);
            for (int j = 0; j < nc; ++j) cscores.push_back(sm[(size_t)aidx * nc + j]);
        }
    }

    // fast NMS. 같은 박스 집합을 클래스마다 그 클래스 점수로 정렬해 따로 거른다.
    const int n = (int)(cboxes.size() / 4);
    const int topk = p.nms_top_k > 0 ? std::min(p.nms_top_k, n) : n;
    std::vector<detection> out;
    std::vector<int> order(n);
    for (int cls = 0; cls < nc; ++cls) {
        for (int i = 0; i < n; ++i) order[i] = i;
        std::partial_sort(order.begin(), order.begin() + topk, order.end(),
                          [&](int a, int b) {
                              return cscores[(size_t)a * nc + cls] > cscores[(size_t)b * nc + cls];
                          });
        // iou_max[j] = 더 높은 순위 i<j 와의 최대 IoU. 임계 **이하**만 살린다(≤).
        for (int j = 0; j < topk; ++j) {
            const float sc = cscores[(size_t)order[j] * nc + cls];
            if (sc <= p.score_thr) continue;      // fast_nms 의 2차 임계(> 만 통과)
            float mx = 0.0f;
            for (int i = 0; i < j; ++i)
                mx = std::max(mx, box_iou4(&cboxes[(size_t)order[i] * 4],
                                           &cboxes[(size_t)order[j] * 4]));
            if (mx <= p.nms_thr) {
                float const* bx = &cboxes[(size_t)order[j] * 4];
                out.push_back({bx[0], bx[1], bx[2], bx[3], sc, cls});
            }
        }
    }
    std::sort(out.begin(), out.end(),
              [](detection const& a, detection const& b) { return a.score > b.score; });
    if (p.max_per_img > 0 && (int)out.size() > p.max_per_img) out.resize(p.max_per_img);
    return out;
}


// ── CornerNet / CentripetalNet ───────────────────────────────────────────────
// mmdet `corner_head.py _decode_heatmap` + `_bboxes_nms`. 앵커가 없다 —
// 코너 두 장에서 각각 top-k 를 뽑아 **k×k 쌍**을 만들고 규칙으로 걸러낸다.
std::vector<detection> detect_corner(
    std::vector<float> const& tl_heat, std::vector<float> const& br_heat,
    std::vector<float> const& tl_off, std::vector<float> const& br_off,
    std::vector<float> const& tl_emb, std::vector<float> const& br_emb,
    std::vector<float> const& tl_shift, std::vector<float> const& br_shift,
    int fh, int fw, corner_params const& p) {

    const int nc = p.num_classes, npos = fh * fw, k = p.topk;
    const int pad = (p.local_max_kernel - 1) / 2;

    // ① 국소 최대만 남긴다(get_local_maximum): maxpool(k, stride 1) 과 같은 자리만 통과.
    //    heatmap 은 CWHN flat 이라 (y*fw+x)*nc + c 다.
    auto local_max = [&](std::vector<float> const& h, std::vector<float>& out) {
        out.assign((size_t)npos * nc, 0.0f);
        for (int y = 0; y < fh; ++y)
            for (int x = 0; x < fw; ++x)
                for (int c = 0; c < nc; ++c) {
                    const float v = sigmoidf(h[((size_t)y * fw + x) * nc + c]);
                    float mx = -1e30f;
                    for (int dy = -pad; dy <= pad; ++dy) {
                        const int yy = y + dy;
                        if (yy < 0 || yy >= fh) continue;
                        for (int dx = -pad; dx <= pad; ++dx) {
                            const int xx = x + dx;
                            if (xx < 0 || xx >= fw) continue;
                            mx = std::max(mx, sigmoidf(h[((size_t)yy * fw + xx) * nc + c]));
                        }
                    }
                    // mmdet 은 `hmax == heat` 로 비교한다 — 같은 값이 여럿이면 둘 다 남는다.
                    if (v >= mx) out[((size_t)y * fw + x) * nc + c] = v;
                }
    };
    std::vector<float> tl_lm, br_lm;
    local_max(tl_heat, tl_lm);
    local_max(br_heat, br_lm);

    // ② 코너별 top-k. mmdet 은 (class, y, x) 를 편 뒤 topk 이므로 **클래스가 인덱스에 섞인다**.
    struct corner { float score; int cls, y, x, pos; };
    auto topk = [&](std::vector<float> const& lm, std::vector<corner>& out) {
        std::vector<std::pair<float, int>> all;
        all.reserve((size_t)npos * nc);
        for (int c = 0; c < nc; ++c)
            for (int i = 0; i < npos; ++i)
                all.emplace_back(lm[(size_t)i * nc + c], c * npos + i);
        const int kk = std::min<int>(k, (int)all.size());
        std::partial_sort(all.begin(), all.begin() + kk, all.end(),
                          [](auto const& a, auto const& b) { return a.first > b.first; });
        out.clear();
        for (int i = 0; i < kk; ++i) {
            const int idx = all[(size_t)i].second, c = idx / npos, pos = idx % npos;
            out.push_back({all[(size_t)i].first, c, pos / fw, pos % fw, pos});
        }
    };
    std::vector<corner> tl, br;
    topk(tl_lm, tl);
    topk(br_lm, br);

    // ③ 코너 좌표 보정(offset) + 픽셀 스케일. off 는 (x,y) 2채널.
    const float sx = p.input_w > 0 ? (float)p.input_w / fw : 1.0f;
    const float sy = p.input_h > 0 ? (float)p.input_h / fh : 1.0f;
    auto corner_xy = [&](corner const& c, std::vector<float> const& off,
                         float& ox, float& oy) {
        ox = c.x + off[(size_t)c.pos * 2 + 0];
        oy = c.y + off[(size_t)c.pos * 2 + 1];
    };

    std::vector<detection> cand;
    for (auto const& a : tl) {
        float ax, ay;
        corner_xy(a, tl_off, ax, ay);
        for (auto const& b : br) {
            if (a.cls != b.cls) continue;                 // 클래스가 같아야 한 물체다
            float bx, by;
            corner_xy(b, br_off, bx, by);
            if (bx <= ax || by <= ay) continue;           // width/height 음수 거부

            // 짝짓기 판정 — 두 방식 중 하나(mmdet 은 assert 로 하나만 켜지게 한다).
            float dist;
            if (p.centripetal) {
                // centripetal shift 로 옮긴 점이 박스 **중앙 영역**에 드는지 본다.
                const float tcx = ax + std::exp(tl_shift[(size_t)a.pos * 2 + 0]);
                const float tcy = ay + std::exp(tl_shift[(size_t)a.pos * 2 + 1]);
                const float bcx = bx - std::exp(br_shift[(size_t)b.pos * 2 + 0]);
                const float bcy = by - std::exp(br_shift[(size_t)b.pos * 2 + 1]);
                // 픽셀로 올린 뒤 판정한다(mmdet 도 스케일 후에 비교한다).
                const float X1 = ax * sx, Y1 = ay * sy, X2 = bx * sx, Y2 = by * sy;
                float T1 = tcx * sx, T2 = tcy * sy, B1 = bcx * sx, B2 = bcy * sy;
                T1 = T1 > 0 ? T1 : 0.0f; T2 = T2 > 0 ? T2 : 0.0f;
                B1 = B1 > 0 ? B1 : 0.0f; B2 = B2 > 0 ? B2 : 0.0f;
                const float area = std::fabs((X2 - X1) * (Y2 - Y1));
                // 논문 4.1 의 상수 — 큰 박스는 중앙 영역을 좁게 잡는다.
                const float mu = area > 3500.0f ? 1.0f / 2.1f : 1.0f / 2.4f;
                const float cx = (X1 + X2) * 0.5f, cy = (Y1 + Y2) * 0.5f;
                const float r0 = cx - mu * (X2 - X1) * 0.5f, r1 = cy - mu * (Y2 - Y1) * 0.5f;
                const float r2 = cx + mu * (X2 - X1) * 0.5f, r3 = cy + mu * (Y2 - Y1) * 0.5f;
                if (T1 <= r0 || T1 >= r2 || T2 <= r1 || T2 >= r3 ||
                    B1 <= r0 || B1 >= r2 || B2 <= r1 || B2 >= r3)
                    continue;                              // 중앙 영역 밖 — 버린다
                const float area_ct = std::fabs((B1 - T1) * (B2 - T2));
                const float area_r = std::fabs((r2 - r0) * (r3 - r1));
                dist = area_ct / (area_r + 1e-12f);
            } else {
                dist = std::fabs(tl_emb[(size_t)a.pos] - br_emb[(size_t)b.pos]);
            }
            if (dist > p.distance_threshold) continue;

            const float score = (a.score + b.score) * 0.5f;
            cand.push_back({ax * sx, ay * sy, bx * sx, by * sy, score, a.cls});
        }
    }

    // ④ 전체에서 num_dets(=max_per_img) 만 남긴 뒤 score_thr — mmdet 순서 그대로다.
    std::sort(cand.begin(), cand.end(),
              [](detection const& a, detection const& b) { return a.score > b.score; });
    if (p.max_per_img > 0 && (int)cand.size() > p.max_per_img) cand.resize(p.max_per_img);
    std::vector<detection> kept;
    for (auto const& d : cand) if (d.score > p.score_thr) kept.push_back(d);

    // ⑤ soft-NMS(gaussian, 클래스별). 하드 NMS 와 달리 **지우지 않고 점수를 깎는다** —
    //    겹친 상자가 살아남되 점수가 exp(−iou²/σ) 배가 된다(mmcv `soft_nms`).
    std::vector<detection> out;
    int maxlabel = 0;
    for (auto const& c : kept) maxlabel = std::max(maxlabel, c.label);
    for (int lab = 0; lab <= maxlabel; ++lab) {
        std::vector<detection> per;
        for (auto const& c : kept) if (c.label == lab) per.push_back(c);
        while (!per.empty()) {
            int best = 0;
            for (int i = 1; i < (int)per.size(); ++i)
                if (per[(size_t)i].score > per[(size_t)best].score) best = i;
            detection m = per[(size_t)best];
            per.erase(per.begin() + best);
            if (m.score < p.soft_nms_min_score) break;     // 남은 것은 더 낮다
            out.push_back(m);
            float mb[4] = {m.x1, m.y1, m.x2, m.y2};
            for (auto& d : per) {
                float db[4] = {d.x1, d.y1, d.x2, d.y2};
                const float ov = box_iou4(mb, db);
                d.score *= std::exp(-ov * ov / p.soft_nms_sigma);
            }
            per.erase(std::remove_if(per.begin(), per.end(),
                                     [&](detection const& d) {
                                         return d.score < p.soft_nms_min_score;
                                     }),
                      per.end());
        }
    }
    std::sort(out.begin(), out.end(),
              [](detection const& a, detection const& b) { return a.score > b.score; });
    if (p.max_per_img > 0 && (int)out.size() > p.max_per_img) out.resize(p.max_per_img);
    return out;
}


// ── CenterNet (heatmap 중심점 디코드) ────────────────────────────────────────
std::vector<detection> detect_centernet(
    std::vector<float> const& heat, std::vector<float> const& wh,
    std::vector<float> const& off, int fh, int fw, centernet_params const& p) {

    const int C = p.num_classes;
    const int HW = fh * fw;
    const int pad = (p.local_max_kernel - 1) / 2;
    // ① 국소 최대만 남긴다(mmcv `get_local_maximum`: maxpool(k, stride 1, pad) == heat).
    //    ⚠️ **maxpool 의 패딩은 0 이 아니라 -inf 다.** 0 으로 채우면 경계에서 음수 값이
    //    최대가 못 되어 중심점이 통째로 사라진다. 여기서는 범위 밖을 아예 안 본다.
    std::vector<std::pair<float, int>> cand;   // (score, flat idx = c*HW + y*fw + x)
    cand.reserve((size_t)HW);
    for (int c = 0; c < C; ++c)
        for (int y = 0; y < fh; ++y)
            for (int x = 0; x < fw; ++x) {
                const float v = heat[((size_t)y * fw + x) * C + c];
                bool is_max = true;
                for (int dy = -pad; dy <= pad && is_max; ++dy)
                    for (int dx = -pad; dx <= pad; ++dx) {
                        const int ny = y + dy, nx = x + dx;
                        if (ny < 0 || ny >= fh || nx < 0 || nx >= fw) continue;
                        if (heat[((size_t)ny * fw + nx) * C + c] > v) { is_max = false; break; }
                    }
                if (is_max) cand.emplace_back(v, c * HW + y * fw + x);
            }
    // ② 전체(클래스 포함)에서 top-k. torch.topk 는 내림차순이고 동점은 인덱스 순이다.
    const int k = std::min((int)cand.size(), std::max(1, p.topk));
    std::partial_sort(cand.begin(), cand.begin() + k, cand.end(),
                      [](auto const& a, auto const& b) {
                          return a.first != b.first ? a.first > b.first : a.second < b.second;
                      });
    // ③ 같은 자리의 wh/offset 으로 상자를 만들고 입력 해상도로 늘린다.
    const float sx = fw > 0 ? (float)p.input_w / (float)fw : 1.0f;
    const float sy = fh > 0 ? (float)p.input_h / (float)fh : 1.0f;
    std::vector<detection> out;
    out.reserve(k);
    for (int i = 0; i < k; ++i) {
        const int idx = cand[i].second;
        const int cls = idx / HW, rem = idx % HW;
        const int y = rem / fw, x = rem % fw;
        const size_t o2 = ((size_t)y * fw + x) * 2;
        const float cx = (float)x + off[o2 + 0], cy = (float)y + off[o2 + 1];
        const float w = wh[o2 + 0], h = wh[o2 + 1];
        out.push_back({(cx - w * 0.5f) * sx, (cy - h * 0.5f) * sy,
                       (cx + w * 0.5f) * sx, (cy + h * 0.5f) * sy,
                       cand[i].first, cls});
    }
    return out;
}


}  // namespace visp