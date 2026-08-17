#include "visp/postproc.h"
#include <algorithm>
#include <cmath>

namespace visp {

// ── anchor 생성 (mmdet AnchorGenerator.gen_single_level_base_anchors + grid) ──
std::vector<float> gen_anchors(int feat_h, int feat_w, float stride, float base_size,
                               std::vector<float> const& scales,
                               std::vector<float> const& ratios, float center_offset) {
    // base anchors: ratio-major, scale-minor (mmdet: w_ratios[:,None]*scales[None,:]).view(-1)
    float xc = center_offset * base_size, yc = center_offset * base_size;
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
                float const means[4], float const stds[4], int max_w, int max_h) {
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
        dw = std::min(std::max(dw, -max_ratio), max_ratio);
        dh = std::min(std::max(dh, -max_ratio), max_ratio);
        float gxc = pxc + pw * dx, gyc = pyc + ph * dy;
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

// ── 앵커-기반 검출 후처리 (mmdet _predict_by_feat_single) ────────────────────
std::vector<detection> detect_anchor(
    std::vector<std::vector<float>> const& cls_scores,
    std::vector<std::vector<float>> const& bbox_preds,
    std::vector<std::pair<int, int>> const& feat_hw, det_params const& p,
    std::vector<std::vector<float>> const* score_factors) {

    int num_base = (int)(p.octave_scales.size() * p.ratios.size());
    int nc = p.num_classes;
    // 후보 수집: (score, label, box) — 레벨별 nms_pre topk + score_thr
    std::vector<detection> cand;
    int nlev = (int)feat_hw.size();
    for (int l = 0; l < nlev; ++l) {
        int fh = feat_hw[l].first, fw = feat_hw[l].second;
        float stride = p.strides[l];
        float base_size = stride * p.octave_base_scale;
        std::vector<float> anchors = gen_anchors(fh, fw, stride, base_size,
                                                 p.octave_scales, p.ratios, p.center_offset);
        int C_cls = num_base * nc, C_box = num_base * 4;
        float const* cls = cls_scores[l].data();  // HWC flat: (h*fw+w)*C_cls + b*nc + j
        float const* box = bbox_preds[l].data();  // HWC flat: (h*fw+w)*C_box + b*4 + k
        int npos = fh * fw;
        // (score,label,anchor_idx) 후보 → score_thr 넘는 것만, 레벨당 nms_pre topk
        std::vector<std::tuple<float, int, int>> lvl;  // score, label, anchor_idx
        for (int pos = 0; pos < npos; ++pos) {
            for (int b = 0; b < num_base; ++b) {
                int aidx = pos * num_base + b;
                float const* cs = cls + (size_t)pos * C_cls + (size_t)b * nc;
                for (int j = 0; j < nc; ++j) {
                    float sc = p.use_sigmoid ? sigmoidf(cs[j]) : cs[j];
                    if (sc > p.score_thr) lvl.emplace_back(sc, j, aidx);
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
            delta2bbox(anchors.data() + (size_t)aidx * 4, delta, 1, outb,
                       p.means, p.stds, p.input_w, p.input_h);
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
            for (int k = 0; k < 4; ++k) dist[k] = box[(size_t)pos * 4 + k];
            float outb[4];
            distance2bbox(pts.data() + (size_t)pos * 2, dist, 1, outb, p.input_w, p.input_h);
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
        std::vector<float> anchors = gen_anchors(fh, fw, stride, base_size, p.octave_scales, p.ratios);
        int C_cls = num_base * 1, C_box = num_base * 4;
        float const* cls = rpn_cls[l].data();   // CWHN: (h*fw+w)*num_base + b
        float const* box = rpn_bbox[l].data();  // CWHN: (h*fw+w)*C_box + b*4 + k
        int npos = fh * fw;
        std::vector<std::pair<float, int>> lvl;  // (objectness, anchor_idx)
        lvl.reserve((size_t)npos * num_base);
        for (int pos = 0; pos < npos; ++pos)
            for (int b = 0; b < num_base; ++b)
                lvl.emplace_back(sigmoidf(cls[(size_t)pos * C_cls + b]), pos * num_base + b);
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
            delta2bbox(anchors.data() + (size_t)aidx * 4, delta, 1, outb,
                       p.means, p.stds, p.input_w, p.input_h);
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
std::vector<float> preprocess(uint8_t const* img, int img_h, int img_w, int img_c,
                              int out_size, float const mean[3], float const std[3],
                              bool to_rgb, int* out_w, int* out_h) {
    if (out_w) *out_w = out_size;
    if (out_h) *out_h = out_size;
    // bilinear resize → out_size×out_size, normalize, CWHN(index=(h*W+w)*C+c)
    std::vector<float> out((size_t)out_size * out_size * 3);
    float sh = (float)img_h / out_size, sw = (float)img_w / out_size;
    for (int oh = 0; oh < out_size; ++oh) {
        float fy = (oh + 0.5f) * sh - 0.5f;
        int y0 = (int)std::floor(fy); float wy = fy - y0;
        int y0c = std::min(std::max(y0, 0), img_h - 1), y1c = std::min(y0 + 1, img_h - 1);
        for (int ow = 0; ow < out_size; ++ow) {
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
                out[((size_t)oh * out_size + ow) * 3 + c] = (v - mean[c]) / std[c];
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

}  // namespace visp
