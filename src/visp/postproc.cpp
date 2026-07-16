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
    std::vector<std::pair<int, int>> const& feat_hw, det_params const& p) {

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
        for (auto const& t : lvl) {
            int aidx = std::get<2>(t), b = aidx % num_base, pos = aidx / num_base;
            float delta[4];
            float const* bp = box + (size_t)pos * C_box + (size_t)b * 4;
            for (int k = 0; k < 4; ++k) delta[k] = bp[k];
            float outb[4];
            delta2bbox(anchors.data() + (size_t)aidx * 4, delta, 1, outb,
                       p.means, p.stds, p.input_w, p.input_h);
            cand.push_back({outb[0], outb[1], outb[2], outb[3], std::get<0>(t), std::get<1>(t)});
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
        auto pts = gen_points(fh, fw, stride);
        float const* cls = cls_scores[l].data();  // HWC: pos*nc + j
        float const* box = bbox_preds[l].data();   // HWC: pos*4 + k
        float const* ctr = centerness[l].data();    // HWC: pos*1
        std::vector<std::tuple<float, int, int>> lvl;
        for (int pos = 0; pos < npos; ++pos) {
            float cen = sigmoidf(ctr[pos]);
            float const* cs = cls + (size_t)pos * nc;
            for (int j = 0; j < nc; ++j) {
                float sc = sigmoidf(cs[j]) * cen;
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
            // bbox_pred 는 traced head(forward_single) 가 이미 scale·relu·*stride 적용한 최종
            // distance 다 → 여기선 그대로 사용(재적용 금지). norm_on_bbox=false 면 raw*stride.
            for (int k = 0; k < 4; ++k)
                dist[k] = box[(size_t)pos * 4 + k] * (p.norm_on_bbox ? 1.0f : stride);
            (void)stride;
            float outb[4];
            distance2bbox(pts.data() + (size_t)pos * 2, dist, 1, outb, p.input_w, p.input_h);
            cand.push_back({outb[0], outb[1], outb[2], outb[3], std::get<0>(t), std::get<1>(t)});
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
    return out;
}

// ── mask (Mask RCNN): mask logit → 박스 영역 이진 mask ───────────────────────
std::vector<uint8_t> paste_mask(float const* mask_logit, int mh, int mw,
                                detection const& box, float thr, int* out_h, int* out_w) {
    int bw = std::max(1, (int)std::round(box.x2 - box.x1));
    int bh = std::max(1, (int)std::round(box.y2 - box.y1));
    if (out_w) *out_w = bw;
    if (out_h) *out_h = bh;
    std::vector<uint8_t> out((size_t)bh * bw);
    // mask logit(mh×mw) → 박스크기(bh×bw) bilinear → sigmoid → threshold
    for (int y = 0; y < bh; ++y) {
        float fy = (y + 0.5f) * mh / bh - 0.5f; int y0 = (int)std::floor(fy); float wy = fy - y0;
        int y0c = std::min(std::max(y0, 0), mh-1), y1c = std::min(y0+1, mh-1);
        for (int x = 0; x < bw; ++x) {
            float fx = (x + 0.5f) * mw / bw - 0.5f; int x0 = (int)std::floor(fx); float wx = fx - x0;
            int x0c = std::min(std::max(x0, 0), mw-1), x1c = std::min(x0+1, mw-1);
            float v = mask_logit[y0c*mw+x0c]*(1-wy)*(1-wx) + mask_logit[y0c*mw+x1c]*(1-wy)*wx +
                      mask_logit[y1c*mw+x0c]*wy*(1-wx) + mask_logit[y1c*mw+x1c]*wy*wx;
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

}  // namespace visp
