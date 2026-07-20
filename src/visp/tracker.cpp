// ByteTracker — mmdet ByteTracker 알고리즘 복제 (Kalman + IoU 2단계 매칭 + track 관리).
#include "visp/tracker.h"
#include <algorithm>
#include <cmath>

namespace visp {

// ── Kalman filter (mmdet KalmanFilter, SORT 8-state cxcyah) ──────────────────
namespace {
constexpr float SP = 1.0f / 20.0f;    // std_weight_position
constexpr float SV = 1.0f / 160.0f;   // std_weight_velocity

// 작은 밀집행렬 헬퍼 (row-major flat)
void matmul(float const* A, float const* B, float* C, int n, int m, int k) {  // A[n×m] B[m×k] → C[n×k]
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < k; ++j) {
            float s = 0;
            for (int t = 0; t < m; ++t) s += A[i * m + t] * B[t * k + j];
            C[i * k + j] = s;
        }
}
// proj_cov(4×4) X = B(4×c) 를 Gauss 소거로 풀어 X 반환
void solve4(float S[16], float const* B, float* X, int c) {
    float M[4][4 + 8];  // augmented (c ≤ 8)
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) M[i][j] = S[i * 4 + j];
        for (int j = 0; j < c; ++j) M[i][4 + j] = B[i * c + j];
    }
    for (int col = 0; col < 4; ++col) {
        int piv = col;
        for (int r = col + 1; r < 4; ++r) if (std::fabs(M[r][col]) > std::fabs(M[piv][col])) piv = r;
        for (int j = 0; j < 4 + c; ++j) std::swap(M[col][j], M[piv][j]);
        float d = M[col][col];
        if (std::fabs(d) < 1e-12f) d = 1e-12f;
        for (int j = 0; j < 4 + c; ++j) M[col][j] /= d;
        for (int r = 0; r < 4; ++r) {
            if (r == col) continue;
            float f = M[r][col];
            for (int j = 0; j < 4 + c; ++j) M[r][j] -= f * M[col][j];
        }
    }
    for (int i = 0; i < 4; ++i) for (int j = 0; j < c; ++j) X[i * c + j] = M[i][4 + j];
}

std::array<float, 4> xyxy_to_cxcyah(detection const& d) {
    float w = d.x2 - d.x1, h = d.y2 - d.y1;
    return {(d.x1 + d.x2) * 0.5f, (d.y1 + d.y2) * 0.5f, (h != 0 ? w / h : 0.0f), h};
}
void cxcyah_to_xyxy(float const* m, float& x1, float& y1, float& x2, float& y2) {
    float w = m[2] * m[3];
    x1 = m[0] - w * 0.5f; y1 = m[1] - m[3] * 0.5f; x2 = m[0] + w * 0.5f; y2 = m[1] + m[3] * 0.5f;
}
float iou_xyxy(float ax1, float ay1, float ax2, float ay2, detection const& b) {
    float x1 = std::max(ax1, b.x1), y1 = std::max(ay1, b.y1);
    float x2 = std::min(ax2, b.x2), y2 = std::min(ay2, b.y2);
    float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float ua = (ax2 - ax1) * (ay2 - ay1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
    return inter / (ua + 1e-9f);
}

void kf_initiate(float const* meas, std::array<float, 8>& mean, std::array<float, 64>& cov) {
    for (int i = 0; i < 4; ++i) { mean[i] = meas[i]; mean[4 + i] = 0; }
    float h = meas[3];
    float std[8] = {2*SP*h, 2*SP*h, 1e-2f, 2*SP*h, 10*SV*h, 10*SV*h, 1e-5f, 10*SV*h};
    cov.fill(0);
    for (int i = 0; i < 8; ++i) cov[i * 8 + i] = std[i] * std[i];
}
void kf_predict(std::array<float, 8>& mean, std::array<float, 64>& cov) {
    float h = mean[3];
    float sp[8] = {SP*h, SP*h, 1e-2f, SP*h, SV*h, SV*h, 1e-5f, SV*h};
    // mean = F @ mean  (F: cx += vcx ...)
    for (int i = 0; i < 4; ++i) mean[i] += mean[4 + i];
    // cov = F cov F^T + Q ; F cov: row i(<4) += row i+4 ; then (·)F^T: col j(<4) += col j+4
    float C[64]; for (int i = 0; i < 64; ++i) C[i] = cov[i];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 8; ++j) C[i * 8 + j] += cov[(i + 4) * 8 + j];
    float C2[64]; for (int i = 0; i < 64; ++i) C2[i] = C[i];
    for (int i = 0; i < 8; ++i) for (int j = 0; j < 4; ++j) C2[i * 8 + j] += C[i * 8 + (j + 4)];
    for (int i = 0; i < 64; ++i) cov[i] = C2[i];
    for (int i = 0; i < 8; ++i) cov[i * 8 + i] += sp[i] * sp[i];
}
void kf_update(std::array<float, 8>& mean, std::array<float, 64>& cov, float const* meas) {
    float h = mean[3];
    float rstd[4] = {SP*h, SP*h, 1e-1f, SP*h};
    float S[16];  // proj_cov = cov[0:4,0:4] + R
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) S[i * 4 + j] = cov[i * 8 + j];
    for (int i = 0; i < 4; ++i) S[i * 4 + i] += rstd[i] * rstd[i];
    // B = (cov @ H^T).T = (cov[:,0:4]).T  → (4×8)
    float B[32];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 8; ++j) B[i * 8 + j] = cov[j * 8 + i];
    float X[32]; solve4(S, B, X, 8);          // S X = B → X(4×8) ; K = X^T (8×4)
    float innov[4]; for (int i = 0; i < 4; ++i) innov[i] = meas[i] - mean[i];
    // new_mean = mean + K innov = mean + X^T innov
    for (int i = 0; i < 8; ++i) { float s = 0; for (int t = 0; t < 4; ++t) s += X[t * 8 + i] * innov[t]; mean[i] += s; }
    // new_cov = cov - K S K^T = cov - X^T S X
    float SX[32]; matmul(S, X, SX, 4, 4, 8);  // (4×8)
    for (int i = 0; i < 8; ++i) for (int j = 0; j < 8; ++j) {
        float s = 0; for (int t = 0; t < 4; ++t) s += X[t * 8 + i] * SX[t * 8 + j];
        cov[i * 8 + j] -= s;
    }
}
}  // namespace

// ── ByteTracker ─────────────────────────────────────────────────────────────
void ByteTracker::reset() { tracks_.clear(); num_tracks_ = 0; }

ByteTracker::Track* ByteTracker::find(int id) {
    for (auto& t : tracks_) if (t.id == id) return &t;
    return nullptr;
}

void ByteTracker::init_new(detection const& d, int frame_id) {
    Track t;
    t.id = num_tracks_++;
    t.label = d.label;
    auto m = xyxy_to_cxcyah(d);
    kf_initiate(m.data(), t.mean, t.cov);
    t.last_frame = frame_id;
    t.hits = 1;
    t.tentative = (frame_id != 0);   // 첫 프레임은 바로 confirmed
    tracks_.push_back(t);
}
void ByteTracker::update_track(Track& t, detection const& d, int frame_id) {
    t.hits++;
    if (t.tentative && t.hits >= p_.num_tentatives) t.tentative = false;
    t.label = d.label;
    t.last_frame = frame_id;
    auto m = xyxy_to_cxcyah(d);
    kf_update(t.mean, t.cov, m.data());
}

// track_ids × dets IoU 매칭 (greedy, cost=1-iou<1-thr). det별 매칭 track 로컬 index(-1).
std::vector<int> ByteTracker::assign(std::vector<int> const& track_ids,
                                     std::vector<detection> const& dets, bool weight, float thr) {
    std::vector<int> det2track(dets.size(), -1);
    if (track_ids.empty() || dets.empty()) return det2track;
    // (iou, ti_local, di) 후보 — label 다르면 제외, iou≤thr 제외
    struct Cand { float iou; int ti, di; };
    std::vector<Cand> cand;
    for (size_t ti = 0; ti < track_ids.size(); ++ti) {
        Track* t = find(track_ids[ti]);
        float x1, y1, x2, y2; cxcyah_to_xyxy(t->mean.data(), x1, y1, x2, y2);
        for (size_t di = 0; di < dets.size(); ++di) {
            if (dets[di].label != t->label) continue;
            float iou = iou_xyxy(x1, y1, x2, y2, dets[di]);
            if (weight) iou *= dets[di].score;
            if (iou > thr) cand.push_back({iou, (int)ti, (int)di});
        }
    }
    std::sort(cand.begin(), cand.end(), [](Cand const& a, Cand const& b) { return a.iou > b.iou; });
    std::vector<char> tused(track_ids.size(), 0), dused(dets.size(), 0);
    for (auto const& c : cand) {
        if (tused[c.ti] || dused[c.di]) continue;
        tused[c.ti] = dused[c.di] = 1;
        det2track[c.di] = c.ti;
    }
    return det2track;
}

std::vector<track_result> ByteTracker::track(std::vector<detection> const& dets, int frame_id) {
    if (frame_id == 0) reset();
    // confirmed 예측 (byte: track lost 면 vh=0)
    std::vector<int> confirmed, unconfirmed;
    for (auto& t : tracks_) (t.tentative ? unconfirmed : confirmed).push_back(t.id);
    for (int id : confirmed) {
        Track* t = find(id);
        if (t->last_frame != frame_id - 1) t->mean[7] = 0;
        kf_predict(t->mean, t->cov);
    }

    std::vector<track_result> out;
    if (tracks_.empty() || dets.empty()) {
        for (auto const& d : dets) if (d.score > p_.init_thr) { init_new(d, frame_id); out.push_back({d, tracks_.back().id}); }
        // 관리
        for (auto it = tracks_.begin(); it != tracks_.end();) {
            bool old = frame_id - it->last_frame >= p_.num_frames_retain;
            bool tent = it->tentative && it->last_frame != frame_id;
            it = (old || tent) ? tracks_.erase(it) : it + 1;
        }
        return out;
    }

    // det 분리
    std::vector<detection> first, second;
    for (auto const& d : dets) {
        if (d.score > p_.high_thr) first.push_back(d);
        else if (d.score > p_.low_thr) second.push_back(d);
    }
    std::vector<int> first_id(first.size(), -1), second_id(second.size(), -1);

    // 1차 매칭: confirmed × first
    auto m1 = assign(confirmed, first, p_.weight_iou_with_scores, p_.match_high);
    std::vector<char> conf_matched(confirmed.size(), 0);
    for (size_t di = 0; di < first.size(); ++di) if (m1[di] >= 0) { first_id[di] = confirmed[m1[di]]; conf_matched[m1[di]] = 1; }

    // first 미매칭
    std::vector<detection> f_un; std::vector<int> f_un_idx;
    for (size_t di = 0; di < first.size(); ++di) if (first_id[di] < 0) { f_un.push_back(first[di]); f_un_idx.push_back((int)di); }

    // tentative 매칭: unconfirmed × first_unmatched
    auto mt = assign(unconfirmed, f_un, p_.weight_iou_with_scores, p_.match_tentative);
    for (size_t k = 0; k < f_un.size(); ++k) if (mt[k] >= 0) first_id[f_un_idx[k]] = unconfirmed[mt[k]];

    // 2차 매칭: (1차 미매칭이고 직전프레임에 있던) confirmed × second
    std::vector<int> unmatched_conf;
    for (size_t ti = 0; ti < confirmed.size(); ++ti) {
        Track* t = find(confirmed[ti]);
        if (!conf_matched[ti] && t->last_frame == frame_id - 1) unmatched_conf.push_back(confirmed[ti]);
    }
    auto m2 = assign(unmatched_conf, second, false, p_.match_low);
    for (size_t di = 0; di < second.size(); ++di) if (m2[di] >= 0) second_id[di] = unmatched_conf[m2[di]];

    // 결과 취합 (mmdet: first 전부 + second 매칭된 것만). first 미매칭은 신규 track.
    struct Item { detection d; int id; };
    std::vector<Item> items;
    for (size_t di = 0; di < first.size(); ++di) items.push_back({first[di], first_id[di]});
    for (size_t di = 0; di < second.size(); ++di) if (second_id[di] >= 0) items.push_back({second[di], second_id[di]});
    // 신규 id
    for (auto& it : items) if (it.id < 0) { init_new(it.d, frame_id); it.id = tracks_.back().id; }

    for (auto& it : items) {
        Track* t = find(it.id);
        if (t->last_frame != frame_id) update_track(*t, it.d, frame_id);  // 신규는 이미 init
        out.push_back({it.d, it.id});
    }

    // 관리: retain 초과 또는 tentative 미매칭 삭제
    for (auto iter = tracks_.begin(); iter != tracks_.end();) {
        bool old = frame_id - iter->last_frame >= p_.num_frames_retain;
        bool tent = iter->tentative && iter->last_frame != frame_id;
        iter = (old || tent) ? tracks_.erase(iter) : iter + 1;
    }
    return out;
}

}  // namespace visp
