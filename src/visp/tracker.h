// tracking (MOT) host 부품 — ByteTrack. 검출(그래프)은 그대로 쓰고, 프레임 간 association
// (Kalman 예측 + IoU 2단계 매칭 + track 관리)만 여기서 한다. 순수 CPU(ggml 무관), 상태 유지 class.
// mmdet ByteTracker 알고리즘/파라미터 복제.
#pragma once
#include "visp/postproc.h"   // detection
#include <array>
#include <vector>

namespace visp {

// 추적 결과: 검출 + 프레임 간 유지되는 instance id.
struct track_result {
    detection det;
    int id;
};

struct byte_params {
    float high_thr = 0.6f;        // obj_score_thrs.high (1차 매칭 대상)
    float low_thr = 0.1f;         // obj_score_thrs.low (2차 매칭 대상)
    float init_thr = 0.7f;        // 신규 track 초기화 점수 (첫 프레임/empty)
    float match_high = 0.1f;      // match_iou_thrs.high
    float match_low = 0.5f;       // match_iou_thrs.low
    float match_tentative = 0.3f; // match_iou_thrs.tentative
    bool weight_iou_with_scores = true;
    int num_tentatives = 3;       // 이만큼 연속 매칭되면 confirmed
    int num_frames_retain = 30;   // 이만큼 안 보이면 삭제
};

// ByteTracker — track(frame_dets) 를 프레임마다 호출. 같은 물체에 같은 id 유지.
class ByteTracker {
public:
    explicit ByteTracker(byte_params const& p = {}) : p_(p) {}
    void reset();
    // 한 프레임의 검출(픽셀좌표 detection) → 추적 결과(id 부여). frame_id 0 이면 reset.
    std::vector<track_result> track(std::vector<detection> const& dets, int frame_id);

private:
    struct Track {
        int id;
        int label;
        std::array<float, 8> mean;       // cx,cy,aspect,h, v...
        std::array<float, 64> cov;       // 8x8
        int last_frame;
        int hits;                        // 매칭 횟수
        bool tentative;
    };
    byte_params p_;
    std::vector<Track> tracks_;
    int num_tracks_ = 0;

    Track* find(int id);
    // ids(track) × dets IoU 매칭(Hungarian, cost_limit=1-thr) → det별 매칭된 track index(-1=미매칭)
    std::vector<int> assign(std::vector<int> const& track_ids,
                            std::vector<detection> const& dets, bool weight, float thr);
    void init_new(detection const& d, int frame_id);
    void update_track(Track& t, detection const& d, int frame_id);
};

}  // namespace visp
