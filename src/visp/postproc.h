// 검출 전/후처리 (mmdet 스타일). vision.cpp 그래프(부품 조립)의 head 원시출력 → 최종 박스.
// 순수 CPU 로직(ggml 무관). 파라미터는 gguf 메타로 런타임 주입(mmdet test_cfg 대응).
#pragma once
#include <cstdint>
#include <vector>

namespace visp {

struct detection {
    float x1, y1, x2, y2;  // 픽셀 좌표
    float score;
    int label;
};

// ── anchor 생성 (mmdet AnchorGenerator) ─────────────────────────────────────
// 한 레벨의 anchor grid: base_anchor(scales×ratios) × (feat_w×feat_h 격자 shift).
// 반환 [N*4] (x1,y1,x2,y2), N = feat_h*feat_w*num_base. octave_scales = scale 배율 목록.
std::vector<float> gen_anchors(int feat_h, int feat_w, float stride, float base_size,
                               std::vector<float> const& octave_scales,
                               std::vector<float> const& ratios, float center_offset = 0.0f);

// ── bbox decode (mmdet DeltaXYWHBBoxCoder.delta2bbox) ────────────────────────
// anchors[N*4], deltas[N*4] → out[N*4]. denorm(mean/std) + exp(dwh) + clamp + clip.
void delta2bbox(float const* anchors, float const* deltas, int n, float* out,
                float const means[4], float const stds[4], int max_w, int max_h);

// ── NMS (mmcv nms, IoU) ─────────────────────────────────────────────────────
std::vector<int> nms(std::vector<detection> const& dets, float iou_thr);

// ── 앵커-기반 검출 후처리 (RetinaNet/ATSS/GFL/RPN 공유) ──────────────────────
struct det_params {
    std::vector<float> strides;         // 레벨별 stride (예: 8,16,32,64,128)
    float octave_base_scale = 4.0f;     // base_size = stride*octave_base_scale
    std::vector<float> octave_scales;   // scales_per_octave 전개 (예: 2^0, 2^(1/3), 2^(2/3))
    std::vector<float> ratios{0.5f, 1.0f, 2.0f};
    float center_offset = 0.0f;
    float means[4] = {0, 0, 0, 0};
    float stds[4] = {1, 1, 1, 1};
    int num_classes = 80;
    bool use_sigmoid = true;            // cls_score 활성 (RetinaNet sigmoid)
    float score_thr = 0.05f;
    float nms_thr = 0.5f;
    int nms_pre = 1000;                 // 레벨별 topk
    int max_per_img = 100;
    int input_w = 0, input_h = 0;
};

// per-level 원시출력: cls_scores[level] = [num_base*num_classes, feat_w, feat_h](CWHN flat),
// bbox_preds[level] = [num_base*4, feat_w, feat_h]. feat 크기는 shapes 로 전달.
// 반환: 최종 detection(픽셀좌표, label, score).
std::vector<detection> detect_anchor(
    std::vector<std::vector<float>> const& cls_scores,
    std::vector<std::vector<float>> const& bbox_preds,
    std::vector<std::pair<int, int>> const& feat_hw,  // 레벨별 (feat_h, feat_w)
    det_params const& p);

// ── anchor-free 검출 (FCOS/FCOS계열) ────────────────────────────────────────
// point 생성 (mmdet MlvlPointGenerator): point = (idx + offset) * stride.
std::vector<float> gen_points(int feat_h, int feat_w, float stride, float offset = 0.5f);
// distance decode (mmdet distance2bbox): box = [px-l, py-t, px+r, py+b].
void distance2bbox(float const* points, float const* distance, int n, float* out,
                   int max_w, int max_h);

struct fcos_params {
    std::vector<float> strides;       // 8,16,32,64,128
    bool norm_on_bbox = true;         // bbox_pred *= stride
    int num_classes = 80;
    float score_thr = 0.05f;
    float nms_thr = 0.5f;
    int nms_pre = 1000;
    int max_per_img = 100;
    int input_w = 0, input_h = 0;
};
// cls_scores[l]=[nc,W,H]HWC · bbox_preds[l]=[4,W,H]HWC · centerness[l]=[1,W,H]HWC.
std::vector<detection> detect_fcos(
    std::vector<std::vector<float>> const& cls_scores,
    std::vector<std::vector<float>> const& bbox_preds,
    std::vector<std::vector<float>> const& centerness,
    std::vector<std::pair<int, int>> const& feat_hw, fcos_params const& p);

// ── YOLOX (anchor-free grid) ─────────────────────────────────────────────────
struct yolox_params {
    std::vector<float> strides{8, 16, 32};
    float prior_offset = 0.0f;   // YOLOX MlvlPointGenerator offset=0
    int num_classes = 80;
    float score_thr = 0.01f;
    float nms_thr = 0.65f;
    int max_per_img = 100;
    int input_w = 0, input_h = 0;
};
// cls[l]=[nc,W,H]HWC · box[l]=[4,W,H]HWC(dx,dy,logw,logh) · obj[l]=[1,W,H]HWC. score=cls_sig*obj_sig.
std::vector<detection> detect_yolox(
    std::vector<std::vector<float>> const& cls, std::vector<std::vector<float>> const& box,
    std::vector<std::vector<float>> const& obj,
    std::vector<std::pair<int, int>> const& feat_hw, yolox_params const& p);

// ── DETR (set prediction, NMS 없음) ─────────────────────────────────────────
struct detr_params {
    int num_queries = 100;
    int num_classes = 80;
    bool use_sigmoid = false;    // 고전 DETR=softmax, Deformable-DETR=sigmoid
    int max_per_img = 100;
    int input_w = 0, input_h = 0;
};
// cls[Q*(nc 또는 nc+1)] logits · bbox[Q*4] sigmoid(cxcywh 정규화[0,1]). topk(NMS 없음).
std::vector<detection> detect_detr(float const* cls, float const* bbox, detr_params const& p);

// ── mask (Mask RCNN): 박스별 mask logit → 박스 영역 이진 mask ────────────────
// mask_logit[mh*mw](해당 클래스 mask) → sigmoid → 박스크기 resize → threshold. out[bh*bw] 0/1.
std::vector<uint8_t> paste_mask(float const* mask_logit, int mh, int mw,
                                detection const& box, float thr = 0.5f,
                                int* out_h = nullptr, int* out_w = nullptr);

// ── keypoint (heatmap argmax) ───────────────────────────────────────────────
// heatmap[K*H*W] → 각 keypoint (x,y,score). out[K*3].
std::vector<float> decode_keypoints(float const* heatmap, int k, int hm_h, int hm_w,
                                    float stride);

// ── two-stage RoI 후처리 (Faster/Mask/Cascade RCNN 최종 박스) ────────────────
// RPN proposal + RoIAlign 은 host(동적). RoI head 출력(cls softmax, bbox delta)만 여기서 decode+NMS.
struct roi_params {
    float means[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float stds[4] = {0.1f, 0.1f, 0.2f, 0.2f};  // RCNN 표준 stds
    int num_classes = 80;
    bool class_agnostic = false;  // bbox_pred: false=[nc*4], true=[4]
    float score_thr = 0.05f;
    float nms_thr = 0.5f;
    int max_per_img = 100;
    int input_w = 0, input_h = 0;
};
// scores[N*(nc+1)] softmax 완료값(배경 포함, 마지막이 배경) · bbox_deltas[N*(nc*4 또는 4)] · proposals[N*4].
std::vector<detection> detect_roi(float const* scores, float const* bbox_deltas,
                                  float const* proposals, int n, roi_params const& p);

// ── 전처리: 이미지(HWC u8) → 모델 입력 텐서(CWHN f32) ───────────────────────
// resize(size×size) + normalize((v-mean)/std). BGR/RGB·mean/std 는 인자.
std::vector<float> preprocess(uint8_t const* img, int img_h, int img_w, int img_c,
                              int out_size, float const mean[3], float const std[3],
                              bool to_rgb, int* out_w = nullptr, int* out_h = nullptr);

}  // namespace visp
