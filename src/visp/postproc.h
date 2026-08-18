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
// `ctr_clamp` > 0 이면 mmdet 의 **add_ctr_clamp** 경로다(YOLOF). 중심 이동량을 픽셀 단위로
// ±ctr_clamp 로 자르고, dw/dh 는 **상한만** 자른다 — 기본 경로(상·하한 모두)와 다르다
// (delta_xywh_bbox_coder.py:346-350).
void delta2bbox(float const* anchors, float const* deltas, int n, float* out,
                float const means[4], float const stds[4], int max_w, int max_h,
                float ctr_clamp = 0.0f);

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
    // FoveaBox 의 `base_edge_list`(레벨별). 비면 안 쓴다 — `fcos_params::base_edge` 로 넘어간다.
    std::vector<float> base_edge;
    // mmdet DeltaXYWHBBoxCoder 의 `add_ctr_clamp`(YOLOF). 0 이면 기본 경로.
    float ctr_clamp = 0.0f;
    // YOLOv3 의 (w,h) 앵커(레벨별)와 objectness 선필터. 비면 안 쓴다.
    std::vector<std::vector<float>> base_sizes;
    float conf_thr = 0.0f;
    // SABL 의 버킷 파라미터. 0 이면 안 쓴다.
    int num_buckets = 0;
    float bucket_scale = 3.0f;
    float anchor_scale = 4.0f;
};

// per-level 원시출력: cls_scores[level] = [num_base*num_classes, feat_w, feat_h](CWHN flat),
// bbox_preds[level] = [num_base*4, feat_w, feat_h]. feat 크기는 shapes 로 전달.
// 반환: 최종 detection(픽셀좌표, label, score).
// score_factors: ATSS/PAA 의 centerness 갈래(레벨별 [num_base, feat_w, feat_h] CWHN flat).
// nullptr 이면 cls 점수만 쓴다. **top-k 뒤에 곱한다** — mmdet 이 그렇다
// (`base_dense_head._bbox_post_process`). 앞에서 곱하면 살아남는 후보가 달라져,
// 점수는 맞는데 경계선 박스가 조용히 사라진다.
std::vector<detection> detect_anchor(
    std::vector<std::vector<float>> const& cls_scores,
    std::vector<std::vector<float>> const& bbox_preds,
    std::vector<std::pair<int, int>> const& feat_hw,  // 레벨별 (feat_h, feat_w)
    det_params const& p,
    std::vector<std::vector<float>> const* score_factors = nullptr);

// ── anchor-free 검출 (FCOS/FCOS계열) ────────────────────────────────────────
// point 생성 (mmdet MlvlPointGenerator): point = (idx + offset) * stride.
std::vector<float> gen_points(int feat_h, int feat_w, float stride, float offset = 0.5f);
// distance decode (mmdet distance2bbox): box = [px-l, py-t, px+r, py+b].
void distance2bbox(float const* points, float const* distance, int n, float* out,
                   int max_w, int max_h);

struct fcos_params {
    std::vector<float> strides;       // 8,16,32,64,128
    // 격자 중심 오프셋. FCOS 의 MlvlPointGenerator 는 0.5, GFL·VFNet·RTMDet 은
    // AnchorGenerator(center_offset=0) 라 0 이다. 틀리면 박스가 반 칸씩 밀리는데,
    // 오차가 stride 에 비례해 커지는 것이 그 증상이다.
    float point_offset = 0.5f;
    int num_classes = 80;
    float score_thr = 0.05f;
    float nms_thr = 0.5f;
    int nms_pre = 1000;
    int max_per_img = 100;
    int input_w = 0, input_h = 0;
    // FoveaBox 는 거리를 **여기서** 만든다: `dist = base_edge[l] · exp(pred)`.
    // ⚠️ 조립기에 넣으면 안 된다 — `FCOSHead.forward` 는 `exp` 를 자기가 걸지만
    //    `FoveaHead.forward` 는 **안 건다**(`_bbox_decode` 에서 건다). 조립기에 넣으면
    //    텐서 검증이 torch 의 원시 출력과 안 맞는다(실측 상대 L1 8.7e+02).
    //    그리고 곱하는 값은 stride 가 아니라 `base_edge_list` 다 — 기본값이 stride 의
    //    2배(16,32,… vs 8,16,…)라 stride 로 두면 박스가 레벨마다 정확히 절반이 된다.
    std::vector<float> base_edge;
    // RepPoints: 조립기가 `points2bbox` 로 만든 값은 **거리가 아니라 중심 기준 xyxy
    // 오프셋**이고 단위가 격자다. mmdet 은 `bboxes = pred·stride + [cx,cy,cx,cy]` 로
    // 픽셀화한다(reppoints_head.py `_predict_by_feat_single`). 거리 디코드처럼
    // l·t 를 빼면 박스가 중심 반대편으로 뒤집힌다.
    bool box_xyxy_offset = false;
};
// cls_scores[l]=[nc,W,H]HWC · bbox_preds[l]=[4,W,H]HWC.
// bbox_preds 는 조립기가 이미 픽셀 거리로 만든 값이다 — 여기서 stride 를 곱하지 않는다.
// centerness 는 **비어 있어도 된다**(GFL·VFNet 은 cls 가 품질을 겸한다). 비면 안 곱한다.
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

// ── YOLO dense (v8/v10/v26 계열, 격자 ltrb) ─────────────────────────────────
// g2c 생성 그래프는 **디코드 앞에서 끊는다** — 격자 단위 ltrb 와 클래스 **로짓**을 낸다.
// 여기서 stride 곱 · 격자점 더하기 · 시그모이드 · top-k 를 한다(전부 호스트).
struct yolo_dense_params {
    std::vector<float> strides{8, 16, 32};
    float point_offset = 0.5f;   // ultralytics make_anchors 규약
    int num_classes = 80;
    float score_thr = 0.25f;
    float nms_thr = 0.7f;
    bool nms_free = true;        // one2one(YOLOv10/26) 은 NMS 없이 top-k 만
    int max_det = 300;
    int input_w = 0, input_h = 0;
};
// box[4*N] · score[nc*N] 은 **채널 우선**(flat[c*N + a]) — 그래프 덤프 레이아웃 그대로다.
// N = Σ(W_l·H_l). feat_hw 는 레벨별 (H, W).
std::vector<detection> detect_yolo_dense(
    float const* box, float const* score,
    std::vector<std::pair<int, int>> const& feat_hw, yolo_dense_params const& p);

// ── SABL (Side-Aware Boundary Localization) ─────────────────────────────────
// 박스를 델타로 한 번에 내지 않는다. **변(l,r,t,d)마다** 앵커를 `num_buckets` 칸으로 나눠
// ① 어느 칸인지 **분류**하고 ② 그 칸 안의 **오프셋**을 회귀한다. 그래서 box 갈래가 둘이다.
struct sabl_params {
    std::vector<float> strides;
    float anchor_scale = 4.0f;   // square anchor: 한 변 = stride · scale (위치당 1개)
    int num_buckets = 14;        // side_num = ceil(num_buckets/2)
    float bucket_scale = 3.0f;   // 디코드 전에 앵커를 이만큼 키운다(coder 의 scale_factor)
    int num_classes = 80;
    float score_thr = 0.05f;
    float nms_thr = 0.5f;
    int nms_pre = 1000;
    int max_per_img = 100;
    int input_w = 0, input_h = 0;
};
// cls[l]=[nc,W,H] · bcls[l]=[side_num*4,W,H] · breg[l]=[side_num*4,W,H] (전부 CWHN flat).
// 채널 순서는 변 우선이다 — [l·side_num, r·side_num, t·side_num, d·side_num].
// 점수는 `cls_sigmoid × loc_confidence` 이고, **top-k 뒤에** 곱한다(mmdet score_factors 규약).
std::vector<detection> detect_sabl(
    std::vector<std::vector<float>> const& cls,
    std::vector<std::vector<float>> const& bcls,
    std::vector<std::vector<float>> const& breg,
    std::vector<std::pair<int, int>> const& feat_hw, sabl_params const& p);

// ── YOLOv3 (레벨당 한 갈래: na×(5+nc) 채널) ─────────────────────────────────
struct yolov3_params {
    // ⚠️ YOLOv3 는 stride 가 **내림차순**이다(32,16,8). FPN 순서와 반대라 뒤집어 쓰면
    //    레벨마다 4배씩 어긋난다. 프론트엔드가 config 순서 그대로 싣는다.
    std::vector<float> strides;
    // 레벨별 앵커를 **(w,h) 쌍**으로 받는다 — scales×ratios 가 아니다.
    // base_sizes[l] = {w0,h0, w1,h1, ...} (레벨당 num_base 쌍).
    std::vector<std::vector<float>> base_sizes;
    int num_classes = 80;
    float conf_thr = 0.005f;   // objectness 선필터(mmdet test_cfg.conf_thr)
    float score_thr = 0.05f;
    float nms_thr = 0.45f;
    int nms_pre = 1000;
    int max_per_img = 100;
    int input_w = 0, input_h = 0;
};
// pred[l] = [na*(5+nc), W, H] CWHN flat. 채널 = a*(5+nc) + {tx,ty,tw,th,obj,cls...}.
// 디코드(YOLOBBoxCoder): 중심 = 앵커중심 + (sigmoid(t)-0.5)·stride,
//                        반폭 = 앵커반폭 · exp(t).  앵커중심은 격자 중심(=stride/2 오프셋).
std::vector<detection> detect_yolov3(
    std::vector<std::vector<float>> const& pred,
    std::vector<std::pair<int, int>> const& feat_hw, yolov3_params const& p);

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

// ── RPN proposal 생성 (mmdet RPNHead.predict_by_feat) ────────────────────────
// rpn_cls/bbox(레벨별) → anchor decode → 레벨별 nms_pre topk → 전체 NMS → max_per_img.
// objectness 단일채널(sigmoid). detect_anchor 와 달리 클래스 없이 박스+점수만.
struct rpn_params {
    std::vector<float> strides;         // [4,8,16,32,64] (P2-P6)
    float octave_base_scale = 8.0f;     // RPN anchor scale (base_size = stride*scale)
    std::vector<float> octave_scales{1.0f};
    std::vector<float> ratios{0.5f, 1.0f, 2.0f};
    float means[4] = {0, 0, 0, 0};
    float stds[4] = {1, 1, 1, 1};
    int nms_pre = 2000;                 // 레벨별 pre-NMS topk
    float nms_thr = 0.7f;
    int max_per_img = 1000;             // 최종 proposal 수
    int input_w = 0, input_h = 0;
};
// rpn_cls[l] = [num_base*1, W, H] objectness(CWHN flat), rpn_bbox[l] = [num_base*4, W, H].
// 점수까지 필요한 쪽(단독 RPN 계열 검증)이 쓴다. `label` 은 클래스가 아니라 **레벨 번호**다
// — mmdet 이 `batched_nms(level_ids)` 로 레벨별 NMS 를 하기 때문이고, 그게 이 계열의 정본이다.
std::vector<detection> detect_rpn(
    std::vector<std::vector<float>> const& rpn_cls,
    std::vector<std::vector<float>> const& rpn_bbox,
    std::vector<std::pair<int, int>> const& feat_hw, rpn_params const& p);
// 같은 계산에서 박스만 평평하게 뽑는다(two-stage 의 proposal 입력). M ≤ max_per_img.
std::vector<float> rpn_proposals(
    std::vector<std::vector<float>> const& rpn_cls,
    std::vector<std::vector<float>> const& rpn_bbox,
    std::vector<std::pair<int, int>> const& feat_hw, rpn_params const& p);

// ── RoIAlign (mmcv RoIAlign, aligned=True, sampling_ratio=0) ─────────────────
// FPN feats(레벨별 CWHN flat: idx=(y*W+x)*C+c) + rois(이미지좌표) → roi_feat.
// 레벨 배정 = clamp(floor(log2(sqrt(w*h)/finest_scale + 1e-6)), 0, L-1).
struct roi_align_params {
    int output_size = 7;
    int channels = 256;
    std::vector<float> strides{4, 8, 16, 32};  // featmap_strides
    float finest_scale = 56.0f;
    bool aligned = true;
    int sampling_ratio = 0;             // 0 → ceil(roi/out) 적응
};
// feats[l] = [C, W, H] CWHN flat · rois[M*4] 이미지좌표.
// 반환 roi_feat [M*C*out*out] = NCHW flat(idx=((m*C+c)*out+ph)*out+pw). (SubB 입력용으로 permute 는 러너가)
//
// `level_rois` — **피라미드 레벨을 고를 때만** 쓸 박스(없으면 `rois` 로 고른다).
// ⚠️ Double-Head 처럼 회귀용 RoI 를 키워서 다시 자르는 계열이 있는데, mmdet 은
//    **키우기 전 박스로 레벨을 정하고 키운 박스로 샘플링한다**
//    (`single_level_roi_extractor.py:97-100` — `map_roi_levels` 가 `roi_rescale` 보다 먼저다).
//    키운 박스로 레벨까지 고르면 경계에 있던 상자가 한 레벨 위로 올라가 **다른 feature 를
//    읽는다.** 크래시는 없고 박스만 몇 px 밀린다.
std::vector<float> roi_align(
    std::vector<std::vector<float>> const& feats,
    std::vector<std::pair<int, int>> const& feat_hw,
    float const* rois, int m, roi_align_params const& p,
    float const* level_rois = nullptr);

// ── 전처리: 이미지(HWC u8) → 모델 입력 텐서(CWHN f32) ───────────────────────
// resize(size×size) + normalize((v-mean)/std). BGR/RGB·mean/std 는 인자.
std::vector<float> preprocess(uint8_t const* img, int img_h, int img_w, int img_c,
                              int out_size, float const mean[3], float const std[3],
                              bool to_rgb, int* out_w = nullptr, int* out_h = nullptr);

}  // namespace visp
