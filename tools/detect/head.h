// mmdet 검출 head 를 **손코딩 C++ 부품**으로 조립한다. g2c 는 backbone/neck 만 그래프
// 데이터(gguf graph.nodes)로 내보내고, head(공유 conv 타워 + 최종 cls/reg conv)는 여기서
// 조립한다 — 교수님 지시("backbone 만 g2c, head 는 vision.cpp 부품"). decode+NMS 는
// postproc.cpp(detect_anchor)가 담당하므로, 이 부품은 raw cls_score/bbox_pred 까지만 낸다.
#pragma once

#include "visp/ml.h"
#include "visp/postproc.h"   // det_params

#include <string>
#include <vector>

namespace visp {

// 어느 조립기를 쓸지. 계열이 늘어도 러너는 안 바뀐다 — mmdet_head_forward 가 갈라준다.
enum class head_kind {
    anchor,     // RetinaNet · ATSS · PAA — cls/reg(+centerness) 타워, Delta 디코드
    // 타워 조립은 anchor 와 같다(레벨별 타워 + obj 갈래). **디코드가 다르다** — 코더가
    // 없고 격자 단위 (dx,dy,log w,log h) 를 내며 점수는 cls×obj 다. 조립만 보고 anchor 로
    // 두면 Delta 디코더로 떨어져 박스가 통째로 틀린다.
    yolox,
    fcos,       // + bbox 에 scale·clamp·stride (anchor-free 거리 디코드)
    gfl,        // + DFL(분포 → 거리 기댓값). cls 가 품질까지 겸한다
    vfnet,      // star deformable refine (전용 함수)
    reppoints,  // 점 집합 → bbox (전용 함수)
    tood,       // task decomposition + deform sampling (전용 함수)
    centernet,  // heatmap / wh / offset 세 갈래 (앵커 없음, 단일 레벨)
    yolof,      // 단일 레벨 · 암묵 objectness (cls 와 obj 를 로그공간에서 합친다)
    yolo,       // YOLOv3 — 레벨당 bridge conv 하나 + 1x1 예측 conv, 출력이 **한 갈래**다
    cornernet,  // 좌상/우하 코너 — corner pooling(=cummax) + 레벨당 6갈래 출력
    detr,             // transformer encoder+decoder. 출력이 픽셀이 아니라 **query** 다
    conditional_detr, // + 질문을 내용/위치로 쪼개 조회 (참조점은 상수)
    dab_detr,         // + query 가 4D 앵커, 층마다 박스를 고쳐 참조점을 갱신
    deformable_detr,  // multi-scale deformable attention (예측한 소수점 좌표에서 샘플링)
    dino,             // + two-stage: encoder 출력에서 상위 k 개를 골라 query 로 쓴다
    ddq,              // + 층마다 NMS 로 중복 query 를 가린다 → **한 그래프로 못 돈다**
};

// head-conv 구조. 값은 mmdet_to_pt.py 가 <name>.postproc.h 에 굽는다(런타임 파일 없음).
//
// 계열별로 함수를 따로 쓰지 않고 **플래그로 가른다** — RetinaNet/ATSS/PAA/FCOS/GFL 은
// "타워 2개 + 출력 conv 몇 개"라는 뼈대가 같고 곁가지만 다르기 때문이다.
// 뼈대가 아예 다른 계열(vfnet/reppoints/tood)만 전용 함수를 갖는다.
struct anchor_head_cfg {
    head_kind kind = head_kind::anchor;
    int stacked_convs = 4;      // 공유 cls/reg conv 타워 깊이
    // reg 타워만 깊이가 다른 계열이 있다(YOLOF: num_cls_convs=2, num_reg_convs=4).
    // 0 이면 stacked_convs 와 같다고 본다.
    int reg_stacked_convs = 0;
    int feat_channels = 256;
    int num_base = 9;           // location 당 anchor 수
    int num_classes = 80;       // cls_out_channels
    std::string cls_convs_prefix = "bbox_head.cls_convs";  // ModuleList prefix
    std::string reg_convs_prefix = "bbox_head.reg_convs";
    std::string cls_head = "bbox_head.retina_cls";         // 최종 cls conv
    std::string reg_head = "bbox_head.retina_reg";         // 최종 reg conv
    // 분기 conv 앞에 **하나만** 있는 공유 conv(+ReLU). RPNHead 의 `rpn_conv` 가 그렇다.
    // 컨테이너(ModuleList/Sequential)가 아니라 맨 Conv2d 라 타워 탐지에 안 걸린다.
    // 빼먹으면 256→256 이라 shape 이 안 변해 **조용히 틀린다**(rpn 실측 L1 5.56).
    std::string pre_conv;
    bool head_has_norm = false;  // 타워가 ConvModule+GN (ATSS·FCOS·GFL) 인가
    int gn_groups = 32;          // 그때 GroupNorm 그룹 수
    // 타워 가중치를 레벨끼리 공유하지 않는 계열(RetinaSepBNHead — efficientnet·nas_fpn).
    // 이름이 `cls_convs.<레벨>.<단>` 으로 한 겹 깊어진다.
    bool per_level_towers = false;
    // 출력 conv 까지 레벨별인 계열(RTMDetSepBNHead). `rtm_cls.<레벨>` 처럼 한 겹 깊다.
    bool per_level_heads = false;
    // 그 레벨 안에서 또 한 겹 들어가는 계열(SSD: `cls_convs.<레벨>.<단>` 이고 타워가 없다).
    std::string per_level_head_tail;

    // 세 번째 출력 갈래(centerness). 비어 있으면 안 만든다 — RetinaNet 이 그렇다.
    std::string centerness_head;      // 예: "bbox_head.atss_centerness"
    bool centerness_on_reg = true;    // reg_feat 에서 뽑나(false 면 cls_feat)
    // YOLACT 의 mask coefficient 갈래는 tanh 를 거친다(품질 점수가 아니라 계수라서).
    bool ctr_tanh = false;
    // CornerHead 의 embedding 갈래(그룹핑용). config 로 끌 수 있어 플래그로 받는다.
    bool corner_emb = false;
    // CentripetalNet: emb 대신 guiding/centripetal shift 갈래 + DCNv1 feature adaption.
    bool corner_centripetal = false;

    // ── DETR ──────────────────────────────────────────────────────────────
    // 출력이 공간 격자가 아니다 — decoder 층마다 (query, ch) 하나씩 나온다.
    int embed_dims = 256;
    int n_heads = 8;
    int enc_layers = 6;
    int dec_layers = 6;
    int n_points = 4;          // deformable: query 하나가 레벨당 볼 지점 수
    int num_queries = 300;     // DETR 계열의 query 개수(two-stage 는 topk 개수이기도 하다)
    float ddq_iou_thr = 0.8f;  // DDQ 의 distinct query selection NMS 임계

    // 레벨별 learnable scale (mmcv Scale). 비면 안 곱한다.
    // **값은 GGUF 에서 이름으로 읽는다** — 체크포인트를 바꿔도 헤더를 다시 굽지 않아도 된다.
    std::string scales_prefix;        // 예: "bbox_head.scales"

    // bbox_pred 후처리. FCOS 만 쓴다(norm_on_bbox 에 따라 둘 중 하나).
    bool bbox_exp = false;            // norm_on_bbox=false → exp(pred)
    bool bbox_clamp_stride = false;   // norm_on_bbox=true  → clamp(pred,0)*stride
    bool bbox_mul_stride = false;     // RTMDet: 거리 예측이 stride 단위라 그대로 곱한다

    // 타워의 활성화. mmdet ConvModule 의 기본은 ReLU 지만 RTMDet 은 SiLU 다
    // (`act_cfg=dict(type='SiLU')`). 값이 조용히 작아질 뿐 shape 는 같아서 안 드러난다.
    bool head_silu = false;
    // LeakyReLU 를 쓰는 계열(YOLOv3, slope 0.1). >0 이면 이 기울기를 쓴다(silu 보다 우선).
    float head_leaky = 0.0f;

    // GFL 의 DFL. >0 이면 reg_head 출력이 4*(reg_max+1) 채널이고,
    // 조립기가 여기서 기댓값을 내 **4채널 거리**로 바꾼다(디코드 쪽을 단순하게 두려고).
    int reg_max = 0;

    std::vector<float> strides;       // 레벨별 stride (bbox_clamp_stride / gfl 에 필요)
    std::vector<float> reg_denoms;    // VFNet 의 레벨별 정규화 범위(regress_range 상한)
    float center_offset = 0.0f;       // 격자 중심 위치. anchor 계열 0, point 계열 0.5
};

// 조립 결과. ctr 은 centerness 갈래가 없는 계열이면 빈 벡터다.
struct head_outputs {
    std::vector<tensor> cls, box, ctr;
    // 갈래가 셋을 넘는 계열용(CornerHead: tl/br × heat·emb·off = 6). 이름은 덤프 파일명이 된다.
    // ⚠️ 셋에 억지로 우겨넣으면 안 잰 갈래가 생기고 "통과"가 거짓말이 된다.
    std::vector<std::pair<std::string, std::vector<tensor>>> extra;
};

// Everything the runner needs to run one detector.
// Once an architecture is fixed these are constants, so mmdet_to_pt.py emits them as
// mmdet_params() in <name>.postproc.h and they are compiled into the runner. Nothing is read
// at run time.
struct mmdet_cfg {
    anchor_head_cfg head;
    det_params det;
    float img_mean[3] = {0, 0, 0};
    float img_std[3] = {1, 1, 1};
    bool to_rgb = false;
};

// FPN features(레벨별, cwhn) → 레벨별 raw cls_score / bbox_pred(cwhn).
//  · cls_out[l] : ne={num_base*num_classes, feat_w, feat_h, 1}
//  · box_out[l] : ne={num_base*4,           feat_w, feat_h, 1}
// conv 가중치는 모든 레벨이 공유(RetinaHead). detect_anchor 가 기대하는 CWHN flat 레이아웃으로 낸다.
void anchor_head_forward(model_ref m, std::vector<tensor> const& feats,
                         anchor_head_cfg const& c,
                         std::vector<tensor>& cls_out, std::vector<tensor>& box_out);

// VFNet head(distance + star-shaped deformable conv). anchor head 와 달리 offset 을
// bbox_pred 로부터 손코딩으로 만들어 conv_2d_deform 에 넣는다("head 는 ggml 로 안 바뀐다"
// → offset 생성이 자동변환 불가한 부분). 값은 <name>.vfhead.json 에서 읽는다.
struct vfnet_head_cfg {
    int stacked_convs = 3;
    int feat_channels = 256;
    int num_classes = 80;         // cls_out_channels (use_vfl → sigmoid)
    int gn_groups = 32;           // ConvModule 의 GroupNorm 그룹 수
    std::vector<float> strides;   // 레벨별 stride (offset 을 feature scale 로 투영)
    std::vector<float> reg_denoms;// 레벨별 reg_denom (bbox_pred = exp(reg)·reg_denom)
    std::string prefix = "bbox_head";
};

// FPN features(레벨별, cwhn) → 레벨별 raw cls_score / bbox_pred_refine(cwhn, inference 모드).
//  · dcn_base : {18,1,1,1} cwhn — dcn_base_offset(고정 3x3 그리드), 런타임이 값을 채워 넘김.
//  · cls_out[l] : ne={num_classes, w, h, 1},  box_out[l] : ne={4, w, h, 1}
// 가중치는 모든 레벨 공유. offset = star_dcn_offset(bbox_pred) - dcn_base 를 그래프로 조립.
void vfnet_head_forward(model_ref m, std::vector<tensor> const& feats,
                        vfnet_head_cfg const& c, tensor dcn_base,
                        std::vector<tensor>& cls_out, std::vector<tensor>& box_out);

// RepPoints. 점 9개를 예측하고 그 집합을 bbox 로 바꾼다(transform_method='moment').
//  · dcn_base : {18,1,1,1} — dcn_base_offset(고정 3x3 그리드). 런타임이 값을 채운다.
// 두 단계다: init 로 점을 놓고, 그 점을 offset 삼아 deform conv 로 refine 한다.
void reppoints_head_forward(model_ref m, std::vector<tensor> const& feats,
                            anchor_head_cfg const& c, tensor dcn_base,
                            head_outputs& out);

// TOOD. 하나의 inter conv 스택에서 cls/reg 를 갈라내고(task decomposition),
// reg 는 예측한 offset 으로 자기 자신을 다시 샘플링한다(deform sampling).
void tood_head_forward(model_ref m, std::vector<tensor> const& feats,
                       anchor_head_cfg const& c, head_outputs& out);

// DETR. encoder 6층(이미지끼리) + decoder 6층(query 100개) + cls/reg 분기.
// 출력은 **decoder 층마다 하나씩** 이라 out.cls/out.box 의 index 는 FPN 레벨이 아니라 층이다.
//  · cls[i] : ne={num_classes+1, num_queries, 1, 1}
//  · box[i] : ne={4,             num_queries, 1, 1}  (sigmoid 까지)
void detr_head_forward(model_ref m, std::vector<tensor> const& feats,
                       anchor_head_cfg const& c, head_outputs& out);
void conditional_detr_head_forward(model_ref m, std::vector<tensor> const& feats,
                                   anchor_head_cfg const& c, head_outputs& out);
void dab_detr_head_forward(model_ref m, std::vector<tensor> const& feats,
                           anchor_head_cfg const& c, head_outputs& out);
// Deformable DETR. feature 를 레벨별로 펴서 이어붙이고, encoder·decoder 모두
// deformable attention 을 쓴다. 출력은 decoder 층마다 (cls, box).
void deformable_detr_head_forward(model_ref m, std::vector<tensor> const& feats,
                                  anchor_head_cfg const& c, head_outputs& out);
// DINO. encoder 출력에서 proposal 을 만들어 **점수 상위 num_queries 개**를 query 의 초기
// 참조 박스로 쓴다(two-stage). 참조점이 4채널이라 deformable 샘플링이 박스 크기에 비례한다.
void dino_head_forward(model_ref m, std::vector<tensor> const& feats,
                       anchor_head_cfg const& c, head_outputs& out);

// ── DDQ ───────────────────────────────────────────────────────────────────────
// **이 계열만 한 그래프로 못 돈다.** decoder 층마다 `batched_nms` 로 중복 query 를 골라내
// 다음 층의 self-attention 마스크를 만드는데(추론 경로에서도), NMS 는 반복 횟수와 분기가
// 실행 중 값에 따라 정해지는 탐욕 알고리즘이라 정적 그래프로 표현할 수 없다.
// → 조립기를 **토막**으로 나누고 러너가 패스 사이에서 호스트 NMS 를 돌린다.
//   패스 0 : ddq_encode_forward  → 호스트 NMS → query 900개
//   패스 i : ddq_layer_forward   → 호스트 NMS → 다음 층 마스크
struct ddq_stage {
    tensor mem = nullptr, om = nullptr, ecls = nullptr, ecoord = nullptr;  // 패스 0
    tensor qmap = nullptr;   // query 내용의 원천 — `query_map(memory)` 다(om 아님)
    tensor query = nullptr, ref = nullptr, cls = nullptr, box = nullptr;   // 패스 i
};
void ddq_encode_forward(model_ref m, std::vector<tensor> const& feats,
                        anchor_head_cfg const& c, ddq_stage& s);
// mask 는 **가산 마스크**(가릴 자리 −inf, 나머지 0)다 — 호스트가 NMS 결과로 채운다.
// 패스 1~6 에는 백본 그래프가 없으므로 feature 대신 **레벨 치수**만 받는다.
struct msd_level { int64_t w, h, off; };
void ddq_levels(std::vector<tensor> const& feats, std::vector<msd_level>& lv);
void ddq_layer_forward(model_ref m, std::vector<msd_level> const& lv,
                       anchor_head_cfg const& c, int layer_index,
                       tensor mem, tensor query, tensor ref, tensor mask, ddq_stage& s);

// CornerNet. 코너 풀링(= `torch.cummax`)을 쓴다. ggml 에 cummax 도 원소별 max 도 없어
// **항등식과 shift 로 만든다** — 자세한 건 head.cpp 의 emax/cummax_axis 주석.
// 출력이 레벨당 6갈래라 out.extra 를 쓴다(cls=tl_heat, box=br_heat, ctr=tl_off, extra=나머지).
void corner_head_forward(model_ref m, std::vector<tensor> const& feats,
                         anchor_head_cfg const& c, head_outputs& out);

// YOLOv3. 레벨마다 `bridge conv(3x3, BN+LeakyReLU) → 1x1 예측 conv` 하나뿐이고,
// 출력이 **한 갈래**다 — 채널이 `na*(5+num_classes)` 로 box/obj/cls 를 다 담는다.
// 그래서 out.box 는 비어 있고 out.cls 에 pred_map 이 그대로 들어간다.
void yolo_head_forward(model_ref m, std::vector<tensor> const& feats,
                       anchor_head_cfg const& c, head_outputs& out);

// YOLOF. 단일 레벨(dilated encoder)이고, objectness 를 따로 예측해 cls 와 **로그공간에서**
// 합친다 — `cls + obj - log(1 + e^cls + e^obj)`. 그래서 obj 는 별도 출력으로 안 나가고
// cls 안으로 흡수된다(out.ctr 이 비어 있다).
void yolof_head_forward(model_ref m, std::vector<tensor> const& feats,
                        anchor_head_cfg const& c, head_outputs& out);

// CenterNet. 앵커도 타워도 없다 — 단일 레벨에 `Conv3x3 → ReLU → Conv1x1` 세 갈래다.
//  · out.cls = heatmap(num_classes, **sigmoid 까지**) · out.box = wh(2) · out.ctr = offset(2)
void centernet_head_forward(model_ref m, std::vector<tensor> const& feats,
                            anchor_head_cfg const& c, head_outputs& out);

// 계열 무관 진입점. `c.kind` 로 갈라 위 조립기 중 하나를 부른다.
// dcn_base 는 vfnet/reppoints 만 쓴다 — 다른 계열이면 nullptr 로 둔다.
void mmdet_head_forward(model_ref m, std::vector<tensor> const& feats,
                        anchor_head_cfg const& c, tensor dcn_base, head_outputs& out);

}  // namespace visp
