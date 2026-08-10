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
    fcos,       // + bbox 에 scale·clamp·stride (anchor-free 거리 디코드)
    gfl,        // + DFL(분포 → 거리 기댓값). cls 가 품질까지 겸한다
    vfnet,      // star deformable refine (전용 함수)
    reppoints,  // 점 집합 → bbox (전용 함수)
    tood,       // task decomposition + deform sampling (전용 함수)
    centernet,  // heatmap / wh / offset 세 갈래 (앵커 없음, 단일 레벨)
};

// head-conv 구조. 값은 mmdet_to_pt.py 가 <name>.postproc.h 에 굽는다(런타임 파일 없음).
//
// 계열별로 함수를 따로 쓰지 않고 **플래그로 가른다** — RetinaNet/ATSS/PAA/FCOS/GFL 은
// "타워 2개 + 출력 conv 몇 개"라는 뼈대가 같고 곁가지만 다르기 때문이다.
// 뼈대가 아예 다른 계열(vfnet/reppoints/tood)만 전용 함수를 갖는다.
struct anchor_head_cfg {
    head_kind kind = head_kind::anchor;
    int stacked_convs = 4;      // 공유 cls/reg conv 타워 깊이
    int feat_channels = 256;
    int num_base = 9;           // location 당 anchor 수
    int num_classes = 80;       // cls_out_channels
    std::string cls_convs_prefix = "bbox_head.cls_convs";  // ModuleList prefix
    std::string reg_convs_prefix = "bbox_head.reg_convs";
    std::string cls_head = "bbox_head.retina_cls";         // 최종 cls conv
    std::string reg_head = "bbox_head.retina_reg";         // 최종 reg conv
    bool head_has_norm = false;  // 타워가 ConvModule+GN (ATSS·FCOS·GFL) 인가
    int gn_groups = 32;          // 그때 GroupNorm 그룹 수
    // 타워 가중치를 레벨끼리 공유하지 않는 계열(RetinaSepBNHead — efficientnet·nas_fpn).
    // 이름이 `cls_convs.<레벨>.<단>` 으로 한 겹 깊어진다.
    bool per_level_towers = false;
    // 출력 conv 까지 레벨별인 계열(RTMDetSepBNHead). `rtm_cls.<레벨>` 처럼 한 겹 깊다.
    bool per_level_heads = false;

    // 세 번째 출력 갈래(centerness). 비어 있으면 안 만든다 — RetinaNet 이 그렇다.
    std::string centerness_head;      // 예: "bbox_head.atss_centerness"
    bool centerness_on_reg = true;    // reg_feat 에서 뽑나(false 면 cls_feat)

    // 레벨별 learnable scale (mmcv Scale). 비면 안 곱한다.
    // **값은 GGUF 에서 이름으로 읽는다** — 체크포인트를 바꿔도 헤더를 다시 굽지 않아도 된다.
    std::string scales_prefix;        // 예: "bbox_head.scales"

    // bbox_pred 후처리. FCOS 만 쓴다(norm_on_bbox 에 따라 둘 중 하나).
    bool bbox_exp = false;            // norm_on_bbox=false → exp(pred)
    bool bbox_clamp_stride = false;   // norm_on_bbox=true  → clamp(pred,0)*stride

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

// CenterNet. 앵커도 타워도 없다 — 단일 레벨에 `Conv3x3 → ReLU → Conv1x1` 세 갈래다.
//  · out.cls = heatmap(num_classes, **sigmoid 까지**) · out.box = wh(2) · out.ctr = offset(2)
void centernet_head_forward(model_ref m, std::vector<tensor> const& feats,
                            anchor_head_cfg const& c, head_outputs& out);

// 계열 무관 진입점. `c.kind` 로 갈라 위 조립기 중 하나를 부른다.
// dcn_base 는 vfnet/reppoints 만 쓴다 — 다른 계열이면 nullptr 로 둔다.
void mmdet_head_forward(model_ref m, std::vector<tensor> const& feats,
                        anchor_head_cfg const& c, tensor dcn_base, head_outputs& out);

}  // namespace visp
