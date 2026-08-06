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

// anchor head(RetinaNet/ATSS 등) 의 head-conv 구조. 값은 <name>.postproc.json 에서 읽는다.
struct anchor_head_cfg {
    int stacked_convs = 4;      // 공유 cls/reg conv 타워 깊이
    int feat_channels = 256;
    int num_base = 9;           // location 당 anchor 수
    int num_classes = 80;       // cls_out_channels
    std::string cls_convs_prefix = "bbox_head.cls_convs";  // ModuleList prefix
    std::string reg_convs_prefix = "bbox_head.reg_convs";
    std::string cls_head = "bbox_head.retina_cls";         // 최종 cls conv
    std::string reg_head = "bbox_head.retina_reg";         // 최종 reg conv
    bool head_has_norm = false;  // 타워에 norm(GN 등) — 이번 PoC(RetinaNet)=false
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

}  // namespace visp
