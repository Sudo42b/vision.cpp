#include "head.h"

#include "visp/nn.h"

#include <ggml.h>

#include <string>

namespace visp {

// features 는 인터프리터 출력(cwhn). 인터프리터 내부 conv 는 contiguous_2d 레이아웃에서 도므로
// (graph_interpret: cwhn_to_contiguous_2d → conv... → contiguous_2d_to_cwhn), head 도 동일하게
// cwhn→contiguous_2d 로 되돌려 conv 타워를 태우고, 결과를 다시 cwhn 으로 낸다(detect_anchor 규약).
void anchor_head_forward(model_ref m, std::vector<tensor> const& feats,
                         anchor_head_cfg const& c,
                         std::vector<tensor>& cls_out, std::vector<tensor>& box_out) {
    for (size_t l = 0; l < feats.size(); ++l) {
        tensor f = cwhn_to_contiguous_2d(m, feats[l]);

        // cls 타워: stacked_convs × (conv3x3 + relu), 가중치 레벨 공유
        tensor cc = f;
        for (int i = 0; i < c.stacked_convs; ++i) {
            std::string p = c.cls_convs_prefix + "." + std::to_string(i) + ".conv";
            cc = ggml_relu(m, conv_2d(m[p.c_str()], cc, 1, 1));
        }
        tensor cls = conv_2d(m[c.cls_head.c_str()], cc, 1, 1);  // → num_base*num_classes 채널

        // reg 타워
        tensor rr = f;
        for (int i = 0; i < c.stacked_convs; ++i) {
            std::string p = c.reg_convs_prefix + "." + std::to_string(i) + ".conv";
            rr = ggml_relu(m, conv_2d(m[p.c_str()], rr, 1, 1));
        }
        tensor box = conv_2d(m[c.reg_head.c_str()], rr, 1, 1);  // → num_base*4 채널

        cls = contiguous_2d_to_cwhn(m, cls);
        box = contiguous_2d_to_cwhn(m, box);
        ggml_format_name(cls, "cls_%zu", l);
        ggml_format_name(box, "box_%zu", l);
        cls_out.push_back(cls);
        box_out.push_back(box);
    }
}

// ── VFNet ─────────────────────────────────────────────────────────────────────
// runner 는 whcn 네이티브(가중치 {KW,KH,Cin,Cout}). head 내부도 contiguous_2d(=whcn) 에서 돈다
// (anchor_head_forward 와 동일 규약). 채널축은 ne[2]. offset/deform 도 whcn 로 조립.
//
// GroupNorm + affine (γ/β). 코어 라이브러리(nn.cpp)에 group_norm 이 없어도 self-contained 하도록
// 여기 로컬로 둔다(교수님 지시: mmdet 은 tools 에서 자족, 코어 무수정). 채널축: whcn=[1,1,C,1], cwhn=ne[0].
static tensor group_norm_affine(model_ref m, tensor x, int groups, float eps = 1e-5f) {
    x = ggml_group_norm(m, x, groups, eps);
    bool whcn = !(m.flags & model_build_flag::cwhn);
    auto rs = [&](tensor t) { return whcn ? ggml_reshape_4d(m, t, 1, 1, t->ne[0], 1) : t; };
    if (tensor weight = m.find("weight")) x = ggml_mul(m, x, rs(weight));
    if (tensor bias = m.find("bias")) x = ggml_add(m, x, rs(bias));
    return x;
}

// ConvModule(conv 3x3 pad1, bias 없음 + GroupNorm + ReLU).
static tensor conv_gn_relu(model_ref m, tensor x, const std::string& p, int gn_groups) {
    x = conv_2d(m[(p + ".conv").c_str()], x, 1, 1);
    x = group_norm_affine(m[(p + ".gn").c_str()], x, gn_groups);
    return ggml_relu(m, x);
}

// star_dcn_offset (mmdet VFNetHead) 의 그래프 버전 (whcn: bbox_pred {W,H,4,1}, 채널=ne[2]).
//   ch: 0=x1 1=y1 2=x2 3=y2. /stride 후 18ch offset 조립 → base 뺌.
//   추론 모드라 gradient_mul 무관(detach==pred). ggml deform 커널은 knl-pad(=base)를 내부에서
//   더하므로 mmdet 과 동일하게 (star - dcn_base) 를 넘긴다. 채널순서 [2i]=off_y [2i+1]=off_x.
static tensor star_dcn_offset(model_ref m, tensor bp, float stride, tensor dcn_base) {
    const int64_t W = bp->ne[0], H = bp->ne[1], N = bp->ne[3];
    auto ch = [&](int c, float s) -> tensor {           // 채널 c 슬라이스(whcn ne2) × s
        tensor v = ggml_view_4d(m, bp, W, H, 1, N, bp->nb[1], bp->nb[2], bp->nb[3],
                                (size_t)c * bp->nb[2]);
        return ggml_scale(m, ggml_cont(m, v), s);
    };
    const float is = 1.0f / stride;
    tensor ny1 = ch(1, -is);   // -y1
    tensor nx1 = ch(0, -is);   // -x1
    tensor px2 = ch(2, is);    //  x2
    tensor py2 = ch(3, is);    //  y2
    tensor z = ch(0, 0.0f);    //  0
    // 채널 순서 = mmdet star_dcn_offset 필드(0..17)
    tensor chans[18] = {ny1, nx1, ny1, z, ny1, px2, z, nx1, z,
                        z, z, px2, py2, nx1, py2, z, py2, px2};
    tensor off = chans[0];
    for (int i = 1; i < 18; ++i) {
        off = ggml_concat(m, off, chans[i], 2);         // 채널축(ne2) concat → {W,H,18,1}
    }
    return ggml_sub(m, off, dcn_base);                  // dcn_base {1,1,18,1} broadcast
}

void vfnet_head_forward(model_ref m, std::vector<tensor> const& feats,
                        vfnet_head_cfg const& c, tensor dcn_base,
                        std::vector<tensor>& cls_out, std::vector<tensor>& box_out) {
    const std::string H = c.prefix;
    for (size_t l = 0; l < feats.size(); ++l) {
        const float stride = c.strides[l];
        const float reg_denom = c.reg_denoms[l];
        tensor x = cwhn_to_contiguous_2d(m, feats[l]);  // → whcn (whcn-mode)

        // cls / reg 타워 (stacked ConvModule)
        tensor cls_feat = x, reg_feat = x;
        for (int i = 0; i < c.stacked_convs; ++i) {
            cls_feat = conv_gn_relu(m, cls_feat, H + ".cls_convs." + std::to_string(i), c.gn_groups);
            reg_feat = conv_gn_relu(m, reg_feat, H + ".reg_convs." + std::to_string(i), c.gn_groups);
        }

        // 초기 bbox_pred = exp(vfnet_reg(vfnet_reg_conv(reg_feat))) · reg_denom  (scale=1.0)
        tensor reg_init = conv_gn_relu(m, reg_feat, H + ".vfnet_reg_conv", c.gn_groups);
        tensor bbox_pred = conv_2d(m[(H + ".vfnet_reg").c_str()], reg_init, 1, 1);
        bbox_pred = ggml_scale(m, ggml_exp(m, bbox_pred), reg_denom);

        // star deformable offset
        tensor offset = star_dcn_offset(m, bbox_pred, stride, dcn_base);

        // refine: reg_feat 를 deform conv → exp·bbox_pred
        tensor w_reg_dcn = m[(H + ".vfnet_reg_refine_dconv").c_str()].weights("weight");
        tensor reg_ref = ggml_relu(m, conv_2d_deform(m, reg_feat, w_reg_dcn, offset, nullptr, 1, 1));
        tensor box = conv_2d(m[(H + ".vfnet_reg_refine").c_str()], reg_ref, 1, 1);
        box = ggml_mul(m, ggml_exp(m, box), bbox_pred);  // scale_refine=1.0

        // iou-aware cls: cls_feat 를 같은 offset 으로 deform conv → cls conv
        tensor w_cls_dcn = m[(H + ".vfnet_cls_dconv").c_str()].weights("weight");
        tensor cls_ref = ggml_relu(m, conv_2d_deform(m, cls_feat, w_cls_dcn, offset, nullptr, 1, 1));
        tensor cls = conv_2d(m[(H + ".vfnet_cls").c_str()], cls_ref, 1, 1);

        cls = contiguous_2d_to_cwhn(m, cls);            // detect 규약(cwhn)
        box = contiguous_2d_to_cwhn(m, box);
        ggml_format_name(cls, "cls_%zu", l);
        ggml_format_name(box, "box_%zu", l);
        cls_out.push_back(cls);
        box_out.push_back(box);
    }
}

}  // namespace visp
