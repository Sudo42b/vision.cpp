#include "visp/arch/mmdet/head.h"

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

}  // namespace visp
