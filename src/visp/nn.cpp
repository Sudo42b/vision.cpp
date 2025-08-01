#include "nn.h"
#include "util/string.h"

namespace visp {

tensor add_bias(model_ref m, tensor x) {
    if (tensor bias = m.find("bias")) {
        x = ggml_add_inplace(m, x, bias);
    }
    return x;
}

tensor linear(model_ref m, tensor x) {
    x = ggml_mul_mat(m, m.weights("weight"), x);
    x = add_bias(m, x);
    return x;
}

tensor layer_norm(model_ref m, tensor x, float eps) {
    x = ggml_norm(m, x, eps);
    x = ggml_mul_inplace(m, x, m.weights("weight"));
    x = ggml_add_inplace(m, x, m.weights("bias"));
    return named(m, x);
}

tensor permute_cwhn_to_whcn(model_ref m, tensor x) {
    return ggml_permute(m, x, 2, 0, 1, 3);
}

tensor permute_whcn_to_cwhn(model_ref m, tensor x) {
    return ggml_permute(m, x, 1, 2, 0, 3);
}

tensor conv_2d(model_ref m, tensor x, int stride, int pad) {
    ASSERT(m.flags & model_build_flag::cwhn);

    tensor weight = m.weights("weight");
    if (weight->ne[1] == 1 && weight->ne[2] == 1 && stride == 1) {
        auto [c, w, h, b] = nelements(x);
        weight = ggml_reshape_2d(m, weight, weight->ne[0], weight->ne[3]);
        x = ggml_reshape_2d(m, x, x->ne[0], w * h * b);
        x = ggml_mul_mat(m, weight, x);
        x = ggml_reshape_4d(m, x, weight->ne[1], w, h, b);

    } else if (m.flags & model_build_flag::conv_2d_direct) {
        weight = permute_cwhn_to_whcn(m, weight);
        x = permute_cwhn_to_whcn(m, x);
        x = ggml_conv_2d(m, weight, x, stride, stride, pad, pad, 1, 1);
        x = permute_whcn_to_cwhn(m, x);

    } else {
        x = permute_cwhn_to_whcn(m, x);
        tensor permuted_weight = permute_cwhn_to_whcn(m, weight);
        tensor cols = ggml_im2col(
            m, permuted_weight, x, stride, stride, pad, pad, 1, 1, true, GGML_TYPE_F32);
        tensor a = ggml_reshape_2d(m, cols, cols->ne[0], cols->ne[1] * cols->ne[2] * cols->ne[3]);
        tensor b = ggml_reshape_2d(
            m, weight, weight->ne[0] * weight->ne[1] * weight->ne[2], weight->ne[3]);
        x = ggml_mul_mat(m, b, a);
        x = ggml_reshape_4d(m, x, weight->ne[3], cols->ne[1], cols->ne[2], cols->ne[3]);
    }
    x = add_bias(m, x);
    return x;
}

tensor conv_2d_depthwise(model_ref m, tensor x, int stride, int pad) {
    ASSERT(m.flags & model_build_flag::cwhn);

    tensor weight = ggml_permute(m, m.weights("weight"), 3, 2, 0, 1);
    x = permute_cwhn_to_whcn(m, x);
    x = ggml_conv_2d_dw_direct(m, weight, x, stride, stride, pad, pad, 1, 1);
    x = permute_whcn_to_cwhn(m, x);

    x = add_bias(m, x);
    return x;
}

tensor conv_transpose_2d(model_ref m, tensor x, int stride) {
    ASSERT(m.flags & model_build_flag::cwhn);

    tensor weight = m.weights("weight");
    if (m.flags & model_build_flag::f16_conv_transpose) {
        // TODO: ggml_conv_transpose_2d_p0 expects fp16 weights (cpu backend)
        weight = ggml_cast(m, weight, GGML_TYPE_F16);
    }
    x = ggml_cont(m, permute_cwhn_to_whcn(m, x));
    x = ggml_conv_transpose_2d_p0(m, weight, x, stride);
    x = ggml_cont(m, permute_whcn_to_cwhn(m, x));
    x = add_bias(m, x);
    return x;
}

tensor conv_2d_deform(
    model_ref m, tensor x, tensor weight, tensor offset, tensor mask, int stride, int pad) {
    ASSERT(m.flags & model_build_flag::cwhn);

    x = permute_cwhn_to_whcn(m, x);
    weight = permute_cwhn_to_whcn(m, weight);
    offset = permute_cwhn_to_whcn(m, offset);
    if (mask) {
        mask = permute_cwhn_to_whcn(m, mask);
    }
    x = ggml_conv_2d_deform(m, weight, x, offset, mask, stride, stride, pad, pad);
    x = permute_whcn_to_cwhn(m, x);
    return x;
}

tensor batch_norm_2d(model_ref m, tensor x) {
    ASSERT(m.flags & model_build_flag::cwhn);
    ASSERT(m.find("running_mean") == nullptr, "Batch norm was not fused");
    ASSERT(m.find("running_var") == nullptr, "Batch norm was not fused");

    x = ggml_mul_inplace(m, x, m.weights("weight"));
    x = ggml_add_inplace(m, x, m.weights("bias"));
    return named(m, x);
}

} // namespace visp