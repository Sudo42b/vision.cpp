#include "nn.h"
#include "util/string.h"

namespace visp {

tensor linear(model_ref m, tensor x) {
    x = ggml_mul_mat(m, m.weights("weight"), x);
    if (tensor bias = m.find("bias")) {
        x = ggml_add(m, x, bias);
    }
    return x;
}

tensor layer_norm(model_ref m, tensor x, float eps) {
    x = ggml_norm(m, x, eps);
    x = ggml_mul(m, x, m.weights("weight"));
    x = ggml_add(m, x, m.weights("bias"));
    return named(m, x);
}

tensor permute_cwhn_to_whcn(model_ref m, tensor x) {
    return ggml_permute(m, x, 2, 0, 1, 3);
}

tensor permute_whcn_to_cwhn(model_ref m, tensor x) {
    return ggml_permute(m, x, 1, 2, 0, 3);
}

std::array<int64_t, 4> nelements_whcn(model_ref const& m, tensor t) {
    auto ne = nelements(t);
    return (m.flags & model_build_flag::cwhn) ? std::array{ne[1], ne[2], ne[0], ne[3]} : ne;
}

tensor cwhn_to_contiguous_2d(model_ref m, tensor x) {
    if (m.flags & model_build_flag::cwhn) {
        return x; // preferred 2D layout is CWHN too
    }
    return ggml_cont(m, permute_cwhn_to_whcn(m, x));
}

tensor whcn_to_contiguous_2d(model_ref m, tensor x) {
    if (m.flags & model_build_flag::cwhn) {
        return ggml_cont(m, permute_whcn_to_cwhn(m, x));
    }
    return x;
}

tensor contiguous_2d_to_cwhn(model_ref m, tensor x) {
    if (m.flags & model_build_flag::cwhn) {
        return x; // x is already CWHN
    }
    return ggml_cont(m, permute_whcn_to_cwhn(m, x));
}

tensor contiguous_2d_to_whcn(model_ref m, tensor x) {
    if (m.flags & model_build_flag::cwhn) {
        return ggml_cont(m, permute_cwhn_to_whcn(m, x));
    }
    return x;
}

tensor space_to_depth_quad(model_ref m, tensor x, int sw, int sh) {
    int64_t W = x->ne[0], H = x->ne[1], C = x->ne[2];
    GGML_ASSERT(x->ne[3] == 1 && W % 2 == 0 && H % 2 == 0);
    // W(ne0) 짝/홀: [W,H,C,1] → [2, W/2, H, C] 후 ne0=1@sw 선택 → [W/2,H,C,1]
    tensor r = ggml_reshape_4d(m, ggml_cont(m, x), 2, W / 2, H, C);
    tensor v = ggml_view_4d(m, r, 1, W / 2, H, C, r->nb[1], r->nb[2], r->nb[3],
                            (size_t) sw * r->nb[0]);
    v = ggml_reshape_4d(m, ggml_cont(m, v), W / 2, H, C, 1);
    // H(ne1) 짝/홀: [W/2,H,C,1] → [W/2, 2, H/2, C] 후 ne1=1@sh 선택 → [W/2,H/2,C,1]
    tensor r2 = ggml_reshape_4d(m, v, W / 2, 2, H / 2, C);
    tensor v2 = ggml_view_4d(m, r2, W / 2, 1, H / 2, C, r2->nb[1], r2->nb[2], r2->nb[3],
                             (size_t) sh * r2->nb[1]);
    return ggml_reshape_4d(m, ggml_cont(m, v2), W / 2, H / 2, C, 1);
}

tensor add_bias_2d(model_ref m, tensor x) {
    if (tensor bias = m.find("bias")) {
        if (!(m.flags & model_build_flag::cwhn)) {
            bias = ggml_reshape_4d(m, bias, 1, 1, bias->ne[0], 1);
        }
        x = ggml_add(m, x, bias);
    }
    return x;
}

// conv_2d 의 본체 — weight 를 인자로 받고 bias 는 붙이지 않는다.
// conv_2d / conv_2d_wt 가 공유한다. dilation 기본값 1 은 기존 동작과 동일하다.
static tensor conv_2d_impl(model_ref m, tensor x, tensor weight, int stride, int pad,
                           int dilation) {
    if (m.flags & model_build_flag::cwhn) {
        if (weight->ne[1] == 1 && weight->ne[2] == 1 && stride == 1 && dilation == 1) {
            auto [c, w, h, b] = nelements(x);
            weight = ggml_reshape_2d(m, weight, weight->ne[0], weight->ne[3]);
            x = ggml_reshape_2d(m, x, x->ne[0], w * h * b);
            x = ggml_mul_mat(m, weight, x);
            x = ggml_reshape_4d(m, x, weight->ne[1], w, h, b);

        } else if (m.flags & model_build_flag::conv_2d_direct_cwhn) {
            weight = permute_cwhn_to_whcn(m, weight);
            x = permute_cwhn_to_whcn(m, x);
            x = ggml_conv_2d_direct(m, weight, x, stride, stride, pad, pad, dilation, dilation);
            x = permute_whcn_to_cwhn(m, x);

        } else {
            weight = ggml_cont(m, permute_cwhn_to_whcn(m, weight));
            x = ggml_cont(m, permute_cwhn_to_whcn(m, x));
            x = ggml_conv_2d(m, weight, x, stride, stride, pad, pad, dilation, dilation);
            x = ggml_cont(m, permute_whcn_to_cwhn(m, x));
        }
    } else { // WHCN layout
        x = ggml_conv_2d_direct(m, weight, x, stride, stride, pad, pad, dilation, dilation);
    }
    return x;
}

tensor conv_2d(model_ref m, tensor x, int stride, int pad, int dilation) {
    x = conv_2d_impl(m, x, m.weights("weight"), stride, pad, dilation);
    return add_bias_2d(m, x);
}

tensor conv_2d_wt(model_ref m, tensor x, tensor weight, tensor bias, int stride, int pad,
                  int dilation) {
    x = conv_2d_impl(m, x, weight, stride, pad, dilation);
    if (bias) {
        // WHCN 은 채널이 ne[2] 라 broadcast 를 위해 [1,1,C,1] 로 편다 (add_bias_2d 와 같은 규칙).
        if (!(m.flags & model_build_flag::cwhn)) {
            bias = ggml_reshape_4d(m, bias, 1, 1, bias->ne[0], 1);
        }
        x = ggml_add_inplace(m, x, bias);
    }
    return x;
}

tensor conv_2d_grouped(model_ref m, tensor x, int stride, int pad, int dilation, int groups) {
    if (groups <= 1) {
        return conv_2d(m, x, stride, pad, dilation);
    }
    tensor weight = m.weights("weight");
    bool cwhn = bool(m.flags & model_build_flag::cwhn);
    if (cwhn) {
        weight = ggml_cont(m, permute_cwhn_to_whcn(m, weight));
        x = ggml_cont(m, permute_cwhn_to_whcn(m, x));
    } else {
        x = ggml_cont(m, x);
    }
    // whcn: weight[kw,kh,Cin/g,Cout], x[W,H,Cin,N]
    int64_t cin_g = weight->ne[2];
    int64_t cout = weight->ne[3];
    int64_t cout_g = cout / groups;
    int64_t n = x->ne[3];
    tensor y = nullptr;
    for (int g = 0; g < groups; g++) {
        tensor wg = ggml_cont(m, ggml_view_4d(m, weight,
            weight->ne[0], weight->ne[1], cin_g, cout_g,
            weight->nb[1], weight->nb[2], weight->nb[3],
            (size_t) g * cout_g * weight->nb[3]));
        tensor xg = ggml_cont(m, ggml_view_4d(m, x,
            x->ne[0], x->ne[1], cin_g, n,
            x->nb[1], x->nb[2], x->nb[3],
            (size_t) g * cin_g * x->nb[2]));
        tensor yg = ggml_conv_2d(m, wg, xg, stride, stride, pad, pad, dilation, dilation);
        y = y ? ggml_concat(m, y, yg, 2) : yg;  // 채널축(ne2) concat
    }
    if (cwhn) {
        y = ggml_cont(m, permute_whcn_to_cwhn(m, y));
    }
    return add_bias_2d(m, y);
}

tensor conv_2d_depthwise(model_ref m, tensor x, int stride, int pad) {
    tensor weight = m.weights("weight");

    if (m.flags & model_build_flag::cwhn) {
        weight = ggml_permute(m, weight, 3, 2, 0, 1);
        x = permute_cwhn_to_whcn(m, x);
        x = ggml_conv_2d_dw_direct(m, weight, x, stride, stride, pad, pad, 1, 1);
        x = permute_whcn_to_cwhn(m, x);
    } else {
        x = ggml_conv_2d_dw_direct(m, weight, x, stride, stride, pad, pad, 1, 1);
    }
    x = add_bias_2d(m, x);
    return x;
}

tensor conv_transpose_2d(model_ref m, tensor x, int stride) {
    tensor weight = m.weights("weight");
    if (m.flags & model_build_flag::cwhn) {
        x = ggml_cont(m, permute_cwhn_to_whcn(m, x));
    }
    x = ggml_conv_transpose_2d_p0(m, weight, x, stride);

    if (m.flags & model_build_flag::cwhn) {
        x = ggml_cont(m, permute_whcn_to_cwhn(m, x));
    }
    x = add_bias_2d(m, x);
    return x;
}

tensor conv_2d_deform(
    model_ref m, tensor x, tensor weight, tensor offset, tensor mask, int stride, int pad) {

    if (m.flags & model_build_flag::cwhn) {
        x = permute_cwhn_to_whcn(m, x);
        weight = permute_cwhn_to_whcn(m, weight);
        offset = permute_cwhn_to_whcn(m, offset);
        if (mask) {
            mask = permute_cwhn_to_whcn(m, mask);
        }
    }
    x = ggml_conv_2d_deform(m, weight, x, offset, mask, stride, stride, pad, pad);

    if (m.flags & model_build_flag::cwhn) {
        x = permute_whcn_to_cwhn(m, x);
    }
    return x;
}

tensor batch_norm_2d(model_ref m, tensor x) {
    // Batch norm is expected to be have been fused into mul+add. See convert.py
    ASSERT(m.find("running_mean") == nullptr, "Batch norm was not fused");
    ASSERT(m.find("running_var") == nullptr, "Batch norm was not fused");

    tensor weight = m.weights("weight");
    tensor bias = m.weights("bias");
    if (!(m.flags & model_build_flag::cwhn)) { // WHCN layout
        weight = ggml_reshape_4d(m, weight, 1, 1, weight->ne[0], 1);
        bias = ggml_reshape_4d(m, bias, 1, 1, bias->ne[0], 1);
    }
    x = ggml_mul(m, x, weight);
    x = ggml_add(m, x, bias);
    return named(m, x);
}

tensor patch_embed(model_ref m, tensor x, int patch_size) {
    ASSERT(x->ne[1] % patch_size == 0 && x->ne[2] % patch_size == 0);
    char const* proj = m.find("proj.weight") ? "proj" : "projection";

    m.flags |= model_build_flag::cwhn;
    x = conv_2d(m[proj], x, patch_size);

    if (m.find("norm.weight")) {
        auto [c, w, h, b] = nelements(x);
        x = ggml_reshape_3d(m, x, c, w * h, b);
        x = layer_norm(m["norm"], x);
        x = ggml_reshape_4d(m, x, c, w, h, b);
    }
    return named(m, x);
}

attention_qkv split_qkv(model_ref m, tensor x, int n_heads, int split_dim) {
    auto [c, n, b, _] = nelements(x);

    tensor qkv = linear(m, x);
    switch (split_dim) {
        case 1:
            qkv = ggml_reshape_4d(m, qkv, c / n_heads, 3, n_heads * n, b);
            qkv = ggml_cont(m, ggml_permute(m, qkv, 0, 3, 1, 2));
            break;
        case 2:
            qkv = ggml_reshape_4d(m, qkv, c / n_heads, n_heads, 3, n * b);
            qkv = ggml_cont(m, ggml_permute(m, qkv, 0, 1, 3, 2));
            break;
        default: ASSERT(false, "Unsupported split_dim");
    }

    auto split = [&](tensor t, size_t index) mutable {
        t = slice(m, t, {}, {}, {}, index);
        t = ggml_reshape_4d(m, t, c / n_heads, n_heads, n, b);
        return t;
    };

    tensor q = split(qkv, 0);
    tensor k = split(qkv, 1);
    tensor v = split(qkv, 2);
    return {q, k, v};
}

tensor attention(
    model_ref m, tensor q, tensor k, tensor v, tensor mask, float scale, model_ref m_out) {

    q = ggml_permute(m, q, 0, 2, 1, 3);
    k = ggml_permute(m, k, 0, 2, 1, 3);

    tensor x = nullptr;
    if (m.flags & model_build_flag::flash_attention) {
        v = ggml_permute(m, v, 0, 2, 1, 3);

        k = ggml_cast(m, k, GGML_TYPE_F16);
        v = ggml_cast(m, v, GGML_TYPE_F16);
        if (mask && mask->type != GGML_TYPE_F16) {
            mask = ggml_cast(m, mask, GGML_TYPE_F16);
        }

        x = ggml_flash_attn_ext(m, q, k, v, mask, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(x, GGML_PREC_F32);

    } else {
        v = ggml_cont(m, ggml_permute(m, v, 1, 2, 0, 3));

        tensor attn = ggml_mul_mat(m, k, q);
        attn = ggml_soft_max_ext(m, attn, mask, scale, 0.0f);
        x = ggml_mul_mat(m, v, attn);

        x = ggml_cont(m, ggml_permute(m, x, 0, 2, 1, 3));
    }

    // [head_dim, n_heads, n_patches, batch] -> [embed_dim, n_patches, batch]
    x = ggml_reshape_3d(m, x, x->ne[0] * x->ne[1], x->ne[2], x->ne[3]);
    x = linear(m_out, x);

    return named(m, x);
}

} // namespace visp