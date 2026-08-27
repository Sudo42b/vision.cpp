#include <vector>
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
static tensor conv_2d_impl(model_ref m, tensor x, tensor weight, int sw, int sh, int pw,
                           int ph, int dw, int dh) {
    // ⚠️ 축별 인자다. ggml 은 원래 축별로 받는데(`s0`=W, `s1`=H) 여기서 하나로 묶어
    //    넘기고 있었다 — 비대칭 conv(ERFNet 의 3x1/1x3)이 **조용히 대칭으로** 돌았다.
    if (m.flags & model_build_flag::cwhn) {
        // 1x1 fast path 는 padding 을 아예 안 태운다 → pad 가 0 일 때만 쓸 수 있다.
        if (weight->ne[1] == 1 && weight->ne[2] == 1 && sw == 1 && sh == 1 && dw == 1 &&
            dh == 1 && pw == 0 && ph == 0) {
            auto [c, w, h, b] = nelements(x);
            weight = ggml_reshape_2d(m, weight, weight->ne[0], weight->ne[3]);
            x = ggml_reshape_2d(m, x, x->ne[0], w * h * b);
            x = ggml_mul_mat(m, weight, x);
            x = ggml_reshape_4d(m, x, weight->ne[1], w, h, b);

        } else if (m.flags & model_build_flag::conv_2d_direct_cwhn) {
            weight = permute_cwhn_to_whcn(m, weight);
            x = permute_cwhn_to_whcn(m, x);
            x = ggml_conv_2d_direct(m, weight, x, sw, sh, pw, ph, dw, dh);
            x = permute_whcn_to_cwhn(m, x);

        } else {
            weight = ggml_cont(m, permute_cwhn_to_whcn(m, weight));
            x = ggml_cont(m, permute_cwhn_to_whcn(m, x));
            x = ggml_conv_2d(m, weight, x, sw, sh, pw, ph, dw, dh);
            x = ggml_cont(m, permute_whcn_to_cwhn(m, x));
        }
    } else { // WHCN layout
        x = ggml_conv_2d_direct(m, weight, x, sw, sh, pw, ph, dw, dh);
    }
    return x;
}

tensor conv_2d(model_ref m, tensor x, int stride, int pad, int dilation) {
    x = conv_2d_impl(m, x, m.weights("weight"), stride, stride, pad, pad, dilation, dilation);
    return add_bias_2d(m, x);
}

// 축별 conv — 커널·padding·dilation 이 H 와 W 에서 다른 경우(ERFNet 의 3x1/1x3 분리 conv).
// 인자 순서는 ggml 과 같은 **W 먼저**다(`ggml_conv_2d(s0=W, s1=H, …)`).
tensor conv_2d_ex(model_ref m, tensor x, int stride_w, int stride_h, int pad_w, int pad_h,
                  int dilation_w, int dilation_h) {
    x = conv_2d_impl(m, x, m.weights("weight"), stride_w, stride_h, pad_w, pad_h, dilation_w,
                     dilation_h);
    return add_bias_2d(m, x);
}

tensor conv_2d_wt(model_ref m, tensor x, tensor weight, tensor bias, int stride, int pad,
                  int dilation) {
    x = conv_2d_impl(m, x, weight, stride, stride, pad, pad, dilation, dilation);
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

// `dilation` 은 torch 와 같은 뜻이다. ⚠️ **안 받으면 조용히 커진다** — 호출자가 padding 만
// 넘기고 dilation 을 못 넘기면 출력이 `in + 2p - (k-1)` 이 되어 입력보다 크게 나오고,
// 그 크기는 **다음 op(concat)에서** 어긋나 죽는다(mmseg DeepLabV3+ 의 depthwise ASPP:
// dilation 12/24/36 인데 64x64 가 86/110/134 로 커졌다).
tensor conv_2d_depthwise(model_ref m, tensor x, int stride, int pad, int dilation) {
    tensor weight = m.weights("weight");

    if (m.flags & model_build_flag::cwhn) {
        weight = ggml_permute(m, weight, 3, 2, 0, 1);
        x = permute_cwhn_to_whcn(m, x);
        x = ggml_conv_2d_dw_direct(m, weight, x, stride, stride, pad, pad, dilation, dilation);
        x = permute_whcn_to_cwhn(m, x);
    } else {
        x = ggml_conv_2d_dw_direct(m, weight, x, stride, stride, pad, pad, dilation, dilation);
    }
    x = add_bias_2d(m, x);
    return x;
}

tensor conv_transpose_2d(model_ref m, tensor x, int stride, int pad, int groups,
                         int output_padding) {
    tensor weight = m.weights("weight");
    // ⚠️ **커널은 F16 이어야 한다.** `ggml_compute_forward_conv_transpose_2d` 는
    //    `GGML_ASSERT(src0->type == GGML_TYPE_F16)` 로 시작한다(ggml-cpu/ops.cpp).
    //    `model_transfer` 에 `preferred_float_type()` 을 주면 CPU 백엔드에서 F32 로 올라와
    //    **로드는 되고 실행에서 죽는다.** 호출자가 알아야 할 사정이 아니므로 여기서 맞춘다
    //    (attention 의 k/v 캐스팅과 같은 규약).
    if (weight->type != GGML_TYPE_F16) {
        weight = ggml_cast(m, weight, GGML_TYPE_F16);
    }
    if (m.flags & model_build_flag::cwhn) {
        x = ggml_cont(m, permute_cwhn_to_whcn(m, x));
    }
    if (groups <= 1) {
        x = ggml_conv_transpose_2d_p0(m, weight, x, stride);
    } else {
        // ⚠️ **ggml 에는 grouped conv_transpose 가 없다.** 그냥 통과시키면 groups 가
        //    조용히 무시되어 채널이 섞인 채 돈다(크래시 없음). 그룹마다 커널과 입력을
        //    잘라 따로 돌리고 채널 축으로 이어붙인다 — 정의 그대로다.
        //    커널 ne = [KW, KH, OC/g, IC] 이므로 IC 는 ne3, 입력 채널은 ne2 다.
        const int64_t icg = weight->ne[3] / groups;   // 그룹당 입력 채널(커널 쪽)
        const int64_t xcg = x->ne[2] / groups;        // 그룹당 입력 채널(피처 쪽)
        GGML_ASSERT(icg > 0 && xcg > 0);
        tensor acc = nullptr;
        for (int g = 0; g < groups; ++g) {
            tensor wg = ggml_cont(m, ggml_view_4d(
                m, weight, weight->ne[0], weight->ne[1], weight->ne[2], icg,
                weight->nb[1], weight->nb[2], weight->nb[3], (size_t)g * icg * weight->nb[3]));
            tensor xg = ggml_cont(m, ggml_view_4d(
                m, x, x->ne[0], x->ne[1], xcg, x->ne[3],
                x->nb[1], x->nb[2], x->nb[3], (size_t)g * xcg * x->nb[2]));
            tensor og = ggml_conv_transpose_2d_p0(m, wg, xg, stride);
            acc = acc ? ggml_concat(m, acc, og, 2) : og;
        }
        x = acc;
    }

    // ⚠️ **`ggml_conv_transpose_2d_p0` 은 이름 그대로 padding 0 전용이다.**
    //    transposed conv 의 padding p 는 "출력 가장자리를 p 픽셀씩 버린다" 와 같으므로
    //    p0 로 크게 뽑아 놓고 여기서 잘라낸다. 안 자르면 출력이 2p 만큼 크고, 그 크기는
    //    **다음 op 에서** 어긋나 죽는다 — 크래시 지점이 원인 지점이 아니다
    //    (centernet 실측: deconv 가 34x34 를 내고 다음 DCN 의 offset 32 와 안 맞았다).
    //    mmdet 의 deconv 는 전부 대칭 padding 이라 양쪽을 같은 값으로 자르면 된다.
    // ⚠️ **`output_padding` 은 「덜 자른다」로 처리한다.** torch 는 출력 오른쪽·아래를
    //    그만큼 더 갖는데, p0 버퍼는 자르기 전이라 그 자리를 대개 이미 갖고 있다.
    //    torch 출력 인덱스 j 는 버퍼 인덱스 j+pad 다. 버퍼가 모자라는 만큼
    //    (output_padding > pad) 은 실제로 기여가 0 인 자리라 0 으로 채운다.
    //    안 처리하면 출력이 output_padding 만큼 작고, 뒤의 residual add 가
    //    `ggml_can_repeat` 로 죽는다 — 크래시 지점이 원인 지점이 아니다
    //    (erfnet UpsamplerBlock: stride 2 · pad 1 · output_padding 1 → 2n-1 vs 2n).
    if (pad > 0 || output_padding > 0) {
        const int64_t bw = x->ne[0], bh = x->ne[1];
        const int64_t ow = bw - 2 * pad + output_padding;
        const int64_t oh = bh - 2 * pad + output_padding;
        GGML_ASSERT(ow > 0 && oh > 0);
        const int64_t vw = ow < bw - pad ? ow : bw - pad;   // 버퍼에서 꺼낼 수 있는 만큼
        const int64_t vh = oh < bh - pad ? oh : bh - pad;
        if (pad > 0 || vw != bw || vh != bh) {
            x = ggml_cont(m, ggml_view_4d(m, x, vw, vh, x->ne[2], x->ne[3],
                                          x->nb[1], x->nb[2], x->nb[3],
                                          pad * x->nb[0] + pad * x->nb[1]));
        }
        if (vw < ow || vh < oh) {
            x = ggml_pad(m, x, (int) (ow - vw), (int) (oh - vh), 0, 0);
        }
    }
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

namespace {

// ggml 축 `dim` 에서 인덱스 `i` 한 칸만 잘라 **연속** 텐서로 만든다.
// `ggml_concat` 은 비연속 src 도 받지만, view 를 그대로 넘기면 stride 해석이 축마다
// 달라져 디버깅이 어렵다 — 한 칸짜리라 복사 비용이 무시할 만하므로 cont 로 고정한다.
tensor pad_reflect_slice(model_ref m, tensor x, int dim, int64_t i) {
    int64_t ne[4] = {x->ne[0], x->ne[1], x->ne[2], x->ne[3]};
    ne[dim] = 1;
    return ggml_cont(m, ggml_view_4d(m, x, ne[0], ne[1], ne[2], ne[3],
                                     x->nb[1], x->nb[2], x->nb[3],
                                     (size_t)i * x->nb[dim]));
}

// 한 축만 거울 반사. torch 규약: out[k] = x[lo-k] (k<lo), out[n+lo+k] = x[n-2-k].
// **경계 자신은 복제하지 않는다** — 그래서 인덱스가 1 부터 시작하고 n-2 에서 내려간다.
tensor pad_reflect_axis(model_ref m, tensor x, int dim, int lo, int hi) {
    if (lo <= 0 && hi <= 0) {
        return x;
    }
    int64_t n = x->ne[dim];
    ASSERT(lo < n && hi < n, "reflect 패딩이 축 길이보다 크다");
    std::vector<tensor> parts;
    for (int k = lo; k >= 1; --k) {
        parts.push_back(pad_reflect_slice(m, x, dim, k));
    }
    parts.push_back(x);
    for (int k = 1; k <= hi; ++k) {
        parts.push_back(pad_reflect_slice(m, x, dim, n - 1 - k));
    }
    return ggml_concat_n(m, parts.data(), (int)parts.size(), dim);
}

}  // namespace

tensor pad_reflect_ext(model_ref m, tensor x, int l0, int r0, int l1, int r1) {
    x = pad_reflect_axis(m, x, 0, l0, r0);
    x = pad_reflect_axis(m, x, 1, l1, r1);
    return x;
}

tensor group_norm(model_ref m, tensor x, int groups, float eps) {
    x = ggml_group_norm(m, x, groups, eps);
    // 채널축 broadcast 규약은 batch_norm_2d 와 같다 — CWHN 은 ne0 이 채널이라 그대로,
    // WHCN 은 ne2 라 [1,1,C,1] 로 편다.
    const bool whcn = !(m.flags & model_build_flag::cwhn);
    auto ch = [&](tensor t) { return whcn ? ggml_reshape_4d(m, t, 1, 1, t->ne[0], 1) : t; };
    if (tensor weight = m.find("weight")) x = ggml_mul(m, x, ch(weight));
    if (tensor bias = m.find("bias")) x = ggml_add(m, x, ch(bias));
    return named(m, x);
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