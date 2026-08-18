#pragma once

#include "visp/ml.h"
#include "visp/util.h"

// Common neural network building blocks

namespace visp {

tensor linear(model_ref, tensor x);
tensor layer_norm(model_ref, tensor x, float eps = 1e-5f);
// GroupNorm = 정규화 + **per-channel affine(γ·x̂+β)**. `ggml_group_norm` 은 정규화만 한다 —
// affine 을 빼먹으면 shape 이 그대로라 크래시가 없고 값만 틀린다. 그리고 γ=1·β=0 으로 초기화되므로
// **랜덤 가중치로는 안 드러난다**(학습된 모델에서만 어긋난다).
// affine=False 인 GroupNorm 은 weight/bias 가 없으므로 그때만 건너뛴다.
tensor group_norm(model_ref, tensor x, int groups, float eps = 1e-5f);

// 거울(reflect) 패딩. ggml 에는 `ggml_pad_reflect_1d` 밖에 없어 2D 를 못 쓴다.
// 인자 순서는 `ggml_pad_ext` 와 맞췄다 — ggml 축 0(W)·1(H) 만 지원한다.
// torch `reflection_pad2d` 와 동일하게 축을 차례로 접으므로 모서리는 이중 반사가 된다.
tensor pad_reflect_ext(model_ref, tensor x, int l0, int r0, int l1, int r1);

// space-to-depth quadrant: x[..., sh::2, sw::2] (YOLO Focus).
// `ggml_view` 는 ne0(=W) 축에 step 을 못 줘서 view 로는 표현이 안 된다 — step 이 무시돼
// 좌상단 연속 블록이 잘려 나오고, shape 은 맞으므로 **조용히 틀린다.**
// sw/sh 는 각각 0 또는 1. 결과 shape 은 [W/2, H/2, C, 1].
tensor space_to_depth_quad(model_ref m, tensor x, int sw, int sh);

// Permute between CWHN and WHCN tensor dimension ordering. Does not rewrite tensor data.
tensor permute_cwhn_to_whcn(model_ref m, tensor x);
tensor permute_whcn_to_cwhn(model_ref m, tensor x);

// "Contiguous 2D" refers to the layout configured in `m` model flags, ie. the preferred
// memory layout for 2D operations like convolution.
inline bool is_whcn(model_ref m) { return !(m.flags & model_build_flag::cwhn); }
inline bool is_cwhn(model_ref m) { return !!(m.flags & model_build_flag::cwhn); }

// These functions convert between memory layouts, ie. they rewrite tensor data.
tensor cwhn_to_contiguous_2d(model_ref m, tensor x);
tensor whcn_to_contiguous_2d(model_ref m, tensor x);
tensor contiguous_2d_to_cwhn(model_ref m, tensor x);
tensor contiguous_2d_to_whcn(model_ref m, tensor x);

// Always returns number of elements of tensor in width-height-channels-batch order,
// even if that's not how they're stored in memory.
std::array<int64_t, 4> nelements_whcn(model_ref const&, tensor t);

// 2D (convolution) functions
// Input and weight are expected to be in "contiguous 2D" layout as configured in `m`.
tensor conv_2d(model_ref m, tensor x, int stride = 1, int pad = 0, int dilation = 1);

// grouped conv (nn.Conv2d(groups=g)). groups<=1 이면 conv_2d 와 동일.
// ggml 에 grouped conv 커널이 없어 그룹별 view → conv → 채널축 concat 으로 분해한다.
tensor conv_2d_grouped(model_ref m, tensor x, int stride = 1, int pad = 0,
                       int dilation = 1, int groups = 1);

// weight/bias 를 **그래프 텐서로** 받는 conv — gguf 에 없고 런타임에 계산되는 가중치용.
// 예: ConvWS2d/ConvAWS2d 의 표준화된 weight. groups==1 전용. bias 는 nullptr 가능.
tensor conv_2d_wt(model_ref m, tensor x, tensor weight, tensor bias,
                  int stride = 1, int pad = 0, int dilation = 1);

tensor conv_2d_depthwise(model_ref m, tensor x, int stride = 1, int pad = 0);
tensor conv_2d_deform(
    model_ref m, tensor x, tensor weight, tensor offset, tensor mask, int stride, int pad);
// `pad` 는 torch 의 ConvTranspose2d padding 과 같은 뜻이다(출력 가장자리를 그만큼 버린다).
// ggml 에는 padding 을 받는 conv_transpose 가 없어 여기서 잘라낸다.
// `groups` 는 torch 와 같은 뜻이다. ggml 에 grouped conv_transpose 가 없어 그룹마다
// 커널·입력을 잘라 돌리고 채널로 이어붙인다.
tensor conv_transpose_2d(model_ref m, tensor x, int stride, int pad = 0, int groups = 1);
tensor batch_norm_2d(model_ref, tensor x);

// 2D image to patch embedding using convolution and optional norm. CWHN input and output.
tensor patch_embed(model_ref, tensor x, int patch_size);

struct attention_qkv {
    tensor q, k, v;
};
// Input: x [head_dim*n_heads, n_patches, batch]
// Output: q, k, v each of shape [head_dim, n_heads, n_patches, batch]
attention_qkv split_qkv(model_ref m, tensor x, int n_heads, int split_dim);

// Attention with optional mask and output linear layer.
tensor attention(
    model_ref m, tensor q, tensor k, tensor v, tensor mask, float scale, model_ref m_out);

} // namespace visp
