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

// 정수 텐서를 f32 로 올린다.
//
// ⚠️ **ggml 의 이항 연산(add/sub/mul/div)은 정수 타입을 모른다.**
//    `ggml-cpu/binary-ops.cpp` 가 `binary_op: unsupported types: dst: i32, src0: i32, src1: i32`
//    로 **abort** 한다. 인덱스 계산이 그 자리다 — `ggml_get_rows` 가 I32 를 강제하므로
//    한 번 i32 가 된 텐서가 그대로 산술까지 흘러온다(mmseg point_rend 의 점 좌표 차).
//
// ⚠️ 그 abort 는 **크래시로 안 보인다.** OpenMP 병렬 구간 안에서 나면 나머지 스레드가
//    교착에 빠져 프로세스가 CPU 0% 로 매달린다 — 「느린 모델」로 오인된다(12분 관측).
//    행의 지문은 「경과 시간은 느는데 CPU 시간이 0」이다.
//
// 좌표·인덱스는 정수라 f32 로 계산해도 값이 안 깨진다(f32 는 2^24 까지 정확).
// **이미 f32 면 그대로 돌려주므로 비용이 없다** — 그래서 무조건 감싸도 된다.
inline tensor as_f32(model_ref m, tensor x) {
    return x->type == GGML_TYPE_F32 ? x : ggml_cast(m, ggml_cont(m, x), GGML_TYPE_F32);
}

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

// 축별 conv — H 와 W 의 padding/stride/dilation 이 다른 경우(ERFNet 의 3x1 · 1x3 분리 conv).
// ⚠️ 인자는 ggml 과 같은 **W 먼저**다. torch 의 (H, W) 순서와 반대이므로 렌더러에서 뒤집는다.
tensor conv_2d_ex(model_ref m, tensor x, int stride_w, int stride_h, int pad_w, int pad_h,
                  int dilation_w = 1, int dilation_h = 1);

// grouped conv (nn.Conv2d(groups=g)). groups<=1 이면 conv_2d 와 동일.
// ggml 에 grouped conv 커널이 없어 그룹별 view → conv → 채널축 concat 으로 분해한다.
tensor conv_2d_grouped(model_ref m, tensor x, int stride = 1, int pad = 0,
                       int dilation = 1, int groups = 1);

// weight/bias 를 **그래프 텐서로** 받는 conv — gguf 에 없고 런타임에 계산되는 가중치용.
// 예: ConvWS2d/ConvAWS2d 의 표준화된 weight. groups==1 전용. bias 는 nullptr 가능.
tensor conv_2d_wt(model_ref m, tensor x, tensor weight, tensor bias,
                  int stride = 1, int pad = 0, int dilation = 1);

// 동적 weight 의 **depthwise** conv (groups == 채널수). DMNet 의 DCM 이 이 자리다 —
// 필터를 입력에서 만들어 채널마다 다른 커널로 돈다.
// weight 는 torch `[C,1,kh,kw]` → ggml `[kw,kh,1,C]` 로 그대로 온다(GGUF 경유가 아니라
// 그래프 텐서라 `conv_2d_depthwise` 의 레이아웃 되돌림이 필요 없다).
tensor conv_2d_wt_dw(model_ref m, tensor x, tensor weight, tensor bias,
                     int stride = 1, int pad = 0, int dilation = 1);

tensor conv_2d_depthwise(model_ref m, tensor x, int stride = 1, int pad = 0, int dilation = 1);
tensor conv_2d_deform(
    model_ref m, tensor x, tensor weight, tensor offset, tensor mask, int stride, int pad);
// `pad` 는 torch 의 ConvTranspose2d padding 과 같은 뜻이다(출력 가장자리를 그만큼 버린다).
// ggml 에는 padding 을 받는 conv_transpose 가 없어 여기서 잘라낸다.
// `groups` 는 torch 와 같은 뜻이다. ggml 에 grouped conv_transpose 가 없어 그룹마다
// 커널·입력을 잘라 돌리고 채널로 이어붙인다.
// `output_padding` 도 torch 와 같은 뜻이다(출력 오른쪽·아래에 그만큼 더 붙인다).
// ⚠️ 렌더러와 **같이** 고쳐야 한다 — 한쪽만 고치면 크기가 1 어긋난 채 조용히 돈다.
tensor conv_transpose_2d(
    model_ref m, tensor x, int stride, int pad = 0, int groups = 1, int output_padding = 0);
tensor batch_norm_2d(model_ref, tensor x);

// PReLU. 기울기 `weight` 는 채널당 하나(또는 전체 하나)다.
// ⚠️ `ggml_leaky_relu(0.25)` 로 근사하지 마라 — 0.25 는 torch 기본 초기값이라
//    **랜덤 가중치에서만 맞는다.** 학습된 모델에서 조용히 틀린다.
tensor prelu(model_ref m, tensor x);

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
