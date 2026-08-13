#include "head.h"

#include <cmath>

#include "visp/nn.h"

#include <ggml.h>

#include <algorithm>
#include <string>

namespace visp {

// ── 공용 부품 ─────────────────────────────────────────────────────────────────
// dcn_base 를 head 내부 레이아웃(contiguous_2d = whcn)의 채널 축에 맞춘다.
// 러너는 평평한 {18,1,1,1} 로 넘긴다 — 그대로 쓰면 ne0 끼리 맞춰보다 `ggml_can_repeat` 로
// 죽는다(offset 은 {W,H,18,1} 이라 채널이 ne2 다). broadcast 가 먹도록 축을 세운다.
static tensor dcn_base_whcn(model_ref m, tensor b) {
    if (!b) return b;
    int64_t n = ggml_nelements(b);
    return (m.flags & model_build_flag::cwhn) ? ggml_reshape_4d(m, b, n, 1, 1, 1)
                                              : ggml_reshape_4d(m, b, 1, 1, n, 1);
}

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

// pad 를 커널 크기에서 정한다. mmdet head 는 3x3(pad 1)과 1x1(pad 0)이 섞여 있고,
// 계열마다 어느 쪽인지 다르다 — cfg 에 pad 를 또 하나 두는 대신 가중치를 보고 정한다.
// **타워 안에서도 섞인다**: NAS-FCOS 는 탐색된 head 라 단마다 커널이 다르다.
// pad 를 1 로 박으면 1×1 단에서 출력이 두 칸 커진다(실측: 64 → 66).
static tensor conv_same(model_ref m, const std::string& p, tensor x, int stride = 1) {
    model_ref sub = m[p.c_str()];
    int k = (int)sub.weights("weight")->ne[0];   // whcn 가중치 {KW,KH,Cin,Cout}
    return conv_2d(sub, x, stride, (k - 1) / 2);
}

// ne2(채널) 축 [off, off+n) 슬라이스.
static tensor slice_ch(model_ref m, tensor x, int64_t off, int64_t n) {
    return ggml_cont(m, ggml_view_4d(m, x, x->ne[0], x->ne[1], n, x->ne[3],
                                     x->nb[1], x->nb[2], x->nb[3], (size_t)off * x->nb[2]));
}

// ConvModule 의 conv 단. 보통은 평범한 conv 지만 **DCNv2(`ModulatedDeformConv2dPack`)일 수도**
// 있다 — NAS-FCOS 처럼 head 구조를 탐색한 계열이 타워 중간중간에 섞어 쓴다.
// 판별은 `conv_offset` 가중치의 유무로 한다(있으면 deform). 이름 표를 두지 않는다.
//
// mmcv 의 forward 그대로: conv_offset 이 3·k·k 채널을 내고 앞 2/3 가 offset, 뒤 1/3 이 mask(sigmoid).
static tensor conv_maybe_deform(model_ref m, const std::string& p, tensor x) {
    model_ref cm = m[p.c_str()];
    tensor ow = cm["conv_offset"].find("weight");
    if (!ow) return conv_same(m, p, x);

    tensor w = cm.weights("weight");
    const int k = (int)w->ne[0], pad = (k - 1) / 2, kk = k * k;
    const int64_t cin = w->ne[2], och = ow->ne[3];

    // conv_offset 출력 채널로 **버전과 그룹 수를 동시에** 알아낸다.
    //   DCNv1(`DeformConv2dPack`)        dg·2·k²      offset 만
    //   DCNv2(`ModulatedDeformConv2dPack`) dg·3·k²    offset + mask
    // dg 를 1 로 박으면 NAS-FCOS(dg=2)가 틀리고, v2 로 박으면 DDOD(v1)에서 mask 를 읽다
    // 텐서 밖으로 나간다(`data_size + view_offs <= nbytes` assert).
    // 54 처럼 양쪽에 걸리는 값은 **입력 채널을 나눌 수 있는 쪽**으로 정한다(v2 dg=2 vs v1 dg=3).
    bool modulated = (och % (3 * kk) == 0) && (cin % (och / (3 * kk)) == 0);
    const int dg = std::max<int>(1, (int)(och / ((modulated ? 3 : 2) * kk)));

    tensor o = conv_2d(cm["conv_offset"], x, 1, pad);
    tensor offset = slice_ch(m, o, 0, 2 * dg * kk);
    tensor mask = modulated ? ggml_sigmoid(m, slice_ch(m, o, 2 * dg * kk, dg * kk)) : nullptr;

    if (dg == 1) {
        x = conv_2d_deform(m, x, w, offset, mask, 1, pad);
    } else {
        // ggml 커널은 dg=1 전제다(`offset->ne[2] == 2·k²` assert). vendored ggml 은 안 고치므로
        // **정의대로 쪼갠다** — 입력 채널을 dg 등분해 각자의 offset/mask 로 돌리고 **더한다**.
        // (deform_groups 는 offset 만 나눈다. 출력 채널은 안 나뉘므로 concat 이 아니라 합이다.)
        const int64_t cg = cin / dg;
        tensor acc = nullptr;
        for (int g = 0; g < dg; ++g) {
            tensor xg = slice_ch(m, x, g * cg, cg);
            tensor wg = ggml_cont(m, ggml_view_4d(m, w, w->ne[0], w->ne[1], cg, w->ne[3],
                                                  w->nb[1], w->nb[2], w->nb[3],
                                                  (size_t)(g * cg) * w->nb[2]));
            tensor og = slice_ch(m, offset, (int64_t)g * 2 * kk, 2 * kk);
            tensor mg = mask ? slice_ch(m, mask, (int64_t)g * kk, kk) : nullptr;
            tensor yg = conv_2d_deform(m, xg, wg, og, mg, 1, pad);
            acc = acc ? ggml_add(m, acc, yg) : yg;
        }
        x = acc;
    }
    if (tensor b = cm.find("bias")) {
        x = ggml_add(m, x, (m.flags & model_build_flag::cwhn)
                               ? b : ggml_reshape_4d(m, b, 1, 1, b->ne[0], 1));
    }
    return x;
}

// ConvModule 의 활성화. 기본은 ReLU, RTMDet 만 SiLU 다.
static tensor head_act(model_ref m, tensor x, anchor_head_cfg const& c) {
    if (c.head_leaky > 0.0f) return ggml_leaky_relu(m, x, c.head_leaky, false);
    return c.head_silu ? ggml_silu(m, x) : ggml_relu(m, x);
}

// ConvModule(conv + GroupNorm + ReLU). 커널 크기는 가중치에서 읽는다.
static tensor conv_gn_relu(model_ref m, tensor x, const std::string& p, int gn_groups) {
    x = conv_maybe_deform(m, p + ".conv", x);
    x = group_norm_affine(m[(p + ".gn").c_str()], x, gn_groups);
    return ggml_relu(m, x);
}

// cls/reg 공유 타워. ConvModule 은 norm 유무로 두 모양이다 — RetinaNet 은 conv+relu,
// ATSS/FCOS/GFL/RepPoints 는 conv+GN+relu.
static tensor conv_tower(model_ref m, tensor x, anchor_head_cfg const& c,
                         const std::string& prefix, size_t level, int nconv = -1) {
    // RetinaSepBNHead 는 레벨마다 타워를 따로 둔다 → 이름이 한 겹 깊다.
    const std::string base = c.per_level_towers
                                 ? prefix + "." + std::to_string(level) : prefix;
    if (nconv < 0) nconv = c.stacked_convs;
    for (int i = 0; i < nconv; ++i) {
        std::string p = base + "." + std::to_string(i);
        x = conv_maybe_deform(m, p + ".conv", x);
        if (c.head_has_norm) x = group_norm_affine(m[(p + ".gn").c_str()], x, c.gn_groups);
        x = head_act(m, x, c);
    }
    return x;
}

// mmcv `Scale` — 레벨마다 스칼라 하나를 곱한다. 값은 GGUF 에서 이름으로 읽으므로
// 체크포인트를 바꿔도 파라미터 헤더를 다시 굽지 않아도 된다. 없으면 그대로 통과.
static tensor apply_scale(model_ref m, tensor x, anchor_head_cfg const& c, size_t l) {
    if (c.scales_prefix.empty()) return x;
    std::string p = c.scales_prefix + "." + std::to_string(l) + ".scale";
    tensor s = m.find(p.c_str());
    return s ? ggml_mul(m, x, s) : x;   // {1,1,1,1} broadcast
}

// GFL 의 DFL(Distribution Focal Loss) 디코드. reg_head 는 방향 4개 × 빈 (reg_max+1) 개의
// **로짓**을 낸다 — 그 분포의 기댓값이 거리다.
//
//     x = softmax(logits over bins);  distance = Σ_j j·x_j
//
// mmdet 은 이걸 디코드(`Integral`)에서 하지만 여기서 한다. 그러면 GFL 의 출력이
// FCOS 와 같은 **4채널 거리**가 되어 디코드 쪽에 계열 분기를 안 만들어도 된다.
//
// 채널 배치: C = i·(reg_max+1) + j (빈 j 가 빠른 축). whcn 이라 C 는 ne2 다.
static tensor dfl_integral(model_ref m, tensor bp, int reg_max) {
    const int64_t W = bp->ne[0], H = bp->ne[1];
    const int B = reg_max + 1;
    // ne2(C) 를 [빈, 방향] 으로 가른다. j 가 빠른 축이라 ne2=B, ne3=4 가 된다.
    tensor t = ggml_reshape_4d(m, ggml_cont(m, bp), W, H, B, 4);
    // softmax 는 ne0 에만 걸리므로 빈 축을 ne0 로 데려온다 → [B, W, H, 4]
    t = ggml_cont(m, ggml_permute(m, t, 1, 2, 0, 3));
    t = ggml_soft_max(m, t);
    t = ggml_mul(m, t, ggml_arange(m, 0.0f, (float)B, 1.0f));   // j 가중
    t = ggml_sum_rows(m, t);                                     // → [1, W, H, 4]
    return ggml_cont(m, ggml_permute(m, t, 3, 0, 1, 2));         // → [W, H, 4, 1]
}

// ── RetinaNet · ATSS · PAA · FCOS · GFL ───────────────────────────────────────
// features 는 인터프리터 출력(cwhn). 인터프리터 내부 conv 는 contiguous_2d 레이아웃에서 도므로
// (graph_interpret: cwhn_to_contiguous_2d → conv... → contiguous_2d_to_cwhn), head 도 동일하게
// cwhn→contiguous_2d 로 되돌려 conv 타워를 태우고, 결과를 다시 cwhn 으로 낸다(detect_anchor 규약).
//
// 다섯 계열의 뼈대가 같아 한 함수로 둔다. 다른 곳은 cfg 플래그로만 갈린다:
//   centerness_head  ATSS·PAA·FCOS 에만 있는 세 번째 갈래
//   scales_prefix    ATSS·PAA·FCOS·GFL 의 레벨별 learnable scale
//   bbox_*           FCOS 의 clamp·stride / exp
//   reg_max          GFL 의 DFL
static void tower_head_forward(model_ref m, std::vector<tensor> const& feats,
                               anchor_head_cfg const& c, head_outputs& out) {
    for (size_t l = 0; l < feats.size(); ++l) {
        tensor f = cwhn_to_contiguous_2d(m, feats[l]);
        // 분기 앞 공유 conv(+ReLU). RPNHead 의 `rpn_conv` — 없으면 그냥 통과한다.
        if (!c.pre_conv.empty()) f = head_act(m, conv_same(m, c.pre_conv, f), c);

        tensor cc = conv_tower(m, f, c, c.cls_convs_prefix, l);
        tensor rr = conv_tower(m, f, c, c.reg_convs_prefix, l);

        // RTMDetSepBNHead 는 출력 conv 도 레벨마다 따로다.
        const std::string lv = c.per_level_heads
                                   ? "." + std::to_string(l) + c.per_level_head_tail : "";
        tensor cls = conv_same(m, c.cls_head + lv, cc);
        tensor box = conv_same(m, c.reg_head + lv, rr);

        box = apply_scale(m, box, c, l);
        if (c.reg_max > 0) box = dfl_integral(m, box, c.reg_max);

        // FCOS 는 둘 중 하나만 쓴다(norm_on_bbox). GFL 은 거리에 stride 를 곱한다.
        float stride = l < c.strides.size() ? c.strides[l] : 1.0f;
        if (c.bbox_clamp_stride) box = ggml_scale(m, ggml_relu(m, box), stride);
        else if (c.bbox_exp)     box = ggml_exp(m, box);
        else if (c.reg_max > 0)  box = ggml_scale(m, box, stride);
        // RTMDet 은 거리를 **stride 단위**로 낸다. 안 곱하면 레벨마다 8·16·32 배씩 작다
        // (실측 배율 0.117 / 0.059 / 0.030 — 정확히 1/8, 1/16, 1/32).
        if (c.bbox_mul_stride)   box = ggml_scale(m, box, stride);

        cls = contiguous_2d_to_cwhn(m, cls);
        box = contiguous_2d_to_cwhn(m, box);
        ggml_format_name(cls, "cls_%zu", l);
        ggml_format_name(box, "box_%zu", l);
        out.cls.push_back(cls);
        out.box.push_back(box);

        if (!c.centerness_head.empty()) {
            tensor ctr = conv_same(m, c.centerness_head + lv, c.centerness_on_reg ? rr : cc);
            if (c.ctr_tanh) ctr = ggml_tanh(m, ctr);
            ctr = contiguous_2d_to_cwhn(m, ctr);
            ggml_format_name(ctr, "ctr_%zu", l);
            out.ctr.push_back(ctr);
        }
    }
}

// ── DETR ──────────────────────────────────────────────────────────────────────
// 여기만 출력이 **공간 격자가 아니다**. query 100개가 이미지를 조회해 답을 하나씩 낸다.
//
// ⚠️ `split_qkv` 를 쓸 수 없다. 그건 q·k·v 가 **같은 입력**에서 나올 때 쓰는 것인데
//    DETR 은 (a) self-attn 조차 q·k 에만 위치 인코딩을 더하고 v 에는 안 더하며
//    (b) cross-attn 은 q 가 query, k·v 가 이미지다. 그래서 packed in_proj 를 직접 자른다.
//
// `nn.MultiheadAttention` 의 in_proj 는 [Wq; Wk; Wv] 를 세로로 쌓은 것이다
// (ne = {embed, 3*embed}) → ne1 방향으로 3등분한다.
static tensor mha_proj(model_ref m, const std::string& p, tensor x, int idx, int n_heads) {
    tensor w = m.weights((p + ".in_proj_weight").c_str());
    tensor b = m.find((p + ".in_proj_bias").c_str());
    const int64_t E = w->ne[0];
    tensor wi = ggml_cont(m, ggml_view_2d(m, w, E, E, w->nb[1], (size_t)idx * E * w->nb[1]));
    tensor y = ggml_mul_mat(m, wi, x);
    if (b) y = ggml_add(m, y, ggml_view_1d(m, b, E, (size_t)idx * E * b->nb[0]));
    // attention() 규약: [head_dim, n_heads, n_tokens, batch]
    return ggml_reshape_4d(m, y, E / n_heads, n_heads, y->ne[1], 1);
}

// mmcv `MultiheadAttention`: q·k 에만 pos 를 더하고, **residual 은 pos 를 더하기 전 query** 다.
static tensor mha(model_ref m, const std::string& p, tensor q_in, tensor q_pos,
                  tensor kv_in, tensor kv_pos, anchor_head_cfg const& c,
                  tensor mask = nullptr) {
    const std::string A = p + ".attn";
    tensor qs = q_pos ? ggml_add(m, q_in, q_pos) : q_in;
    tensor ks = kv_pos ? ggml_add(m, kv_in, kv_pos) : kv_in;
    tensor q = mha_proj(m, A, qs,    0, c.n_heads);
    tensor k = mha_proj(m, A, ks,    1, c.n_heads);
    tensor v = mha_proj(m, A, kv_in, 2, c.n_heads);   // v 에는 pos 를 안 더한다
    const float scale = 1.0f / std::sqrt((float)(c.embed_dims / c.n_heads));
    tensor a = attention(m, q, k, v, mask, scale, m[(A + ".out_proj").c_str()]);
    return ggml_add(m, q_in, a);                      // residual = pos 더하기 **전**
}

// mmdet `MLP` — Linear 를 n 단 쌓고 **마지막 뺀 전부**에 ReLU 를 건다.
static tensor mlp(model_ref m, const std::string& p, tensor x, int n) {
    for (int i = 0; i < n; ++i) {
        x = linear(m[(p + ".layers." + std::to_string(i)).c_str()], x);
        if (i + 1 < n) x = ggml_relu(m, x);
    }
    return x;
}

// mmcv `FFN`: Linear → ReLU → Linear, residual 포함.
static tensor ffn(model_ref m, const std::string& p, tensor x) {
    tensor y = linear(m[(p + ".layers.0.0").c_str()], x);
    // ⚠️ 활성화가 ReLU 가 아닌 계열이 있다 — DAB-DETR 은 **PReLU**(기울기가 학습 파라미터).
    //    `ggml_leaky_relu` 는 기울기를 float 로만 받으므로 정의대로 조립한다:
    //        PReLU(x) = relu(x) - a * relu(-x)
    //    가중치가 있으면 PReLU, 없으면 ReLU — 이름으로 계열을 가르지 않는다.
    if (tensor a = m.find((p + ".layers.0.1.weight").c_str()))
        y = ggml_sub(m, ggml_relu(m, y), ggml_mul(m, ggml_relu(m, ggml_neg(m, y)), a));
    else
        y = ggml_relu(m, y);
    y = linear(m[(p + ".layers.1").c_str()], y);
    return ggml_add(m, x, y);
}

// Conditional DETR 의 attention. `nn.MultiheadAttention` 이 아니라 mmdet 자체
// `ConditionalAttention` 이라 q/k/v 투영이 **따로** 있다.
//   self  : q = qcontent(x) + qpos(qpos),  k = kcontent(x) + kpos(qpos), v = v_proj(x)
//   cross : q = [qcontent(x)(+qpos) ; qpos_sine(ref)] , k = [kcontent(mem) ; kpos(pos)] ,
//           v = v_proj(mem)                      ← q·k 만 head 축으로 **이어붙여** head_dim 이 2배
// ⚠️ `qpos_proj` 는 **0번 층에만** 있다(mmdet 이 나머지 층에서 지운다). 그래서 is_first 로 가른다.
static tensor cond_head_split(model_ref m, tensor x, int n_heads) {
    return ggml_reshape_4d(m, x, x->ne[0] / n_heads, n_heads, x->ne[1], 1);
}

static tensor cond_attn(model_ref m, const std::string& p, tensor x, tensor q_pos,
                        tensor kv, tensor kv_pos, tensor ref_sine, bool is_first,
                        anchor_head_cfg const& c) {
    const int H = c.n_heads;
    const bool cross = (kv != x);
    tensor qc = linear(m[(p + ".qcontent_proj").c_str()], x);
    tensor kc = linear(m[(p + ".kcontent_proj").c_str()], kv);
    tensor v  = linear(m[(p + ".v_proj").c_str()], kv);
    tensor kp = kv_pos ? linear(m[(p + ".kpos_proj").c_str()], kv_pos) : nullptr;

    tensor q, k;
    if (!cross) {
        tensor qp = linear(m[(p + ".qpos_proj").c_str()], q_pos);
        q = cond_head_split(m, ggml_add(m, qc, qp), H);
        k = cond_head_split(m, ggml_add(m, kc, kp), H);
    } else {
        // 0번 층만 질문에 위치를 더한다. 나머지 층은 내용만 쓰고 위치는 sine 쪽으로 간다.
        // ⚠️ 0번 층은 q 와 **k 둘 다** 위치를 더한다. 그리고 k_pos 는 더한 뒤 **또 이어붙인다** —
        //    한 번만 쓴다고 착각하기 쉽다(실측: k 에 안 더하면 층0 부터 rel_L1 0.105).
        tensor qq = is_first ? ggml_add(m, qc, linear(m[(p + ".qpos_proj").c_str()], q_pos)) : qc;
        tensor kk = is_first ? ggml_add(m, kc, kp) : kc;
        tensor qs = linear(m[(p + ".qpos_sine_proj").c_str()], ref_sine);
        // head 축(ne0)으로 이어붙인다 — torch 의 `cat(dim=3)` 과 같은 자리다.
        q = ggml_concat(m, cond_head_split(m, qq, H), cond_head_split(m, qs, H), 0);
        k = ggml_concat(m, cond_head_split(m, kk, H), cond_head_split(m, kp, H), 0);
    }
    tensor vv = cond_head_split(m, v, H);
    const float scale = 1.0f / std::sqrt((float)(q->ne[0]));
    tensor a = attention(m, q, k, vv, nullptr, scale, m[(p + ".out_proj").c_str()]);
    return ggml_add(m, x, a);
}

// encoder 는 세 계열이 같다(표준 MultiheadAttention · post-norm). 여기만 공유한다.
static tensor detr_encode(model_ref m, std::vector<tensor> const& feats,
                          anchor_head_cfg const& c, tensor& pos_out) {
    if (feats.empty()) { fprintf(stderr, "detr: feature 가 없다\n"); abort(); }
    // cwhn {C,W,H,1} → {C, W*H, 1}. mmdet 의 `flatten(2).permute(0,2,1)` 과 같은 순서다
    // (공간 index = h*W + w). pos_embed 도 프론트엔드가 같은 순서로 구워 둔다.
    tensor f = feats[0];
    tensor x = ggml_reshape_3d(m, ggml_cont(m, f), f->ne[0], f->ne[1] * f->ne[2], 1);
    tensor pos = m.weights("pos_embed");
    for (int i = 0; i < c.enc_layers; ++i) {
        const std::string L = "encoder.layers." + std::to_string(i);
        x = mha(m, L + ".self_attn", x, pos, x, pos, c);
        x = layer_norm(m[(L + ".norms.0").c_str()], x);   // post-norm 이다
        x = ffn(m, L + ".ffn", x);
        x = layer_norm(m[(L + ".norms.1").c_str()], x);
    }
    pos_out = pos;
    return x;
}

static void detr_emit(head_outputs& out, int i, tensor cls, tensor box) {
    ggml_format_name(cls, "cls_%d", i);
    ggml_format_name(box, "box_%d", i);
    out.cls.push_back(cls);
    out.box.push_back(box);
}

// 원조 DETR. decoder 도 표준 MultiheadAttention 이고 query 는 0 에서 시작한다.
void detr_head_forward(model_ref m, std::vector<tensor> const& feats,
                       anchor_head_cfg const& c, head_outputs& out) {
    tensor pos = nullptr;
    tensor mem = detr_encode(m, feats, c, pos);
    tensor q_pos = m.weights("query_embedding.weight");
    tensor q = ggml_scale(m, q_pos, 0.0f);               // zeros_like(query_pos)

    for (int i = 0; i < c.dec_layers; ++i) {
        const std::string L = "decoder.layers." + std::to_string(i);
        q = mha(m, L + ".self_attn",  q, q_pos, q,   q_pos, c);
        q = layer_norm(m[(L + ".norms.0").c_str()], q);
        q = mha(m, L + ".cross_attn", q, q_pos, mem, pos,   c);
        q = layer_norm(m[(L + ".norms.1").c_str()], q);
        q = ffn(m, L + ".ffn", q);
        q = layer_norm(m[(L + ".norms.2").c_str()], q);

        // 층마다 post_norm 을 거친 값이 head 로 간다(deep supervision). 마지막만이 아니다.
        tensor h = layer_norm(m["decoder.post_norm"], q);
        tensor cls = linear(m["bbox_head.fc_cls"], h);
        // reg_ffn: Linear → ReLU → Linear. ⚠️ `add_identity=False` 라 residual 을 더하면 안 된다.
        tensor r = linear(m["bbox_head.reg_ffn.layers.0.0"], h);
        r = linear(m["bbox_head.reg_ffn.layers.1"], ggml_relu(m, r));
        tensor box = ggml_sigmoid(m, linear(m["bbox_head.fc_reg"], ggml_relu(m, r)));
        detr_emit(out, i, cls, box);
    }
}

// Conditional DETR. 질문을 내용/위치로 쪼개 조회한다. 참조점은 query_embedding 에서만
// 나오므로 **상수**다 — sine 인코딩과 inverse_sigmoid 를 프론트엔드가 구워 뒀다.
void conditional_detr_head_forward(model_ref m, std::vector<tensor> const& feats,
                                   anchor_head_cfg const& c, head_outputs& out) {
    tensor pos = nullptr;
    tensor mem = detr_encode(m, feats, c, pos);
    tensor q_pos = m.weights("query_embedding.weight");
    tensor q = ggml_scale(m, q_pos, 0.0f);
    tensor ref_sine0 = m.weights("ref_sine_embed");

    for (int i = 0; i < c.dec_layers; ++i) {
        const std::string L = "decoder.layers." + std::to_string(i);
        // 0번 층은 배율 1, 그 뒤로는 query 에서 뽑은 배율을 곱한다(MLP 2단).
        tensor rs = ref_sine0;
        if (i > 0) rs = ggml_mul(m, ref_sine0, mlp(m, "decoder.query_scale", q, 2));
        // self-attn 의 key 위치도 query_pos 다(`kpos_proj(query_pos)`). 널을 넘기면 죽는다.
        q = cond_attn(m, L + ".self_attn", q, q_pos, q, q_pos, nullptr, i == 0, c);
        q = layer_norm(m[(L + ".norms.0").c_str()], q);
        q = cond_attn(m, L + ".cross_attn", q, q_pos, mem, pos, rs, i == 0, c);
        q = layer_norm(m[(L + ".norms.1").c_str()], q);
        q = ffn(m, L + ".ffn", q);
        q = layer_norm(m[(L + ".norms.2").c_str()], q);

        tensor h = layer_norm(m["decoder.post_norm"], q);
        tensor cls = linear(m["bbox_head.fc_cls"], h);
        tensor r = linear(m["bbox_head.reg_ffn.layers.0.0"], h);
        r = linear(m["bbox_head.reg_ffn.layers.1"], ggml_relu(m, r));
        tensor reg = linear(m["bbox_head.fc_reg"], ggml_relu(m, r));
        // 박스 앞 2채널에 참조점(inverse_sigmoid)을 더한다. 뒤 2채널이 0 인 상수를 구워 뒀다.
        tensor box = ggml_sigmoid(m, ggml_add(m, reg, m.weights("ref_inv_pad")));
        detr_emit(out, i, cls, box);
    }
}

// mmdet `inverse_sigmoid`. 입력이 이미 [0,1] 이라 clamp 는 eps 하한만 의미가 있다.
// ⚠️ `ggml_clamp` 은 in-place 라 원본을 덮어쓴다 — 반드시 사본에 건다.
static tensor inv_sigmoid(model_ref m, tensor x, float eps) {
    tensor a = ggml_clamp(m, ggml_cont(m, x), eps, 1.0f);
    tensor b = ggml_clamp(m, ggml_cont(m, ggml_scale_bias(m, x, -1.0f, 1.0f)), eps, 1.0f);
    return ggml_log(m, ggml_div(m, a, b));
}

// mmdet `coordinate_to_encoding` — 좌표(cx,cy,w,h)를 sine 인코딩한다.
// DAB 은 참조점이 **층마다 갱신**돼 상수로 못 굽는다. 그래서 여기서 계산한다.
//
// dim_t 는 인접한 두 칸이 같은 값이라, 결과는 짝수 칸 sin · 홀수 칸 cos 이 번갈아 놓인 꼴이다.
// 그 자리 선택을 view 로 하면 ne0 에 step 이 필요해 안 된다 → **0/1 마스크 상수**로 고른다.
// `2π/dim_t`·짝수 마스크·홀수 마스크 셋 다 프론트엔드가 mmdet 공식 그대로 구웠다.
static tensor coord_encode(model_ref m, tensor coord) {
    tensor inv = m.weights("dab_inv_dim_t");     // {F}  = 2π / dim_t
    tensor ev  = m.weights("dab_even");
    tensor od  = m.weights("dab_odd");
    const int64_t F = inv->ne[0], nq = coord->ne[1];
    const int order[4] = {1, 0, 2, 3};           // mmdet 은 (y, x, w, h) 순으로 잇는다
    tensor outp = nullptr;
    for (int i = 0; i < 4; ++i) {
        tensor ci = ggml_cont(m, ggml_view_2d(m, coord, 1, nq, coord->nb[1],
                                              (size_t)order[i] * coord->nb[0]));
        // ggml 에 축 broadcast 가 없어 **외적**으로 {F, nq} 를 만든다.
        tensor p = ggml_mul_mat(m, ggml_reshape_2d(m, inv, 1, F),
                                   ggml_reshape_2d(m, ci, 1, nq));
        tensor e = ggml_add(m, ggml_mul(m, ggml_sin(m, p), ev),
                               ggml_mul(m, ggml_cos(m, p), od));
        outp = outp ? ggml_concat(m, outp, e, 0) : e;
    }
    return outp;                                  // {4F, nq}
}

// DAB-DETR. query 가 **4차원 앵커 박스**이고, 층마다 박스를 고쳐 다음 층 참조점으로 쓴다
// (iterative refinement). 그래서 박스 분기가 decoder 안에 들어와 있다.
void dab_detr_head_forward(model_ref m, std::vector<tensor> const& feats,
                           anchor_head_cfg const& c, head_outputs& out) {
    if (feats.empty()) { fprintf(stderr, "dab_detr: feature 가 없다\n"); abort(); }
    // ⚠️ DAB 은 **encoder 도 다르다** — 층마다 위치 인코딩에 배율을 건다
    //    (`query_pos * query_scale(query)`). 공용 encoder 를 쓰면 조용히 틀린다(실측 L1 3.84).
    tensor f0 = feats[0];
    tensor mem = ggml_reshape_3d(m, ggml_cont(m, f0), f0->ne[0], f0->ne[1] * f0->ne[2], 1);
    tensor pos = m.weights("pos_embed");
    for (int i = 0; i < c.enc_layers; ++i) {
        const std::string L = "encoder.layers." + std::to_string(i);
        tensor p = ggml_mul(m, pos, mlp(m, "encoder.query_scale", mem, 2));
        mem = mha(m, L + ".self_attn", mem, p, mem, p, c);
        mem = layer_norm(m[(L + ".norms.0").c_str()], mem);
        mem = ffn(m, L + ".ffn", mem);
        mem = layer_norm(m[(L + ".norms.1").c_str()], mem);
    }
    const int E = c.embed_dims, H = E / 2;
    tensor q = nullptr;
    tensor ref = ggml_sigmoid(m, m.weights("query_embedding.weight"));   // {4, nq}
    const int64_t nq = ref->ne[1];

    for (int i = 0; i < c.dec_layers; ++i) {
        const std::string L = "decoder.layers." + std::to_string(i);
        tensor rse = coord_encode(m, ref);                        // {2E, nq}
        tensor q_pos = mlp(m, "decoder.ref_point_head", rse, 2);  // 2E → E
        if (q == nullptr) q = ggml_scale(m, q_pos, 0.0f);         // output 은 0 에서 시작
        tensor rs = ggml_cont(m, ggml_view_2d(m, rse, E, nq, rse->nb[1], 0));
        if (i > 0) rs = ggml_mul(m, rs, mlp(m, "decoder.query_scale", q, 2));

        // 박스 크기로 attention 을 변조한다. 앞 절반은 h, 뒤 절반은 w 로 나눈다.
        tensor hw = ggml_sigmoid(m, mlp(m, "decoder.ref_anchor_head", q, 2));   // {2, nq}
        auto row = [&](tensor t, int64_t k) {
            return ggml_cont(m, ggml_view_2d(m, t, 1, nq, t->nb[1], (size_t)k * t->nb[0]));
        };
        tensor lo = ggml_mul(m, ggml_cont(m, ggml_view_2d(m, rs, H, nq, rs->nb[1], 0)),
                             ggml_div(m, row(hw, 1), row(ref, 3)));
        tensor hi = ggml_mul(m, ggml_cont(m, ggml_view_2d(m, rs, H, nq, rs->nb[1],
                                                          (size_t)H * rs->nb[0])),
                             ggml_div(m, row(hw, 0), row(ref, 2)));
        rs = ggml_concat(m, lo, hi, 0);

        q = cond_attn(m, L + ".self_attn", q, q_pos, q, q_pos, nullptr, i == 0, c);
        q = layer_norm(m[(L + ".norms.0").c_str()], q);
        q = cond_attn(m, L + ".cross_attn", q, q_pos, mem, pos, rs, i == 0, c);
        q = layer_norm(m[(L + ".norms.1").c_str()], q);
        q = ffn(m, L + ".ffn", q);
        q = layer_norm(m[(L + ".norms.2").c_str()], q);

        tensor rinv = inv_sigmoid(m, ref, 1e-3f);
        // ⚠️ **다음 층 참조점은 post_norm 이전 출력으로 계산한다.** head 로 나가는 박스는
        //    post_norm 을 거친 값으로 계산하고 — mmdet 이 두 값을 따로 쓴다.
        //    한쪽으로 통일하면 층 0 은 맞고 층 1 부터 어긋난다(실측 L1 5.15).
        ref = ggml_sigmoid(m, ggml_add(m, mlp(m, "bbox_head.fc_reg", q, 3), rinv));

        tensor h = layer_norm(m["decoder.post_norm"], q);
        tensor cls = linear(m["bbox_head.fc_cls"], h);
        tensor box = ggml_sigmoid(m, ggml_add(m, mlp(m, "bbox_head.fc_reg", h, 3), rinv));
        detr_emit(out, i, cls, box);
    }
}

static tensor emax(model_ref m, tensor a, tensor b);   // CornerNet 절에 정의

// ── Deformable attention (Deformable DETR · DINO · DDQ) ───────────────────────
// query 가 **자기가 볼 지점을 예측**하고 그 **소수점 좌표**에서 값을 읽는다.
// ggml 에 grid_sample 이 없어 정의대로 조립한다 — 그래서 필요한 것이 둘이다:
//   ① 실행 중에 만든 F32 좌표 → I32 인덱스   : `ggml_cast`(CPU 백엔드가 f32→i32 를 지원한다)
//   ② 그 인덱스로 행 뽑기                     : `ggml_get_rows`
// 나머지(floor·clamp·곱·합)는 이미 있는 op 이다. vendor ggml 은 안 고친다.
//
// 좌표 규약은 mmcv 의 CPU 구현(`F.grid_sample(align_corners=False)`)에서 나온다.
//   grid = 2·s − 1  →  x_pixel = s_x·W − 0.5      (범위 밖은 0 — padding_mode='zeros')

// 정수 좌표가 [0, n) 안인가 → 1/0. floor 결과라 정수값이므로 clamp 두 번으로 만든다.
static tensor in_range(model_ref m, tensor v, int64_t n) {
    tensor lo = ggml_clamp(m, ggml_cont(m, ggml_scale_bias(m, v, 1.0f, 1.0f)), 0.0f, 1.0f);
    tensor hi = ggml_clamp(m, ggml_cont(m, ggml_scale_bias(m, v, -1.0f, (float)n)), 0.0f, 1.0f);
    return ggml_mul(m, lo, hi);
}

// {2 또는 2L, X} 에서 성분 하나를 {1,1,1,X} 로 뽑는다(ne1·ne2 는 broadcast 로 늘어난다).
static tensor comp(model_ref m, tensor r, int64_t k) {
    const int64_t X = r->ne[1];
    return ggml_cont(m, ggml_view_4d(m, r, 1, 1, 1, X, r->nb[1], r->nb[1], r->nb[1],
                                     (size_t)k * r->nb[0]));
}

// mmcv `MultiScaleDeformableAttention`.
//   query {E,N} · value {E,T} · ref {2,N}(디코더) 또는 {2L,T}(인코더, 레벨별)
static tensor msdeform(model_ref m, const std::string& p, tensor query, tensor q_pos,
                       tensor value_src, tensor ref, std::vector<msd_level> const& lv,
                       int n_heads, int n_points) {
    const int64_t E = query->ne[0], N = query->ne[1], T = value_src->ne[1];
    const int L = (int)lv.size(), H = n_heads, P = n_points, D = E / H;
    tensor q = q_pos ? ggml_add(m, query, q_pos) : query;   // identity 는 더하기 **전** query

    // value 를 {D, T*H} 로 눕힌다 — 행 index = t + T*h 라 head 까지 한 번에 gather 한다.
    tensor v = linear(m[(p + ".value_proj").c_str()], value_src);
    v = ggml_cont(m, ggml_permute(m, ggml_reshape_4d(m, v, D, H, T, 1), 0, 2, 1, 3));
    v = ggml_reshape_2d(m, v, D, T * H);
    tensor head_off = ggml_scale(m, ggml_arange(m, 0.0f, (float)H, 1.0f), (float)T);
    head_off = ggml_reshape_4d(m, head_off, 1, 1, H, 1);

    // 채널 배치는 torch 의 view 순서 그대로다: ((h·L + l)·P + p)·2 + c
    tensor off = linear(m[(p + ".sampling_offsets").c_str()], q);
    off = ggml_reshape_3d(m, off, 2 * P * L, H, N);
    tensor aw = linear(m[(p + ".attention_weights").c_str()], q);
    aw = ggml_soft_max(m, ggml_reshape_3d(m, aw, L * P, H, N));   // (L·P) 축에 softmax

    tensor acc = nullptr;
    for (int l = 0; l < L; ++l) {
        // 이 레벨의 offset 만 잘라 {2,P,H,N} 으로 편다(ne0 안에서 l 이 바깥이라 연속이다).
        tensor ol = ggml_cont(m, ggml_view_3d(m, off, 2 * P, H, N, off->nb[1], off->nb[2],
                                              (size_t)(l * 2 * P) * off->nb[0]));
        ol = ggml_reshape_4d(m, ol, 2, P, H, N);
        tensor ox = ggml_cont(m, ggml_view_4d(m, ol, 1, P, H, N, ol->nb[1], ol->nb[2],
                                              ol->nb[3], 0));
        tensor oy = ggml_cont(m, ggml_view_4d(m, ol, 1, P, H, N, ol->nb[1], ol->nb[2],
                                              ol->nb[3], ol->nb[0]));
        const int64_t W = lv[l].w, Hh = lv[l].h;
        // s = ref + offset/(W,H) → 픽셀 좌표 = s·(W,H) − 0.5. 둘을 합치면 offset 은 그대로다.
        // ref 가 4채널(cx,cy,w,h)이면 offset 을 **박스 크기에 비례**해 흩뿌린다:
        //   s = ref_xy + offset/num_points · ref_wh · 0.5      (2채널이면 s = ref + offset/(W,H))
        tensor px, py;
        if (ref->ne[0] == 4) {
            const float k = 0.5f / (float)P;
            // ⚠️ `ggml_add(a,b)` 는 **b 를 a 모양으로** broadcast 한다 — 큰 쪽이 앞이어야 한다.
            //    작은 쪽을 앞에 두면 `GGML_ASSERT(ggml_can_repeat(b, a))` 로 죽는다.
            px = ggml_add(m, ggml_mul(m, ggml_scale(m, ox, k * (float)W), comp(m, ref, 2)),
                          ggml_scale(m, comp(m, ref, 0), (float)W));
            py = ggml_add(m, ggml_mul(m, ggml_scale(m, oy, k * (float)Hh), comp(m, ref, 3)),
                          ggml_scale(m, comp(m, ref, 1), (float)Hh));
        } else {
            const int64_t rk = (ref->ne[0] == 2) ? 0 : 2 * l;
            px = ggml_add(m, ox, ggml_scale(m, comp(m, ref, rk), (float)W));
            py = ggml_add(m, oy, ggml_scale(m, comp(m, ref, rk + 1), (float)Hh));
        }
        px = ggml_scale_bias(m, px, 1.0f, -0.5f);
        py = ggml_scale_bias(m, py, 1.0f, -0.5f);
        tensor x0 = ggml_floor(m, px), y0 = ggml_floor(m, py);
        tensor fx = ggml_sub(m, px, x0), fy = ggml_sub(m, py, y0);

        tensor lvl_acc = nullptr;
        for (int corner = 0; corner < 4; ++corner) {
            const float dx = (float)(corner & 1), dy = (float)(corner >> 1);
            tensor xi = ggml_scale_bias(m, x0, 1.0f, dx);
            tensor yi = ggml_scale_bias(m, y0, 1.0f, dy);
            // 범위 밖은 0 으로 샘플링된다(zeros padding) → 가중치를 0 으로 만든다.
            tensor ok = ggml_mul(m, in_range(m, xi, W), in_range(m, yi, Hh));
            tensor wx = dx > 0 ? fx : ggml_scale_bias(m, fx, -1.0f, 1.0f);
            tensor wy = dy > 0 ? fy : ggml_scale_bias(m, fy, -1.0f, 1.0f);
            tensor wgt = ggml_mul(m, ggml_mul(m, wx, wy), ok);

            tensor xc = ggml_clamp(m, ggml_cont(m, xi), 0.0f, (float)(W - 1));
            tensor yc = ggml_clamp(m, ggml_cont(m, yi), 0.0f, (float)(Hh - 1));
            tensor idx = ggml_add(m, ggml_add(m, xc, ggml_scale(m, yc, (float)W)),
                                  ggml_scale_bias(m, head_off, 1.0f, (float)lv[l].off));
            idx = ggml_cast(m, ggml_reshape_1d(m, ggml_cont(m, idx), P * H * N), GGML_TYPE_I32);
            tensor g = ggml_reshape_4d(m, ggml_get_rows(m, v, idx), D, P, H, N);
            g = ggml_mul(m, g, wgt);
            lvl_acc = lvl_acc ? ggml_add(m, lvl_acc, g) : g;
        }
        // 이 레벨의 attention weight 를 곱한다(softmax 는 이미 L·P 전체에 걸려 있다).
        tensor al = ggml_cont(m, ggml_view_3d(m, aw, P, H, N, aw->nb[1], aw->nb[2],
                                              (size_t)(l * P) * aw->nb[0]));
        lvl_acc = ggml_mul(m, lvl_acc, ggml_reshape_4d(m, al, 1, P, H, N));
        acc = acc ? ggml_add(m, acc, lvl_acc) : lvl_acc;
    }
    // P 축을 더한다. ne1 을 줄이는 op 이 없어 조각을 더한다(P 는 4 라 싸다).
    tensor sum = nullptr;
    for (int pp = 0; pp < P; ++pp) {
        tensor sl = ggml_cont(m, ggml_view_4d(m, acc, D, 1, H, N, acc->nb[1], acc->nb[2],
                                              acc->nb[3], (size_t)pp * acc->nb[1]));
        sum = sum ? ggml_add(m, sum, sl) : sl;
    }
    tensor o = ggml_reshape_2d(m, ggml_cont(m, sum), E, N);   // 채널 = h·D + d
    return ggml_add(m, query, linear(m[(p + ".output_proj").c_str()], o));
}

// 레벨별 feature 를 {E, HW} 로 펴서 잇고(mmdet `feat_flatten` 순서) deformable encoder 를 돈다.
// 다중 레벨 위치 인코딩(level_embed 포함)과 encoder 참조점은 상수라 구워 뒀다.
void ddq_levels(std::vector<tensor> const& feats, std::vector<msd_level>& lv) {
    int64_t off = 0;
    for (tensor f : feats) {
        lv.push_back({f->ne[1], f->ne[2], off});
        off += f->ne[1] * f->ne[2];
    }
}

static tensor msd_encode(model_ref m, std::vector<tensor> const& feats,
                         anchor_head_cfg const& c, std::vector<msd_level>& lv) {
    tensor mem = nullptr;
    for (tensor f : feats) {
        tensor t = ggml_reshape_2d(m, ggml_cont(m, f), f->ne[0], f->ne[1] * f->ne[2]);
        mem = mem ? ggml_concat(m, mem, t, 1) : t;
    }
    ddq_levels(feats, lv);
    tensor pos = m.weights("pos_embed");
    tensor enc_ref = m.weights("enc_ref");
    for (int i = 0; i < c.enc_layers; ++i) {
        const std::string L = "encoder.layers." + std::to_string(i);
        mem = msdeform(m, L + ".self_attn", mem, pos, mem, enc_ref, lv, c.n_heads, c.n_points);
        mem = layer_norm(m[(L + ".norms.0").c_str()], mem);
        mem = ffn(m, L + ".ffn", mem);
        mem = layer_norm(m[(L + ".norms.1").c_str()], mem);
    }
    return mem;
}

// `nn.Sequential(Linear, ReLU, Linear, ReLU, Linear)` — 인덱스가 0·2·4 다(MLP 와 다르다).
static tensor seq_mlp3(model_ref m, const std::string& p, tensor x) {
    x = linear(m[(p + ".0").c_str()], x);
    x = linear(m[(p + ".2").c_str()], ggml_relu(m, x));
    return linear(m[(p + ".4").c_str()], ggml_relu(m, x));
}

void deformable_detr_head_forward(model_ref m, std::vector<tensor> const& feats,
                                  anchor_head_cfg const& c, head_outputs& out) {
    std::vector<msd_level> lv;
    tensor mem = msd_encode(m, feats, c, lv);

    // query_embedding 은 {2E, nq} 로 (query_pos, query) 가 붙어 있다. torch 의 split(dim=-1)
    // 이라 ne0 앞쪽이 query_pos 다.
    tensor qe = m.weights("query_embedding.weight");
    const int64_t E = c.embed_dims, nq = qe->ne[1];
    tensor q_pos = ggml_cont(m, ggml_view_2d(m, qe, E, nq, qe->nb[1], 0));
    tensor q = ggml_cont(m, ggml_view_2d(m, qe, E, nq, qe->nb[1], (size_t)E * qe->nb[0]));
    tensor ref = ggml_sigmoid(m, linear(m["reference_points_fc"], q_pos));   // {2, nq}

    for (int i = 0; i < c.dec_layers; ++i) {
        const std::string L = "decoder.layers." + std::to_string(i);
        // self-attn 은 평범한 MultiheadAttention 이다(query 끼리).
        q = mha(m, L + ".self_attn", q, q_pos, q, q_pos, c);
        q = layer_norm(m[(L + ".norms.0").c_str()], q);
        q = msdeform(m, L + ".cross_attn", q, q_pos, mem, ref, lv, c.n_heads, c.n_points);
        q = layer_norm(m[(L + ".norms.1").c_str()], q);
        q = ffn(m, L + ".ffn", q);
        q = layer_norm(m[(L + ".norms.2").c_str()], q);

        // cls/reg branch 는 층마다 따로 있다(공유 config 면 같은 가중치가 복사돼 있다).
        const std::string B = "bbox_head." + std::string("cls_branches.") + std::to_string(i);
        const std::string R = "bbox_head.reg_branches." + std::to_string(i);
        tensor cls = linear(m[B.c_str()], q);
        tensor reg = seq_mlp3(m, R, q);
        // reference 는 2차원이라 앞 2채널에만 더한다. {2,nq} 를 {4,nq} 자리에 맞춰 0 을 잇는다.
        tensor rinv = inv_sigmoid(m, ref, 1e-5f);
        rinv = ggml_concat(m, rinv, ggml_scale(m, rinv, 0.0f), 0);          // {4, nq}
        tensor box = ggml_sigmoid(m, ggml_add(m, reg, rinv));
        detr_emit(out, i, cls, box);
    }
}

void dino_head_forward(model_ref m, std::vector<tensor> const& feats,
                       anchor_head_cfg const& c, head_outputs& out) {
    std::vector<msd_level> lv;
    tensor mem = msd_encode(m, feats, c, lv);
    const int64_t E = c.embed_dims;
    const std::string HB = "bbox_head.";
    const std::string enc_i = std::to_string(c.dec_layers);   // 마지막+1 번째 분기가 encoder 용

    // ── two-stage: encoder 출력에서 proposal 을 만들어 점수 상위 k 개를 고른다 ──
    // 유효하지 않은 자리는 memory 를 0 으로 덮는다(`masked_fill`). 마스크는 상수라 구워 뒀다.
    tensor om = ggml_mul(m, mem, m.weights("enc_valid"));
    om = layer_norm(m["memory_trans_norm"], linear(m["memory_trans_fc"], om));
    tensor ecls = linear(m[(HB + "cls_branches." + enc_i).c_str()], om);          // {ncls, T}
    tensor ecoord = ggml_add(m, seq_mlp3(m, HB + "reg_branches." + enc_i, om),
                             m.weights("enc_proposals"));                        // {4, T}

    // 클래스축 최댓값 → 토큰 점수. ggml 에 max-reduce 가 없어 채널을 하나씩 접는다
    // (`emax` = (a+b+|a-b|)/2). ncls 는 80 이라 싸다.
    const int64_t T = ecls->ne[1], ncls = ecls->ne[0];
    auto ch1 = [&](tensor t, int64_t k) {
        return ggml_cont(m, ggml_view_2d(m, t, 1, T, t->nb[1], (size_t)k * t->nb[0]));
    };
    tensor score = ch1(ecls, 0);
    for (int64_t k = 1; k < ncls; ++k) score = emax(m, score, ch1(ecls, k));

    // ⚠️ `ggml_top_k` 를 쓰면 안 된다 — **순서를 보장하지 않는다.** CPU 커널이
    //    "order is not important" 라며 앞 두 개를 **일부러 뒤바꾼다.**
    //    여기서는 query 순서가 곧 출력 순서라 정렬이 필요하다 → argsort 기반을 쓴다.
    //    (실측: 고른 900개 집합은 같은데 순서만 달라 rel_L1 0.489 → 정렬 후 3.3e-04)
    tensor idx = ggml_argsort_top_k(m, ggml_reshape_1d(m, score, T), (int)c.num_queries);
    tensor ref = ggml_sigmoid(m, ggml_get_rows(m, ecoord, idx));                  // {4, nq}

    tensor q = m.weights("query_embedding.weight");                              // {E, nq}
    for (int i = 0; i < c.dec_layers; ++i) {
        const std::string L = "decoder.layers." + std::to_string(i);
        // 참조 박스의 sine 인코딩에서 query 위치를 만든다(층마다 다시 — ref 가 갱신되므로).
        tensor q_pos = mlp(m, "decoder.ref_point_head", coord_encode(m, ref), 2);
        q = mha(m, L + ".self_attn", q, q_pos, q, q_pos, c);
        q = layer_norm(m[(L + ".norms.0").c_str()], q);
        q = msdeform(m, L + ".cross_attn", q, q_pos, mem, ref, lv, c.n_heads, c.n_points);
        q = layer_norm(m[(L + ".norms.1").c_str()], q);
        q = ffn(m, L + ".ffn", q);
        q = layer_norm(m[(L + ".norms.2").c_str()], q);

        const std::string R = HB + "reg_branches." + std::to_string(i);
        tensor rinv = inv_sigmoid(m, ref, 1e-5f);
        // ⚠️ DAB 과 같은 함정: **다음 층 참조점은 norm 이전 출력**으로, head 출력은 norm 이후로.
        tensor next_ref = ggml_sigmoid(m, ggml_add(m, seq_mlp3(m, R, q), rinv));
        tensor h = layer_norm(m["decoder.norm"], q);
        tensor cls = linear(m[(HB + "cls_branches." + std::to_string(i)).c_str()], h);
        tensor box = ggml_sigmoid(m, ggml_add(m, seq_mlp3(m, R, h), rinv));
        detr_emit(out, i, cls, box);
        ref = next_ref;
    }
}

// ── DDQ ───────────────────────────────────────────────────────────────────────
// 패스 0: encoder 까지 돌고 proposal 점수/박스를 낸다. 여기서 멈추는 이유는 head.h 참조.
void ddq_encode_forward(model_ref m, std::vector<tensor> const& feats,
                        anchor_head_cfg const& c, ddq_stage& s) {
    std::vector<msd_level> lv;
    s.mem = msd_encode(m, feats, c, lv);
    const std::string HB = "bbox_head.", ei = std::to_string(c.dec_layers);
    // 유효하지 않은 자리는 memory 를 0 으로 덮는다(mmdet `masked_fill`). 마스크는 상수다.
    tensor om = ggml_mul(m, s.mem, m.weights("enc_valid"));
    s.om = layer_norm(m["memory_trans_norm"], linear(m["memory_trans_fc"], om));
    s.ecls = linear(m[(HB + "cls_branches." + ei).c_str()], s.om);
    s.ecoord = ggml_add(m, seq_mlp3(m, HB + "reg_branches." + ei, s.om),
                        m.weights("enc_proposals"));
    // ⚠️ query 내용은 `output_memory` 가 아니라 **`query_map(memory)`** 에서 뽑는다.
    //    mmdet 주석: "we fuse the feature map embedding of distinct positions as the
    //    content part". om 을 쓰면 층 0 부터 어긋난다(실측 rel_L1 0.107).
    s.qmap = linear(m["query_map"], s.mem);
    for (tensor x : {s.mem, s.om, s.ecls, s.ecoord, s.qmap}) ggml_set_output(x);
}

// 패스 i: decoder 한 층. query·ref·mask 는 호스트가 채운 그래프 입력이다.
void ddq_layer_forward(model_ref m, std::vector<msd_level> const& lv,
                       anchor_head_cfg const& c, int layer_index,
                       tensor mem, tensor query, tensor ref, tensor mask, ddq_stage& s) {
    const std::string L = "decoder.layers." + std::to_string(layer_index);
    const std::string HB = "bbox_head.", R = HB + "reg_branches." + std::to_string(layer_index);

    tensor q_pos = mlp(m, "decoder.ref_point_head", coord_encode(m, ref), 2);
    tensor q = mha(m, L + ".self_attn", query, q_pos, query, q_pos, c, mask);
    q = layer_norm(m[(L + ".norms.0").c_str()], q);
    q = msdeform(m, L + ".cross_attn", q, q_pos, mem, ref, lv, c.n_heads, c.n_points);
    q = layer_norm(m[(L + ".norms.1").c_str()], q);
    q = ffn(m, L + ".ffn", q);
    q = layer_norm(m[(L + ".norms.2").c_str()], q);

    tensor rinv = inv_sigmoid(m, ref, 1e-3f);
    // DINO 와 같은 규약: 참조점 갱신은 norm **이전** 출력으로, head 출력은 norm 이후로.
    s.ref = ggml_sigmoid(m, ggml_add(m, seq_mlp3(m, R, q), rinv));
    tensor h = layer_norm(m["decoder.norm"], q);
    s.query = q;
    s.cls = linear(m[(HB + "cls_branches." + std::to_string(layer_index)).c_str()], h);
    s.box = ggml_sigmoid(m, ggml_add(m, seq_mlp3(m, R, h), rinv));
    for (tensor t : {s.query, s.ref, s.cls, s.box}) ggml_set_output(t);
}

// ── CornerNet ─────────────────────────────────────────────────────────────────
// corner pooling 은 mmcv 에서 `torch.cummax` 다(방향에 따라 flip 을 앞뒤로 건다).
// ggml 에는 cummax 도, **원소별 max 조차** 없다. 둘 다 있는 것으로 만든다.

// max(a,b) = (a + b + |a-b|) / 2. ggml_abs·add·sub·scale 만 쓴다.
static tensor emax(model_ref m, tensor a, tensor b) {
    tensor s = ggml_add(m, a, b);
    tensor d = ggml_abs(m, ggml_sub(m, a, b));
    return ggml_scale(m, ggml_add(m, s, d), 0.5f);
}

// x 를 축 ax(0=W, 1=H)로 k 칸 민다. 밀려 들어오는 자리는 **0** 이다.
//   fwd=true  → out[i] = x[i-k]   (i<k    이면 0)
//   fwd=false → out[i] = x[i+k]   (i>=N-k 이면 0)
// `ggml_view` 는 ne0 축에 step 을 못 주지만, pad 로 늘린 뒤 앞/뒤를 잘라내는 것은 된다.
static tensor shift_axis(model_ref m, tensor x, int ax, int k, bool fwd) {
    const int64_t N = x->ne[ax];
    int lp[2] = {0, 0}, rp[2] = {0, 0};
    (fwd ? lp : rp)[ax] = k;
    tensor p = ggml_pad_ext(m, ggml_cont(m, x), lp[0], rp[0], lp[1], rp[1], 0, 0, 0, 0);
    int64_t ne[4] = {p->ne[0], p->ne[1], p->ne[2], p->ne[3]};
    ne[ax] = N;
    const size_t off = fwd ? 0 : (size_t)k * p->nb[ax];
    return ggml_cont(m, ggml_view_4d(m, p, ne[0], ne[1], ne[2], ne[3],
                                     p->nb[1], p->nb[2], p->nb[3], off));
}

// torch.cummax 를 **prefix/suffix max 스캔**으로 낸다 — k = 1,2,4,… 로 log2(N) 단계다
// (128 이면 7단계). 각 단계는 "자기 자신과 k 칸 민 자기 자신의 max" 다.
//   fwd=true  → out[i] = max(x[0..i])    (mmcv 의 flip=False)
//   fwd=false → out[i] = max(x[i..N-1])  (flip=True — flip 대신 반대로 민다)
//
// ⚠️ **밀려 들어오는 자리를 0 으로 채운다.** 0 이 max 의 항등원이려면 입력이 음수가 아니어야
//    한다. CornerPool 입력은 `ConvModule(BN+ReLU)` 뒤라 항상 ≥0 이므로 성립한다.
//    음수가 들어오는 자리에 이 함수를 쓰면 **크래시 없이 값만 틀린다.**
static tensor cummax_axis(model_ref m, tensor x, int ax, bool fwd) {
    for (int64_t k = 1; k < x->ne[ax]; k <<= 1)
        x = emax(m, x, shift_axis(m, x, ax, (int)k, fwd));
    return x;
}

// mmdet `BiCornerPool`. topleft=true 면 방향이 ['top','left'], false 면 ['bottom','right'].
// mmcv 의 cummax_dim_flip: bottom=(H,prefix) top=(H,suffix) right=(W,prefix) left=(W,suffix).
// whcn 이라 H=ne1, W=ne0 이다.
static tensor bi_corner_pool(model_ref m, tensor x, const std::string& p, bool topleft) {
    tensor d1 = ggml_relu(m, conv_same(m, p + ".direction1_conv.conv", x));
    tensor d2 = ggml_relu(m, conv_same(m, p + ".direction2_conv.conv", x));
    tensor f1 = cummax_axis(m, d1, 1, !topleft);          // top(suffix) / bottom(prefix)
    tensor f2 = cummax_axis(m, d2, 0, !topleft);          // left(suffix) / right(prefix)
    // aftpool_conv 와 conv1 은 act_cfg=None 이다 — ReLU 를 붙이면 안 된다.
    tensor a  = conv_same(m, p + ".aftpool_conv.conv", ggml_add(m, f1, f2));
    tensor c1 = conv_same(m, p + ".conv1.conv", x);
    tensor r  = ggml_relu(m, ggml_add(m, a, c1));
    return ggml_relu(m, conv_same(m, p + ".conv2.conv", r));
}

// `_make_layers` = Sequential(ConvModule(3x3, norm 없음, ReLU), ConvModule(1x1, norm·act 없음)).
// **둘 다 ConvModule 이다** — 두 번째도 `.1.conv` 지 `.1` 이 아니다.
static tensor corner_pred(model_ref m, tensor x, const std::string& p) {
    return conv_same(m, p + ".1.conv", ggml_relu(m, conv_same(m, p + ".0.conv", x)));
}

void corner_head_forward(model_ref m, std::vector<tensor> const& feats,
                         anchor_head_cfg const& c, head_outputs& out) {
    const std::string H = "bbox_head";
    out.extra.push_back({"brof", {}});
    if (c.corner_emb) { out.extra.push_back({"tlemb", {}}); out.extra.push_back({"bremb", {}}); }
    if (c.corner_centripetal) {
        out.extra.push_back({"tlgs", {}}); out.extra.push_back({"brgs", {}});
        out.extra.push_back({"tlcs", {}}); out.extra.push_back({"brcs", {}});
    }

    for (size_t l = 0; l < feats.size(); ++l) {
        const std::string lv = "." + std::to_string(l);
        tensor f = cwhn_to_contiguous_2d(m, feats[l]);
        tensor tl = bi_corner_pool(m, f, H + ".tl_pool" + lv, true);
        tensor br = bi_corner_pool(m, f, H + ".br_pool" + lv, false);

        auto emit = [&](std::vector<tensor>& dst, const char* tag, tensor t) {
            t = contiguous_2d_to_cwhn(m, t);
            ggml_format_name(t, "%s_%zu", tag, l);
            dst.push_back(t);
        };
        emit(out.cls, "cls", corner_pred(m, tl, H + ".tl_heat" + lv));
        emit(out.box, "box", corner_pred(m, br, H + ".br_heat" + lv));
        emit(out.ctr, "ctr", corner_pred(m, tl, H + ".tl_off"  + lv));
        emit(out.extra[0].second, "brof", corner_pred(m, br, H + ".br_off" + lv));
        if (c.corner_emb) {
            emit(out.extra[1].second, "tlemb", corner_pred(m, tl, H + ".tl_emb" + lv));
            emit(out.extra[2].second, "bremb", corner_pred(m, br, H + ".br_emb" + lv));
        }
        if (c.corner_centripetal) {
            // guiding shift(2채널) → 1x1 conv(bias 없음) 로 DCN offset 18채널 → **DCNv1**
            // (mask 없음) 으로 pool feature 를 재샘플링 → centripetal shift.
            // ⚠️ mmcv DeformConv2d 의 offset 은 **기준 격자로부터의 델타**다. vfnet 처럼
            //    base 를 빼면 안 된다 — 거기선 절대 위치를 만들어 놓고 뺐던 것이다.
            auto branch = [&](tensor pool, const char* side, const char* tag_gs, const char* tag_cs,
                              std::vector<tensor>& dgs, std::vector<tensor>& dcs) {
                const std::string P = H + "." + side;
                tensor gs = corner_pred(m, pool, P + "_guiding_shift" + lv);
                tensor off = conv_same(m, P + "_dcn_offset" + lv + ".conv", gs);
                tensor fa = conv_2d_deform(m, pool,
                                           m.find((P + "_feat_adaption" + lv + ".weight").c_str()),
                                           off, nullptr, 1, 1);
                emit(dgs, tag_gs, gs);
                emit(dcs, tag_cs, corner_pred(m, fa, P + "_centripetal_shift" + lv));
            };
            branch(tl, "tl", "tlgs", "tlcs", out.extra[1].second, out.extra[3].second);
            branch(br, "br", "brgs", "brcs", out.extra[2].second, out.extra[4].second);
        }
    }
}

// ── YOLOv3 ────────────────────────────────────────────────────────────────────
// 레벨마다 bridge conv 하나(ConvModule: conv+BN+LeakyReLU, BN 은 프론트엔드가 접었다) 뒤에
// 1x1 예측 conv 가 붙는다. 타워 반복이 없어 `cls_convs_prefix.<레벨>.conv` 로 한 번만 태운다.
void yolo_head_forward(model_ref m, std::vector<tensor> const& feats,
                       anchor_head_cfg const& c, head_outputs& out) {
    for (size_t l = 0; l < feats.size(); ++l) {
        const std::string lv = "." + std::to_string(l);
        tensor x = cwhn_to_contiguous_2d(m, feats[l]);
        x = head_act(m, conv_same(m, c.cls_convs_prefix + lv + ".conv", x), c);
        tensor pred = contiguous_2d_to_cwhn(m, conv_same(m, c.cls_head + lv, x));
        ggml_format_name(pred, "cls_%zu", l);
        out.cls.push_back(pred);
    }
}

// ── YOLOF ─────────────────────────────────────────────────────────────────────
// 다른 계열과 두 군데가 다르다.
//  ① cls/reg 타워 깊이가 서로 다르다(num_cls_convs=2, num_reg_convs=4).
//  ② objectness 를 따로 예측해 **로그공간에서 cls 에 흡수**시킨다. 그래서 세 번째 출력이 없다.
//
//     normalized_cls = cls + obj - log(1 + e^cls + e^obj)
//
// mmdet 은 `view(N, -1, num_classes, H, W)` 로 채널을 [anchor][class] 로 가른 뒤 obj 를
// 클래스축으로 broadcast 한다. 여기서도 같은 순서로 가른다 — 채널 index = a*nc + c 이므로
// **nc 를 안쪽 축**에 두어야 한다.
void yolof_head_forward(model_ref m, std::vector<tensor> const& feats,
                        anchor_head_cfg const& c, head_outputs& out) {
    const int nreg = c.reg_stacked_convs > 0 ? c.reg_stacked_convs : c.stacked_convs;
    for (size_t l = 0; l < feats.size(); ++l) {
        tensor f = cwhn_to_contiguous_2d(m, feats[l]);
        tensor cc = conv_tower(m, f, c, c.cls_convs_prefix, l, c.stacked_convs);
        tensor rr = conv_tower(m, f, c, c.reg_convs_prefix, l, nreg);

        tensor cls = conv_same(m, c.cls_head, cc);
        tensor box = conv_same(m, c.reg_head, rr);
        tensor obj = conv_same(m, c.centerness_head, rr);   // object_pred (na 채널)

        const int64_t W = cls->ne[0], H = cls->ne[1];
        const int64_t nc = c.num_classes, na = c.num_base;
        tensor c4 = ggml_reshape_4d(m, ggml_cont(m, cls), W, H, nc, na);
        tensor o4 = ggml_reshape_4d(m, ggml_cont(m, obj), W, H, 1,  na);
        // ggml_add 는 src1 을 broadcast 한다(ne2: 1 → nc). clamp(max=INF) 는 항등이라 생략.
        tensor sum = ggml_add(m, c4, o4);
        tensor exp = ggml_add(m, ggml_exp(m, c4), ggml_exp(m, o4));
        // 1 + e 를 만들 상수 텐서가 없으므로 scale_bias(a*1 + 1) 로 낸다.
        tensor nrm = ggml_sub(m, sum, ggml_log(m, ggml_scale_bias(m, exp, 1.0f, 1.0f)));
        cls = ggml_reshape_4d(m, ggml_cont(m, nrm), W, H, nc * na, 1);

        cls = contiguous_2d_to_cwhn(m, cls);
        box = contiguous_2d_to_cwhn(m, box);
        ggml_format_name(cls, "cls_%zu", l);
        ggml_format_name(box, "box_%zu", l);
        out.cls.push_back(cls);
        out.box.push_back(box);
    }
}

// 종전 진입점 — RetinaNet PoC 러너가 쓰던 이름을 유지한다(centerness 없는 계열 전용).
void anchor_head_forward(model_ref m, std::vector<tensor> const& feats,
                         anchor_head_cfg const& c,
                         std::vector<tensor>& cls_out, std::vector<tensor>& box_out) {
    head_outputs o;
    tower_head_forward(m, feats, c, o);
    cls_out = std::move(o.cls);
    box_out = std::move(o.box);
}

// ── VFNet ─────────────────────────────────────────────────────────────────────
// runner 는 whcn 네이티브(가중치 {KW,KH,Cin,Cout}). head 내부도 contiguous_2d(=whcn) 에서 돈다
// (anchor_head_forward 와 동일 규약). 채널축은 ne[2]. offset/deform 도 whcn 로 조립.
//
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
    tensor base = dcn_base_whcn(m, dcn_base);
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

        // 초기 bbox_pred = exp(scale · vfnet_reg(vfnet_reg_conv(reg_feat))) · reg_denom
        // ⚠️ `scale` 은 **학습되는 값**이다. 1.0 으로 박으면 랜덤 초기화에서만 맞는다.
        tensor reg_init = conv_gn_relu(m, reg_feat, H + ".vfnet_reg_conv", c.gn_groups);
        tensor bbox_pred = conv_2d(m[(H + ".vfnet_reg").c_str()], reg_init, 1, 1);
        if (tensor s = m.find((H + ".scales." + std::to_string(l) + ".scale").c_str()))
            bbox_pred = ggml_mul(m, bbox_pred, s);
        bbox_pred = ggml_scale(m, ggml_exp(m, bbox_pred), reg_denom);

        // star deformable offset
        tensor offset = star_dcn_offset(m, bbox_pred, stride, base);

        // refine: reg_feat 를 deform conv → exp·bbox_pred
        tensor w_reg_dcn = m[(H + ".vfnet_reg_refine_dconv").c_str()].weights("weight");
        tensor reg_ref = ggml_relu(m, conv_2d_deform(m, reg_feat, w_reg_dcn, offset, nullptr, 1, 1));
        tensor box = conv_2d(m[(H + ".vfnet_reg_refine").c_str()], reg_ref, 1, 1);
        if (tensor s = m.find((H + ".scales_refine." + std::to_string(l) + ".scale").c_str()))
            box = ggml_mul(m, box, s);
        box = ggml_mul(m, ggml_exp(m, box), bbox_pred);

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

// ── RepPoints ─────────────────────────────────────────────────────────────────
// ne2(채널) 축 reduce. ggml 의 mean/sum 은 ne0 만 줄이므로 축을 데려왔다 되돌린다.
static tensor reduce_mean_ne2(model_ref m, tensor x) {
    tensor t = ggml_cont(m, ggml_permute(m, x, 2, 1, 0, 3));   // ne0 ↔ ne2
    t = ggml_mean(m, t);
    return ggml_cont(m, ggml_permute(m, t, 2, 1, 0, 3));
}

// 표본표준편차(torch.std 기본 = unbiased). 평균을 빼도 std 는 안 변하므로 원본에 바로 건다.
static tensor reduce_std_ne2(model_ref m, tensor x, int n) {
    tensor d = ggml_sub(m, x, reduce_mean_ne2(m, x));          // {W,H,1,1} broadcast
    tensor var = reduce_mean_ne2(m, ggml_sqr(m, d));
    return ggml_sqrt(m, ggml_scale(m, var, (float)n / (float)(n - 1)));
}

// 점 집합 → bbox (transform_method='moment').
// pts 는 {W,H,2·num_points,1}, 채널이 y_first 로 [y0,x0,y1,x1,…] 이다 — y 와 x 가 **한 칸씩
// 번갈아** 놓여 있어서 stride 2 짜리 view 로 갈라낸다(연속 복사 없이).
static tensor points2bbox_moment(model_ref m, tensor pts, tensor moment_transfer,
                                 int num_points) {
    const int64_t W = pts->ne[0], H = pts->ne[1], N = pts->ne[3];
    auto every2 = [&](int start) {          // 채널 start, start+2, … → {W,H,num_points,1}
        return ggml_cont(m, ggml_view_4d(m, pts, W, H, num_points, N,
                                         pts->nb[1], 2 * pts->nb[2], pts->nb[3],
                                         (size_t)start * pts->nb[2]));
    };
    tensor py = every2(0), px = every2(1);

    tensor y_mean = reduce_mean_ne2(m, py), x_mean = reduce_mean_ne2(m, px);
    tensor y_std = reduce_std_ne2(m, py, num_points);
    tensor x_std = reduce_std_ne2(m, px, num_points);

    // 추론에서는 moment_mul 항이 상쇄된다(detach 가 값 그대로) → moment_transfer 자체.
    auto elem = [&](int i) {
        return ggml_view_1d(m, moment_transfer, 1, (size_t)i * moment_transfer->nb[0]);
    };
    tensor half_w = ggml_mul(m, x_std, ggml_exp(m, ggml_cont(m, elem(0))));
    tensor half_h = ggml_mul(m, y_std, ggml_exp(m, ggml_cont(m, elem(1))));

    tensor b = ggml_sub(m, x_mean, half_w);                       // x1
    b = ggml_concat(m, b, ggml_sub(m, y_mean, half_h), 2);        // y1
    b = ggml_concat(m, b, ggml_add(m, x_mean, half_w), 2);        // x2
    b = ggml_concat(m, b, ggml_add(m, y_mean, half_h), 2);        // y2
    return b;
}

void reppoints_head_forward(model_ref m, std::vector<tensor> const& feats,
                            anchor_head_cfg const& c, tensor dcn_base,
                            head_outputs& out) {
    const std::string H = "bbox_head";
    // dcn_base 원소 수(18)에서 점 개수를 얻는다 — cfg 에 또 하나 두지 않는다.
    const int num_points = (int)ggml_nelements(dcn_base) / 2;
    tensor base = dcn_base_whcn(m, dcn_base);

    for (size_t l = 0; l < feats.size(); ++l) {
        tensor f = cwhn_to_contiguous_2d(m, feats[l]);
        tensor cls_feat = conv_tower(m, f, c, c.cls_convs_prefix, l);
        tensor pts_feat = conv_tower(m, f, c, c.reg_convs_prefix, l);

        // ① 점을 놓는다. center_init=True 라 points_init = 0 이므로 더할 게 없다.
        tensor pts_init = conv_same(m, H + ".reppoints_pts_init_out",
                                    ggml_relu(m, conv_same(m, H + ".reppoints_pts_init_conv",
                                                           pts_feat)));

        // ② 그 점을 offset 삼아 deform conv. 추론에서는 gradient_mul 항이 상쇄된다.
        //    ggml deform 커널이 기준 격자를 내부에서 더하므로 mmdet 과 같이 base 를 뺀 값을 준다.
        tensor dcn_offset = ggml_sub(m, pts_init, base);

        tensor w_cls = m[(H + ".reppoints_cls_conv").c_str()].weights("weight");
        tensor cls = conv_same(m, H + ".reppoints_cls_out",
                               ggml_relu(m, conv_2d_deform(m, cls_feat, w_cls, dcn_offset,
                                                           nullptr, 1, 1)));

        tensor w_ref = m[(H + ".reppoints_pts_refine_conv").c_str()].weights("weight");
        tensor pts_ref = conv_same(m, H + ".reppoints_pts_refine_out",
                                   ggml_relu(m, conv_2d_deform(m, pts_feat, w_ref, dcn_offset,
                                                               nullptr, 1, 1)));
        pts_ref = ggml_add(m, pts_ref, pts_init);   // refine 은 init 에 대한 잔차다

        tensor box = points2bbox_moment(m, pts_ref, m.weights((H + ".moment_transfer").c_str()),
                                        num_points);

        cls = contiguous_2d_to_cwhn(m, cls);
        box = contiguous_2d_to_cwhn(m, box);
        ggml_format_name(cls, "cls_%zu", l);
        ggml_format_name(box, "box_%zu", l);
        out.cls.push_back(cls);
        out.box.push_back(box);
    }
}

// ── TOOD ──────────────────────────────────────────────────────────────────────
// ne2(채널) 축을 [off, off+n) 구간만 잘라낸다.
static tensor slice_ne2(model_ref m, tensor x, int64_t off, int64_t n) {
    return ggml_view_4d(m, x, x->ne[0], x->ne[1], n, x->ne[3],
                        x->nb[1], x->nb[2], x->nb[3], (size_t)off * x->nb[2]);
}

// TaskDecomposition. mmdet 은 layer attention 을 **conv 가중치에 먼저 곱해** 동적 커널을
// 만들고 bmm 한다(메모리 절약). 여기서는 분배법칙으로 뒤집는다 —
//
//     Σ_{s,i} (a_s · W[o, s,i]) · x[s,i]  =  Σ_{s,i} W[o, s,i] · (a_s · x[s,i])
//
// 즉 **입력 청크에 먼저 곱하면** 커널이 정적으로 남아 평범한 1×1 conv 가 된다.
// 값은 같고, 동적 가중치를 그래프에 만들 필요가 없다.
static tensor task_decomp(model_ref m, tensor feat, tensor avg_feat,
                          const std::string& p, int stacked, int chunk_c, int gn_groups) {
    // layer_attention: conv1x1 → relu → conv1x1 → sigmoid  (→ {1,1,stacked,1})
    tensor a = conv_2d(m[(p + ".layer_attention.0").c_str()], avg_feat, 1, 0);
    a = conv_2d(m[(p + ".layer_attention.2").c_str()], ggml_relu(m, a), 1, 0);
    a = ggml_sigmoid(m, a);

    tensor scaled = nullptr;
    for (int s = 0; s < stacked; ++s) {
        tensor as = ggml_view_4d(m, a, 1, 1, 1, 1, a->nb[1], a->nb[2], a->nb[3],
                                 (size_t)s * a->nb[2]);          // 스칼라 a_s
        tensor xs = ggml_mul(m, ggml_cont(m, slice_ne2(m, feat, (int64_t)s * chunk_c, chunk_c)),
                             ggml_cont(m, as));
        scaled = scaled ? ggml_concat(m, scaled, xs, 2) : xs;
    }
    tensor y = conv_2d(m[(p + ".reduction_conv.conv").c_str()], scaled, 1, 0);
    y = group_norm_affine(m[(p + ".reduction_conv.gn").c_str()], y, gn_groups);
    return ggml_relu(m, y);
}

// 격자 중심점(stride 로 나눈 좌표계). **오프셋을 0.5 로 박으면 안 된다** — TOOD 는
// ATSSHead 를 상속해 AnchorGenerator(center_offset=0)를 쓰므로 중심이 정확히 i 다.
// FCOS 계열의 MlvlPointGenerator 만 i+0.5 다. 0.5 를 잘못 넣으면 전 레벨에서 박스가
// 반 칸씩 밀린다(실측: cpp−ref 가 어느 레벨에서나 +0.47).
static tensor grid_centers(model_ref m, int64_t W, int64_t H, bool along_x, float off) {
    tensor v = ggml_arange(m, off, (float)(along_x ? W : H) + off, 1.0f);
    if (along_x) return ggml_repeat_4d(m, ggml_reshape_4d(m, v, W, 1, 1, 1), W, H, 1, 1);
    return ggml_repeat_4d(m, ggml_reshape_4d(m, v, 1, H, 1, 1), W, H, 1, 1);
}

void tood_head_forward(model_ref m, std::vector<tensor> const& feats,
                       anchor_head_cfg const& c, head_outputs& out) {
    const std::string H = "bbox_head";
    const int S = c.stacked_convs, C = c.feat_channels;

    for (size_t l = 0; l < feats.size(); ++l) {
        tensor x = cwhn_to_contiguous_2d(m, feats[l]);
        const int64_t W = x->ne[0], FH = x->ne[1];

        // ① inter conv 스택. 각 단 출력을 전부 모아 이어붙인다(= task interactive feature).
        tensor feat = nullptr;
        for (int i = 0; i < S; ++i) {
            x = conv_gn_relu(m, x, c.cls_convs_prefix + "." + std::to_string(i), c.gn_groups);
            feat = feat ? ggml_concat(m, feat, x, 2) : x;
        }
        tensor avg = ggml_pool_2d(m, feat, GGML_OP_POOL_AVG, W, FH, W, FH, 0, 0);

        // ② task decomposition — cls 와 reg 가 같은 feat 에서 서로 다른 관점을 뽑는다.
        tensor cls_feat = task_decomp(m, feat, avg, H + ".cls_decomp", S, C, c.gn_groups);
        tensor reg_feat = task_decomp(m, feat, avg, H + ".reg_decomp", S, C, c.gn_groups);

        // ③ cls: 분류 로짓과 정렬 확률의 기하평균. sqrt(σ(a)·σ(b)) 로 그대로 편다
        //    (mmdet 은 autograd.Function 이지만 그건 역전파용 최적화다).
        tensor logits = conv_same(m, H + ".tood_cls", cls_feat);
        tensor prob = conv_2d(m[(H + ".cls_prob_module.0").c_str()], feat, 1, 0);
        prob = conv_same(m, H + ".cls_prob_module.2", ggml_relu(m, prob));
        tensor cls = ggml_sqrt(m, ggml_mul(m, ggml_sigmoid(m, logits), ggml_sigmoid(m, prob)));

        // ④ reg: 거리 → 격자 좌표계 bbox. distance2bbox 를 그래프로 편다.
        tensor dist = ggml_exp(m, conv_same(m, H + ".tood_reg", reg_feat));
        dist = apply_scale(m, dist, c, l);
        tensor px = grid_centers(m, W, FH, true, c.center_offset),
               py = grid_centers(m, W, FH, false, c.center_offset);
        auto d = [&](int i) { return ggml_cont(m, slice_ne2(m, dist, i, 1)); };
        tensor rb = ggml_sub(m, px, d(0));                          // x1 = px - l
        rb = ggml_concat(m, rb, ggml_sub(m, py, d(1)), 2);          // y1 = py - t
        rb = ggml_concat(m, rb, ggml_add(m, px, d(2)), 2);          // x2 = px + r
        rb = ggml_concat(m, rb, ggml_add(m, py, d(3)), 2);          // y2 = py + b

        // ⑤ deform sampling — 1×1 ones 커널 · 채널별 offset. 실질은 채널마다 다른 위치에서
        //    bbox 를 다시 읽는 bilinear 재샘플이다. groups=4 를 커널이 못 받으므로
        //    (vendored ggml 은 안 고친다) **채널 4갈래로 쪼개 groups=1 로 부르고 잇는다.**
        tensor off = conv_2d(m[(H + ".reg_offset_module.0").c_str()], feat, 1, 0);
        off = conv_same(m, H + ".reg_offset_module.2", ggml_relu(m, off));
        tensor one = ggml_reshape_4d(m, ggml_arange(m, 1.0f, 2.0f, 1.0f), 1, 1, 1, 1);
        tensor ones = one;                       // 1×1 ones 커널 {1,1,1,1}
        tensor bp = nullptr;
        for (int i = 0; i < 4; ++i) {
            tensor xi = ggml_cont(m, slice_ne2(m, rb, i, 1));
            tensor oi = ggml_cont(m, slice_ne2(m, off, 2 * i, 2));
            tensor yi = conv_2d_deform(m, xi, ones, oi, nullptr, 1, 0);
            bp = bp ? ggml_concat(m, bp, yi, 2) : yi;
        }

        // ⑥ 재샘플이 좌상단/우하단을 뒤집어 놓은 자리는 원래 박스로 되돌린다.
        //    ggml 에 where/비교가 없어 step 으로 마스크를 만든다.
        auto b = [&](int i) { return ggml_cont(m, slice_ne2(m, bp, i, 1)); };
        tensor bad = ggml_add(m, ggml_step(m, ggml_sub(m, b(0), b(2))),
                              ggml_step(m, ggml_sub(m, b(1), b(3))));
        bad = ggml_clamp(m, bad, 0.0f, 1.0f);                       // 둘 중 하나라도 참 → 1
        tensor keep = ggml_sub(m, ggml_repeat_4d(m, one, W, FH, 1, 1), bad);  // 1 - bad
        tensor box = ggml_add(m, ggml_mul(m, bp, keep), ggml_mul(m, rb, bad));

        // 박스는 **격자 좌표계 그대로** 낸다 — mmdet 의 head 출력과 같은 규약이다.
        // stride 곱하기는 디코드(predict_by_feat)의 몫이라 여기서 하면 두 번 곱해진다.

        cls = contiguous_2d_to_cwhn(m, cls);
        box = contiguous_2d_to_cwhn(m, box);
        ggml_format_name(cls, "cls_%zu", l);
        ggml_format_name(box, "box_%zu", l);
        out.cls.push_back(cls);
        out.box.push_back(box);
    }
}

// ── CenterNet ─────────────────────────────────────────────────────────────────
// 앵커도 타워도 없다. 세 갈래가 각각 `Conv3x3 → ReLU → Conv1x1` 이고, 이름은
// `heatmap_head.0/.2` 처럼 nn.Sequential 인덱스다(1 번은 ReLU 라 가중치가 없다).
static tensor centernet_branch(model_ref m, tensor x, const std::string& p) {
    return conv_same(m, p + ".2", ggml_relu(m, conv_same(m, p + ".0", x)));
}

void centernet_head_forward(model_ref m, std::vector<tensor> const& feats,
                            anchor_head_cfg const& c, head_outputs& out) {
    const std::string H = "bbox_head";
    for (size_t l = 0; l < feats.size(); ++l) {
        tensor f = cwhn_to_contiguous_2d(m, feats[l]);
        // heatmap 만 sigmoid 를 **여기서** 건다 — mmdet 의 forward_single 이 그렇다.
        tensor cls = ggml_sigmoid(m, centernet_branch(m, f, H + ".heatmap_head"));
        tensor box = centernet_branch(m, f, H + ".wh_head");
        tensor ctr = centernet_branch(m, f, H + ".offset_head");

        cls = contiguous_2d_to_cwhn(m, cls);
        box = contiguous_2d_to_cwhn(m, box);
        ctr = contiguous_2d_to_cwhn(m, ctr);
        ggml_format_name(cls, "cls_%zu", l);
        ggml_format_name(box, "box_%zu", l);
        ggml_format_name(ctr, "ctr_%zu", l);
        out.cls.push_back(cls);
        out.box.push_back(box);
        out.ctr.push_back(ctr);
    }
}

// ── 계열 분기 ─────────────────────────────────────────────────────────────────
void mmdet_head_forward(model_ref m, std::vector<tensor> const& feats,
                        anchor_head_cfg const& c, tensor dcn_base, head_outputs& out) {
    switch (c.kind) {
        case head_kind::anchor:
        case head_kind::fcos:
        case head_kind::gfl:
            tower_head_forward(m, feats, c, out);
            break;
        case head_kind::yolof:
            yolof_head_forward(m, feats, c, out);
            break;
        case head_kind::yolo:
            yolo_head_forward(m, feats, c, out);
            break;
        case head_kind::centernet:
            centernet_head_forward(m, feats, c, out);
            break;
        case head_kind::cornernet:
            corner_head_forward(m, feats, c, out);
            break;
        case head_kind::detr:
            detr_head_forward(m, feats, c, out);
            break;
        case head_kind::conditional_detr:
            conditional_detr_head_forward(m, feats, c, out);
            break;
        case head_kind::dab_detr:
            dab_detr_head_forward(m, feats, c, out);
            break;
        case head_kind::deformable_detr:
            deformable_detr_head_forward(m, feats, c, out);
            break;
        case head_kind::dino:
            dino_head_forward(m, feats, c, out);
            break;
        case head_kind::reppoints:
            reppoints_head_forward(m, feats, c, dcn_base, out);
            break;
        case head_kind::tood:
            tood_head_forward(m, feats, c, out);
            break;
        case head_kind::vfnet: {
            // vfnet 은 자기 cfg 를 따로 갖는다 — 공통 필드에서 채워 넘긴다.
            vfnet_head_cfg v;
            v.stacked_convs = c.stacked_convs;
            v.feat_channels = c.feat_channels;
            v.num_classes = c.num_classes;
            v.gn_groups = c.gn_groups;
            v.strides = c.strides;
            // ⚠️ reg_denom 은 stride 에서 못 얻는다. mmdet 은 regress_ranges 상한을 쓰고
            //    **마지막 레벨만 그 두 배**다([64,128,256,512,1024]). 프론트엔드가 준 값을 쓴다.
            v.reg_denoms = c.reg_denoms;
            vfnet_head_forward(m, feats, v, dcn_base, out.cls, out.box);
            break;
    }
    // ⚠️ switch 에 case 를 안 넣으면 **아무 일도 안 하고 빈 출력**이 나간다 — centernet 이
    //    그랬고, 러너가 그 빈 벡터를 인덱싱해 SIGSEGV 로 죽었다. 원인 지점에서 말하게 한다.
    if (out.cls.empty()) {
        fprintf(stderr, "mmdet_head_forward: head_kind %d 에 조립기가 없다 (head.cpp switch 확인)\n",
                (int)c.kind);
        abort();
        }
    }
}

}  // namespace visp
