#include "head.h"

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

// ConvModule(conv + GroupNorm + ReLU). 커널 크기는 가중치에서 읽는다.
static tensor conv_gn_relu(model_ref m, tensor x, const std::string& p, int gn_groups) {
    x = conv_maybe_deform(m, p + ".conv", x);
    x = group_norm_affine(m[(p + ".gn").c_str()], x, gn_groups);
    return ggml_relu(m, x);
}

// cls/reg 공유 타워. ConvModule 은 norm 유무로 두 모양이다 — RetinaNet 은 conv+relu,
// ATSS/FCOS/GFL/RepPoints 는 conv+GN+relu.
static tensor conv_tower(model_ref m, tensor x, anchor_head_cfg const& c,
                         const std::string& prefix, size_t level) {
    // RetinaSepBNHead 는 레벨마다 타워를 따로 둔다 → 이름이 한 겹 깊다.
    const std::string base = c.per_level_towers
                                 ? prefix + "." + std::to_string(level) : prefix;
    for (int i = 0; i < c.stacked_convs; ++i) {
        std::string p = base + "." + std::to_string(i);
        x = c.head_has_norm ? conv_gn_relu(m, x, p, c.gn_groups)
                            : ggml_relu(m, conv_maybe_deform(m, p + ".conv", x));
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

        tensor cc = conv_tower(m, f, c, c.cls_convs_prefix, l);
        tensor rr = conv_tower(m, f, c, c.reg_convs_prefix, l);

        // RTMDetSepBNHead 는 출력 conv 도 레벨마다 따로다.
        const std::string lv = c.per_level_heads ? "." + std::to_string(l) : "";
        tensor cls = conv_same(m, c.cls_head + lv, cc);
        tensor box = conv_same(m, c.reg_head + lv, rr);

        box = apply_scale(m, box, c, l);
        if (c.reg_max > 0) box = dfl_integral(m, box, c.reg_max);

        // FCOS 는 둘 중 하나만 쓴다(norm_on_bbox). GFL 은 거리에 stride 를 곱한다.
        float stride = l < c.strides.size() ? c.strides[l] : 1.0f;
        if (c.bbox_clamp_stride) box = ggml_scale(m, ggml_relu(m, box), stride);
        else if (c.bbox_exp)     box = ggml_exp(m, box);
        else if (c.reg_max > 0)  box = ggml_scale(m, box, stride);

        cls = contiguous_2d_to_cwhn(m, cls);
        box = contiguous_2d_to_cwhn(m, box);
        ggml_format_name(cls, "cls_%zu", l);
        ggml_format_name(box, "box_%zu", l);
        out.cls.push_back(cls);
        out.box.push_back(box);

        if (!c.centerness_head.empty()) {
            tensor ctr = conv_same(m, c.centerness_head + lv, c.centerness_on_reg ? rr : cc);
            ctr = contiguous_2d_to_cwhn(m, ctr);
            ggml_format_name(ctr, "ctr_%zu", l);
            out.ctr.push_back(ctr);
        }
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
    }
}

}  // namespace visp
