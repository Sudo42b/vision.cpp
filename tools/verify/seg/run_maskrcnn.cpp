// run_maskrcnn.cpp — Mask R-CNN E2E 러너 (run_frcnn + mask 분기).
//   Faster R-CNN(SubA→rpn→roialign→SubB→detect_roi) → 박스 → mask RoIAlign(out=14) →
//   SubC(mask_head, g2c) → paste_mask(host) → 인스턴스 마스크.
#include "FRCNN_SubA.h"
#include "FRCNN_SubB.h"
#include "MaskRCNN_SubC.h"
#include "visp/ml.h"
#include "visp/postproc.h"

#include <ggml.h>
#include <nlohmann/json.hpp>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <span>
#include <string>
#include <vector>
using namespace visp;
using json = nlohmann::json;

static std::vector<float> load_bin(const std::string& p, size_t n) {
    std::vector<float> v(n); std::ifstream f(p, std::ios::binary);
    f.read(reinterpret_cast<char*>(v.data()), n * sizeof(float)); return v;
}
static std::vector<float> grab(compute_graph& g, const char* name) {
    tensor t = ggml_graph_get_tensor(g.graph, name);
    std::vector<float> v(ggml_nelements(t));
    transfer_from_backend(t, std::span<float>(v.data(), v.size())); return v;
}
// g2c 그래프 1회 실행 → out_i 추출 (범용). input 은 미리 채운다.
struct sub_run {
    model_file file; model_weights w; compute_graph g;
    sub_run(const char* gguf, backend_device& b) : file(model_load(gguf)),
        w(model_init(file.n_tensors())), g(compute_graph_init(131072)) {
        model_transfer(file, w, b, b.preferred_float_type(), file.tensor_layout());
    }
};

int main(int argc, char** argv) {
    if (argc < 7) {
        fprintf(stderr, "usage: %s <SubA.gguf> <SubB.gguf> <SubC.gguf> <mask.json> <input.bin> <out_prefix> [size=800]\n", argv[0]);
        return 1;
    }
    json j = json::parse(std::ifstream(argv[4]));
    int SZ = argc > 7 ? atoi(argv[7]) : 800;
    std::string outp = argv[6];
    std::vector<std::pair<int, int>> feat_hw;
    for (auto const& hw : j["feat_hw"]) feat_hw.emplace_back(hw[0], hw[1]);
    backend_device backend = backend_init();

    // ── 1) SubA ──
    sub_run A(argv[1], backend); model_ref mA(A.w, A.g);
    tensor inA = compute_graph_input(mA, GGML_TYPE_F32, {3, SZ, SZ, 1}, "x");
    ggml_build_forward_expand(A.g, inA);
    ggml_build_forward_expand(A.g, FRCNN_SubA_forward(mA, inA, FRCNN_SubA_detect_params(A.file)));
    compute_graph_allocate(A.g, backend);
    auto ib = load_bin(argv[5], (size_t)3 * SZ * SZ);
    transfer_to_backend(inA, std::span<const float>(ib.data(), ib.size()));
    compute(A.g, backend);
    std::vector<std::vector<float>> P(4), rc(5), rb(5);
    for (int l = 0; l < 4; ++l) P[l] = grab(A.g, ("out_" + std::to_string(l)).c_str());
    for (int l = 0; l < 5; ++l) rc[l] = grab(A.g, ("out_" + std::to_string(4 + l)).c_str());
    for (int l = 0; l < 5; ++l) rb[l] = grab(A.g, ("out_" + std::to_string(9 + l)).c_str());

    // ── 2) RPN proposals ──
    rpn_params rp;
    rp.strides = j["rpn_strides"].get<std::vector<float>>(); rp.octave_base_scale = j["rpn_scale"];
    rp.octave_scales = {1.0f}; rp.ratios = j["rpn_ratios"].get<std::vector<float>>();
    for (int i = 0; i < 4; ++i) { rp.means[i] = j["rpn_means"][i]; rp.stds[i] = j["rpn_stds"][i]; }
    rp.nms_pre = j["rpn_nms_pre"]; rp.nms_thr = j["rpn_nms_thr"]; rp.max_per_img = j["rpn_max"];
    rp.input_w = SZ; rp.input_h = SZ;
    std::vector<float> proposals = rpn_proposals(rc, rb, feat_hw, rp);
    int N = (int)(proposals.size() / 4);

    // ── 3) RoIAlign(7) → SubB → detect_roi ──
    auto nchw2nhwc = [](std::vector<float> const& s, int n, int C, int O) {
        std::vector<float> d((size_t)n * C * O * O);
        for (int m = 0; m < n; ++m) for (int c = 0; c < C; ++c) for (int ph = 0; ph < O; ++ph)
            for (int pw = 0; pw < O; ++pw)
                d[(((size_t)m * O + ph) * O + pw) * C + c] = s[(((size_t)m * C + c) * O + ph) * O + pw];
        return d;
    };
    roi_align_params ap;
    ap.output_size = j["roi_out"]; ap.channels = 256; ap.strides = j["roi_strides"].get<std::vector<float>>();
    ap.finest_scale = j["roi_finest_scale"]; ap.sampling_ratio = j["roi_sampling_ratio"]; ap.aligned = j["roi_aligned"];
    std::vector<std::pair<int, int>> roi_hw(feat_hw.begin(), feat_hw.begin() + 4);
    std::vector<float> rf7 = nchw2nhwc(roi_align(P, roi_hw, proposals.data(), N, ap), N, 256, ap.output_size);

    sub_run B(argv[2], backend); model_ref mB(B.w, B.g);
    tensor inB = compute_graph_input(mB, GGML_TYPE_F32, {256, ap.output_size, ap.output_size, N}, "x");
    ggml_build_forward_expand(B.g, inB);
    ggml_build_forward_expand(B.g, FRCNN_SubB_forward(mB, inB, FRCNN_SubB_detect_params(B.file)));
    compute_graph_allocate(B.g, backend);
    transfer_to_backend(inB, std::span<const float>(rf7.data(), rf7.size()));
    compute(B.g, backend);
    std::vector<float> cls = grab(B.g, "out_0"), reg = grab(B.g, "out_1");
    int nc = j["num_classes"];
    std::vector<float> sm((size_t)N * (nc + 1));
    for (int i = 0; i < N; ++i) {
        float const* s = cls.data() + (size_t)i * (nc + 1); float mx = s[0];
        for (int k = 1; k <= nc; ++k) mx = std::max(mx, s[k]);
        float su = 0; for (int k = 0; k <= nc; ++k) { float e = std::exp(s[k] - mx); sm[(size_t)i * (nc + 1) + k] = e; su += e; }
        for (int k = 0; k <= nc; ++k) sm[(size_t)i * (nc + 1) + k] /= su;
    }
    roi_params dp; dp.num_classes = nc; dp.class_agnostic = j["class_agnostic"];
    for (int i = 0; i < 4; ++i) { dp.means[i] = j["rcnn_means"][i]; dp.stds[i] = j["rcnn_stds"][i]; }
    dp.score_thr = j["rcnn_score_thr"]; dp.nms_thr = j["rcnn_nms_thr"]; dp.max_per_img = j["rcnn_max"];
    dp.input_w = SZ; dp.input_h = SZ;
    std::vector<detection> dets = detect_roi(sm.data(), reg.data(), proposals.data(), N, dp);
    int M = (int)dets.size();
    printf("- RPN %d → 검출 %d\n", N, M);

    // ── 4) mask RoIAlign(14) on 최종 박스 → SubC → mask_logits(M,80,28,28) ──
    std::vector<float> boxes((size_t)M * 4);
    for (int m = 0; m < M; ++m) { boxes[m*4] = dets[m].x1; boxes[m*4+1] = dets[m].y1; boxes[m*4+2] = dets[m].x2; boxes[m*4+3] = dets[m].y2; }
    roi_align_params mp;
    mp.output_size = j["mask_roi_out"]; mp.channels = 256; mp.strides = j["mask_strides"].get<std::vector<float>>();
    mp.finest_scale = j["mask_finest_scale"]; mp.sampling_ratio = 0; mp.aligned = true;
    int O = mp.output_size;
    std::vector<float> mf14 = nchw2nhwc(roi_align(P, roi_hw, boxes.data(), M, mp), M, 256, O);
    int MS = 28;
    float thr = j["mask_thr_binary"];

    // SubC 그래프는 batch=1 로 (ggml_conv_transpose_2d_p0 가 N>1 미지원 → roi 별 실행).
    sub_run C(argv[3], backend); model_ref mC(C.w, C.g);
    tensor inC = compute_graph_input(mC, GGML_TYPE_F32, {256, O, O, 1}, "x");
    ggml_build_forward_expand(C.g, inC);
    ggml_build_forward_expand(C.g, MaskRCNN_SubC_forward(mC, inC, MaskRCNN_SubC_detect_params(C.file)));
    compute_graph_allocate(C.g, backend);

    // ── 5) roi 별 SubC → paste_mask → (M, SZ, SZ) 이진 마스크 + 박스 덤프 ──
    FILE* fm = fopen((outp + ".masks.bin").c_str(), "wb");   // (M, SZ, SZ) uint8
    FILE* fb = fopen((outp + ".boxes.bin").c_str(), "wb");   // (M, 6) f32
    size_t roi_stride = (size_t)256 * O * O;
    for (int m = 0; m < M; ++m) {
        transfer_to_backend(inC, std::span<const float>(mf14.data() + (size_t)m * roi_stride, roi_stride));
        compute(C.g, backend);
        std::vector<float> ml = grab(C.g, "out_0");   // cwhn {80,28,28,1}: idx = c + pw*80 + ph*80*28
        int lab = dets[m].label;
        std::vector<float> logit((size_t)MS * MS);
        for (int ph = 0; ph < MS; ++ph) for (int pw = 0; pw < MS; ++pw)
            logit[ph * MS + pw] = ml[(size_t)lab + pw * 80 + ph * 80 * MS];
        int bh = 0, bw = 0;
        std::vector<uint8_t> bm = paste_mask(logit.data(), MS, MS, dets[m], thr, &bh, &bw);
        std::vector<uint8_t> canvas((size_t)SZ * SZ, 0);
        int x0 = (int)std::floor(dets[m].x1), y0 = (int)std::floor(dets[m].y1);
        for (int y = 0; y < bh; ++y) for (int x = 0; x < bw; ++x) {
            int iy = y0 + y, ix = x0 + x;
            if (iy >= 0 && iy < SZ && ix >= 0 && ix < SZ) canvas[(size_t)iy * SZ + ix] = bm[(size_t)y * bw + x];
        }
        fwrite(canvas.data(), 1, canvas.size(), fm);
        float rec[6] = {dets[m].x1, dets[m].y1, dets[m].x2, dets[m].y2, dets[m].score, (float)lab};
        fwrite(rec, sizeof(float), 6, fb);
    }
    fclose(fm); fclose(fb);
    printf("- 인스턴스 마스크 %d개 → %s.masks.bin (%dx%d), 박스 → %s.boxes.bin\n", M, outp.c_str(), SZ, SZ, outp.c_str());
    return 0;
}
