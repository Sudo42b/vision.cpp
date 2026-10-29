// run_frcnn.cpp — Faster R-CNN two-stage E2E 러너 (오케스트레이션).
//   SubA(g2c: backbone+neck+RPN) → rpn_proposals(host) → roi_align(host) →
//   SubB(g2c: bbox_head) → detect_roi(host) → 박스.
// mmdet 지식은 frcnn.json(config) 로만 주입. g2c 코어·라이브러리 무결합(head 처럼 러너와 컴파일).
#include "FRCNN_SubA.h"
#include "FRCNN_SubB.h"
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
    std::vector<float> v(n);
    std::ifstream f(p, std::ios::binary);
    f.read(reinterpret_cast<char*>(v.data()), n * sizeof(float));
    return v;
}
static std::vector<float> grab(compute_graph& g, const char* name) {
    tensor t = ggml_graph_get_tensor(g.graph, name);
    std::vector<float> v(ggml_nelements(t));
    transfer_from_backend(t, std::span<float>(v.data(), v.size()));
    return v;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <SubA.gguf> <SubB.gguf> <frcnn.json> <input_cwhn.bin> [size=800]\n", argv[0]);
        return 1;
    }
    json j = json::parse(std::ifstream(argv[3]));
    int SZ = argc > 5 ? atoi(argv[5]) : 800;
    std::vector<std::pair<int, int>> feat_hw;  // P2..P6 (5)
    for (auto const& hw : j["feat_hw"]) feat_hw.emplace_back(hw[0], hw[1]);

    backend_device backend = backend_init();

    // ── 1) SubA: 이미지 → P2-P5 + rpn_cls×5 + rpn_bbox×5 (out_0..13) ──
    model_file fA = model_load(argv[1]);
    model_weights wA = model_init(fA.n_tensors());
    model_transfer(fA, wA, backend, backend.preferred_float_type(), fA.tensor_layout());
    compute_graph gA = compute_graph_init(131072);
    model_ref mA(wA, gA);
    tensor inA = compute_graph_input(mA, GGML_TYPE_F32, {3, SZ, SZ, 1}, "x");
    ggml_build_forward_expand(gA, inA);
    tensor lastA = FRCNN_SubA_forward(mA, inA, FRCNN_SubA_detect_params(fA));
    ggml_build_forward_expand(gA, lastA);
    compute_graph_allocate(gA, backend);
    std::vector<float> inbuf = load_bin(argv[4], (size_t)3 * SZ * SZ);
    transfer_to_backend(inA, std::span<const float>(inbuf.data(), inbuf.size()));
    compute(gA, backend);

    std::vector<std::vector<float>> P(4), rpn_cls(5), rpn_bbox(5);
    for (int l = 0; l < 4; ++l) P[l] = grab(gA, ("out_" + std::to_string(l)).c_str());       // P2-P5
    for (int l = 0; l < 5; ++l) rpn_cls[l] = grab(gA, ("out_" + std::to_string(4 + l)).c_str());
    for (int l = 0; l < 5; ++l) rpn_bbox[l] = grab(gA, ("out_" + std::to_string(9 + l)).c_str());

    // ── 2) host: RPN proposals ──
    rpn_params rp;
    rp.strides = j["rpn_strides"].get<std::vector<float>>();
    rp.octave_base_scale = j["rpn_scale"]; rp.octave_scales = {1.0f};
    rp.ratios = j["rpn_ratios"].get<std::vector<float>>();
    for (int i = 0; i < 4; ++i) { rp.means[i] = j["rpn_means"][i]; rp.stds[i] = j["rpn_stds"][i]; }
    rp.nms_pre = j["rpn_nms_pre"]; rp.nms_thr = j["rpn_nms_thr"]; rp.max_per_img = j["rpn_max"];
    rp.input_w = SZ; rp.input_h = SZ;
    std::vector<float> proposals = rpn_proposals(rpn_cls, rpn_bbox, feat_hw, rp);
    int N = (int)(proposals.size() / 4);
    printf("- RPN proposals: %d\n", N);

    // ── 3) host: RoIAlign (P2-P5) → roi_feat NCHW ──
    roi_align_params ap;
    ap.output_size = j["roi_out"]; ap.channels = 256;
    ap.strides = j["roi_strides"].get<std::vector<float>>();
    ap.finest_scale = j["roi_finest_scale"]; ap.sampling_ratio = j["roi_sampling_ratio"];
    ap.aligned = j["roi_aligned"];
    std::vector<std::pair<int, int>> roi_hw(feat_hw.begin(), feat_hw.begin() + 4);
    std::vector<float> roi_nchw = roi_align(P, roi_hw, proposals.data(), N, ap);  // [N,256,7,7]

    // NCHW → NHWC (SubB cwhn 입력 {C,W,H,N} = NHWC 메모리)
    int C = 256, O = ap.output_size;
    std::vector<float> roi_nhwc((size_t)N * C * O * O);
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c)
            for (int ph = 0; ph < O; ++ph)
                for (int pw = 0; pw < O; ++pw)
                    roi_nhwc[(((size_t)n * O + ph) * O + pw) * C + c] =
                        roi_nchw[(((size_t)n * C + c) * O + ph) * O + pw];

    // ── 4) SubB: roi_feat → cls_score(N,81) + bbox_pred(N,320) ──
    model_file fB = model_load(argv[2]);
    model_weights wB = model_init(fB.n_tensors());
    model_transfer(fB, wB, backend, backend.preferred_float_type(), fB.tensor_layout());
    compute_graph gB = compute_graph_init(131072);
    model_ref mB(wB, gB);
    tensor inB = compute_graph_input(mB, GGML_TYPE_F32, {C, O, O, N}, "x");  // {256,7,7,N}
    ggml_build_forward_expand(gB, inB);
    tensor lastB = FRCNN_SubB_forward(mB, inB, FRCNN_SubB_detect_params(fB));
    ggml_build_forward_expand(gB, lastB);
    compute_graph_allocate(gB, backend);
    transfer_to_backend(inB, std::span<const float>(roi_nhwc.data(), roi_nhwc.size()));
    compute(gB, backend);
    std::vector<float> cls = grab(gB, "out_0");   // [81,N] → n*81+j
    std::vector<float> reg = grab(gB, "out_1");   // [320,N] → n*320+k
    int nc = j["num_classes"];

    // cls softmax per proposal (detect_roi 는 softmax 완료값 기대)
    std::vector<float> cls_sm((size_t)N * (nc + 1));
    for (int i = 0; i < N; ++i) {
        float const* s = cls.data() + (size_t)i * (nc + 1);
        float mx = s[0];
        for (int k = 1; k <= nc; ++k) mx = std::max(mx, s[k]);
        float sum = 0;
        for (int k = 0; k <= nc; ++k) { float e = std::exp(s[k] - mx); cls_sm[(size_t)i * (nc + 1) + k] = e; sum += e; }
        for (int k = 0; k <= nc; ++k) cls_sm[(size_t)i * (nc + 1) + k] /= sum;
    }

    // ── 5) host: detect_roi ──
    roi_params dp;
    dp.num_classes = nc; dp.class_agnostic = j["class_agnostic"];
    for (int i = 0; i < 4; ++i) { dp.means[i] = j["rcnn_means"][i]; dp.stds[i] = j["rcnn_stds"][i]; }
    dp.score_thr = j["rcnn_score_thr"]; dp.nms_thr = j["rcnn_nms_thr"]; dp.max_per_img = j["rcnn_max"];
    dp.input_w = SZ; dp.input_h = SZ;
    std::vector<detection> dets = detect_roi(cls_sm.data(), reg.data(), proposals.data(), N, dp);

    printf("- 최종 검출: %zu\n", dets.size());
    FILE* f = fopen("frcnn_boxes.bin", "wb");
    for (auto const& d : dets) {
        float rec[6] = {d.x1, d.y1, d.x2, d.y2, d.score, (float)d.label};
        fwrite(rec, sizeof(float), 6, f);
    }
    fclose(f);
    for (size_t i = 0; i < dets.size() && i < 10; ++i)
        printf("  [%zu] label=%d score=%.3f box=(%.0f,%.0f,%.0f,%.0f)\n",
               i, dets[i].label, dets[i].score, dets[i].x1, dets[i].y1, dets[i].x2, dets[i].y2);
    return 0;
}
