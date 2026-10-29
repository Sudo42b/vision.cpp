// run_rpn_verify.cpp — rpn_proposals(postproc.cpp) 격리 검증. rpn_head raw(cls/bbox) 를 넣어
// proposal 을 뽑고 torch RPN predict_by_feat golden 과 IoU 매칭. (proposal 집합 비교)
#include "visp/postproc.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
using namespace visp;
using json = nlohmann::json;

static std::vector<float> load_bin(const std::string& p, size_t n = 0) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    size_t sz = f.tellg(); f.seekg(0);
    size_t cnt = n ? n : sz / sizeof(float);
    std::vector<float> v(cnt);
    f.read(reinterpret_cast<char*>(v.data()), cnt * sizeof(float));
    return v;
}
static float iou(const float* a, const float* b) {
    float x1 = std::max(a[0], b[0]), y1 = std::max(a[1], b[1]);
    float x2 = std::min(a[2], b[2]), y2 = std::min(a[3], b[3]);
    float inter = std::max(0.f, x2 - x1) * std::max(0.f, y2 - y1);
    float ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter;
    return inter / (ua + 1e-9f);
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <rpn.json> <dir>\n", argv[0]); return 1; }
    json j = json::parse(std::ifstream(argv[1]));
    std::string d = argv[2];
    rpn_params p;
    p.strides = j["strides"].get<std::vector<float>>();
    p.octave_base_scale = j["octave_base_scale"];
    p.octave_scales = {1.0f};
    p.ratios = j["ratios"].get<std::vector<float>>();
    for (int i = 0; i < 4; ++i) { p.means[i] = j["means"][i]; p.stds[i] = j["stds"][i]; }
    p.nms_pre = j["nms_pre"]; p.nms_thr = j["nms_thr"]; p.max_per_img = j["max_per_img"];
    p.input_w = j["input_w"]; p.input_h = j["input_h"];
    int L = (int)p.strides.size();

    std::vector<std::vector<float>> cls, bbox;
    std::vector<std::pair<int, int>> feat_hw;
    for (int l = 0; l < L; ++l) {
        cls.push_back(load_bin(d + "/cls." + std::to_string(l) + ".bin"));
        bbox.push_back(load_bin(d + "/bbox." + std::to_string(l) + ".bin"));
        feat_hw.emplace_back(j["feat_hw"][l][0], j["feat_hw"][l][1]);
    }
    std::vector<float> props = rpn_proposals(cls, bbox, feat_hw, p);
    int M = (int)(props.size() / 4);
    std::vector<float> gold = load_bin(d + "/proposals.gold.bin");
    int G = (int)(gold.size() / 4);

    // golden 각 proposal 에 대해 best-IoU C++ proposal
    int m99 = 0, m90 = 0; double sumiou = 0;
    for (int g = 0; g < G; ++g) {
        float best = 0;
        for (int i = 0; i < M; ++i) best = std::max(best, iou(gold.data() + (size_t)g * 4, props.data() + (size_t)i * 4));
        sumiou += best;
        if (best > 0.99f) ++m99;
        if (best > 0.90f) ++m90;
    }
    printf("RPN proposals: C++ M=%d  torch G=%d\n", M, G);
    printf("  golden 매칭: IoU>0.99 %d/%d (%.1f%%),  IoU>0.90 %d/%d (%.1f%%),  평균 best-IoU %.4f\n",
           m99, G, 100.0 * m99 / G, m90, G, 100.0 * m90 / G, sumiou / G);
    return (m90 > G * 0.95) ? 0 : 3;
}
