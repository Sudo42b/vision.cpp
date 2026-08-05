// run_roi_verify.cpp — RoIAlign(postproc.cpp) 격리 검증. torch feats+rois 를 넣어 roi_feat 를
// 뽑고 torch bbox_roi_extractor golden 과 cosine 비교. (ggml/모델 무관, 순수 C++ op 검증)
#include "visp/postproc.h"
#include <nlohmann/json.hpp>
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

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <roi.json> <dir>\n", argv[0]); return 1; }
    json j = json::parse(std::ifstream(argv[1]));
    std::string d = argv[2];
    int N = j["N"], C = j["channels"], out = j["output_size"];
    roi_align_params p;
    p.channels = C; p.output_size = out;
    p.finest_scale = j["finest_scale"];
    p.strides = j["strides"].get<std::vector<float>>();
    p.sampling_ratio = j["sampling_ratio"];
    p.aligned = j["aligned"];
    int L = (int)p.strides.size();

    std::vector<std::vector<float>> feats;
    std::vector<std::pair<int, int>> feat_hw;
    for (int l = 0; l < L; ++l) {
        feats.push_back(load_bin(d + "/feat." + std::to_string(l) + ".bin"));
        feat_hw.emplace_back(j["feat_hw"][l][0], j["feat_hw"][l][1]);
    }
    std::vector<float> rois = load_bin(d + "/rois.bin", (size_t)N * 4);

    std::vector<float> got = roi_align(feats, feat_hw, rois.data(), N, p);
    std::vector<float> gold = load_bin(d + "/roifeat.gold.bin");

    if (got.size() != gold.size()) {
        fprintf(stderr, "size mismatch: got %zu gold %zu\n", got.size(), gold.size());
        return 2;
    }
    double dot = 0, na = 0, nb = 0, maxabs = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        dot += (double)got[i] * gold[i]; na += (double)got[i] * got[i]; nb += (double)gold[i] * gold[i];
        maxabs = std::max(maxabs, (double)std::fabs(got[i] - gold[i]));
    }
    double cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
    printf("RoIAlign: N=%d roi_feat=%zu  cos=%.7f  max|Δ|=%.2e\n", N, got.size(), cos, maxabs);
    return cos > 0.999 ? 0 : 3;
}
