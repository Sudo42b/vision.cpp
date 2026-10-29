// run_vfnet_head.cpp — VFNet head C++ 부품(vfnet_head_forward) 격리 검증 harness.
// torch FPN features(cwhn bin)를 직접 입력으로 넣어 head 만 돌리고 cls/box_refine 를 덤프한다
// (backbone.cpp 무관 → 신규 head 코드만 검증). 가중치는 g2c MMDetBackbone.gguf(bbox_head.* 포함).
//
// 컴파일: build_vfnet_head.sh (libvisioncpp 링크)
// 실행:  run_vfnet_head <gguf> <vfhead.json> <feat_prefix> <out_prefix>
#include "head.h"
#include "visp/ml.h"

#include <ggml.h>
#include <nlohmann/json.hpp>

#include <cstdio>
#include <fstream>
#include <span>
#include <string>
#include <vector>
using namespace visp;
using json = nlohmann::json;

static std::vector<float> load_bin(const std::string& path, size_t n) {
    std::vector<float> v(n);
    std::ifstream f(path, std::ios::binary);
    f.read(reinterpret_cast<char*>(v.data()), n * sizeof(float));
    return v;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <gguf> <vfhead.json> <feat_prefix> <out_prefix>\n", argv[0]);
        return 1;
    }
    const char* gguf = argv[1];
    json cfg = json::parse(std::ifstream(argv[2]));
    std::string featp = argv[3];
    std::string outp = argv[4];

    vfnet_head_cfg c;
    c.num_classes = cfg["num_classes"];
    c.feat_channels = cfg["feat_channels"];
    c.stacked_convs = cfg["stacked_convs"];
    c.gn_groups = cfg["gn_groups"];
    c.strides = cfg["strides"].get<std::vector<float>>();
    c.reg_denoms = cfg["reg_denoms"].get<std::vector<float>>();
    std::vector<float> dcn_base_vals = cfg["dcn_base"].get<std::vector<float>>();
    int L = cfg["levels"];
    auto feat_shapes = cfg["feat_shapes"];  // [[N,C,H,W], ...]

    backend_device backend = backend_init();
    model_file file = model_load(gguf);
    model_weights weights = model_init(file.n_tensors());
    model_transfer(file, weights, backend, backend.preferred_float_type(), file.tensor_layout());

    compute_graph graph = compute_graph_init(262144);
    model_ref m(weights, graph);

    // FPN feature 입력 (cwhn {C,W,H,1}) + 채워넣을 데이터 준비
    std::vector<tensor> feats;
    std::vector<std::vector<float>> feat_data;
    for (int l = 0; l < L; ++l) {
        int C = feat_shapes[l][1], Hh = feat_shapes[l][2], Ww = feat_shapes[l][3];
        tensor f = compute_graph_input(m, GGML_TYPE_F32, {C, Ww, Hh, 1},
                                       ("feat_" + std::to_string(l)).c_str());
        ggml_build_forward_expand(graph, f);
        feats.push_back(f);
        feat_data.push_back(load_bin(featp + "." + std::to_string(l) + ".bin", (size_t)C * Ww * Hh));
    }
    // dcn_base 입력 {1,1,18,1} (whcn: 채널=ne2, W/H 로 broadcast)
    tensor dcn_base = compute_graph_input(m, GGML_TYPE_F32, {1, 1, 18, 1}, "dcn_base");
    ggml_build_forward_expand(graph, dcn_base);

    // head 조립
    std::vector<tensor> cls_t, box_t;
    vfnet_head_forward(m, feats, c, dcn_base, cls_t, box_t);
    for (tensor t : cls_t) ggml_build_forward_expand(graph, t);
    for (tensor t : box_t) ggml_build_forward_expand(graph, t);

    compute_graph_allocate(graph, backend);
    for (int l = 0; l < L; ++l) {
        transfer_to_backend(feats[l], std::span<const float>(feat_data[l].data(), feat_data[l].size()));
    }
    transfer_to_backend(dcn_base, std::span<const float>(dcn_base_vals.data(), dcn_base_vals.size()));
    compute(graph, backend);

    auto dump = [&](tensor t, const std::string& path) {
        size_t ne = ggml_nelements(t);
        std::vector<float> d(ne);
        transfer_from_backend(t, std::span<float>(d.data(), d.size()));
        FILE* f = fopen(path.c_str(), "wb");
        fwrite(d.data(), sizeof(float), ne, f);
        fclose(f);
        return ne;
    };
    for (int l = 0; l < L; ++l) {
        size_t nc = dump(cls_t[l], outp + ".cls." + std::to_string(l) + ".bin");
        size_t nb = dump(box_t[l], outp + ".box." + std::to_string(l) + ".bin");
        printf("level %d: cls ne=%zu [%lld %lld %lld] box ne=%zu\n", l, nc,
               (long long)cls_t[l]->ne[0], (long long)cls_t[l]->ne[1], (long long)cls_t[l]->ne[2], nb);
    }
    printf("dumped %d levels → %s.{cls,box}.*.bin\n", L, outp.c_str());
    return 0;
}
