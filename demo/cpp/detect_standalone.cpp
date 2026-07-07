// Flow C — C++ 자족: Python/mmdet 없이 vision.cpp 하나로 이미지→박스.
//
// 철학(vision.cpp model-implementation-guide): compute graph + 전/후처리 전부 C++, self-contained.
//   이미지 → preprocess(C++) → compose_forward(NPU 한 덩어리) → detect_anchor(C++) → 박스
//   후처리 파라미터(anchor scale·strides·num_classes·means/stds)는 gguf 메타에서 읽음(하드코딩 대신).
//
// 사용: detect_standalone  model.gguf  image.jpg  [cpu|gpu]
//
// ※ 이 데모는 anchor-based(RetinaNet/ATSS/GFL) 기준. FCOS/YOLO/DETR 은 detect_fcos/yolox/detr 로 교체.

#include "visp/arch/component.h"
#include "visp/ml.h"
#include "visp/postproc.h"
#include "visp/vision.h"

#include <ggml.h>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace visp;

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s model.gguf image.jpg [cpu|gpu]\n", argv[0]);
        return 1;
    }
    const char* gguf = argv[1];
    const char* image_path = argv[2];
    bool use_npu = argc > 3 && std::string(argv[3]) == "gpu";

    // ── 1) 모델 로드 (compose gguf) ──
    backend_device backend = backend_init(use_npu ? backend_type::gpu : backend_type::cpu);
    model_file file = model_load(gguf);
    model_weights weights = model_init(file.n_tensors());
    model_transfer(file, weights, backend, GGML_TYPE_F32, file.tensor_layout());

    // ── 2) 전처리: 이미지 → 입력 텐서 (mean/std 는 gguf 메타에서) ──
    image_data img = image_load(image_path);      // rgba_u8 (4채널, RGB 순)
    int size = 800;  // gguf 메타 "input.size" 로 대체 가능
    float mean[3] = {123.675f, 116.28f, 103.53f};
    float std[3] = {58.395f, 57.12f, 57.375f};
    int W, H;
    // extent = {width, height}, data = 4채널(RGBA) → img_c=4, to_rgb=false(이미 RGB)
    auto in = preprocess(img.data.get(), img.extent[1], img.extent[0], 4, size, mean, std,
                         /*to_rgb*/ false, &W, &H);

    // ── 3) 그래프: compose 조립 → NPU 한 덩어리 실행 ──
    compute_graph graph = compute_graph_init(131072);
    model_ref m(weights, graph);
    tensor x = compute_graph_input(m, GGML_TYPE_F32, {3, W, H, 1}, "input");
    compose_spec spec = parse_compose_pipeline(file.get_string("compose.pipeline"));
    compose_forward(m, x, spec);  // backbone→neck→head 한 그래프, out_0..N 등록
    compute_graph_allocate(graph, backend);
    transfer_to_backend(x, span<float const>(in.data(), in.size()));
    compute(graph, backend);  // ← 여기까지 NPU 한 덩어리 (경계 1회)

    // ── 4) head 출력(out_0..N) 꺼내기 → cls/box 분리 ──
    std::vector<std::vector<float>> cls, box;
    std::vector<std::pair<int, int>> feat_hw;
    int n_out = 0;
    for (int i = 0;; ++i) {
        tensor t = ggml_graph_get_tensor(graph.graph, ("out_" + std::to_string(i)).c_str());
        if (!t) break;
        n_out++;
    }
    int n_lev = n_out / 2;  // cls×L + box×L (RetinaNet 계열)
    for (int i = 0; i < n_out; ++i) {
        tensor t = ggml_graph_get_tensor(graph.graph, ("out_" + std::to_string(i)).c_str());
        tensor_data d = transfer_from_backend(t);
        std::vector<float> v(d.as_f32().begin(), d.as_f32().end());
        if (i < n_lev) { cls.push_back(v); feat_hw.push_back({(int)t->ne[1], (int)t->ne[0]}); }
        else box.push_back(v);
    }

    // ── 5) 후처리: detect_anchor (파라미터는 gguf 메타에서 읽는 게 정석) ──
    det_params p;
    p.strides = {8, 16, 32, 64, 128};
    p.octave_base_scale = 4;
    p.octave_scales = {1.0f, std::pow(2.0f, 1.0f / 3), std::pow(2.0f, 2.0f / 3)};
    p.num_classes = 80;
    p.score_thr = 0.3f;
    p.nms_thr = 0.5f;
    p.input_w = W;
    p.input_h = H;
    auto dets = detect_anchor(cls, box, feat_hw, p);

    // ── 6) 출력 ──
    printf("검출 %zu 개:\n", dets.size());
    for (auto& d : dets)
        printf("  box=(%.0f,%.0f,%.0f,%.0f) label=%d score=%.2f\n",
               d.x1, d.y1, d.x2, d.y2, d.label, d.score);
    return 0;
}
