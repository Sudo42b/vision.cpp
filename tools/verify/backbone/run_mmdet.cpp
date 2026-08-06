// run_mmdet.cpp — mmdet 검출기 러너 (main 의 run_yolo_cpp.cpp 패턴).
//
//   백본  = g2c 가 생성한 output/<ARCH>.cpp (그대로 컴파일) → <ARCH>_forward
//   head  = tools/detect/head.cpp 부품 (러너와 함께 컴파일, 라이브러리 아님)
//   decode+NMS = src/visp/postproc.cpp detect_anchor (라이브러리)
//   cfg   = <name>.postproc.h (mmdet_params() from mmdet_to_pt.py, compiled in)
//
// 백본을 arch/ 로 복사하거나 cli REG 에 등록하지 않는다 — output/.cpp 를 직접 컴파일해
// libvisioncpp 와 링크(build_mmdet_cpp.sh). run_yolo_cpp 와 동일한 -DARCH 매크로 방식.
//
// 컴파일: -DARCH=<클래스명> -DVISP_ARCH_HEADER='"<gen>/<ARCH>.h"'
// run:   run_mmdet <gguf> <input> <out.png> [size=512]
//        an output ending in .bin holds raw f32 for comparison; anything else is an image

#include VISP_ARCH_HEADER                 // 백본: <ARCH>_forward / <ARCH>_params / <ARCH>_detect_params
#include "head.h"                          // head 부품: anchor_head_forward (같은 폴더)
#include "draw.h"                          // draws detections onto the image (same folder)
#include MMDET_PARAMS_HEADER              // generated mmdet_params(); values live in the binary
#include "visp/image.h"                   // image_load (이미지 입력 pre)
#include "visp/ml.h"
#include "visp/postproc.h"                // detect_anchor, preprocess, det_params, detection

#include <ggml.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <string>
#include <vector>

using namespace visp;

#define CAT_(a, b) a##b
#define CAT(a, b) CAT_(a, b)
#define FWD CAT(ARCH, _forward)
#define PARAMS_T CAT(ARCH, _params)
#define DETECT_PARAMS CAT(ARCH, _detect_params)

static std::vector<float> load_bin(const char* path, size_t n) {
    std::vector<float> v(n);
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    size_t got = fread(v.data(), sizeof(float), n, f);
    fclose(f);
    if (got != n) { fprintf(stderr, "short read %s: %zu/%zu\n", path, got, n); exit(1); }
    return v;
}

static std::vector<float> to_vec(tensor t) {
    int64_t ne = ggml_nelements(t);
    std::vector<float> d((size_t)ne);
    transfer_from_backend(t, std::span<float>(d.data(), d.size()));
    return d;
}

// An image input goes through preprocess() with the generated constants;
// a .bin is taken as an already pre-processed CWHN f32 tensor.
static bool has_ext(std::string const& s, const char* e) {
    size_t n = std::strlen(e);
    return s.size() >= n && s.compare(s.size() - n, n, e) == 0;
}

static bool is_image_path(std::string const& s) {
    return has_ext(s, ".jpg") || has_ext(s, ".jpeg") || has_ext(s, ".png") || has_ext(s, ".bmp");
}

// A non-empty `source` means the input was an image, so the result can be drawn on it.
static std::vector<float> load_input(const char* path, int SZ, mmdet_cfg const& c,
                                     image_data* source) {
    std::string s(path);
    if (is_image_path(s)) {
        image_data img = image_load(path);
        int iw = img.extent[0], ih = img.extent[1];
        int ic = n_channels(img.format);   // stbi_load(...,0)=네이티브 채널수 (JPEG=3, PNG+α=4)
        float const (&mean)[3] = c.img_mean;
        float const (&sd)[3] = c.img_std;
        bool to_rgb = c.to_rgb;
        printf("- preprocess: image %dx%dx%d → %dx%d (mean %.1f,%.1f,%.1f std %.1f,%.1f,%.1f to_rgb=%d)\n",
            iw, ih, ic, SZ, SZ, mean[0], mean[1], mean[2], sd[0], sd[1], sd[2], (int)to_rgb);
        auto tensor_data = preprocess(img.data.get(), ih, iw, ic, SZ, mean, sd, to_rgb);
        if (source) {
            *source = std::move(img);
        }
        return tensor_data;
    }
    return load_bin(path, (size_t)3 * SZ * SZ);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr,
            "usage: %s <gguf> <input> <output> [size=512]\n"
            "       output ending in .bin holds raw float32 detections;\n"
            "       any other extension is an image with the boxes drawn on it\n", argv[0]);
        return 1;
    }
    const char* gguf = argv[1];
    const char* inp = argv[2];
    const char* outp = argv[3];
    const int SZ = argc > 4 ? atoi(argv[4]) : 512;

    // 1) 가중치 (백본 + head 전부 이 gguf 에)
    backend_device backend = backend_init();
    model_file file = model_load(gguf);
    model_weights weights = model_init(file.n_tensors());
    model_transfer(file, weights, backend, backend.preferred_float_type(), file.tensor_layout());

    compute_graph graph = compute_graph_init(131072);
    model_ref m(weights, graph);
    PARAMS_T p = DETECT_PARAMS(file);

    // 2) 백본 (g2c output/.cpp) — FPN features(out_0..L-1) 를 그래프에 조립
    tensor input = compute_graph_input(m, GGML_TYPE_F32, {3, SZ, SZ, 1}, "x");
    ggml_build_forward_expand(graph, input);
    tensor bb = FWD(m, input, p);
    ggml_build_forward_expand(graph, bb);

    // 3) Configuration -- constants fixed at compile time. No file is read.
    mmdet_cfg cfg = mmdet_params();
    anchor_head_cfg& hc = cfg.head;
    det_params& dp = cfg.det;
    dp.input_w = SZ;
    dp.input_h = SZ;
    int L = (int)dp.strides.size();
    if (L == 0) {
        fprintf(stderr, "no FPN strides — this config's head was not recognised at export\n");
        return 1;
    }

    // 4) 백본 features(out_0..L-1) 를 잡아 head 부품 조립
    std::vector<tensor> feats;
    for (int l = 0; l < L; ++l) {
        tensor f = ggml_graph_get_tensor(graph.graph, ("out_" + std::to_string(l)).c_str());
        if (!f) { fprintf(stderr, "백본 출력 out_%d 없음 (백본 .cpp 출력 규약 확인)\n", l); return 3; }
        feats.push_back(f);
    }
    std::vector<tensor> cls_t, box_t;
    anchor_head_forward(m, feats, hc, cls_t, box_t);
    for (tensor t : cls_t) ggml_build_forward_expand(graph, t);
    for (tensor t : box_t) ggml_build_forward_expand(graph, t);
    printf("- mmdet runner: 백본 out_0..%d + C++ head(%d convs, %s)\n",
        L - 1, hc.stacked_convs, hc.cls_head.c_str());

    // 5) 계산 (입력: 이미지면 preprocess, .bin 이면 전처리된 텐서)
    compute_graph_allocate(graph, backend);
    image_data source;
    auto in_data = load_input(inp, SZ, cfg, &source);
    if (const char* dp = std::getenv("MMDET_DUMP_PRE")) {   // 디버그: 전처리 텐서 덤프
        FILE* f = fopen(dp, "wb"); if (f) { fwrite(in_data.data(), sizeof(float), in_data.size(), f); fclose(f); }
    }
    if (in_data.size() != (size_t)3 * SZ * SZ) {
        fprintf(stderr, "입력 크기 %zu != %d (전처리/크기 확인)\n", in_data.size(), 3 * SZ * SZ);
        return 2;
    }
    transfer_to_backend(input, std::span<const float>(in_data.data(), in_data.size()));
    compute(graph, backend);

    // 6) raw cls/box → detect_anchor (decode + NMS)
    std::vector<std::vector<float>> cls_v(L), box_v(L);
    std::vector<std::pair<int, int>> feat_hw(L);
    for (int l = 0; l < L; ++l) {
        feat_hw[l] = { (int)cls_t[l]->ne[2], (int)cls_t[l]->ne[1] };  // (fh, fw)
        cls_v[l] = to_vec(cls_t[l]);
        box_v[l] = to_vec(box_t[l]);
    }
    std::vector<detection> dets = detect_anchor(cls_v, box_v, feat_hw, dp);

    // An image by default, as with every other entry point here. Raw numbers on request.
    std::string out_s(outp);
    bool want_raw = has_ext(out_s, ".bin");
    if (!want_raw && source.extent[0] == 0) {
        fprintf(stderr, "- input was a tensor, so there is no image to draw on; writing raw\n");
        want_raw = true;
    }

    if (want_raw) {
        FILE* f = fopen(outp, "wb");
        if (!f) { fprintf(stderr, "cannot write %s\n", outp); return 1; }
        for (detection const& d : dets) {
            float rec[6] = { d.x1, d.y1, d.x2, d.y2, d.score, (float)d.label };
            fwrite(rec, sizeof(float), 6, f);
        }
        fclose(f);
        printf("- detect(anchor): %zu boxes → %s (x1,y1,x2,y2,score,label f32*6)\n",
            dets.size(), outp);
    } else {
        float thr = 0.3f;
        if (const char* e = std::getenv("VISP_DRAW_THRESHOLD")) {
            thr = (float)atof(e);
        }
        // Coordinates are in the square input, so scale them back to the original resolution.
        float sx = float(source.extent[0]) / float(SZ);
        float sy = float(source.extent[1]) / float(SZ);
        int drawn = draw_detections(source, dets, sx, sy, thr);
        image_save(source, outp);
        printf("- detect(anchor): %zu boxes, %d drawn at score >= %.2f → %s\n",
            dets.size(), drawn, thr, outp);
    }

    // The highest-scoring detections, so a run says something without opening the output.
    // VISP_PRINT_DETS sets how many; 0 turns it off.
    int n_print = 10;
    if (const char* e = std::getenv("VISP_PRINT_DETS")) {
        n_print = atoi(e);
    }
    n_print = std::min<int>(n_print, (int)dets.size());
    if (n_print > 0) {
        printf("\n  %5s %8s %8s %8s %8s %8s %6s\n", "#", "x1", "y1", "x2", "y2", "score", "label");
        for (int i = 0; i < n_print; ++i) {
            detection const& d = dets[i];
            printf("  %5d %8.1f %8.1f %8.1f %8.1f %8.3f %6d\n",
                i, d.x1, d.y1, d.x2, d.y2, d.score, d.label);
        }
        if ((int)dets.size() > n_print) {
            printf("  %5s  (%zu more)\n", "...", dets.size() - n_print);
        }
        printf("\n");
    }
    return 0;
}
