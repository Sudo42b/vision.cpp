// run_mmdet.cpp — mmdet 검출기 러너 (main 의 run_yolo_cpp.cpp 패턴).
//
//   백본  = g2c 가 생성한 output/<ARCH>.cpp (그대로 컴파일) → <ARCH>_forward
//   head  = tools/detect/head.cpp 부품 (러너와 함께 컴파일, 라이브러리 아님)
//   decode+NMS = src/visp/postproc.cpp detect_anchor (라이브러리)
//   cfg   = <name>.postproc.json (tools/frontend/mmdet/mmdet_to_pt.py 가 생성)
//
// 백본을 arch/ 로 복사하거나 cli REG 에 등록하지 않는다 — output/.cpp 를 직접 컴파일해
// libvisioncpp 와 링크(build_mmdet_cpp.sh). run_yolo_cpp 와 동일한 -DARCH 매크로 방식.
//
// 컴파일: -DARCH=<클래스명> -DVISP_ARCH_HEADER='"<gen>/<ARCH>.h"'
// 실행:  run_mmdet <gguf> <input_cwhn.bin> <postproc.json> <out.bin> [size=512]

#include VISP_ARCH_HEADER                 // 백본: <ARCH>_forward / <ARCH>_params / <ARCH>_detect_params
#include "head.h"                          // head 부품: anchor_head_forward (같은 폴더)
#include "visp/image.h"                   // image_load (이미지 입력 pre)
#include "visp/ml.h"
#include "visp/postproc.h"                // detect_anchor, preprocess, det_params, detection

#include <ggml.h>
#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstring>
#include <fstream>
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

// 러너가 필요로 하는 설정 전부. 예전엔 postproc.json 에서만 왔고, 이제 gguf 메타에서도 온다.
struct mmdet_cfg {
    anchor_head_cfg head;
    det_params det;
    float img_mean[3] = {0, 0, 0};
    float img_std[3] = {1, 1, 1};
    bool to_rgb = false;
};

// gguf 메타(`mmdet.*`) 에서 읽는다 — mmdet_inject.py 가 심어둔 값.
static mmdet_cfg cfg_from_gguf(model_file const& f) {
    mmdet_cfg c;
    auto arr = [&](const char* k, std::vector<float>& out) {
        std::string key = std::string("mmdet.") + k;
        out.resize(f.get_array_size(key.c_str()));
        f.get_array(key.c_str(), std::span<float>(out.data(), out.size()));
    };
    auto arr4 = [&](const char* k, float (&out)[4]) {
        std::vector<float> v;
        arr(k, v);
        for (int i = 0; i < 4 && i < (int)v.size(); ++i) out[i] = v[i];
    };
    auto str = [&](const char* k, std::string dflt) {
        std::string key = std::string("mmdet.") + k;
        return f.has_key(key.c_str()) ? std::string(f.get_string(key.c_str())) : dflt;
    };
    auto num = [&](const char* k, int dflt) {
        std::string key = std::string("mmdet.") + k;
        return f.has_key(key.c_str()) ? f.get_int(key.c_str()) : dflt;
    };

    c.head.stacked_convs = num("stacked_convs", 4);
    c.head.feat_channels = num("feat_channels", 256);
    c.head.num_base = num("num_base", 9);
    c.head.num_classes = num("num_classes", 80);
    c.head.cls_convs_prefix = str("cls_convs_prefix", "bbox_head.cls_convs");
    c.head.reg_convs_prefix = str("reg_convs_prefix", "bbox_head.reg_convs");
    c.head.cls_head = str("cls_head", "bbox_head.retina_cls");
    c.head.reg_head = str("reg_head", "bbox_head.retina_reg");

    arr("strides", c.det.strides);
    arr("octave_scales", c.det.octave_scales);
    arr("ratios", c.det.ratios);
    arr4("means", c.det.means);
    arr4("stds", c.det.stds);
    c.det.octave_base_scale = f.has_key("mmdet.octave_base_scale")
                                  ? f.get_float("mmdet.octave_base_scale") : 4.0f;
    c.det.center_offset = f.has_key("mmdet.center_offset")
                              ? f.get_float("mmdet.center_offset") : 0.0f;
    c.det.num_classes = c.head.num_classes;
    c.det.use_sigmoid = f.has_key("mmdet.use_sigmoid") ? f.get_bool("mmdet.use_sigmoid") : true;

    std::vector<float> v;
    if (f.has_key("mmdet.img_mean")) { arr("img_mean", v); for (int i = 0; i < 3; ++i) c.img_mean[i] = v[i]; }
    if (f.has_key("mmdet.img_std"))  { arr("img_std", v);  for (int i = 0; i < 3; ++i) c.img_std[i] = v[i]; }
    c.to_rgb = f.has_key("mmdet.to_rgb") ? f.get_bool("mmdet.to_rgb") : false;
    return c;
}

// 사이드카 JSON 에서 읽는다 — gguf 에 메타가 없는 구형 산출물용.
static mmdet_cfg cfg_from_json(nlohmann::json const& j) {
    mmdet_cfg c;
    c.head.stacked_convs = j.value("stacked_convs", 4);
    c.head.feat_channels = j.value("feat_channels", 256);
    c.head.num_base = j.value("num_base", 9);
    c.head.num_classes = j.value("num_classes", 80);
    c.head.cls_convs_prefix = j.value("cls_convs_prefix", std::string("bbox_head.cls_convs"));
    c.head.reg_convs_prefix = j.value("reg_convs_prefix", std::string("bbox_head.reg_convs"));
    c.head.cls_head = j.value("cls_head", std::string("bbox_head.retina_cls"));
    c.head.reg_head = j.value("reg_head", std::string("bbox_head.retina_reg"));

    c.det.strides = j["strides"].get<std::vector<float>>();
    c.det.octave_base_scale = j.value("octave_base_scale", 4.0f);
    c.det.octave_scales = j["octave_scales"].get<std::vector<float>>();
    c.det.ratios = j["ratios"].get<std::vector<float>>();
    c.det.center_offset = j.value("center_offset", 0.0f);
    c.det.num_classes = c.head.num_classes;
    c.det.use_sigmoid = j.value("use_sigmoid", true);
    { auto mn = j["means"].get<std::vector<float>>(); auto sd = j["stds"].get<std::vector<float>>();
      for (int i = 0; i < 4; ++i) { c.det.means[i] = mn[i]; c.det.stds[i] = sd[i]; } }

    if (j.contains("img_mean")) { auto v = j["img_mean"].get<std::vector<float>>(); for (int i = 0; i < 3; ++i) c.img_mean[i] = v[i]; }
    if (j.contains("img_std"))  { auto v = j["img_std"].get<std::vector<float>>();  for (int i = 0; i < 3; ++i) c.img_std[i] = v[i]; }
    c.to_rgb = j.value("to_rgb", false);
    return c;
}

// 입력이 이미지(.jpg/.png…)면 preprocess()(resize+normalize+to_rgb)로 텐서 생성.
// .bin 이면 이미 전처리된 CWHN f32 텐서로 간주. → yolo run_yolo_cpp 는 .bin(외부 전처리)만, 여기선 둘 다.
static std::vector<float> load_input(const char* path, int SZ, mmdet_cfg const& c) {
    std::string s(path);
    auto ext = [&](const char* e) {
        size_t n = std::strlen(e);
        return s.size() >= n && s.compare(s.size() - n, n, e) == 0;
    };
    if (ext(".jpg") || ext(".jpeg") || ext(".png") || ext(".bmp")) {
        image_data img = image_load(path);
        int iw = img.extent[0], ih = img.extent[1];
        int ic = n_channels(img.format);   // stbi_load(...,0)=네이티브 채널수 (JPEG=3, PNG+α=4)
        printf("- preprocess: image %dx%dx%d → %dx%d (mean %.1f,%.1f,%.1f std %.1f,%.1f,%.1f to_rgb=%d)\n",
            iw, ih, ic, SZ, SZ, c.img_mean[0], c.img_mean[1], c.img_mean[2],
            c.img_std[0], c.img_std[1], c.img_std[2], (int)c.to_rgb);
        return preprocess(img.data.get(), ih, iw, ic, SZ, c.img_mean, c.img_std, c.to_rgb);
    }
    return load_bin(path, (size_t)3 * SZ * SZ);
}

int main(int argc, char** argv) {
    auto usage = [&] {
        fprintf(stderr,
            "usage: %s <gguf> <input> <out.bin> [size=512]\n"
            "       %s <gguf> <input> <postproc.json> <out.bin> [size=512]   (older exports)\n",
            argv[0], argv[0]);
    };
    if (argc < 4) { usage(); return 1; }
    const char* gguf = argv[1];
    const char* inp = argv[2];

    // 1) 가중치 (백본 + head 전부 이 gguf 에)
    backend_device backend = backend_init();
    model_file file = model_load(gguf);

    // 설정이 gguf 안에 있으면 사이드카 인자가 없다 — 그만큼 argv 가 한 칸씩 당겨진다.
    const bool from_gguf = file.has_key("mmdet.strides");
    const char* jsonp = from_gguf ? nullptr : argv[3];
    const char* outp = from_gguf ? argv[3] : (argc > 4 ? argv[4] : nullptr);
    const int SZ = from_gguf ? (argc > 4 ? atoi(argv[4]) : 512)
                             : (argc > 5 ? atoi(argv[5]) : 512);
    if (!outp) { usage(); return 1; }
    printf("- config: %s\n", from_gguf ? "gguf metadata" : jsonp);

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

    // 3) 설정: gguf 메타가 있으면 거기서, 없으면 사이드카 JSON 에서.
    mmdet_cfg cfg = from_gguf ? cfg_from_gguf(file) : [&] {
        nlohmann::json j;
        std::ifstream jf(jsonp);
        if (!jf) { fprintf(stderr, "cannot open %s\n", jsonp); exit(1); }
        jf >> j;
        return cfg_from_json(j);
    }();
    anchor_head_cfg& hc = cfg.head;
    det_params& dp = cfg.det;
    dp.input_w = SZ;
    dp.input_h = SZ;
    int L = (int)dp.strides.size();
    if (L == 0) { fprintf(stderr, "no FPN strides in the configuration\n"); return 1; }

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
    auto in_data = load_input(inp, SZ, cfg);
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

    FILE* f = fopen(outp, "wb");
    for (detection const& d : dets) {
        float rec[6] = { d.x1, d.y1, d.x2, d.y2, d.score, (float)d.label };
        fwrite(rec, sizeof(float), 6, f);
    }
    fclose(f);
    printf("- detect(anchor): %zu boxes → %s (x1,y1,x2,y2,score,label f32*6)\n", dets.size(), outp);
    return 0;
}
