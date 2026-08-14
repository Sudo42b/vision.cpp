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
#include "head.h"

#include <cmath>                          // head 부품: anchor_head_forward (같은 폴더)
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
    // deformable 을 쓰는 계열(vfnet/reppoints)은 고정 3×3 격자가 필요하다. mmdet 의
    // dcn_base_offset 과 같은 값 — [y,x] 가 번갈아 놓인 18개다. 계열이 안 쓰면 그냥 남는다.
    // ⚠️ 안 쓰는 계열에서 만들면 그래프에 안 실려 버퍼가 없고, add_buffer 가
    //    "tensor buffer not set" 으로 죽는다. 쓰는 계열에서만 만든다.
    tensor dcn_base = nullptr;
    if (hc.kind == head_kind::vfnet || hc.kind == head_kind::reppoints) {
        dcn_base = compute_graph_input(m, GGML_TYPE_F32, {18, 1, 1, 1}, "dcn_base");
        tensor_data d = tensor_alloc(dcn_base);
        std::span<float> v = d.as_f32();
        for (int i = 0; i < 9; ++i) {
            v[2 * i] = float(i / 3 - 1);        // y: -1,-1,-1, 0,0,0, 1,1,1
            v[2 * i + 1] = float(i % 3 - 1);    // x: -1,0,1 반복
        }
        m.add_buffer(std::move(d));
    }

    // ── DDQ 만 다중 패스다 ──────────────────────────────────────────────────
    // decoder 층마다 호스트 NMS 로 중복 query 를 가려야 해서 한 그래프로 못 돈다(head.h).
    // 패스 0(백본+encoder) → NMS → 패스 1..N(층 하나씩, 사이마다 NMS).
    if (hc.kind == head_kind::ddq) {
        std::vector<msd_level> lv;
        ddq_levels(feats, lv);
        ddq_stage s0;
        ddq_encode_forward(m, feats, hc, s0);
        for (tensor x : {s0.mem, s0.om, s0.ecls, s0.ecoord, s0.qmap})
            ggml_build_forward_expand(graph, x);
        compute_graph_allocate(graph, backend);
        image_data src0;
        auto in0 = load_input(inp, SZ, cfg, &src0);
        transfer_to_backend(input, std::span<const float>(in0.data(), in0.size()));
        compute(graph, backend);

        const int64_t T = s0.mem->ne[1], E = s0.mem->ne[0], NC = s0.ecls->ne[0];
        const int64_t NQ = hc.num_queries;
        std::vector<float> mem_v = to_vec(s0.mem), qm_v = to_vec(s0.qmap);
        std::vector<float> ec_v = to_vec(s0.ecls), eb_v = to_vec(s0.ecoord);

        // 호스트: 점수 = sigmoid(클래스 최댓값), 박스 = cxcywh→xyxy(sigmoid(coord)).
        auto sigm = [](float v) { return 1.0f / (1.0f + std::exp(-v)); };
        std::vector<detection> dets((size_t)T);
        for (int64_t i = 0; i < T; ++i) {
            float best = -1e30f;
            for (int64_t k = 0; k < NC; ++k) best = std::max(best, ec_v[i * NC + k]);
            float cx = sigm(eb_v[i * 4 + 0]), cy = sigm(eb_v[i * 4 + 1]);
            float w = sigm(eb_v[i * 4 + 2]), h = sigm(eb_v[i * 4 + 3]);
            dets[i] = {cx - w * 0.5f, cy - h * 0.5f, cx + w * 0.5f, cy + h * 0.5f, sigm(best), 0};
        }
        // 이미 있는 NMS 를 그대로 쓴다(postproc.cpp — "mmcv nms 동일").
        std::vector<int> keep = nms(dets, hc.ddq_iou_thr);
        if ((int64_t)keep.size() > NQ) keep.resize((size_t)NQ);
        if ((int64_t)keep.size() < NQ) {
            fprintf(stderr, "ddq: NMS 후 %zu 개 < num_queries %lld — 그래프 모양이 안 맞는다\n",
                    keep.size(), (long long)NQ);
            return 5;
        }
        // DDQ 는 query 내용을 학습 embedding 이 아니라 **선택된 위치의 feature** 로 채운다.
        std::vector<float> q_v((size_t)E * NQ), r_v((size_t)4 * NQ);
        for (int64_t i = 0; i < NQ; ++i) {
            const int64_t t0 = keep[(size_t)i];
            std::copy_n(&qm_v[(size_t)t0 * E], E, &q_v[(size_t)i * E]);
            for (int k = 0; k < 4; ++k) r_v[(size_t)i * 4 + k] = sigm(eb_v[(size_t)t0 * 4 + k]);
        }
        fprintf(stderr, "[ddq] NMS 후보 %lld → 유지 %zu (임계 %.2f)\n",
                (long long)T, keep.size(), hc.ddq_iou_thr);
        std::vector<float> mask_v((size_t)NQ * NQ, 0.0f);   // 가산 마스크(0 = 통과)
        std::vector<char> alive((size_t)NQ, 1);             // 0층은 전부 살아있다

        std::vector<std::vector<float>> cls_out, box_out;
        for (int li = 0; li < hc.dec_layers; ++li) {
            compute_graph g = compute_graph_init(65536);
            model_ref mi(weights, g);
            tensor memT = compute_graph_input(mi, GGML_TYPE_F32, {E, T, 1, 1}, "mem");
            tensor qT = compute_graph_input(mi, GGML_TYPE_F32, {E, NQ, 1, 1}, "q");
            tensor rT = compute_graph_input(mi, GGML_TYPE_F32, {4, NQ, 1, 1}, "ref");
            tensor kT = compute_graph_input(mi, GGML_TYPE_F32, {NQ, NQ, 1, 1}, "mask");
            for (tensor x : {memT, qT, rT, kT}) ggml_build_forward_expand(g, x);
            ddq_stage si;
            ddq_layer_forward(mi, lv, hc, li, memT, qT, rT, kT, si);
            for (tensor x : {si.query, si.ref, si.cls, si.box}) ggml_build_forward_expand(g, x);
            compute_graph_allocate(g, backend);
            transfer_to_backend(memT, std::span<const float>(mem_v.data(), mem_v.size()));
            transfer_to_backend(qT, std::span<const float>(q_v.data(), q_v.size()));
            transfer_to_backend(rT, std::span<const float>(r_v.data(), r_v.size()));
            transfer_to_backend(kT, std::span<const float>(mask_v.data(), mask_v.size()));
            compute(g, backend);

            cls_out.push_back(to_vec(si.cls));
            box_out.push_back(to_vec(si.box));
            q_v = to_vec(si.query);
            r_v = to_vec(si.ref);
            if (li + 1 < hc.dec_layers) {
                // 다음 층 마스크. mmdet 규약 두 가지를 그대로 따른다:
                //  ① NMS 는 **지금까지 살아남은 query 끼리만** 돈다(집합이 단조 감소한다).
                //  ② 셀 (i,j) 는 **행이나 열 중 하나라도** 살아있으면 통과, 아니면 가린다.
                //     (mmdet 주석: "if it requires to keep index i, then all cells in row
                //      or column i should be kept")
                std::vector<int> ori;
                for (int64_t i = 0; i < NQ; ++i) if (alive[(size_t)i]) ori.push_back((int)i);
                std::vector<detection> dq(ori.size());
                for (size_t n = 0; n < ori.size(); ++n) {
                    const int64_t i = ori[n];
                    float best = -1e30f;
                    for (int64_t k = 0; k < NC; ++k)
                        best = std::max(best, cls_out.back()[(size_t)i * NC + k]);
                    float cx = r_v[(size_t)i * 4 + 0], cy = r_v[(size_t)i * 4 + 1];
                    float w = r_v[(size_t)i * 4 + 2], h = r_v[(size_t)i * 4 + 3];
                    dq[n] = {cx - w * 0.5f, cy - h * 0.5f, cx + w * 0.5f, cy + h * 0.5f,
                             sigm(best), 0};
                }
                std::vector<char> next((size_t)NQ, 0);
                for (int k : nms(dq, hc.ddq_iou_thr)) next[(size_t)ori[(size_t)k]] = 1;
                alive.swap(next);
                const float NEG = -1e30f;
                for (int64_t i = 0; i < NQ; ++i)
                    for (int64_t j = 0; j < NQ; ++j)
                        mask_v[(size_t)i * NQ + j] =
                            (alive[(size_t)i] || alive[(size_t)j]) ? 0.0f : NEG;
            }
        }
        if (const char* pre = std::getenv("MMDET_DUMP_HEAD")) {
            auto dump = [&](const char* kind, size_t l, std::vector<float> const& v) {
                std::string path = std::string(pre) + "." + kind + "." + std::to_string(l) + ".bin";
                if (FILE* f = fopen(path.c_str(), "wb")) {
                    fwrite(v.data(), sizeof(float), v.size(), f); fclose(f);
                }
            };
            for (size_t l = 0; l < cls_out.size(); ++l) dump("cls", l, cls_out[l]);
            for (size_t l = 0; l < box_out.size(); ++l) dump("box", l, box_out[l]);
            printf("- ddq: %d 패스 (호스트 NMS %d 회)\n", hc.dec_layers + 1, hc.dec_layers);
        }
        return 0;
    }

    head_outputs ho;
    mmdet_head_forward(m, feats, hc, dcn_base, ho);
    std::vector<tensor>&cls_t = ho.cls, &box_t = ho.box;
    for (tensor t : cls_t) ggml_build_forward_expand(graph, t);
    for (tensor t : box_t) ggml_build_forward_expand(graph, t);
    for (tensor t : ho.ctr) ggml_build_forward_expand(graph, t);
    // ⚠️ extra 갈래도 **그래프에 넣어야** 버퍼가 생긴다. 안 넣고 읽으면
    //    `GGML_ASSERT(buf != NULL && "tensor buffer not set")` 로 죽는다.
    for (auto const& e : ho.extra)
        for (tensor t : e.second) ggml_build_forward_expand(graph, t);
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

    // 5b) head 검증용 원시 덤프. **디코드 전** 값을 그대로 내보내 torch 의 bbox_head 출력과
    //     직접 대조한다 — NMS 를 거치면 어디가 틀렸는지 못 짚는다.
    if (const char* pre = std::getenv("MMDET_DUMP_HEAD")) {
        auto dump = [&](const char* kind, size_t l, tensor t) {
            std::vector<float> v = to_vec(t);
            std::string path = std::string(pre) + "." + kind + "." + std::to_string(l) + ".bin";
            if (FILE* f = fopen(path.c_str(), "wb")) {
                fwrite(v.data(), sizeof(float), v.size(), f);
                fclose(f);
            }
        };
        for (size_t l = 0; l < cls_t.size(); ++l) dump("cls", l, cls_t[l]);
        for (size_t l = 0; l < box_t.size(); ++l) dump("box", l, box_t[l]);
        for (size_t l = 0; l < ho.ctr.size(); ++l) dump("ctr", l, ho.ctr[l]);
        // 갈래가 셋을 넘는 계열(CornerHead 6갈래). 이름은 조립기가 정한다.
        for (auto const& e : ho.extra)
            for (size_t l = 0; l < e.second.size(); ++l) dump(e.first.c_str(), l, e.second[l]);
        printf("- head raw dump: %s.{cls,box%s}.<level>.bin\n", pre,
               ho.ctr.empty() ? "" : ",ctr");
    }

    // 코너 계열은 anchor 디코드가 아예 없다(코너 짝짓기 + embedding 그룹핑). 조립·덤프까지가
    // 이 러너의 몫이고, 그 뒤를 억지로 detect_anchor 에 넣으면 의미 없는 박스가 나온다.
    if (hc.kind == head_kind::cornernet) return 0;

    // 조립기가 레벨 수를 못 채우면 아래 인덱싱이 널을 읽는다. 여기서 말한다.
    if ((int)cls_t.size() < L || (int)box_t.size() < L) {
        fprintf(stderr, "head 출력이 %d 레벨에 못 미친다 (cls %zu, box %zu)\n",
                L, cls_t.size(), box_t.size());
        return 4;
    }

    // 6) raw cls/box(+ctr) → detect_anchor (decode + NMS)
    std::vector<std::vector<float>> cls_v(L), box_v(L), ctr_v;
    std::vector<std::pair<int, int>> feat_hw(L);
    for (int l = 0; l < L; ++l) {
        feat_hw[l] = { (int)cls_t[l]->ne[2], (int)cls_t[l]->ne[1] };  // (fh, fw)
        cls_v[l] = to_vec(cls_t[l]);
        box_v[l] = to_vec(box_t[l]);
    }
    // centerness 갈래가 있으면 score factor 로 넘긴다(ATSS·PAA·DDOD…). mmdet 은 이걸
    // top-k 뒤에 곱한다 — 안 넘기면 **박스는 맞고 점수만** 높게 나온다(실측 Δ0.071).
    // ⚠️ YOLACT 의 coeff 갈래는 점수가 아니다 — `ctr_tanh` 로 가른다.
    if ((int)ho.ctr.size() >= L && !hc.ctr_tanh) {
        ctr_v.resize(L);
        for (int l = 0; l < L; ++l) ctr_v[l] = to_vec(ho.ctr[l]);
    }
    std::vector<detection> dets =
        detect_anchor(cls_v, box_v, feat_hw, dp, ctr_v.empty() ? nullptr : &ctr_v);

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
