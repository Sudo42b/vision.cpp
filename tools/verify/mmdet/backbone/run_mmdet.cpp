// run_mmdet.cpp — mmdet 검증 하네스.
//
// 배포 경로는 `vision-cli <arch> -m <gguf> -i <img> -o out.png` 하나다. 이 러너는 그
// 경로에 없는 두 가지를 위해 남아 있다.
//   · 전처리를 마친 `.bin` 텐서를 그대로 입력으로 받는다 — torch 와 같은 입력으로 맞춰
//     비교하려면 이미지 디코딩 차이를 빼야 한다.
//   · `MMDET_DUMP_HEAD` 로 디코드 전 중간 텐서를 덤프한다.
// 백본은 arch/ 에 설치하지 않고 g2c 산출 `.cpp` 를 그대로 컴파일한다(build_mmdet_cpp.sh).
//
// 조립·디코드는 라이브러리의 `mmdet_run` 이 한다 — vision-cli 와 같은 코드다.
//
// 컴파일: -DARCH=<클래스명> -DVISP_ARCH_HEADER='"<gen>/<ARCH>.h"'
// 실행:   run_mmdet <gguf> <input> <out.png|out.bin> [size=512]

#include VISP_ARCH_HEADER                 // 백본: <ARCH>_forward / <ARCH>_detect_params
#include MMDET_PARAMS_HEADER              // 생성된 mmdet_params()

#include "draw.h"
#include "visp/image.h"
#include "visp/ml.h"
#include "visp/mmdet_run.h"
#include "visp/postproc.h"

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
#define DETECT_PARAMS CAT(ARCH, _detect_params)


static tensor backbone(model_ref m, tensor x, model_file const& f) {
    return FWD(m, x, DETECT_PARAMS(f));
}

static std::vector<float> load_bin(const char* path, size_t n) {
    std::vector<float> v(n);
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    size_t got = fread(v.data(), sizeof(float), n, f);
    fclose(f);
    if (got != n) { fprintf(stderr, "short read %s: %zu/%zu\n", path, got, n); exit(1); }
    return v;
}

static bool has_ext(std::string const& s, const char* e) {
    size_t n = std::strlen(e);
    return s.size() >= n && s.compare(s.size() - n, n, e) == 0;
}

static bool is_image_path(std::string const& s) {
    return has_ext(s, ".jpg") || has_ext(s, ".jpeg") || has_ext(s, ".png") || has_ext(s, ".bmp");
}

// 이미지는 생성된 상수로 preprocess 를 거치고, `.bin` 은 이미 전처리된 CWHN f32 로 본다.
static std::vector<float> load_input(const char* path, int SZ, mmdet_cfg const& c,
                                     image_data* source) {
    std::string s(path);
    if (!is_image_path(s)) {
        return load_bin(path, (size_t)3 * SZ * SZ);
    }
    image_data img = image_load(path);
    int iw = img.extent[0], ih = img.extent[1];
    int ic = n_channels(img.format);   // stbi_load(...,0)=네이티브 채널수 (JPEG=3, PNG+α=4)
    printf("- preprocess: image %dx%dx%d → %dx%d (mean %.1f,%.1f,%.1f std %.1f,%.1f,%.1f to_rgb=%d)\n",
        iw, ih, ic, SZ, SZ, c.img_mean[0], c.img_mean[1], c.img_mean[2],
        c.img_std[0], c.img_std[1], c.img_std[2], (int)c.to_rgb);
    auto data = preprocess(img.data.get(), ih, iw, ic, SZ, c.img_mean, c.img_std, c.to_rgb);
    if (source) *source = std::move(img);
    return data;
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

    backend_device backend = backend_init();
    model_file file = model_load(gguf);
    model_weights weights = model_init(file.n_tensors());
    model_transfer(file, weights, backend, backend.preferred_float_type(), file.tensor_layout());

    mmdet_cfg cfg = mmdet_params();
    image_data source;
    std::vector<float> in_data = load_input(inp, SZ, cfg, &source);
    if (const char* dp = std::getenv("MMDET_DUMP_PRE")) {
        if (FILE* f = fopen(dp, "wb")) {
            fwrite(in_data.data(), sizeof(float), in_data.size(), f);
            fclose(f);
        }
    }

    std::vector<detection> dets;
    try {
        dets = mmdet_run(backend, weights, file, &backbone, cfg,
                         std::span<const float>(in_data.data(), in_data.size()), SZ, SZ);
    } catch (std::exception const& e) {
        fprintf(stderr, "%s\n", e.what());
        return 4;
    }

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
        printf("- detect: %zu boxes → %s (x1,y1,x2,y2,score,label f32*6)\n", dets.size(), outp);
    } else {
        float thr = 0.3f;
        if (const char* e = std::getenv("VISP_DRAW_THRESHOLD")) thr = (float)atof(e);
        // 좌표는 정사각 입력 기준이라 원본 해상도로 되돌린다.
        float sx = float(source.extent[0]) / float(SZ);
        float sy = float(source.extent[1]) / float(SZ);
        int drawn = draw_detections(source, dets, sx, sy, thr);
        image_save(source, outp);
        printf("- detect: %zu boxes, %d drawn at score >= %.2f → %s\n",
            dets.size(), drawn, thr, outp);
    }

    // 출력 파일을 열지 않아도 한 번의 실행이 무엇을 냈는지 보이게 상위 몇 개를 찍는다.
    // VISP_PRINT_DETS 가 개수를 정하고, 0 이면 끈다.
    int n_print = 10;
    if (const char* e = std::getenv("VISP_PRINT_DETS")) n_print = atoi(e);
    n_print = std::min<int>(n_print, (int)dets.size());
    if (n_print > 0) {
        printf("\n  %5s %8s %8s %8s %8s %8s %6s\n", "#", "x1", "y1", "x2", "y2", "score", "label");
        for (int i = 0; i < n_print; ++i) {
            detection const& d = dets[i];
            printf("  %5d %8.1f %8.1f %8.1f %8.1f %8.3f %6d\n",
                i, d.x1, d.y1, d.x2, d.y2, d.score, d.label);
        }
        if ((int)dets.size() > n_print) printf("  %5s  (%zu more)\n", "...", dets.size() - n_print);
        printf("\n");
    }
    return 0;
}
