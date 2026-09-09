#include <cstdlib>
// gtx_smoke.cpp — NPU 경로를 **계층별로** 최소 비용 검증한다.
//
// 모델 전체(LPRNet ~1시간, ISS)를 돌려 실패 지점을 찾는 건 너무 비싸다. 아래 사다리는
// 각 단계가 한 겹씩만 확인하므로 초 단위로 어디서 깨지는지 갈린다:
//
//   L0 backend  : GTX 백엔드 init            (라이브러리·transport)
//   L1 alloc    : GTX 버퍼 할당               (device memory)
//   L2 roundtrip: host→device→host 데이터 왕복 (**DDR 직접 채널** — CPY 트랩이 난 층)
//   L3 compute  : 커널 1개 실행(ADD) 후 수치 대조 (RPC launch + 커널)
//   L4 cpy      : CONT/CPY 경로 (LPRNet 이 죽은 op)
//
// 텐서는 아주 작게(기본 64x32) 잡아 ISS 부담과 로그 폭증을 피한다.
//
// 빌드: vision.cpp/tools/build_gtx_smoke.sh
// 실행: gtx_smoke [원소수]   (기본 2048)

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static int g_fail = 0;

static void step(const char* name, bool ok, const std::string& detail = "") {
    printf("  [%s] %-12s %s\n", ok ? "PASS" : "FAIL", name, detail.c_str());
    if (!ok) ++g_fail;
}

int main(int argc, char** argv) {
    const int64_t N = argc > 1 ? atoll(argv[1]) : 2048;
    printf("GTX 스모크 사다리 (원소 %lld)\n", (long long) N);

    // ── L0: 백엔드 init ──────────────────────────────────────────────
    ggml_backend_t gtx = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    step("L0 backend", gtx != nullptr, gtx ? ggml_backend_name(gtx) : "GPU 백엔드 없음");
    if (!gtx) return 1;

    ggml_init_params ip{};
    ip.mem_size = 32u * 1024 * 1024;
    ip.no_alloc = true;
    ggml_context* ctx = ggml_init(ip);

    ggml_tensor* a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N); ggml_set_input(a);
    ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N); ggml_set_input(b);

    // ── L1: 디바이스 버퍼 할당 ────────────────────────────────────────
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, gtx);
    step("L1 alloc", buf != nullptr, buf ? "" : "GTX 버퍼 할당 실패");
    if (!buf) return 2;

    // ── L2: host→device→host 왕복 (DDR 직접 채널) ─────────────────────
    std::vector<float> ha(N), hb(N), back(N);
    for (int64_t i = 0; i < N; ++i) { ha[i] = (float)(i % 97) - 48.0f; hb[i] = (float)(i % 13) - 6.0f; }
    ggml_backend_tensor_set(a, ha.data(), 0, N * sizeof(float));
    ggml_backend_tensor_set(b, hb.data(), 0, N * sizeof(float));
    ggml_backend_tensor_get(a, back.data(), 0, N * sizeof(float));
    double worst = 0;
    for (int64_t i = 0; i < N; ++i) worst = std::fmax(worst, std::fabs(back[i] - ha[i]));
    {
        char d[64]; snprintf(d, sizeof d, "max|Δ|=%.3g", worst);
        step("L2 roundtrip", worst == 0.0, d);
    }

    // ── L3: 커널 1개 실행 (ADD) ──────────────────────────────────────
    ggml_tensor* sum = ggml_add(ctx, a, b);
    ggml_set_output(sum);
    ggml_cgraph* cg = ggml_new_graph_custom(ctx, 256, false);
    ggml_build_forward_expand(cg, sum);
    ggml_backend_buffer_t obuf = ggml_backend_alloc_ctx_tensors(ctx, gtx);

    enum ggml_status st = ggml_backend_graph_compute(gtx, cg);
    if (st != GGML_STATUS_SUCCESS) {
        step("L3 compute", false, "graph_compute 실패(status=" + std::to_string((int)st) + ")");
    } else {
        std::vector<float> got(N);
        ggml_backend_tensor_get(sum, got.data(), 0, N * sizeof(float));
        double w = 0;
        for (int64_t i = 0; i < N; ++i) w = std::fmax(w, std::fabs(got[i] - (ha[i] + hb[i])));
        char d[64]; snprintf(d, sizeof d, "max|Δ|=%.3g (fp16 오차 허용)", w);
        step("L3 compute", w < 0.5, d);
    }

    // ── L3b: UNARY(ReLU) ─────────────────────────────────────────────
    // ⚠️ **L3 는 ADD 하나만 돌린다.** 그래서 다른 커널이 틀려도 ALL PASS 가 나온다 —
    //    실측(2026-09-09): ResNet-18 이 장치에서 argmax 282 → 228 로 틀렸는데
    //    스모크는 통과했다. 원인은 ReLU 였다. 커널마다 한 단계씩 둔다.
    {
        ggml_init_params ip3 = ip;
        ggml_context* c3 = ggml_init(ip3);
        // 모델의 relu 는 4차원 WHCN 이다. 1차원만 재면 모양·stride 취급을 못 본다.
        const char* shp = std::getenv("GTX_SMOKE_RELU_4D");
        ggml_tensor* x = shp ? ggml_new_tensor_4d(c3, GGML_TYPE_F32, 56, 56, 64, 1)
                             : ggml_new_tensor_1d(c3, GGML_TYPE_F32, N);
        ggml_set_input(x);
        const int64_t NR = ggml_nelements(x);
        ggml_tensor* r = ggml_relu(c3, x);
        ggml_set_output(r);
        ggml_cgraph* g3 = ggml_new_graph_custom(c3, 256, false);
        ggml_build_forward_expand(g3, r);
        ggml_backend_buffer_t rbuf = ggml_backend_alloc_ctx_tensors(c3, gtx);
        std::vector<float> hx(NR);
        for (int64_t i = 0; i < NR; ++i) hx[i] = float((i % 41) - 20);  // 음수·0·양수를 고루
        ggml_backend_tensor_set(x, hx.data(), 0, NR * sizeof(float));
        enum ggml_status st3 = ggml_backend_graph_compute(gtx, g3);
        if (st3 != GGML_STATUS_SUCCESS) {
            step("L3b relu", false, "graph_compute 실패(status=" + std::to_string((int)st3) + ")");
        } else {
            std::vector<float> got(NR);
            ggml_backend_tensor_get(r, got.data(), 0, NR * sizeof(float));
            double w = 0;
            int64_t bad = -1;
            for (int64_t i = 0; i < NR; ++i) {
                double want = hx[i] > 0 ? hx[i] : 0.0;
                double diff = std::fabs(got[i] - want);
                if (diff > w) { w = diff; bad = i; }
            }
            char d[128];
            snprintf(d, sizeof d, "max|Δ|=%.3g @i=%lld (got %.3g, want %.3g)", w,
                     (long long)bad, bad >= 0 ? got[bad] : 0.0,
                     bad >= 0 ? (hx[bad] > 0 ? hx[bad] : 0.0) : 0.0);
            step("L3b relu", w < 0.5, d);
        }
        ggml_backend_buffer_free(rbuf);
        ggml_free(c3);
    }

    // ── L4: CPY/CONT 경로 (LPRNet 이 죽은 지점) ───────────────────────
    {
        ggml_init_params ip2 = ip;
        ggml_context* c2 = ggml_init(ip2);
        ggml_tensor* x = ggml_new_tensor_2d(c2, GGML_TYPE_F32, 64, N / 64); ggml_set_input(x);
        ggml_tensor* t = ggml_cont(c2, ggml_transpose(c2, x));   // CONT → 실제 복사 발생
        ggml_set_output(t);
        ggml_cgraph* g2 = ggml_new_graph_custom(c2, 256, false);
        ggml_build_forward_expand(g2, t);
        ggml_backend_buffer_t b2 = ggml_backend_alloc_ctx_tensors(c2, gtx);
        if (!b2) {
            step("L4 cpy", false, "버퍼 할당 실패");
        } else {
            ggml_backend_tensor_set(x, ha.data(), 0, N * sizeof(float));
            enum ggml_status s2 = ggml_backend_graph_compute(gtx, g2);
            step("L4 cpy", s2 == GGML_STATUS_SUCCESS,
                 s2 == GGML_STATUS_SUCCESS ? "CONT(transpose) 성공"
                                           : "status=" + std::to_string((int)s2));
            ggml_backend_buffer_free(b2);
        }
        ggml_free(c2);
    }

    printf("\nRESULT: %s (%d 실패)\n", g_fail ? "FAIL" : "ALL PASS", g_fail);
    if (obuf) ggml_backend_buffer_free(obuf);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(gtx);
    return g_fail ? 3 : 0;
}
