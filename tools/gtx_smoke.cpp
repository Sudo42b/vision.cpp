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


// ── 커널별 크기 한계 찾기 ────────────────────────────────────────────
// ⚠️ **커널마다 감당하는 크기가 다르고, 넘으면 조용히 틀린 값을 낸다.**
//    ADD 는 전송을 블록으로 쪼개 65535 까지 맞지만, unary 계열(relu·clamp·leaky_relu)은
//    쪼개지 않고 폭을 `(uint16_t)(elems * 2)` 로 만들어 **32768 부터 감긴다.**
//    한 커널만 재고 나머지를 대표시키면 그 결함이 통과한다 — 실측으로 ResNet-18 이
//    argmax 282 → 228 로 틀렸는데 ADD 만 재던 스모크는 ALL PASS 였다.
//    `gtx_smoke --limits` 로 커널마다 통과하는 최대 크기를 이분해 표로 낸다.

using build_fn = ggml_tensor * (*)(ggml_context *, ggml_tensor *);
using want_fn  = double (*)(double);

static ggml_tensor * bld_relu(ggml_context * c, ggml_tensor * x) { return ggml_relu(c, x); }
static ggml_tensor * bld_clamp(ggml_context * c, ggml_tensor * x) { return ggml_clamp(c, x, -5.0f, 5.0f); }
static ggml_tensor * bld_leaky(ggml_context * c, ggml_tensor * x) { return ggml_leaky_relu(c, x, 0.1f, false); }
static ggml_tensor * bld_neg(ggml_context * c, ggml_tensor * x) { return ggml_neg(c, x); }
static ggml_tensor * bld_scale(ggml_context * c, ggml_tensor * x) { return ggml_scale(c, x, 2.0f); }

static double want_relu(double v) { return v > 0 ? v : 0.0; }
static double want_clamp(double v) { return v < -5.0 ? -5.0 : (v > 5.0 ? 5.0 : v); }
static double want_leaky(double v) { return v > 0 ? v : v * 0.1; }
static double want_neg(double v) { return -v; }
static double want_scale(double v) { return v * 2.0; }

// 한 크기에서 그 op 이 맞는 값을 내는지 본다. 계산 실패도 틀린 것으로 센다.
static bool op_ok_at(ggml_backend_t gtx, build_fn build, want_fn want, int64_t n) {
    ggml_init_params ip = {};
    ip.mem_size = ggml_tensor_overhead() * 64 + ggml_graph_overhead();
    ip.no_alloc = true;
    ggml_context * c = ggml_init(ip);
    if (c == nullptr) return false;
    ggml_tensor * x = ggml_new_tensor_1d(c, GGML_TYPE_F32, n);
    ggml_set_input(x);
    ggml_tensor * y = build(c, x);
    ggml_set_output(y);
    ggml_cgraph * g = ggml_new_graph_custom(c, 256, false);
    ggml_build_forward_expand(g, y);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(c, gtx);
    if (buf == nullptr) { ggml_free(c); return false; }
    std::vector<float> hx((size_t) n);
    for (int64_t i = 0; i < n; ++i) hx[(size_t) i] = float((i % 41) - 20);
    ggml_backend_tensor_set(x, hx.data(), 0, (size_t) n * sizeof(float));
    bool ok = ggml_backend_graph_compute(gtx, g) == GGML_STATUS_SUCCESS;
    if (ok) {
        std::vector<float> got((size_t) n);
        ggml_backend_tensor_get(y, got.data(), 0, (size_t) n * sizeof(float));
        for (int64_t i = 0; i < n && ok; ++i) {
            if (std::fabs(got[(size_t) i] - want(hx[(size_t) i])) > 0.5) ok = false;
        }
    }
    ggml_backend_buffer_free(buf);
    ggml_free(c);
    return ok;
}

// 통과하는 최대 크기를 이분한다. lo 는 통과가 확인된 값, hi 는 실패가 확인된 값.
static int64_t find_limit(ggml_backend_t gtx, build_fn build, want_fn want, int64_t hi_cap) {
    if (!op_ok_at(gtx, build, want, 1024)) return 0;          // 작은 것도 못 하면 0
    int64_t lo = 1024, hi = hi_cap;
    if (op_ok_at(gtx, build, want, hi)) return hi;            // 상한까지 다 된다
    while (hi - lo > 1) {
        int64_t mid = lo + (hi - lo) / 2;
        if (op_ok_at(gtx, build, want, mid)) lo = mid; else hi = mid;
    }
    return lo;
}

static void run_limits(ggml_backend_t gtx) {
    struct entry { const char * name; build_fn build; want_fn want; };
    const entry ops[] = {
        {"relu",       bld_relu,  want_relu},
        {"clamp",      bld_clamp, want_clamp},
        {"leaky_relu", bld_leaky, want_leaky},
        {"neg",        bld_neg,   want_neg},
        {"scale",      bld_scale, want_scale},
    };
    printf("커널별 크기 한계 (원소 수, 통과하는 최대값)\n");
    for (const entry & e : ops) {
        const int64_t lim = find_limit(gtx, e.build, e.want, 131072);
        printf("  %-12s %8lld", e.name, (long long) lim);
        if (lim == 0)            printf("   장치가 안 맡는다(또는 작은 크기부터 틀린다)");
        else if (lim >= 131072)  printf("   상한(131072)까지 통과");
        else                     printf("   %lld 부터 틀린다", (long long) (lim + 1));
        printf("\n");
    }
}

int main(int argc, char** argv) {
    const bool want_limits = argc > 1 && std::string(argv[1]) == "--limits";
    const int64_t N = (!want_limits && argc > 1) ? atoll(argv[1]) : 2048;
    if (!want_limits) {
        printf("GTX 스모크 사다리 (원소 %lld)\n", (long long) N);
    }

    // ── L0: 백엔드 init ──────────────────────────────────────────────
    ggml_backend_t gtx = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!want_limits) {
        step("L0 backend", gtx != nullptr, gtx ? ggml_backend_name(gtx) : "GPU 백엔드 없음");
    }
    if (!gtx) return 1;
    if (want_limits) {
        run_limits(gtx);
        return 0;
    }

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
