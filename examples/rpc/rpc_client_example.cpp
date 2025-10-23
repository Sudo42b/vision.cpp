#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-rpc.h"

#include <vector>
#include <cstdio>

static void log_cb(ggml_log_level level, const char * text, void * user) {
    (void) level; (void) user; fputs(text, stderr); fflush(stderr);
}

int main(int argc, char ** argv) {
    ggml_log_set(log_cb, nullptr);
    ggml_time_init();

    const char * endpoint = argc > 1 ? argv[1] : "127.0.0.1:50051";

    // 1) RPC 백엔드 초기화
    ggml_backend_t backend = ggml_backend_rpc_init(endpoint);
    if (!backend) {
        fprintf(stderr, "failed to init RPC backend (endpoint=%s)\n", endpoint);
        return 1;
    }

    // 2) 텐서 메타데이터 컨텍스트 생성 (no-alloc)
    const int rows_A = 4, cols_A = 2;
    const int rows_B = 3, cols_B = 2;
    const int num_tensors = 2;

    struct ggml_init_params ip = {
        /*.mem_size   =*/ ggml_tensor_overhead() * num_tensors,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(ip);

    struct ggml_tensor * A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols_A, rows_A);
    struct ggml_tensor * B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols_B, rows_B);

    // 3) 원격 버퍼에 텐서들을 할당
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        fprintf(stderr, "alloc_ctx_tensors failed\n");
        return 1;
    }

    // 4) 호스트 데이터 준비 및 업로드
    float a_data[rows_A * cols_A] = {
        2, 8,
        5, 1,
        4, 2,
        8, 6
    };
    float b_data[rows_B * cols_B] = {
        10, 5,
        9, 9,
        5, 4
    };
    ggml_backend_tensor_set(A, a_data, 0, ggml_nbytes(A));
    ggml_backend_tensor_set(B, b_data, 0, ggml_nbytes(B));

    // 5) 그래프 구성 (A * B^T)
    size_t buf_sz = ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    std::vector<uint8_t> tmp(buf_sz);
    struct ggml_init_params ip2 = {
        /*.mem_size   =*/ buf_sz,
        /*.mem_buffer =*/ tmp.data(),
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * gctx = ggml_init(ip2);
    struct ggml_cgraph * gf = ggml_new_graph(gctx);
    struct ggml_tensor * R = ggml_mul_mat(gctx, A, B);
    ggml_build_forward_expand(gf, R);

    // 6) 원격 그래프 실행
    ggml_backend_graph_compute(backend, gf);

    // 7) 결과 읽기
    std::vector<float> out(ggml_nelements(R));
    ggml_backend_tensor_get(R, out.data(), 0, ggml_nbytes(R));

    printf("Result (%d x %d):\n[", (int) R->ne[0], (int) R->ne[1]);
    for (int j = 0; j < (int)R->ne[1]; ++j) {
        if (j) printf("\n");
        for (int i = 0; i < (int)R->ne[0]; ++i) {
            printf(" %.2f", out[j*R->ne[0] + i]);
        }
    }
    printf(" ]\n");

    // 정리
    ggml_free(gctx);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}


