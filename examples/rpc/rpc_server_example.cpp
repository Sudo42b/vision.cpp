#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-rpc.h"

#include <cstdio>
#include <cstdlib>

int main(int argc, char ** argv) {
    const char * endpoint = argc > 1 ? argv[1] : "127.0.0.1:50051";
    const char * cache_dir = argc > 2 ? argv[2] : nullptr; // optional: large tensor cache

    ggml_time_init();

    // 서버에서 실제 연산을 수행할 백엔드 선택 (여기서는 CPU)
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        fprintf(stderr, "failed to init CPU backend\n");
        return 1;
    }

    size_t free_mem = 0, total_mem = 0;
    // CPU backend는 정확한 값이 없을 수 있으므로 0 전달 OK (클라이언트 조회용)

    // 단일 스레드 동기 서버 루프 (클라이언트 1개씩 순차 처리)
    ggml_backend_rpc_start_server(backend, endpoint, cache_dir, free_mem, total_mem);

    ggml_backend_free(backend);
    return 0;
}


