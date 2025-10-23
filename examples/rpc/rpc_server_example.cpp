#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-rpc.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <unistd.h>

struct rpc_server_params {
    std::string host        = "0.0.0.0";
    int         port        = 50052;
    size_t      backend_mem = 0;
};

static void print_usage(int /*argc*/, char ** argv, rpc_server_params params) {
    fprintf(stderr, "Usage: %s [options]\n\n", argv[0]);
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help            show this help message and exit\n");
    fprintf(stderr, "  -H HOST, --host HOST  host to bind to (default: %s)\n", params.host.c_str());
    fprintf(stderr, "  -p PORT, --port PORT  port to bind to (default: %d)\n", params.port);
    fprintf(stderr, "  -m MEM, --mem MEM     backend memory size (in MB)\n");
    fprintf(stderr, "\n");
}

static bool rpc_server_params_parse(int argc, char ** argv, rpc_server_params & params) {
    std::string arg;
    for (int i = 1; i < argc; i++) {
        arg = argv[i];
        if (arg == "-H" || arg == "--host") {
            if (++i >= argc) {
                return false;
            }
            params.host = argv[i];
        } else if (arg == "-p" || arg == "--port") {
            if (++i >= argc) {
                return false;
            }
            params.port = std::stoi(argv[i]);
            if (params.port <= 0 || params.port > 65535) {
                return false;
            }
        } else if (arg == "-m" || arg == "--mem") {
            if (++i >= argc) {
                return false;
            }
            params.backend_mem = std::stoul(argv[i]) * 1024 * 1024;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argc, argv, params);
            exit(0);
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            print_usage(argc, argv, params);
            exit(0);
        }
    }
    return true;
}

static ggml_backend_t create_backend() {
    ggml_backend_t backend = NULL;
#ifdef GGML_USE_CUDA
    fprintf(stderr, "%s: using CUDA backend\n", __func__);
    backend = ggml_backend_cuda_init(0); // init device 0
    if (!backend) {
        fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
    }
#elif GGML_USE_METAL
    fprintf(stderr, "%s: using Metal backend\n", __func__);
    backend = ggml_backend_metal_init();
    if (!backend) {
        fprintf(stderr, "%s: ggml_backend_metal_init() failed\n", __func__);
    }
#elif GGML_USE_SYCL
    fprintf(stderr, "%s: using SYCL backend\n", __func__);
    backend = ggml_backend_sycl_init(0); // init device 0
    if (!backend) {
        fprintf(stderr, "%s: ggml_backend_sycl_init() failed\n", __func__);
    }
#endif

    // if there aren't GPU Backends fallback to CPU backend
    if (!backend) {
        fprintf(stderr, "%s: using CPU backend\n", __func__);
        backend = ggml_backend_cpu_init();
    }
    return backend;
}
static void get_backend_memory(size_t * free_mem, size_t * total_mem) {
    #ifdef GGML_USE_CUDA
        ggml_backend_cuda_get_device_memory(0, free_mem, total_mem);
    #else
        #ifdef _WIN32
            MEMORYSTATUSEX status;
            status.dwLength = sizeof(status);
            GlobalMemoryStatusEx(&status);
            *total_mem = status.ullTotalPhys;
            *free_mem = status.ullAvailPhys;
        #else
            long pages = sysconf(_SC_PHYS_PAGES);
            long page_size = sysconf(_SC_PAGE_SIZE);
            *total_mem = pages * page_size;
            *free_mem = *total_mem;
        #endif
    #endif
    }
int main(int argc, char ** argv) {

    rpc_server_params params;
    if (!rpc_server_params_parse(argc, argv, params)) {
        fprintf(stderr, "Invalid parameters\n");
        return 1;
    }
    std::string cache_dir = "/tmp/ggml-rpc-cache";
    std::string endpoint_str = params.host + ":" + std::to_string(params.port);

    // 서버에서 실제 연산을 수행할 로컬 백엔드 생성 (예: CPU)
    ggml_backend_t backend = create_backend();
    if (!backend) {
        fprintf(stderr, "failed to init backend\n");
        return 1;
    }

    size_t free_mem = 0, total_mem = 0;
    if (params.backend_mem > 0) {
        free_mem = params.backend_mem;
        total_mem = params.backend_mem;
    } else {
        get_backend_memory(&free_mem, &total_mem);
    }

    // 단일 스레드 동기 서버 루프 (클라이언트 1개씩 순차 처리)
    ggml_backend_rpc_start_server(backend, endpoint_str.c_str(), cache_dir.c_str(), free_mem, total_mem);

    ggml_backend_free(backend);
    return 0;
}



