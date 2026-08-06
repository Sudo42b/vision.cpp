// run_dump.cpp — g2c 가 생성한 whole-detector <ARCH>.cpp 를 실행하고 등록된 모든
// compute_graph_output("out_i") 를 bin 으로 덤프하는 제네릭 검증 harness.
// 러너(run_mmdet)와 달리 head/detect 없이 그래프 raw 출력만 → torch golden 과 cosine 비교용.
//
// 컴파일: -DARCH=<클래스명> -DVISP_ARCH_HEADER='"<gen>/<ARCH>.h"'
#include VISP_ARCH_HEADER              // <ARCH>_forward / <ARCH>_params / <ARCH>_detect_params
#include "visp/ml.h"
#include <ggml.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <span>
#include <string>
#include <vector>
using namespace visp;

#define CAT_(a, b) a##b
#define CAT(a, b) CAT_(a, b)
#define FWD CAT(ARCH, _forward)
#define DETECT_PARAMS CAT(ARCH, _detect_params)

static std::vector<float> load_bin(const char* path, size_t n) {
    std::vector<float> v(n);
    std::ifstream f(path, std::ios::binary);
    f.read(reinterpret_cast<char*>(v.data()), n * sizeof(float));
    return v;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <gguf> <in_cwhn.bin> <out_prefix> [size=512]\n", argv[0]);
        return 1;
    }
    const char* gguf = argv[1];
    const char* inb = argv[2];
    std::string pref = argv[3];
    const int SZ = argc > 4 ? atoi(argv[4]) : 512;

    // 기본은 backend_init()(=init_best). GTX 백엔드가 빌드돼 있으면 그걸 고르는데,
    // **codegen 정확성(축 A) 검증은 CPU 로 해야 한다** — NPU 가 안 떠 있으면
    // "Failed to open PCIe connection" 후 buffer alloc assert 로 죽는다.
    // VISP_BACKEND=cpu 로 강제(미설정 시 기존 동작 그대로).
    const char* be_env = std::getenv("VISP_BACKEND");
    backend_device backend = (be_env && std::string(be_env) == "cpu")
                                 ? backend_init(backend_type::cpu)
                                 : backend_init();
    model_file file = model_load(gguf);
    model_weights weights = model_init(file.n_tensors());
    model_transfer(file, weights, backend, backend.preferred_float_type(), file.tensor_layout());

    compute_graph graph = compute_graph_init(262144);
    model_ref m(weights, graph);
    auto p = DETECT_PARAMS(file);

    tensor input = compute_graph_input(m, GGML_TYPE_F32, {3, SZ, SZ, 1}, "x");
    ggml_build_forward_expand(graph, input);
    tensor last = FWD(m, input, p);   // 내부에서 compute_graph_output 이 out_i 를 그래프에 등록
    ggml_build_forward_expand(graph, last);

    // G2C_DUMP=<dir> → **모든 중간 노드**를 <이름>.bin 으로 덤프한다(cli.cpp 와 동일).
    // 최종 출력만 비교하면 "어디서 갈라졌는지" 를 못 찾는다. 노드 단위 이분탐색용.
    const bool dump_all = std::getenv("G2C_DUMP") != nullptr;
    if (dump_all) {
        for (int i = 0; i < ggml_graph_n_nodes(graph.graph); ++i) {
            ggml_set_output(ggml_graph_node(graph.graph, i));
        }
    }

    compute_graph_allocate(graph, backend);
    auto in = load_bin(inb, (size_t)3 * SZ * SZ);
    transfer_to_backend(input, std::span<const float>(in.data(), in.size()));
    compute(graph, backend);

    if (dump_all) {
        const std::string dir(std::getenv("G2C_DUMP"));
        int dn = 0;
        for (int i = 0; i < ggml_graph_n_nodes(graph.graph); ++i) {
            ggml_tensor* t = ggml_graph_node(graph.graph, i);
            const char* nm = ggml_get_name(t);
            if (!nm || !nm[0] || t->type != GGML_TYPE_F32) continue;
            std::vector<float> d(ggml_nelements(t));
            ggml_backend_tensor_get(t, d.data(), 0, d.size() * sizeof(float));
            std::ofstream of(dir + "/" + nm + ".bin", std::ios::binary);
            of.write(reinterpret_cast<char const*>(d.data()), d.size() * sizeof(float));
            ++dn;
        }
        printf("- G2C_DUMP: %d nodes -> %s\n", dn, dir.c_str());
    }

    int n = 0;
    for (;; ++n) {
        std::string nm = "out_" + std::to_string(n);
        tensor t = ggml_graph_get_tensor(graph.graph, nm.c_str());
        if (!t) break;
        size_t ne = ggml_nelements(t);
        std::vector<float> d(ne);
        transfer_from_backend(t, std::span<float>(d.data(), d.size()));
        std::string op = pref + ".out." + std::to_string(n) + ".bin";
        FILE* f = fopen(op.c_str(), "wb");
        fwrite(d.data(), sizeof(float), ne, f);
        fclose(f);
        printf("out_%d: ne=%zu [%lld %lld %lld %lld]\n", n, ne,
               (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3]);
    }
    printf("dumped %d outputs → %s.out.*.bin\n", n, pref.c_str());
    // 이 프로그램은 모델별로 한 번 실행하고 끝나는 검증용 subprocess다. 현재 deploy
    // CPU runtime은 모든 출력 저장 뒤 정적/RAII teardown 중 segfault할 수 있어 정상 계산도
    // RUN_FAIL로 오분류된다. 파일과 로그를 명시적으로 flush한 뒤 destructor를 건너뛴다.
    // 제품 runtime의 종료 결함을 고치는 우회가 아니라 CPU 수치 sweep의 경계 분리다.
    fflush(nullptr);
    std::_Exit(EXIT_SUCCESS);
}
