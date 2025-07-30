#include "testing.hpp"
#include "visp/image.hpp"
#include "visp/ml.hpp"
#include "visp/util.hpp"
#include "visp/vision.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <string>

using namespace visp;

void run_benchmark(compute_graph& graph, backend_device& backend, char const* model_name) {
    // Warm-up
    compute(graph, backend);

    const int iterations = 10;
    std::vector<double> timings;

    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        compute(graph, backend);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start;
        timings.push_back(elapsed.count());
    }

    double mean = std::accumulate(timings.begin(), timings.end(), 0.0) / timings.size();
    double sq_sum = std::inner_product(timings.begin(), timings.end(), timings.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / timings.size() - mean * mean);
    printf("%s: %.1f +/- %.1f ms\n", model_name, mean, stdev);
}

void benchmark_sam(backend_device& backend) {
    path model_path = test_dir().models / "MobileSAM-F16.gguf";
    path input_path = test_dir().input / "cat-and-hat.jpg";

    sam_model model = sam_load_model(model_path.string().c_str(), backend);
    image_data input = image_load(input_path.string().c_str());
    sam_encode(model, image_view(input));

    i32x2 point = {input.extent[0] / 2, input.extent[1] / 2};
    image_data mask = sam_compute(model, point);

    run_benchmark(model.encoder, backend, "sam-encoder");
    run_benchmark(model.decoder, backend, "sam-decoder");
}

void benchmark_birefnet(backend_device& backend) {
    path model_path = test_dir().models / "BiRefNet-lite-F16.gguf";
    path input_path = test_dir().input / "wardrobe.jpg";

    birefnet_model model = birefnet_load_model(model_path.string().c_str(), backend);
    image_data input = image_load(input_path.string().c_str());

    birefnet_compute(model, input);
    run_benchmark(model.graph, backend, "birefnet");
}

void benchmark_migan(backend_device& backend) {
    path model_path = test_dir().models / "MIGAN-512-places2-F16.gguf";
    path image_path = test_dir().input / "bench-image.jpg";
    path mask_path = test_dir().input / "bench-mask.png";

    migan_model model = migan_load_model(model_path.string().c_str(), backend);
    image_data image = image_load(image_path.string().c_str());
    image_data mask = image_load(mask_path.string().c_str());
    migan_compute(model, image, mask);
    run_benchmark(model.graph, backend, "migan");
}

void benchmark_esrgan(backend_device& backend) {
    path model_path = test_dir().models / "RealESRGAN-x4plus_anime-6B-F16.gguf";
    path input_path = test_dir().input / "vase-and-bowl.jpg";

    esrgan_model model = esrgan_load_model(model_path.string().c_str(), backend);
    image_data input = image_load(input_path.string().c_str());
    esrgan_compute(model, input);
    run_benchmark(model.graph, backend, "esrgan");
}

backend_device initialize_backend(std::string_view backend_type) {
    if (backend_type == "cpu") {
        return backend_init(backend_type::cpu);
    } else if (backend_type == "gpu") {
        return backend_init(backend_type::gpu);
    } else {
        throw std::invalid_argument("Invalid backend type. Use 'cpu' or 'gpu'.");
    }
}

void benchmark_model(std::string_view model_name, backend_device& backend) {
    if (model_name == "sam") {
        benchmark_sam(backend);
    } else if (model_name == "birefnet") {
        benchmark_birefnet(backend);
    } else if (model_name == "migan") {
        benchmark_migan(backend);
    } else if (model_name == "esrgan") {
        benchmark_esrgan(backend);
    } else {
        fprintf(stderr, "Unknown model: %s\n", model_name.data());
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: vision-bench <model_name> <backend_type (cpu|gpu)>\n");
        return 1;
    }

    std::string_view model_name = argv[1];
    std::string_view backend_type = argv[2];

    try {
        backend_device backend = initialize_backend(backend_type);
        benchmark_model(model_name, backend);
    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }

    return 0;
}
