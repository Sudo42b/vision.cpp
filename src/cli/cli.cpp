#include "util/string.hpp"
#include "visp/vision.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <optional>
#include <string_view>
#include <vector>

namespace visp {
using std::filesystem::path;

enum class cli_command { sam, birefnet, migan, esrgan };

struct cli_args {
    cli_command command;
    std::vector<std::string_view> inputs;   // -i --input
    std::string_view output = "output.png"; // -o --output
    std::string_view model;                 // -m --model
    std::vector<std::string_view> prompt;
    // int threads = -1; // -t --threads
    // bool verbose = false; // -v --verbose
    std::optional<backend_type> backend_type; // -b --backend
    // std::string_view device = 0; // -d --device
    // ggml_type float_type = GGML_TYPE_COUNT; // -f32 -f16

    // path composite; // --composite
};

std::string_view next_arg(int argc, char** argv, int& i) {
    if (++i < argc) {
        return argv[i];
    } else {
        throw error("Missing argument after {}", argv[i - 1]);
    }
}

std::vector<std::string_view> collect_args(int argc, char** argv, int& i, char delim = '-') {
    std::vector<std::string_view> r;
    do {
        r.push_back(next_arg(argc, argv, i));
    } while (i + 1 < argc && argv[i + 1][0] != delim);
    if (r.empty()) {
        throw error("Missing argument after {}", argv[i - 1]);
    }
    return r;
}

int parse_int(std::string_view arg) {
    int value = 0;
    auto [ptr, ec] = std::from_chars(arg.data(), arg.data() + arg.size(), value);
    if (ec != std::errc()) {
        throw error("Invalid integer argument: {}", arg);
    }
    return value;
}

std::string_view validate_path(std::string_view arg) {
    if (!exists(path(arg))) {
        throw error("File not found: {}", arg);
    }
    return arg;
}

cli_args cli_parse(int argc, char** argv) {
    cli_args r;
    if (argc < 2) {
        throw error("Missing command.\nUsage: {} <command> [options]", argv[0]);
    }

    std::string_view arg1 = argv[1];
    if (arg1 == "sam") {
        r.command = cli_command::sam;
    } else if (arg1 == "birefnet") {
        r.command = cli_command::birefnet;
    } else if (arg1 == "migan") {
        r.command = cli_command::migan;
    } else if (arg1 == "esrgan") {
        r.command = cli_command::esrgan;
    } else {
        throw error("Unknown command: {}", arg1);
    }

    for (int i = 2; i < argc; ++i) {
        std::string_view arg = argv[i];
        if (arg == "-i" || arg == "--input") {
            r.inputs = collect_args(argc, argv, i);
            for_each(r.inputs.begin(), r.inputs.end(), validate_path);
        } else if (arg == "-o" || arg == "--output") {
            r.output = next_arg(argc, argv, i);
        } else if (arg == "-m" || arg == "--model") {
            r.model = validate_path(next_arg(argc, argv, i));
        } else if (arg == "-p" || arg == "--prompt") {
            r.prompt = collect_args(argc, argv, i, '-');
        } else if (arg == "-b" || arg == "--backend") {
            std::string_view backend_arg = next_arg(argc, argv, i);
            if (backend_arg == "cpu") {
                r.backend_type = backend_type::cpu;
            } else if (backend_arg == "gpu") {
                r.backend_type = backend_type::gpu;
            } else {
                throw error("Unknown backend type '{}', must be one of: cpu, gpu", backend_arg);
            }
        } else if (arg.starts_with("-")) {
            throw error("Unknown argument: {}", arg);
        }
    }
    return r;
}

struct timer {
    int64_t start;
    fixed_string<16> string;

    timer() : start(ggml_time_us()) {}

    int64_t elapsed() const { return ggml_time_us() - start; }
    float elapsed_ms() const { return float(elapsed()) / 1000.0f; }

    char const* elapsed_str() {
        format(string, "{:.1f} ms", elapsed_ms());
        return string.c_str();
    }
};

backend backend_init(cli_args const& args) {
    timer t;
    printf("Initializing backend... ");

    backend b;
    if (args.backend_type) {
        b = backend_init(*args.backend_type);
    } else {
        b = backend_init();
    }
    printf("done (%s)\n", t.elapsed_str());

    ggml_backend_dev_t dev = ggml_backend_get_device(b);
    char const* dev_name = ggml_backend_dev_name(dev);
    char const* dev_desc = ggml_backend_dev_description(dev);
    printf("- selected device %s - %s\n", dev_name, dev_desc);
    return b;
}

void compute_timed(compute_graph const& g, backend const& b) {
    timer t;
    printf("Running inference... ");
    compute(g, b);
    printf("complete (%s)\n", t.elapsed_str());
}

struct sam_prompt {
    i32x2 point1 = {-1, -1};
    i32x2 point2 = {-1, -1};

    bool is_point() const { return point2[0] == -1 || point2[1] == -1; }
    bool is_box() const { return !is_point(); }
};

sam_prompt sam_prompt_parse(std::span<std::string_view const> args, i32x2 extent) {
    if (args.empty()) {
        throw error(
            "SAM requires a prompt with coordinates for a point or box"
            "eg. '--prompt 100 200' to pick the point at pixel (x=100, y=200)");
    }
    if (args.size() < 2 || args.size() > 4) {
        throw error(
            "Invalid number of arguments for SAM prompt. Expected 2 (point) or 4 (box) numbers, "
            "got {}",
            args.size());
    }
    i32x2 a{-1, -1};
    if (args.size() >= 2) {
        a = {parse_int(args[0]), parse_int(args[1])};
        if (a[0] < 0 || a[1] < 0 || a[0] >= extent[0] || a[1] >= extent[1]) {
            throw error("Invalid image coordinates: ({}, {})", a[0], a[1]);
        }
    }
    i32x2 b{-1, -1};
    if (args.size() == 4) {
        b = {parse_int(args[2]), parse_int(args[3])};
        if (b[0] < 0 || b[1] < 0 || b[0] >= extent[0] || b[1] >= extent[1]) {
            throw error("Invalid image coordinates: ({}, {})", b[0], b[1]);
        }
        if (a[0] >= b[0] || a[1] >= b[1]) {
            throw error("Invalid box coordinates: ({}, {}) to ({}, {})", a[0], a[1], b[0], b[1]);
        }
    }
    return sam_prompt{a, b};
};

void run_sam(cli_args const& args) {
    backend backend = backend_init(args);

    char const* model_path = args.model.empty() ? "models/mobile-sam.gguf" : args.model.data();
    model_load_params load_params = {
        .float_type = backend.preferred_float_type(),
    };
    model_weights weights = model_load(model_path, backend, load_params);

    sam_params params{};

    image_data image = image_load(args.inputs[0].data());
    image_data_f32 image_data_ = sam_preprocess_image(image, params);

    sam_prompt prompt = sam_prompt_parse(args.prompt, image.extent);
    f32x4 prompt_data = prompt.is_point()
        ? sam_preprocess_point(prompt.point1, image.extent, params)
        : sam_preprocess_box(prompt.point1, prompt.point2, image.extent, params);

    compute_graph graph = compute_graph_init();
    model_ref m(weights, graph);

    tensor image_tensor = create_input(m, GGML_TYPE_F32, {3, 1024, 1024, 1}, "image");
    tensor point_tensor = create_input(m, GGML_TYPE_F32, {2, 2, 1, 1}, "points");

    tensor image_embed = sam_encode_image(m, image_tensor, params);
    tensor prompt_embed = prompt.is_point() ? sam_encode_points(m, point_tensor)
                                            : sam_encode_box(m, point_tensor);

    sam_prediction output = sam_predict(m, image_embed, prompt_embed);

    allocate(graph, backend);
    transfer_to_backend(image_tensor, image_data_);
    transfer_to_backend(point_tensor, span(prompt_data.v, 4));

    compute_timed(graph, backend);

    timer t_post;
    printf("Postprocessing output... ");

    tensor_data iou = transfer_from_backend(output.iou);
    tensor_data mask_data = transfer_from_backend(output.masks);

    image_data mask = sam_postprocess_mask(mask_data.as_f32(), 2, image.extent, params);
    printf("complete (%s)\n", t_post.elapsed_str());

    image_save(mask, args.output.data());

    printf("-> estimated accuracy (IoU): %f, %f, %f\n", iou.as_f32()[0], iou.as_f32()[1], iou.as_f32()[2]);
    printf("-> mask saved to %s\n", args.output.data());
}

} // namespace visp

int main(int argc, char** argv) {
    using namespace visp;
    try {
        ggml_time_init();

        cli_args args = cli_parse(argc, argv);
        run_sam(args);

    } catch (std::exception const& e) {
        printf("Error: %s\n", e.what());
        return 1;
    } catch (...) {
        return -1;
    }
    return 0;
}