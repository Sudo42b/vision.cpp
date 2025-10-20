#include <iostream>
using namespace std;
#include "ggml.h"
#include "util/math.h"
#include "util/string.h"
#include "visp/arch/yolov9t.h"
#include "visp/ml.h"
#include "visp/nn.h"
#include "visp/vision.h"
#include <algorithm>
#include <charconv>
#include <cstdio>
#include <filesystem>
#include <optional>
#include <string_view>
#include <vector>

namespace visp {
using std::filesystem::path;

enum class cli_command { none, sam, birefnet, depth_anything, migan, esrgan, yolov9t };

struct cli_args {
    cli_command command = cli_command::none;
    std::vector<char const*> inputs;   // -i --input
    char const* output = "output.png"; // -o --output
    char const* model = nullptr;       // -m --model
    std::vector<char const*> prompt;   // -p --prompt
    // int threads = -1; // -t --threads
    // bool verbose = false; // -v --verbose
    std::optional<backend_type> bknd_type; // -b --backend
    // std::string_view device = 0; // -d --device
    // ggml_type float_type = GGML_TYPE_COUNT; // -f32 -f16

    char const* composite = nullptr; // --composite
    int tile_size = -1;              // --tile
    // Debug dump options
    bool dump_all = true;       // --dump-all
    std::vector<int> dump_keys; // --dump-keys <k1> <k2> ...
};

void print_usage() {
    char const* const usage = R"(
Usage: vision-cli <command> [options]

Commands:
    sam       - MobileSAM image segmentation
    birefnet  - BirefNet background removal
    depthany  - Depth-Anything depth estimation
    migan     - MI-GAN inpainting
    esrgan    - ESRGAN/Real-ESRGAN upscaling
    yolov9t   - YOLOv9t object detection

Options:
    -i, --input <image1> [<image2> ...]  Input image(s)
    -o, --output <file>                  Output file (default: output.png)
    -m, --model <file>                   Model file (.gguf)
    -p, --prompt <x> [<y> ...]           Prompt (eg. pixel coordinates)
    -b, --backend <cpu|gpu>              Backend type (default: auto)
    -h, --help                           Print usage and exit
    --composite <file>                   Composite input image with mask
    --tile <size>                        Tile size to split large images

Examples:
    vision-cli sam -m MobileSAM-F16.gguf -i image.jpg -p 100 200 -o mask.png
    vision-cli birefnet -m BiRefNet-F16.gguf -i image.jpg -o mask.png --composite output.png
    vision-cli migan -m MIGAN-F16.gguf -i image.jpg mask.png -o output.png
    vision-cli esrgan -m ESRGAN-x4-F16.gguf -i image.jpg -o upscaled.png
    vision-cli yolov9t -m yolov9t_converted-F16.gguf -i image.jpg -o detections.png
)";
    printf("%s", usage);
}

char const* const short_usage = R"(
Usage: vision-cli <command> [options]
See 'vision-cli --help' for more details.
)";

char const* next_arg(int argc, char** argv, int& i) {
    if (++i < argc) {
        return argv[i];
    } else {
        throw except("Missing argument after {}", argv[i - 1]);
    }
}

std::vector<char const*> collect_args(int argc, char** argv, int& i, char delim = '-') {
    std::vector<char const*> r;
    do {
        r.push_back(next_arg(argc, argv, i));
    } while (i + 1 < argc && argv[i + 1][0] != delim);
    if (r.empty()) {
        throw except("Missing argument after {}", argv[i - 1]);
    }
    return r;
}

int parse_int(std::string_view arg) {
    int value = 0;
    auto [ptr, ec] = std::from_chars(arg.data(), arg.data() + arg.size(), value);
    if (ec != std::errc()) {
        throw except("Invalid integer argument: {}", arg);
    }
    return value;
}

char const* validate_path(char const* arg) {
    if (!exists(path(arg))) {
        throw except("File not found: {}", arg);
    }
    return arg;
}

void require_inputs(std::span<char const* const> inputs, int n_required, char const* names) {
    if (inputs.size() != size_t(n_required)) {
        throw except(
            "Expected -i to be followed by {} inputs: {} - but found {}.", n_required, names,
            inputs.size());
    }
}

cli_args cli_parse(int argc, char** argv) {
    cli_args r;
    if (argc < 2) {
        throw except("Missing command.\n{}", short_usage);
    }

    std::string_view arg1 = argv[1];
    if (arg1 == "yolov9t") {
        r.command = cli_command::yolov9t;
    } else if (arg1 == "-h" || arg1 == "--help") {
        print_usage();
    } else {
        throw except("Unknown command: '{}'\n{}", arg1, short_usage);
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
                r.bknd_type = backend_type::cpu;
            } else if (backend_arg == "gpu") {
                r.bknd_type = backend_type::gpu;
            } else {
                throw except("Unknown backend type '{}', must be one of: cpu, gpu", backend_arg);
            }
        } else if (arg == "--composite") {
            r.composite = next_arg(argc, argv, i);
        } else if (arg == "--tile") {
            r.tile_size = parse_int(next_arg(argc, argv, i));
        } else if (arg == "--dump-all") {
            r.dump_all = true;
        } else if (arg == "--dump-keys") {
            auto keys = collect_args(argc, argv, i);
            for (char const* k : keys) {
                r.dump_keys.push_back(parse_int(k));
            }
        } else if (arg.starts_with("-")) {
            throw except("Unknown argument: {}\n{}", arg, short_usage);
        }
    }
    return r;
}

void run_yolov9t(cli_args const&);

} // namespace visp

//
// main

int main(int argc, char** argv) {
    using namespace visp;
    try {
        ggml_time_init();

        cli_args args = cli_parse(argc, argv);
        switch (args.command) {
            case cli_command::yolov9t: run_yolov9t(args); break;
            case cli_command::none: break;
        }

    } catch (std::exception const& e) {
        printf("Error: %s\n", e.what());
        return 1;
    } catch (...) {
        return -1;
    }
    return 0;
}

namespace visp {

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

//
// Common helpers

backend_device backend_init(cli_args const& args) {
    timer t;
    printf("Initializing backend... ");

    backend_device b;
    if (args.bknd_type) {
        b = backend_init(*args.bknd_type);
    } else {
        b = backend_init();
    }
    printf("done (%s)\n", t.elapsed_str());

    ggml_backend_dev_t dev = ggml_backend_get_device(b);
    char const* dev_name = ggml_backend_dev_name(dev);
    char const* dev_desc = ggml_backend_dev_description(dev);
    printf("- device: %s - %s\n", dev_name, dev_desc);
    return b;
}

char const* to_string(tensor_data_layout l) {
    switch (l) {
        case tensor_data_layout::cwhn: return "cwhn";
        case tensor_data_layout::whcn: return "whcn";
        default: return "unknown";
    }
}

std::tuple<model_file, model_weights> load_model_weights(
    cli_args const& args,
    backend_device const& dev,
    char const* default_model,
    int n_tensors = 0,
    tensor_data_layout preferred_layout = tensor_data_layout::unknown) {

    timer t;
    char const* model_path = args.model ? args.model : default_model;
    printf("Loading model weights from '%s'... ", model_path);

    model_file file = model_load(model_path);
    model_weights weights = model_init(file.n_tensors() + n_tensors);
    if (preferred_layout == tensor_data_layout::unknown) {
        preferred_layout = file.tensor_layout();
    }
    model_transfer(file, weights, dev, dev.preferred_float_type(), preferred_layout);

    printf("done (%s)\n", t.elapsed_str());
    printf("- float type: %s\n", ggml_type_name(weights.float_type()));
    if (preferred_layout != tensor_data_layout::unknown) {
        printf("- tensor layout: %s\n", to_string(preferred_layout));
    }
    return {std::move(file), std::move(weights)};
}

void print_model_flags(model_ref const& m) {
    bool flash_attn = !!(m.flags & model_build_flag::flash_attention);
    printf("- flash attention: %s\n", flash_attn ? "on" : "off");
}

void compute_timed(compute_graph const& g, backend_device const& b) {
    timer t;
    printf("Running inference... ");
    compute(g, b);
    printf("complete (%s)\n", t.elapsed_str());
}

// YOLOv9t
void run_yolov9t(cli_args const& args) {
    using namespace visp::yolov9t;

    backend_device backend = backend_init(args);
    auto [file, weights] = load_model_weights(
        args, backend, "../../../models/yolov9t_converted.gguf", 0, backend.preferred_layout());

    require_inputs(args.inputs, 1, "<image>");
    image_data image = image_load(args.inputs[0]);
    yolov9t_params params = yolov9t_detect_params(file, image.extent, backend.max_alloc());

    int img_sz = check_img_size(params.input_size);
    i32x2 extent = {img_sz, img_sz};
    printf("- model image size: %dx%d\n", extent[0], extent[1]);
    printf("- original inference image size: %dx%d\n", image.extent[0], image.extent[1]);

    printf(
        "- tensor layout: %s\n",
        (backend.preferred_layout() == visp::tensor_data_layout::cwhn) ? "CWHN" : "WHCN");

    // Build compute graph
    compute_graph graph = compute_graph_init();
    model_ref m(weights, graph);
    print_model_flags(m);

    yolov9_buffers buffers = yolov9_precompute(m, params); // make_anchors
    
    image_data input_data = yolov9t_process_input2(image, params);
    printf("- resized inference image size: %dx%d\n", input_data.extent[0], input_data.extent[1]);
    
    tensor input = compute_graph_input(m, GGML_TYPE_F32, {3, params.input_size, params.input_size, 1}, "input");

    ggml_build_forward_expand(m.graph, input); // predictions
    std::vector<tensor> d = yolov9t_forward(m, input);

    for (size_t i = 0; i < d.size(); ++i) {
        if (d[i] != nullptr) {
            ggml_build_forward_expand(m.graph, d[i]);
        }
    }

    // Allocate and compute
    compute_graph_allocate(graph, backend);

    // Upload input data
    transfer_to_backend(input, input_data);
    
    for (tensor_data const& buf : buffers) {
        transfer_to_backend(buf);
    }

    std::vector<detected_obj> detections;

    compute_timed(graph, backend);

    printf("Forward pass built\n");

    // Post-processing
    //timer t_post;
    // std::vector<detected_obj> detections = non_max_suppression(outputs);
    // scale_boxes(detections, {img_sz, img_sz}, orig_extent);

    // Draw and save
    // image_data output_image = image_load(args.inputs[0]);
    // std::vector<std::string> const& class_names = get_coco_class_names();
    // draw_detections(output_image, detections, class_names);
    // image_save(output_image, args.output);

    // printf("Postprocessing complete (%s)\n", t_post.elapsed_str());
    // printf("Found %zu objects\n", detections.size());
}





} // namespace visp