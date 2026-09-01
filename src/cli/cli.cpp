#include "util/math.h"
#include "util/string.h"
#include "visp/arch_registry.h"
#include "visp/draw.h"
#include "visp/postproc.h"
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

// `generated` = g2c 산출 arch. **모델마다 항목을 늘리지 않는다** — 이름은 런타임에
// `arch_registry` 에서 찾는다. 손코딩 arch 5개만 여기 남는다(전처리·후처리가 제각각).
enum class cli_command { none, sam, birefnet, depth_anything, migan, esrgan, generated };

struct cli_args {
    cli_command command = cli_command::none;
    std::string_view arch;             // command == generated 일 때 arch 이름
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
    <arch>    - g2c generated model (see "Generated archs" below)

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
)";
    printf("%s", usage);

    // 등록된 생성 arch 는 **빌드에 무엇이 들어갔느냐**에 달렸다 — 하드코딩할 수 없다.
    auto archs = arch_all();
    printf("Generated archs (%zu):\n", archs.size());
    if (archs.empty()) {
        printf("    (none - build one with g2c, then tools/install_arch.py)\n");
    }
    for (arch_entry const& e : archs) {
        printf("    %.*s\n", (int)e.name.size(), e.name.data());
    }
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
    if (arg1 == "sam") {
        r.command = cli_command::sam;
    } else if (arg1 == "birefnet") {
        r.command = cli_command::birefnet;
    } else if (arg1 == "depthany" || arg1 == "depth-anything") {
        r.command = cli_command::depth_anything;
    } else if (arg1 == "migan") {
        r.command = cli_command::migan;
    } else if (arg1 == "esrgan") {
        r.command = cli_command::esrgan;
    } else if (arg1 == "-h" || arg1 == "--help") {
        print_usage();
    } else if (arch_find(arg1)) {
        // g2c 생성 arch. `src/visp/arch/<name>_register.cpp` 가 스스로 등록한 것.
        r.command = cli_command::generated;
        r.arch = arg1;
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
            r.model = next_arg(argc, argv, i);
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
        } else if (arg.starts_with("-")) {
            throw except("Unknown argument: {}\n{}", arg, short_usage);
        }
    }
    return r;
}

void run_sam(cli_args const&);
void run_birefnet(cli_args const&);
void run_depth_anything(cli_args const&);
void run_migan(cli_args const&);
void run_esrgan(cli_args const&);
void run_generated(cli_args const&);

} // namespace visp

//
// main

int main(int argc, char** argv) {
    using namespace visp;
    try {
        ggml_time_init();

        cli_args args = cli_parse(argc, argv);
        switch (args.command) {
            case cli_command::sam: run_sam(args); break;
            case cli_command::birefnet: run_birefnet(args); break;
            case cli_command::depth_anything: run_depth_anything(args); break;
            case cli_command::migan: run_migan(args); break;
            case cli_command::esrgan: run_esrgan(args); break;
            case cli_command::generated: run_generated(args); break;
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

path find_model(char const* model_name_or_path) {
    path p = path(model_name_or_path);
    if (exists(p) || p.is_absolute()) {
        return p;
    }
    path search_paths[5];
    search_paths[0] = path("models");
    if (char const* vision_model_dir = getenv("VISION_MODEL_DIR")) {
        search_paths[1] = path(vision_model_dir);
    }
    if (char const* xdg_data_home = getenv("XDG_DATA_HOME")) {
        search_paths[2] = path(xdg_data_home) / "visioncpp";
    }
    if (char const* home = getenv("HOME")) {
        search_paths[3] = path(home) / ".local/share/visioncpp";
    }
    if constexpr (VISP_MODEL_INSTALL_DIR[0] != '\0') {
        search_paths[4] = path(VISP_MODEL_INSTALL_DIR);
    }
    for (auto& sp : search_paths) {
        if (!sp.empty()) {
            path candidate = sp / p;
            if (exists(candidate)) {
                return candidate;
            }
        }
    }
    printf("Looking for %s\n", p.generic_string().c_str());
    for (auto& sp : search_paths) {
        if (!sp.empty()) {
            printf("Looking for %s\n", (sp / p).generic_string().c_str());
        }
    }
    throw except("Model file not found: {}", model_name_or_path);
}

std::tuple<model_file, model_weights> load_model_weights(
    cli_args const& args,
    backend_device const& dev,
    char const* default_model,
    int n_tensors = 0,
    tensor_data_layout preferred_layout = tensor_data_layout::unknown) {

    timer t;
    path model_path = find_model(args.model ? args.model : default_model);
    auto model_path_str = model_path.generic_string();
    printf("Loading model weights from '%s'... ", model_path_str.c_str());

    model_file file = model_load(model_path_str.c_str());
    model_weights weights = model_init(file.n_tensors() + n_tensors);
    if (preferred_layout == tensor_data_layout::unknown) {
        preferred_layout = file.tensor_layout();
    }
    model_transfer(file, weights, dev, dev.preferred_float_type(), preferred_layout);
    printf("done (%s)\n", t.elapsed_str());

    ggml_type ftype = file.float_type();
    if (ftype == GGML_TYPE_COUNT) {
        ftype = weights.float_type();
    }
    printf("- float type: %s\n", ggml_type_name(ftype));
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

void composite_image_with_mask(image_view image, image_view mask, char const* output_path) {
    if (!output_path) {
        return;
    }
    image_data image_f32_data;
    if (!is_float(image.format)) {
        image_f32_data = image_u8_to_f32(image, image_format::rgba_f32);
        image = image_f32_data;
    }
    image_data mask_f32_data;
    if (!is_float(mask.format)) {
        mask_f32_data = image_u8_to_f32(mask, image_format::alpha_f32);
        mask = mask_f32_data;
    }

    image_data foreground = image_estimate_foreground(image, mask);

    image_data output = image_f32_to_u8(foreground, image_format::rgba_u8);
    image_save(output, output_path);
    printf("-> image composited and saved to %s\n", output_path);
}

//
// SAM

struct sam_prompt {
    i32x2 point1 = {-1, -1};
    i32x2 point2 = {-1, -1};

    bool is_point() const { return point2[0] == -1 || point2[1] == -1; }
    bool is_box() const { return !is_point(); }
};

sam_prompt sam_parse_prompt(std::span<char const* const> args, i32x2 extent) {
    if (args.empty()) {
        throw except(
            "SAM requires a prompt with coordinates for a point or box"
            "eg. '--prompt 100 200' to pick the point at pixel (x=100, y=200)");
    }
    if (args.size() < 2 || args.size() > 4) {
        throw except(
            "Invalid number of arguments for SAM prompt. Expected 2 (point) or 4 (box) numbers, "
            "got {}",
            args.size());
    }
    i32x2 a{-1, -1};
    if (args.size() >= 2) {
        a = {parse_int(args[0]), parse_int(args[1])};
        if (a[0] < 0 || a[1] < 0 || a[0] >= extent[0] || a[1] >= extent[1]) {
            throw except("Invalid image coordinates: ({}, {})", a[0], a[1]);
        }
    }
    i32x2 b{-1, -1};
    if (args.size() == 4) {
        b = {parse_int(args[2]), parse_int(args[3])};
        if (b[0] < 0 || b[1] < 0 || b[0] >= extent[0] || b[1] >= extent[1]) {
            throw except("Invalid image coordinates: ({}, {})", b[0], b[1]);
        }
        if (a[0] >= b[0] || a[1] >= b[1]) {
            throw except("Invalid box coordinates: ({}, {}) to ({}, {})", a[0], a[1], b[0], b[1]);
        }
    }
    return sam_prompt{a, b};
};

void run_sam(cli_args const& args) {
    backend_device backend = backend_init(args);
    auto [file, weights] = load_model_weights(
        args, backend, "MobileSAM-F16.gguf", 0, backend.preferred_layout());
    sam_params params{};

    require_inputs(args.inputs, 1, "<image>");
    image_data image = image_load(args.inputs[0]);
    image_data image_data_ = sam_process_input(image, params);

    sam_prompt prompt = sam_parse_prompt(args.prompt, image.extent);
    f32x4 prompt_data = prompt.is_point()
        ? sam_process_point(prompt.point1, image.extent, params)
        : sam_process_box({prompt.point1, prompt.point2}, image.extent, params);

    compute_graph graph = compute_graph_init();
    model_ref m(weights, graph);

    tensor image_tensor = compute_graph_input(m, GGML_TYPE_F32, {3, 1024, 1024, 1}, "image");
    tensor point_tensor = compute_graph_input(m, GGML_TYPE_F32, {2, 2, 1, 1}, "points");

    tensor image_embed = sam_encode_image(m, image_tensor, params);
    tensor prompt_embed = prompt.is_point() ? sam_encode_points(m, point_tensor)
                                            : sam_encode_box(m, point_tensor);

    sam_prediction output = sam_predict_mask(m, image_embed, prompt_embed);

    compute_graph_allocate(graph, backend);
    transfer_to_backend(image_tensor, image_data_);
    transfer_to_backend(point_tensor, span(prompt_data.v, 4));

    compute_timed(graph, backend);

    timer t_post;
    printf("Postprocessing output... ");

    tensor_data iou = transfer_from_backend(output.iou);
    tensor_data mask_data = transfer_from_backend(output.masks);

    image_data mask = sam_process_mask(mask_data.as_f32(), 2, image.extent, params);
    printf("complete (%s)\n", t_post.elapsed_str());

    image_save(mask, args.output);

    auto ious = iou.as_f32();
    printf("-> estimated accuracy (IoU): %f, %f, %f\n", ious[0], ious[1], ious[2]);
    printf("-> mask saved to %s\n", args.output);

    composite_image_with_mask(image, mask, args.composite);
}

//
// BirefNet

//
// g2c 생성 arch (레지스트리 경유). **모델이 늘어도 이 함수는 안 바뀐다.**

void run_generated(cli_args const& args) {
    arch_entry const* e = arch_find(args.arch);
    ASSERT(e != nullptr, "arch not registered");   // 파싱에서 이미 확인했다

    backend_device backend = backend_init(args);
    auto [file, weights] = load_model_weights(args, backend, nullptr, 0, backend.preferred_layout());

    // gguf 이름과 부를 함수가 어긋나면 **엉뚱한 그래프에 남의 가중치**를 태운다.
    // 크래시 없이 값만 틀리므로 여기서 막는다.
    if (file.arch() != e->name) {
        throw except("Model arch is '{}' but command is '{}'", file.arch(), e->name);
    }

    require_inputs(args.inputs, 1, "<image>");
    image_data image = image_load(args.inputs[0]);

    arch_task const& task = e->task;
    // 비정사각 모델은 `input_w`/`input_h` 를 싣는다. 0 이면 정사각(`input_size`).
    const int IW = task.input_w ? task.input_w : task.input_size;
    const int IH = task.input_h ? task.input_h : task.input_size;
    const int SZ = IW;  // 정사각 경로의 옛 이름 — 아래 박스 되돌리기가 쓴다
    // 리사이즈 + **등록된** mean/std. `install_arch.py --mean/--std` 가 박는다.
    // **letterbox 가 아니라 단순 리사이즈**다 — 종횡비가 바뀌므로 박스를 되돌릴 때
    // x·y 배율을 따로 쓴다.
    const int nch = n_channels(image.format);
    std::vector<float> input_cwhn = preprocess(image.data.get(), image.extent[1], image.extent[0],
                                               nch, IW, IH, task.mean.data(), task.stdv.data(),
                                               /*to_rgb=*/false);

    compute_graph graph = compute_graph_init(262144);
    model_ref m(weights, graph);
    tensor input = compute_graph_input(m, GGML_TYPE_F32, {3, IW, IH, 1}, "x");
    ggml_build_forward_expand(graph, input);
    ggml_build_forward_expand(graph, e->forward(m, input, file));
    compute_graph_allocate(graph, backend);
    transfer_to_backend(input, std::span<const float>(input_cwhn.data(), input_cwhn.size()));
    compute_timed(graph, backend);

    // 등록된 out_0.. 를 전부 읽는다. **개수를 강제하지 않는다** — 계열마다 분기 수가 다르다
    // (YOLO26 은 one2many/one2one 두 벌을 낸다).
    std::vector<std::vector<float>> outs;
    std::vector<std::pair<int, int>> hw;
    for (int i = 0;; ++i) {
        tensor o = ggml_graph_get_tensor(graph, ("out_" + std::to_string(i)).c_str());
        if (!o) {
            break;
        }
        std::vector<float> d((size_t)ggml_nelements(o));
        transfer_from_backend(o, std::span<float>(d.data(), d.size()));
        outs.push_back(std::move(d));
        hw.push_back({(int)o->ne[2], (int)o->ne[1]});
    }
    printf("- outputs: %zu\n", outs.size());

    if (task.kind != arch_kind::detect_yolo) {
        for (size_t i = 0; i < outs.size(); ++i) {
            std::string path = std::string(args.output) + "." + std::to_string(i) + ".bin";
            if (FILE* f = fopen(path.c_str(), "wb")) {
                fwrite(outs[i].data(), sizeof(float), outs[i].size(), f);
                fclose(f);
            }
        }
        printf("-> raw outputs saved to %s.<i>.bin\n", args.output);
        return;
    }

    // 박스/점수 출력 고르기. 지정이 없으면 **뒤에서부터** 찾는다 — YOLO26 은 one2many 가 앞이고
    // 추론에 쓰는 one2one 이 뒤다. 앞 것을 쓰면 조용히 다른 분기를 재게 된다.
    int bi = task.box_out, si = task.score_out;
    if (bi < 0 || si < 0) {
        for (int i = (int)outs.size() - 1; i >= 0; --i) {
            if (si < 0 && hw[i].first == task.num_classes) {
                si = i;
            } else if (si >= 0 && bi < 0 && hw[i].first == 4) {
                bi = i;
            }
        }
    }
    if (bi < 0 || si < 0) {
        throw except("Could not find box/score outputs (num_classes={})", task.num_classes);
    }
    const int n_anchor = hw[si].second;
    printf("- decode: box=out_%d score=out_%d anchors=%d\n", bi, si, n_anchor);

    // 레벨별 격자는 stride 와 입력 크기로 정해진다. 합이 앵커 수와 안 맞으면 stride 가 틀린 것 —
    // 그대로 디코드하면 박스가 통째로 엉뚱한 데 찍힌다.
    std::vector<std::pair<int, int>> feat_hw;
    int sum = 0;
    for (float s : task.strides) {
        const int g = int(float(SZ) / s);
        feat_hw.push_back({g, g});
        sum += g * g;
    }
    if (sum != n_anchor) {
        throw except("anchor mismatch: strides give {} but graph has {}", sum, n_anchor);
    }

    yolo_dense_params dp;
    dp.strides = task.strides;
    dp.num_classes = task.num_classes;
    dp.score_thr = task.score_thr;
    dp.nms_thr = task.nms_thr;
    dp.nms_free = task.nms_free;
    dp.max_det = task.max_det;
    dp.input_w = SZ;
    dp.input_h = SZ;
    std::vector<detection> dets =
        detect_yolo_dense(outs[bi].data(), outs[si].data(), feat_hw, dp);

    const float sx = float(image.extent[0]) / float(SZ);
    const float sy = float(image.extent[1]) / float(SZ);

    // 좌표까지 찍는다. 그리기만 하면 결과를 수치로 확인할 방법이 없어서, 참조 구현과
    // 맞춰 보려면 디코더를 밖에서 다시 짜야 했다. 여기 이미 다 있는 값이다.
    printf("  %4s %9s %9s %9s %9s %8s  %s\n", "#", "x1", "y1", "x2", "y2", "score", "class");
    for (size_t i = 0; i < dets.size(); ++i) {
        detection const& d = dets[i];
        char const* name = (d.label >= 0 && size_t(d.label) < task.class_names.size())
                               ? task.class_names[d.label].c_str()
                               : "";
        printf("  %4zu %9.2f %9.2f %9.2f %9.2f %8.4f  %d %s\n", i,
               d.x1 * sx, d.y1 * sy, d.x2 * sx, d.y2 * sy, d.score, d.label, name);
    }

    draw_detections(image_span(image), dets, task.class_names, sx, sy);
    image_save(image, args.output);
    printf("-> %zu boxes drawn, saved to %s\n", dets.size(), args.output);
}

void run_birefnet(cli_args const& args) {
    backend_device backend = backend_init(args);
    auto [file, weights] = load_model_weights(
        args, backend, "BiRefNet-lite-F16.gguf", 0, backend.preferred_layout());

    require_inputs(args.inputs, 1, "<image>");
    image_data image = image_load(args.inputs[0]);
    birefnet_params params = birefnet_detect_params(file, image.extent, backend.max_alloc());
    image_data input_data = birefnet_process_input(image, params);

    i32x2 extent = params.image_extent;
    char const* image_size_str = params.image_size < 0 ? " (dynamic)" : "";
    printf("- model image size: %d%s\n", params.image_size, image_size_str);
    printf("- inference image size: %dx%d\n", extent[0], extent[1]);

    compute_graph graph = compute_graph_init(6 * 1024);
    model_ref m(weights, graph);
    print_model_flags(m);

    birefnet_buffers buffers = birefnet_precompute(m, params);
    tensor input = compute_graph_input(m, GGML_TYPE_F32, {3, extent[0], extent[1], 1});
    tensor output = birefnet_predict(m, input, params);

    compute_graph_allocate(graph, backend);
    transfer_to_backend(input, input_data);
    for (tensor_data const& buf : buffers) {
        transfer_to_backend(buf);
    }

    compute_timed(graph, backend);

    tensor_data mask_data = transfer_from_backend(output);
    image_view mask_output(extent, mask_data.as_f32());
    image_data mask_resized = image_scale(mask_output, image.extent);
    image_data mask = image_f32_to_u8(mask_resized, image_format::alpha_u8);
    image_save(mask, args.output);
    printf("-> mask saved to %s\n", args.output);

    composite_image_with_mask(image, mask_resized, args.composite);
}

//
// Depth Anything

void run_depth_anything(cli_args const& args) {
    backend_device backend = backend_init(args);
    auto [file, weights] = load_model_weights(
        args, backend, "DepthAnythingV2-Small-F32.gguf", 0, backend.preferred_layout());

    require_inputs(args.inputs, 1, "<image>");
    image_data image = image_load(args.inputs[0]);
    depthany_params params = depthany_detect_params(file, image.extent);
    image_data input_data = depthany_process_input(image, params);

    i32x2 extent = params.image_extent;
    printf("- model image size: %d\n", params.image_size);
    printf("- inference image size: %dx%d\n", params.image_extent[0], params.image_extent[1]);

    compute_graph graph = compute_graph_init();
    model_ref m(weights, graph);
    print_model_flags(m);

    tensor input = compute_graph_input(m, GGML_TYPE_F32, {3, extent[0], extent[1], 1});
    tensor output = depthany_predict(m, input, params);

    compute_graph_allocate(graph, backend);
    transfer_to_backend(input, input_data);

    compute_timed(graph, backend);

    tensor_data output_data = transfer_from_backend(output);
    image_data depth_raw = depthany_process_output(output_data.as_f32(), image.extent, params);
    image_data depth_image = image_f32_to_u8(depth_raw, image_format::alpha_u8);
    image_save(depth_image, args.output);
    printf("-> depth image saved to %s\n", args.output);
}

//
// MI-GAN

void run_migan(cli_args const& args) {
    backend_device backend = backend_init(args);
    auto [file, weights] = load_model_weights(
        args, backend, "MIGAN-512-places2-F16.gguf", backend.preferred_layout());
    migan_params params = migan_detect_params(file);
    params.invert_mask = true; // -> inpaint opaque areas

    require_inputs(args.inputs, 2, "<image> <mask>");
    image_data image = image_load(args.inputs[0]);
    image_data mask = image_load(args.inputs[1]);
    if (mask.format != image_format::alpha_u8) {
        mask = image_to_mask(mask);
    }
    image_data input_data = migan_process_input(image, mask, params);

    compute_graph graph = compute_graph_init();
    model_ref m(weights, graph);

    i64x4 input_shape = {4, params.resolution, params.resolution, 1};
    tensor input = compute_graph_input(m, GGML_TYPE_F32, input_shape);
    tensor output = migan_generate(m, input, params);

    compute_graph_allocate(graph, backend);
    transfer_to_backend(input, input_data);

    compute_timed(graph, backend);

    tensor_data output_data = transfer_from_backend(output);
    image_data output_image = migan_process_output(output_data.as_f32(), image.extent, params);
    image_data mask_resized = image_scale(mask, image.extent);
    image_data composited = image_alpha_composite(output_image, image, mask_resized);
    image_save(composited, args.output);
    printf("-> output image saved to %s\n", args.output);
}

//
// ESRGAN

void run_esrgan(cli_args const& args) {
    backend_device backend = backend_init(args);
    auto [file, weights] = load_model_weights(
        args, backend, "RealESRGAN-x4.gguf", 0, backend.preferred_layout());
    esrgan_params params = esrgan_detect_params(file);
    printf("- scale: %dx\n", params.scale);
    printf("- block count: %d\n", params.n_blocks);

    require_inputs(args.inputs, 1, "<image>");
    image_data image = image_load(args.inputs[0]);
    int tile_size = args.tile_size > 0 ? args.tile_size : 224;

    tile_layout tiles = tile_layout(image.extent, tile_size, 16);
    tile_layout tiles_out = tile_scale(tiles, params.scale);
    image_data input_tile = image_alloc(tiles.tile_size, image_format::rgb_f32);
    image_data output_tile = image_alloc(tiles_out.tile_size, image_format::rgb_f32);
    image_data output_image = image_alloc(image.extent * params.scale, image_format::rgb_f32);
    image_clear(output_image);

    compute_graph graph = compute_graph_init(esrgan_estimate_graph_size(params));
    model_ref m(weights, graph);

    i64x4 input_shape = {3, tiles.tile_size[0], tiles.tile_size[1], 1};
    tensor input = compute_graph_input(m, GGML_TYPE_F32, input_shape);
    tensor output = esrgan_generate(m, input, params);

    compute_graph_allocate(graph, backend);

    timer total;
    printf(
        "Using tile size %d with %d overlap -> %dx%d tiles\n", //
        tile_size, tiles.overlap[0], tiles.n_tiles[0], tiles.n_tiles[1]);

    for (int t = 0; t < tiles.total(); ++t) {
        printf("\rRunning inference... tile %d of %d", t + 1, tiles.total());
        i32x2 tile_coord = tiles.coord(t);
        i32x2 tile_offset = tiles.start(tile_coord);

        image_u8_to_f32(image, input_tile, f32x4(0), f32x4(1), tile_offset);
        transfer_to_backend(input, input_tile);

        compute(graph, backend);

        transfer_from_backend(output, output_tile);
        tile_merge(output_tile, output_image, tile_coord, tiles_out);
    }
    printf("\rRunning inference... complete (%s)\n", total.elapsed_str());

    image_data output_u8 = image_f32_to_u8(output_image, image_format::rgba_u8);
    image_save(output_u8, args.output);
    printf("-> output image saved to %s\n", args.output);
}

} // namespace visp
