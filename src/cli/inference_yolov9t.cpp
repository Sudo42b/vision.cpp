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
#include <cfloat>   // FLT_MAX

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
            case cli_command::sam:
            case cli_command::birefnet:
            case cli_command::depth_anything:
            case cli_command::migan:
            case cli_command::esrgan:
                printf("Command not implemented yet.\n");
                break;
            case cli_command::yolov9t:
                run_yolov9t(args);
                break;
            case cli_command::none:
                break;
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
    yolov9t_params params = yolov9t_detect_params(file);

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

    // Prepare input
    // image_data input_data = yolov9t_process_input2(image, params);
    auto input_data = yolov9t_process_input(std::move(image), params);





    printf("- resized inference image size: %dx%d\n", input_data.img.extent[0], input_data.img.extent[1]);

    tensor input = compute_graph_input(
        m, GGML_TYPE_F32, {3, params.input_size, params.input_size, 1}, "input");

    ggml_build_forward_expand(m.graph, input);

    // Forward pass
    DetectOutput d = yolov9t_forward(m, input);

    if (d.debug_cls_logits) {
        ggml_build_forward_expand(m.graph, d.debug_cls_logits);
    }

    // Build forward graph with predictions
    ggml_build_forward_expand(m.graph, d.predictions_cls);
    ggml_build_forward_expand(m.graph, d.predictions_bbox);

    


    // Allocate and run
    compute_graph_allocate(graph, backend);

    // Upload input tensor
    transfer_to_backend(input, input_data.img);

    // 1) DFL projection (0..reg_max-1)
    transfer_to_backend(d.dfl_proj,
        std::span<const float>(d.dfl_proj_host_data.data(), d.dfl_proj_host_data.size()));

    // 2) Anchor points: [2, total_anchors]
    
    transfer_to_backend(d.anchor_points,
        std::span<const float>(d.anchor_host_data.data(), d.anchor_host_data.size()));
    
    // 3) Stride tensor: [1, total_anchors]
    transfer_to_backend(d.strides_points,
        std::span<const float>(d.stride_host_data.data(), d.stride_host_data.size()));


    transfer_to_backend(d.strides_points,
        std::span<const float>(d.stride_host_data.data(), d.stride_host_data.size()));

        
    compute_timed(graph, backend);
    printf("Forward pass complete.\n");

    // (Optional) 후처리 가능
    // std::vector<detected_obj> detections =
    //     yolov9t_postprocess(d, params, image.extent);
    // draw_detections(image, detections, get_coco_class_names());
    // image_save(image, args.output);
    timer t_post;
    // std::vector<detected_obj> detections = non_max_suppression(outputs);
    // scale_boxes(detections, {img_sz, img_sz}, orig_extent);
    std::vector<detected_obj> detections =
    non_max_suppression(d, 0.25f, 0.45f, 300, 30000, 640); // conf, iou, max_det 등
    // scale_boxes(detections, {img_sz, img_sz}, image.extent);
    scale_boxes(detections,
        {img_sz, img_sz},         // model shape
        image.extent,             // original shape
        input_data.gain,          // ✅ from letterbox
        input_data.pad_w,         // ✅ from letterbox
        input_data.pad_h); 


    
    // Draw and save
    image_data output_image = image_load(args.inputs[0]);
    std::vector<std::string> const& class_names = get_coco_class_names();
    draw_detections(output_image, detections, class_names);
    image_save(output_image, args.output);
    
    printf("Postprocessing complete (%s)\n", t_post.elapsed_str());
    printf("Found %zu objects\n", detections.size());

    printf("Inference finished.\n");
}

















// void run_yolov9t(cli_args const& args) {
//     using namespace visp::yolov9t;

//     backend_device backend = backend_init(args);
//     auto [file, weights] = load_model_weights(
//         args, backend, "../../../models/yolov9t_converted.gguf", 0, backend.preferred_layout());

//     require_inputs(args.inputs, 1, "<image>");
//     image_data image = image_load(args.inputs[0]);
//     yolov9t_params params = yolov9t_detect_params(file);

//     int img_sz = check_img_size(params.input_size);
//     i32x2 extent = {img_sz, img_sz};
//     printf("- model image size: %dx%d\n", extent[0], extent[1]);
//     printf("- original inference image size: %dx%d\n", image.extent[0], image.extent[1]);
//     printf("- tensor layout: %s\n",
//         (backend.preferred_layout() == visp::tensor_data_layout::cwhn) ? "CWHN" : "WHCN");

//     // Build graph
//     compute_graph graph = compute_graph_init();
//     model_ref m(weights, graph);
//     print_model_flags(m);

//     // Prepare input
//     // 변경: letterbox + [0,1] 정규화 경로
//     // image_data in_f32 = yolov9t_process_input(std::move(img), params);
//     image_data input_data = yolov9t_process_input(std::move(image), params);

//     // image_data input_data = yolov9t_process_input2(image, params);
//     printf("- resized inference image size: %dx%d\n", input_data.extent[0], input_data.extent[1]);

//     // quick sanity on input range
//     {
//         // input_data는 f32로 [0,1]이어야 함
//         size_t n = (size_t)input_data.extent[0]*input_data.extent[1]*3;
//         double mn=1e9, mx=-1e9, mean=0.0;
//         size_t n_nan=0, n_inf=0;
//         float* p = reinterpret_cast<float*>(input_data.data.get());
//         for (size_t i=0;i<n;i++){
//             float v = p[i];
//             if (!std::isfinite(v)){ if (std::isnan(v)) n_nan++; else n_inf++; continue; }
//             mn = std::min(mn, (double)v);
//             mx = std::max(mx, (double)v);
//             mean += v;
//         }
//         if (n) mean /= double(n);
//         printf("[DBG] input f32 stats: min=%.6f max=%.6f mean=%.6f nan=%zu inf=%zu\n", mn, mx, mean, n_nan, n_inf);
//     }

//     tensor input = compute_graph_input(m, GGML_TYPE_F32, {3, params.input_size, params.input_size, 1}, "input");
//     ggml_build_forward_expand(m.graph, input);

//     // Forward pass build
//     DetectOutput d = yolov9t_forward(m, input);
//     ggml_build_forward_expand(m.graph, d.predictions_cls);
//     ggml_build_forward_expand(m.graph, d.predictions_bbox);

//     // Allocate
//     compute_graph_allocate(graph, backend);

//     // Upload input
//     transfer_to_backend(input, input_data);

//     // Upload aux (DFL proj, anchors, strides)
//     transfer_to_backend(d.dfl_proj, std::span<const float>(d.dfl_proj_host_data.data(), d.dfl_proj_host_data.size()));
//     transfer_to_backend(d.anchor_points, std::span<const float>(d.anchor_host_data.data(), d.anchor_host_data.size()));
//     transfer_to_backend(d.strides_points, std::span<const float>(d.stride_host_data.data(), d.stride_host_data.size()));

//     // Sanity: dump proj/anchors/strides heads BEFORE compute
//     dbg_print_shape("dfl_proj", d.dfl_proj);
//     dbg_print_stats("dfl_proj", d.dfl_proj, 8);
//     dbg_dump_anchors_and_strides(d.anchor_points, d.strides_points, 8);


//     // compute 전에
//     auto ap_before = transfer_from_backend(d.anchor_points).as_f32();
//     auto st_before = transfer_from_backend(d.strides_points).as_f32();

//     // Run
//     timer t_fw;
//     compute_timed(graph, backend);
//     printf("Forward pass complete. (%s)\n", t_fw.elapsed_str());

//     {
//         auto ap_dev  = transfer_from_backend(d.anchor_points).as_f32();
//         auto st_dev  = transfer_from_backend(d.strides_points).as_f32();
//         const auto& ap_host = d.anchor_host_data;   // [x0,y0,x1,y1,...]
//         const auto& st_host = d.stride_host_data;   // [s0,s1,...]
    
//         auto feq = [](float a, float b){ return std::fabs(a-b) < 1e-6f; };
//         bool anchors_ok = true;
//         int A = (int)st_host.size();
    
//         // 길이 먼저 체크
//         if ((int)ap_dev.size() != 2*A) anchors_ok = false;
    
//         for (int j = 0; anchors_ok && j < std::min(A, 32); ++j) { // 앞부분만 스팟 체크
//             float xh = ap_host[2*j+0], yh = ap_host[2*j+1];
//             float xd = ap_dev [2*j+0], yd = ap_dev [2*j+1];
//             if (!feq(xh, xd) || !feq(yh, yd)) anchors_ok = false;
//         }
//         bool strides_ok = true;
//         if ((int)st_dev.size() != A) strides_ok = false;
//         for (int j = 0; strides_ok && j < std::min(A, 32); ++j)
//             if (!feq(st_host[j], st_dev[j])) strides_ok = false;
    
//         printf("[CHK] anchors equal (host vs device): %s\n", anchors_ok? "YES":"NO");
//         printf("[CHK] strides equal (host vs device): %s\n", strides_ok? "YES":"NO");
    
//         // 디버그로 host/dev 헤드 각각 명확히
//         // Inferred correction for line 719
//         printf("HOST anchor head: ");
//         for (int j=0;j<8 && j<A;++j) {
//             printf("%.2f %.2f ", ap_host[2*j], ap_host[2*j+1]);
//         }
//         puts(""); // This now correctly runs *after* the loop is finished

//         // Inferred correction for line 721
//         printf("DEV  anchor head: ");
//         for (int j=0;j<8 && j<A;++j) {
//             printf("%.2f %.2f ", ap_dev [2*j], ap_dev [2*j+1]);
//         }
//         puts(""); // This now correctly runs *after* the loop is finished
//     }
    


//     // ===== Post forward diagnostics =====
//     dbg_print_shape("predictions_cls", d.predictions_cls);
//     dbg_print_shape("predictions_bbox", d.predictions_bbox);
//     dbg_print_stats("predictions_cls", d.predictions_cls, 10);
//     dbg_print_stats("predictions_bbox", d.predictions_bbox, 12);

//     // top-K class probs & a few bbox samples
//     int64_t top_anchor = -1; int top_class = -1; float top_prob = -1.0f;
//     dbg_topk_cls(d.predictions_cls, /*K=*/5, top_anchor, top_class, top_prob);

//     // show sample boxes (xyxy, already multiplied by stride in graph)
//     dbg_sample_boxes(d.predictions_bbox, 5);

//     // quick candidate count above thresholds
//     {
//         tensor_data td_cls = transfer_from_backend(d.predictions_cls);
//         const float* cp = td_cls.as_f32().data();
//         int64_t nc = d.predictions_cls->ne[0];
//         int64_t na = d.predictions_cls->ne[1];
//         size_t cnt25=0, cnt50=0, cnt70=0, cnt90=0;
//         for (int64_t j=0;j<na;++j){
//             float best=-1.0f;
//             for (int64_t c=0;c<nc;++c) best = std::max(best, cp[c + j*nc]);
//             if (best>=0.25f) cnt25++;
//             if (best>=0.50f) cnt50++;
//             if (best>=0.70f) cnt70++;
//             if (best>=0.90f) cnt90++;
//         }
//         printf("[DBG] candidate anchors by conf: >=0.25:%zu  >=0.50:%zu  >=0.70:%zu  >=0.90:%zu  (na=%lld)\n",
//                cnt25, cnt50, cnt70, cnt90, (long long)na);
//     }

//     // (기존) 간단 덤프
//     printf("[DEBUG] ======== cls tensor dump ========\n");
//     printf("cls tensor shape: [%ld, %ld, %ld, %ld]\n",
//         d.predictions_cls->ne[0], d.predictions_cls->ne[1],
//         d.predictions_cls->ne[2], d.predictions_cls->ne[3]);
//     {
//         tensor_data td_cls = transfer_from_backend(d.predictions_cls);
//         auto cls = td_cls.as_f32().data();
//         for (int i = 0; i < 10; ++i) printf("cls[%d] = %.4f\n", i, cls[i]);
//     }
//     printf("[DEBUG] ====================================\n\n");

//     {
//         tensor_data td_anchor = transfer_from_backend(d.anchor_points);
//         auto a = td_anchor.as_f32();
//         printf("anchor_points[0..7]: ");
//         for (int i = 0; i < 8 && i < (int)a.size(); ++i) printf("%.2f ", a[i]);
//         printf("\n");
//     }

//     // NMS + scale
//     timer t_post;
//     std::vector<detected_obj> detections =
//         non_max_suppression(d, 0.01f, 0.45f, 300, 30000, 640);
//     scale_boxes(detections, {img_sz, img_sz}, image.extent);

//     // Summary of NMS
//     size_t per_cls[80] = {0};
//     for (auto const& o : detections) if (0 <= o.class_id && o.class_id < 80) per_cls[o.class_id]++;
//     size_t kept = detections.size();
//     printf("[DBG] NMS kept=%zu (max_det=300)\n", kept);
//     for (int c=0;c<80;++c) if (per_cls[c]) printf("  class %d: %zu\n", c, per_cls[c]);

//     // Draw and save
//     image_data output_image = image_load(args.inputs[0]);
//     std::vector<std::string> const& class_names = get_coco_class_names();
//     draw_detections(output_image, detections, class_names);



//     // ... compute_timed(graph, backend);

//     // compute 후
//     auto ap_after = transfer_from_backend(d.anchor_points).as_f32();
//     auto st_after = transfer_from_backend(d.strides_points).as_f32();

//     auto same = [](auto a, auto b){
//         if (a.size()!=b.size()) return false;
//         for (size_t i=0;i<a.size();++i) if (std::abs(a[i]-b[i])>1e-6f) return false;
//         return true;
//     };
//     printf("[CHK] anchors equal after compute: %s\n", same(ap_before, ap_after) ? "YES" : "NO");
//     printf("[CHK] strides equal after compute: %s\n", same(st_before, st_after) ? "YES" : "NO");


//     image_save(output_image, args.output);

//     printf("Postprocessing complete (%s)\n", t_post.elapsed_str());
//     printf("Found %zu objects\n", detections.size());
//     if (top_anchor >= 0) {
//         const char* cname = (top_class>=0 && (size_t)top_class < class_names.size()) ? class_names[top_class].c_str() : "unknown";
//         printf("[DBG] Global top-1: prob=%.4f, cls=%d(%s), anchor=%lld\n",
//                top_prob, top_class, cname, (long long)top_anchor);
//     }
//     printf("Inference finished.\n");
// }
























} // namespace visp