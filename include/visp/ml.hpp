#pragma once

#include "visp/image.hpp"
#include "visp/util.hpp"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpp.h>
#include <ggml.h>

#include <array>
#include <cstddef>
#include <memory>
#include <span>
#include <vector>

namespace visp {
using std::byte;
using tensor_name = fixed_string<GGML_MAX_NAME>;
using tensor = ggml_tensor*;

//
// Backend

enum class backend_type { cpu = 1, gpu = 2 };

struct backend {
    ggml_backend_ptr handle;

    backend_type type() const;
    ggml_type preferred_float_type() const;

    operator ggml_backend_t() const { return handle.get(); }
};

backend backend_init();
backend backend_init(backend_type);

//
// Model weights
//   Loads, converts and stores model weights.
//   Allocates and transfers tensor data to backend buffers.

struct model_weights {
    ggml_context_ptr context;
    backend_type backend_type = backend_type::cpu;
    ggml_backend_buffer_ptr weights_buffer;
    std::vector<ggml_backend_buffer_ptr> extra_buffers;

    ggml_type float_type() const;
};

// Creates a GGML context with storage for a fixed number of tensors.
// Does not allocate any backend buffers.
model_weights model_init(backend const&, size_t n_tensors);

struct model_load_params {
    ggml_type float_type = GGML_TYPE_COUNT; // default: use type stored in GGUF file
    int n_extra_tensors = 0;                // number of extra tensors to allocate in the context
};

// Loads model weights from a GGUF file and transfers them to backend buffers.
model_weights model_load(char const* filepath, backend const&, model_load_params = {});

// Allocates backend buffers for the model weights if needed. Does not transfer data.
// Returns false and does nothing if all tensors already have an associated backend buffer.
bool allocate(model_weights&, backend const&);

//
// Compute graph - wrapper for ggml_cgraph and its associated backend memory

struct compute_graph {
    ggml_context_ptr context;
    ggml_cgraph* graph = nullptr;
    ggml_gallocr_ptr allocr;
};

// Initializes a compute graph and associated backend allocator.
compute_graph compute_graph_init(size_t size = GGML_DEFAULT_GRAPH_SIZE);

// Allocates memory for inputs, outputs and computations on the backend.
bool allocate(compute_graph&, backend const&);

// Runs inference. Blocks until done.
void compute(compute_graph const&, backend const&);

//
// Model ref - represents a ML model
//   Allows access to the model's weights by name, with an optional name prefix
//   to support nested modules. Main helper for building compute graphs.

struct model_ref {
    ggml_context* weights_context = nullptr;
    ggml_context* graph_context = nullptr;
    ggml_cgraph* graph = nullptr;
    backend_type backend = backend_type::cpu;
    tensor_name prefix;

    model_ref() = default;
    model_ref(model_weights& m);
    model_ref(model_weights& m, compute_graph& g);

    explicit model_ref(
        ggml_context* weights_context,
        ggml_context* graph_context = nullptr,
        ggml_cgraph* graph = nullptr,
        tensor_name prefix = {},
        backend_type backend = backend_type::cpu);

    // Find weights tensor by name, prepends the current prefix.
    tensor find(char const* name) const;    // returns null if not found
    tensor weights(char const* name) const; // asserts if not found

    model_ref with_prefix(tensor_name new_prefix) const;

    // Returns a model_ref with prefix set to <current prefix>.<sub_module>
    model_ref operator[](char const* sub_module) const;
    model_ref operator[](tensor_name sub_module) const;
    model_ref operator[](int sub_module) const;

    operator ggml_context*() { return graph_context; }
};

// Sets the name of a tensor to the current model prefix.
tensor named(model_ref&, tensor);

// Creates a new input tensor as part of the model graph, where input data is stored.
tensor create_input(model_ref&, ggml_type type, i64x4 shape, tensor_name = "input");

// Marks a tensor as an output of the compute graph.
tensor mark_output(model_ref&, tensor, tensor_name = "output");

struct tensor_data {
    tensor x;
    std::unique_ptr<byte[]> data;

    std::span<float> as_f32();
    std::span<int32_t> as_i32();
};

// Allocates data for a tensor in main memory, outside of context and backend buffers.
tensor_data tensor_alloc(tensor x);

// Loads tensor data from a file storing raw numbers as binary.
tensor_data tensor_load(tensor x, char const* filepath);

// Copies data to the tensor's backend buffer (which should already be allocated).
void transfer_to_backend(tensor_data const&);
void transfer_to_backend(tensor x, std::span<const float> data);
void transfer_to_backend(tensor x, image_cspan const& data);

// Copies tensor data from the backend buffer to main memory.
tensor_data transfer_from_backend(tensor x);
void transfer_from_backend(tensor x, std::span<float> dst, size_t offset = 0);

//
// Tensor operations

// Returns tensor shape. Allows structured binding: `auto [c, w, h, n] = nelements(t);`
inline std::array<int64_t, 4> nelements(tensor t) {
    return {t->ne[0], t->ne[1], t->ne[2], t->ne[3]};
}

struct slice_t {
    int64_t begin;
    int64_t end;
    int64_t step;

    static constexpr int64_t max = std::numeric_limits<int64_t>::max();

    // Default: selects the entire range for a dimension (ie. `tensor[:]`)
    constexpr slice_t() : begin(0), end(max), step(1) {}

    // Selects a single slice of a dimension (ie. `tensor[index]`)
    constexpr slice_t(int64_t index) : begin(index), end(index + 1), step(1) {}

    // Selects a range [begin, end) with an optional step (ie. `tensor[begin:end:step]`)
    constexpr slice_t(int64_t begin, int64_t end, int64_t step = 1)
        : begin(begin), end(end), step(step) {}
};

// Slice a tensor along one or more dimensions similar to numpy/torch. Returns a view.
// Example: `x[0, 0:64, 16:32, :]` becomes `slice(m, x, {}, {16, 32}, {0, 64}, 0)`
tensor slice(model_ref&, tensor x, slice_t s0, slice_t s1 = {}, slice_t s2 = {}, slice_t s3 = {});

// Concatenate multiple tensors along a specified dimension.
tensor concat(model_ref&, std::array<tensor, GGML_MAX_SRC> src, int dim);

// Up- or downsample a 2D tensor (WHCN) to target width x height.
tensor interpolate(model_ref&, tensor x, i64x2 target, int32_t mode);

//
// Mobile SAM

struct sam_params {
    int image_size = 1024;
    int mask_size = 256;
};

struct sam_prediction {
    tensor masks;
    tensor iou;
};

image_data_f32 sam_preprocess_image(image_view image, sam_params const&);
f32x4 sam_preprocess_point(i32x2 point, i32x2 image_extent, sam_params const&);
f32x4 sam_preprocess_box(i32x2 top_left, i32x2 bottom_right, i32x2 image_extent, sam_params const&);

tensor sam_encode_image(model_ref, tensor image, sam_params const&);
tensor sam_encode_points(model_ref, tensor coords);
tensor sam_encode_box(model_ref, tensor coords);

sam_prediction sam_predict(model_ref m, tensor image_embed, tensor prompt_embed);

image_data sam_postprocess_mask(
    std::span<float const> mask_data, int mask_index, i32x2 target_extent, sam_params const&);

} // namespace visp