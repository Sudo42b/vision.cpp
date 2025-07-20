#include "visp/ml.hpp"
#include "util/string.hpp"

#include <algorithm>
#include <array>
#include <thread>
#include <vector>

namespace visp {

//
// backend

enum ggml_backend_dev_type convert(backend_type type) {
    switch (type) {
        case backend_type::cpu: return GGML_BACKEND_DEVICE_TYPE_CPU;
        case backend_type::gpu: return GGML_BACKEND_DEVICE_TYPE_GPU;
        default:
            ASSERT(false, "Unsupported backend type");
            return GGML_BACKEND_DEVICE_TYPE_CPU; // fallback
    }
}

bool load_ggml_backends() {
    static const bool loaded = []() {
        if (ggml_backend_reg_count() > 0) {
            return true; // already loaded
        }
        ggml_backend_load_all();
        return true;
    }();
    return loaded;
}

bool backend_is_available(backend_type type) {
    load_ggml_backends();
    return ggml_backend_dev_by_type(convert(type)) != nullptr;
}

backend_device backend_init() {
    load_ggml_backends();
    backend_device b{ggml_backend_ptr(ggml_backend_init_best())};
    ASSERT(b.handle, "Failed to initialize backend");
    return b;
}

backend_device backend_init(backend_type type) {
    load_ggml_backends();

    backend_device b;
    b.handle.reset(ggml_backend_init_by_type(convert(type), nullptr));
    if (!b.handle) {
        throw error("Failed to initialize backend, no suitable device available");
    }
    b.device = ggml_backend_get_device(b.handle.get());

    int nthreads = std::max(1, (int)std::thread::hardware_concurrency() - 2);
    backend_set_n_threads(b, nthreads);
    return b;
}

backend_type backend_device::type() const {
    ggml_backend_dev_t dev = ggml_backend_get_device(handle.get());
    switch (ggml_backend_dev_type(dev)) {
        case GGML_BACKEND_DEVICE_TYPE_CPU: return backend_type::cpu;
        case GGML_BACKEND_DEVICE_TYPE_GPU: return backend_type::gpu;
        default: ASSERT(false, "Unsupported backend device type"); return backend_type::cpu;
    }
}

typedef bool (*ggml_backend_dev_supports_f16_t)(ggml_backend_dev_t);

ggml_type backend_device::preferred_float_type() const {
    if (type() == backend_type::cpu) {
        return GGML_TYPE_F32; // not all operations support F16
    } else {
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
        if (void* f = ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_supports_f16")) {
            bool supports_f16 = ((ggml_backend_dev_supports_f16_t)f)(device);
            if (!supports_f16) {
                return GGML_TYPE_F32;
            }
        }
    }
    return GGML_TYPE_COUNT; // no preference, use float type of model weights
}

size_t backend_device::total_memory() const {
    ggml_backend_dev_t dev = ggml_backend_get_device(handle.get());
    size_t free, total;
    ggml_backend_dev_memory(dev, &free, &total);
    return total;
}

void backend_set_n_threads(backend_device& b, int n_threads) {
    if (b.type() != backend_type::cpu) {
        return;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(b.device);
    ggml_backend_set_n_threads_t set_n_threads =
        (ggml_backend_set_n_threads_t)ggml_backend_reg_get_proc_address(
            reg, "ggml_backend_set_n_threads");
    ASSERT(set_n_threads, "Failed to get backend set_n_threads function");
    set_n_threads(b.handle.get(), n_threads);
}

//
// model_weights

bool is_float_type(ggml_type t) {
    return t != GGML_TYPE_I8 && t != GGML_TYPE_I16 && t != GGML_TYPE_I32 && t != GGML_TYPE_I64;
}

struct float_converter {
    ggml_type target;
    ggml_type_traits const* dst_traits = nullptr;
    std::vector<float> f32_buffer;
    std::vector<byte> dst_buffer;

    explicit float_converter(ggml_type target_type) : target(target_type) {
        if (target != GGML_TYPE_COUNT) {
            dst_traits = ggml_get_type_traits(target_type);
        }
    }

    ggml_type target_type(ggml_tensor const* t) const {
        if (target == GGML_TYPE_COUNT || !is_float_type(t->type)) {
            return t->type;
        }
        return target;
    }

    void const* operator()(ggml_tensor const* src, ggml_tensor const* dst) {
        if (target == GGML_TYPE_COUNT || src->type == dst->type) {
            return src->data;
        }
        ASSERT(dst->type == target);

        float const* f32_data = reinterpret_cast<float const*>(src->data);
        if (src->type != GGML_TYPE_F32) {
            if (int64_t(f32_buffer.size()) < ggml_nelements(src)) {
                f32_buffer.resize(ggml_nelements(src));
            }
            ggml_type_traits const* src_traits = ggml_get_type_traits(src->type);
            src_traits->to_float(src->data, f32_buffer.data(), ggml_nelements(src));
            f32_data = f32_buffer.data();
        }
        void const* dst_data = f32_data;
        if (target != GGML_TYPE_F32) {
            if (dst_buffer.size() < ggml_nbytes(dst)) {
                dst_buffer.resize(ggml_nbytes(dst));
            }
            dst_traits->from_float_ref(f32_data, dst_buffer.data(), ggml_nelements(dst));
            dst_data = dst_buffer.data();
        }
        return dst_data;
    }
};

model_weights model_init(backend_device const& be, size_t size) {
    ggml_init_params params{};
    params.mem_size = size * ggml_tensor_overhead();
    params.no_alloc = true;
    ggml_context_ptr ctx(ggml_init(params));

    return model_weights{std::move(ctx), be.type(), {}, {}};
}

model_weights model_load(char const* filepath, backend_device const& backend, model_load_params p) {

    ggml_context* data_ctx;
    gguf_init_params params;
    params.no_alloc = false;
    params.ctx = &data_ctx;

    gguf_context_ptr gguf_ctx(gguf_init_from_file(filepath, params));
    if (!gguf_ctx) {
        throw std::runtime_error("Failed to load GGUF model");
    }
    ggml_context_ptr data_ctx_ptr(data_ctx);
    int64_t n_weights = gguf_get_n_tensors(gguf_ctx.get());

    ggml_init_params model_ctx_params{};
    model_ctx_params.mem_size = (n_weights + p.n_extra_tensors) * ggml_tensor_overhead();
    model_ctx_params.no_alloc = true;
    ggml_context_ptr model_ctx(ggml_init(model_ctx_params));

    float_converter convert(p.float_type);
    for (int64_t i = 0; i < gguf_get_n_tensors(gguf_ctx.get()); ++i) {
        auto name = gguf_get_tensor_name(gguf_ctx.get(), i);
        tensor orig = ggml_get_tensor(data_ctx, name);
        tensor dup = ggml_new_tensor(
            model_ctx.get(), convert.target_type(orig), GGML_MAX_DIMS, orig->ne);
        ggml_set_name(dup, name);
    }

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(model_ctx.get(), backend));

    for (ggml_tensor* t = ggml_get_first_tensor(model_ctx.get()); t != nullptr;
         t = ggml_get_next_tensor(model_ctx.get(), t)) {
        tensor data_tensor = ggml_get_tensor(data_ctx, ggml_get_name(t));
        void const* data = convert(data_tensor, t);
        ggml_backend_tensor_set(t, data, 0, ggml_nbytes(t));
    }
    return model_weights{std::move(model_ctx), backend.type(), std::move(buffer), {}};
}

bool model_allocate(model_weights& m, backend_device const& b) {
    ASSERT(m.buffer_type == b.type(), "Model weights must all be on the same backend");

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(m.context.get(), b.handle.get()));
    if (!buffer) {
        return false; // context contains nothing to allocate
    }
    m.extra_buffers.push_back(std::move(buffer));
    return true;
}

ggml_type model_weights::float_type() const {
    for (ggml_tensor* t = ggml_get_first_tensor(context.get()); t != nullptr;
         t = ggml_get_next_tensor(context.get(), t)) {
        if (is_float_type(t->type)) {
            return t->type; // return first float type found
        }
    }
    return GGML_TYPE_COUNT;
}

//
// compute_graph

compute_graph compute_graph_init(size_t size) {
    ggml_init_params graph_ctx_params{};
    graph_ctx_params.mem_size = size * ggml_tensor_overhead() + ggml_graph_overhead();
    graph_ctx_params.no_alloc = true;
    ggml_context* ctx = ggml_init(graph_ctx_params);
    ggml_context_ptr ctx_ptr(ctx);
    ggml_cgraph* graph = ggml_new_graph_custom(ctx, size, false);
    return compute_graph{std::move(ctx_ptr), graph, nullptr};
}

bool compute_graph_allocate(compute_graph& g, backend_device const& backend) {
    if (!g.allocr) {
        g.allocr.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)));
    }
    bool result = ggml_gallocr_alloc_graph(g.allocr.get(), g.graph);
    if (!result) {
        throw std::runtime_error("Failed to allocate buffer for graph");
    }
    return result;
}

void compute(compute_graph const& g, backend_device const& b) {
    ggml_backend_graph_compute(b, g.graph);
}

//
// model_ref

model_build_flags default_backend_flags(backend_type type) {
    using enum model_build_flag;
    switch (type) {
        case backend_type::cpu:
            return cwhn | conv_2d_direct | fused_batch_norm | f16_conv_transpose | window_partition;
        case backend_type::gpu: return cwhn;
    }
    return {};
}

model_ref::model_ref(model_weights& m)
    : weights_context(m.context.get()),
      graph_context(m.context.get()),
      graph(nullptr),
      flags(default_backend_flags(m.buffer_type)) {}

model_ref::model_ref(model_weights& m, compute_graph& g)
    : weights_context(m.context.get()),
      graph_context(g.context.get()),
      graph(g.graph),
      flags(default_backend_flags(m.buffer_type)) {}

model_ref::model_ref(
    ggml_context* weights_context,
    ggml_context* graph_context,
    ggml_cgraph* graph,
    model_build_flags flags,
    tensor_name prefix)
    : weights_context(weights_context),
      graph_context(graph_context ? graph_context : weights_context),
      graph(graph),
      flags(flags),
      prefix(prefix) {}

tensor model_ref::find(char const* name) const {
    auto full_name = tensor_name();
    if (prefix) {
        name = format(full_name, "{}.{}", prefix.c_str(), name);
    }
    return ggml_get_tensor(weights_context, name);
}

tensor model_ref::weights(char const* name) const {
    if (tensor result = find(name)) {
        return result;
    }
    throw error("tensor not found: {}.{}", prefix.view(), name);
}

model_ref model_ref::with_prefix(tensor_name new_prefix) const {
    return model_ref{weights_context, graph_context, graph, flags, new_prefix};
}

template <typename Stringable>
model_ref chain_prefix(model_ref const& m, Stringable sub_module) {
    if (m.prefix) {
        return m.with_prefix(format<tensor_name>("{}.{}", m.prefix.view(), sub_module));
    } else {
        return m.with_prefix(format<tensor_name>("{}", sub_module));
    }
}

model_ref model_ref::operator[](char const* sub_module) const {
    return chain_prefix(*this, sub_module);
}
model_ref model_ref::operator[](tensor_name sub_module) const {
    return chain_prefix(*this, sub_module.view());
}
model_ref model_ref::operator[](int sub_module) const {
    return chain_prefix(*this, sub_module);
}

tensor named(model_ref const& m, tensor tensor) {
    ggml_set_name(tensor, m.prefix.c_str());
    return tensor;
}

//
// tensor creation and data handling

tensor compute_graph_input(model_ref const& m, ggml_type type, i64x4 shape, tensor_name name) {
    tensor x = ggml_new_tensor_4d(m, type, shape[0], shape[1], shape[2], shape[3]);
    ggml_set_name(x, name.c_str());
    ggml_set_input(x);
    return x;
}

tensor compute_graph_output(model_ref const& m, tensor x, tensor_name name) {
    ggml_set_name(x, name.c_str());
    ggml_set_output(x);
    ggml_build_forward_expand(m.graph, x);
    return x;
}

tensor_data tensor_alloc(tensor x) {
    return {x, std::unique_ptr<byte[]>(new byte[ggml_nbytes(x)])};
}

tensor_data tensor_load(tensor x, char const* filepath) {
    FILE* file = fopen(filepath, "rb");
    if (!file) {
        throw error("Failed to open file: {}", filepath);
    }
    tensor_data result = tensor_alloc(x);
    size_t read = fread(result.data.get(), 1, ggml_nbytes(x), file);
    fclose(file);
    if (read != ggml_nbytes(x)) {
        throw error("Failed to read data from file: {}", filepath);
    }
    return result;
}

std::span<float> tensor_data::as_f32() {
    ASSERT(x->type == GGML_TYPE_F32);
    return span(reinterpret_cast<float*>(data.get()), ggml_nelements(x));
}

std::span<float const> tensor_data::as_f32() const {
    ASSERT(x->type == GGML_TYPE_F32);
    return span(reinterpret_cast<float const*>(data.get()), ggml_nelements(x));
}

std::span<int32_t> tensor_data::as_i32() {
    ASSERT(x->type == GGML_TYPE_I32);
    return span(reinterpret_cast<int32_t*>(data.get()), ggml_nelements(x));
}

std::span<int32_t const> tensor_data::as_i32() const {
    ASSERT(x->type == GGML_TYPE_I32);
    return span(reinterpret_cast<int32_t const*>(data.get()), ggml_nelements(x));
}

void transfer_to_backend(tensor_data const& d) {
    ggml_backend_tensor_set(d.x, d.data.get(), 0, ggml_nbytes(d.x));
}

void transfer_to_backend(tensor x, std::span<const byte> data) {
    ASSERT(ggml_nbytes(x) == data.size_bytes());
    ggml_backend_tensor_set(x, data.data(), 0, ggml_nbytes(x));
}

void transfer_to_backend(tensor x, std::span<const float> data) {
    ASSERT(ggml_nbytes(x) == data.size_bytes());
    ggml_backend_tensor_set(x, data.data(), 0, ggml_nbytes(x));
}

void transfer_to_backend(tensor x, image_view const& img) {
    ASSERT(ggml_nbytes(x) == n_bytes(img));
    ggml_backend_tensor_set(x, img.data, 0, ggml_nbytes(x));
}

tensor_data transfer_from_backend(tensor x) {
    tensor_data result = tensor_alloc(x);
    ggml_backend_tensor_get(x, result.data.get(), 0, ggml_nbytes(x));
    return result;
}

void transfer_from_backend(tensor x, span<float> dst, size_t offset) {
    size_t size = std::min(dst.size_bytes(), ggml_nbytes(x) - offset);
    ggml_backend_tensor_get(x, dst.data(), offset, size);
}

void transfer_from_backend(tensor x, image_span const& dst) {
    ASSERT(ggml_nbytes(x) == n_bytes(dst));
    ggml_backend_tensor_get(x, dst.data, 0, ggml_nbytes(x));
}

//
// tensor operations

tensor slice(model_ref const& m, tensor x, slice_t s0, slice_t s1, slice_t s2, slice_t s3) {
    ASSERT(s0.step == 1 && "Slice step must be 1 for the begin dimension");

    auto ne = std::array{x->ne[0], x->ne[1], x->ne[2], x->ne[3]};
    auto nb = std::array{x->nb[0], x->nb[1], x->nb[2], x->nb[3]};
    auto slices = std::array{s0, s1, s2, s3};
    size_t offset = 0;

    for (int dim = 0; dim < 4; ++dim) {
        auto [begin, end, step] = slices[dim];
        end = end == slice_t::max ? x->ne[dim] : end;
        end = end < 0 ? x->ne[dim] + end : end;
        begin = begin < 0 ? x->ne[dim] + begin : begin;
        ASSERT(begin >= 0 && end <= x->ne[dim] && "Slice indices out of bounds");
        ASSERT(begin < end && "Begin index must be less than end index");

        ne[dim] = (end - begin + step - 1) / step;
        nb[dim] = x->nb[dim] * step;
        offset += begin * x->nb[dim];
    }
    return ggml_view_4d(m, x, ne[0], ne[1], ne[2], ne[3], nb[1], nb[2], nb[3], offset);
}

tensor concat(model_ref const& m, std::array<tensor, GGML_MAX_SRC> src, int dim) {
    int n = (int)std::count_if(src.begin(), src.end(), [](tensor t) { return t != nullptr; });
    if (m.flags & model_build_flag::concat_n) {
        return ggml_concat_n(m, src.data(), n, dim);
    } else {
        tensor x = src[0];
        for (int i = 1; i < n; ++i) {
            x = ggml_concat(m, x, src[i], dim);
        }
        return x;
    }
}

tensor interpolate(model_ref const& m, tensor x, i64x2 target, int32_t mode) {
    return ggml_upscale_ext(
        m, x, int(target[0]), int(target[1]), int(x->ne[2]), int(x->ne[3]), mode);
}

} // namespace visp