#pragma once
#include "visp/mmlab.h"
#include "visp/nn.h"
#include "nlohmann/json.hpp"
#include <functional>

namespace visp::mmlab {
using Json = nlohmann::json;
using Tensors = std::vector<tensor>;
using Builder = Tensors (*)(model_ref, Json const&, Tensors const&);
Json read_config(model_file const&);
arch_task task_from_metadata(Json const&, i64x4 shape);
Tensors run(model_ref, Json const&, Tensors const&);
tensor one(model_ref, Json const&, tensor);
Json const& child(Json const&, std::string const&);
tensor layer(model_ref, Json const&, std::string const&, tensor);
model_ref weights(model_ref, Json const&);
tensor parameter(model_ref, Json const&, char const*);
tensor cat(model_ref, Tensors const&, int axis = 2);
tensor resize(model_ref, tensor, int64_t w, int64_t h, bool bilinear, bool align = false);
tensor adaptive_pool(model_ref, tensor, int w, int h, bool maximum = false);
tensor sample_tokens(model_ref, tensor, int64_t width, int64_t height, tensor x, tensor y);
Tensors modules(model_ref, Json const&, Tensors const&);
Tensors backbones(model_ref, Json const&, Tensors const&);
Tensors heads(model_ref, Json const&, Tensors const&);
Tensors transformers(model_ref, Json const&, Tensors const&);
Tensors models(model_ref, Json const&, Tensors const&);
Tensors build_architecture(model_ref, std::string const&, Json const&, Tensors const&);
tensor config_forward(model_ref, tensor, model_file const&);
arch_task config_task(model_file const&);
i64x4 config_input_shape(model_file const&);
} // namespace visp::mmlab

namespace visp { void register_mmlab_config(); }
