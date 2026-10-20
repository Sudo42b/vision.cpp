#pragma once
#include "arch_registry.h"

namespace visp {
// A single architecture for config-derived models. No model-specific translation unit.
tensor runtime_graph_forward(model_ref, tensor, model_file const&);
arch_task runtime_graph_task(model_file const&);
i64x4 runtime_graph_input_shape(model_file const&);
tensor_data_layout runtime_graph_weight_layout(model_file const&, tensor_data_layout preferred);
} // namespace visp
