#pragma once
#include "visp/arch_registry.h"
namespace visp::mmlab {
tensor config_forward(model_ref, tensor, model_file const&);
arch_task config_task(model_file const&);
i64x4 config_input_shape(model_file const&);
}
namespace visp { void register_mmlab_config(); }
