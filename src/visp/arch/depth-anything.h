#pragma once

#include "visp/ml.h"
#include "visp/vision.h"

namespace visp::dpt {

tensor residual_conv(model_ref m, tensor x);
tensor feature_fusion(model_ref m, tensor x0, tensor x1, int64_t const* size = nullptr);
tensor head(model_ref m, span<tensor> features, int64_t patch_w, int64_t patch_h);

} // namespace visp::dpt
