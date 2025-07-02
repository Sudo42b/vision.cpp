#pragma once

#include "visp/ml.hpp"
#include "visp/util.hpp"

// Common neural network building blocks

namespace visp {

tensor linear(model_ref, tensor x);
tensor layer_norm(model_ref, tensor x, float eps = 1e-5f);

// The following 2D operations use channel-contiguous memory layout (CWHN)
inline namespace cwhn {

tensor conv_2d(model_ref, tensor x, int stride = 1, int pad = 0);
tensor conv_2d_depthwise(model_ref, tensor x, int stride = 1, int pad = 0);
tensor conv_2d_deform(
    model_ref, tensor x, tensor weight, tensor offset, tensor mask, int stride, int pad);
tensor conv_transpose_2d(model_ref m, tensor x, int stride);

tensor batch_norm_2d(model_ref, tensor x);

} // namespace cwhn

} // namespace visp
