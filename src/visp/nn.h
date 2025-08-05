#pragma once

#include "visp/ml.h"
#include "visp/util.h"

// Common neural network building blocks

namespace visp {

tensor linear(model_ref, tensor x);
tensor layer_norm(model_ref, tensor x, float eps = 1e-5f);

// Permute between CWHN and WHCN tensor dimension ordering. Does not rewrite tensor data.
tensor permute_cwhn_to_whcn(model_ref m, tensor x);
tensor permute_whcn_to_cwhn(model_ref m, tensor x);

// Input must be CWHN layout, output is WHCN or CWHN (unchanged) depending on model flags.
tensor to_contiguous_2d(model_ref m, tensor x);
// Reverts `to_contiguous_2d`, input must be as configured in `m`, output is CWHN layout.
tensor to_contiguous_channels(model_ref m, tensor x);

// Convolution

tensor conv_2d(model_ref, tensor x, int stride = 1, int pad = 0);
tensor conv_2d_depthwise(model_ref, tensor x, int stride = 1, int pad = 0);
tensor conv_2d_deform(
    model_ref, tensor x, tensor weight, tensor offset, tensor mask, int stride, int pad);
tensor conv_transpose_2d(model_ref m, tensor x, int stride);

// Misc 2D

tensor batch_norm_2d(model_ref, tensor x);

} // namespace visp
