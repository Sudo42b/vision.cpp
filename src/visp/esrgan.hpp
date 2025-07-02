#pragma once

#include "ml.hpp"

namespace visp {

struct esrgan_params {
    int scale = 4;
    int n_blocks = 23;
};

esrgan_params esrgan_detect_params(model_ref);

tensor esrgan_upscale(model_ref, tensor image, esrgan_params const&);

namespace esrgan {

tensor upsample(model_ref m, tensor x);
tensor conv_block(model_ref m, tensor x);
tensor risidual_dense_block(model_ref m, tensor x);
tensor rrdb(model_ref m, tensor x);

} // namespace esrgan
} // namespace visp