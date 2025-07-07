#pragma once

#include "visp/ml.hpp"
#include "visp/util.hpp"

namespace visp {

//
// Mobile SAM - image segmentation with prompt (point or box)

struct image_rect {
    i32x2 top_left;
    i32x2 bottom_right;
};

struct sam_model;

sam_model sam_load_model(char const* filepath, backend const&);

void sam_encode(sam_model&, image_view image, backend const&);

image_data sam_compute(sam_model&, i32x2 point, backend const&);
image_data sam_compute(sam_model&, image_rect box, backend const&);

//

struct sam_params {
    int image_size = 1024;
    int mask_size = 256;
};

struct sam_prediction {
    tensor masks;
    tensor iou;
};

image_data sam_process_input(image_view image, sam_params const&);
f32x4 sam_process_point(i32x2 point, i32x2 image_extent, sam_params const&);
f32x4 sam_process_box(image_rect box, i32x2 image_extent, sam_params const&);

tensor sam_encode_image(model_ref, tensor image, sam_params const&);
tensor sam_encode_points(model_ref, tensor coords);
tensor sam_encode_box(model_ref, tensor coords);

sam_prediction sam_predict(model_ref m, tensor image_embed, tensor prompt_embed);

image_data sam_process_mask(
    std::span<float const> mask_data, int mask_index, i32x2 target_extent, sam_params const&);

//
// BiRefNet - dichomous image segmentation (background removal)

struct birefnet_model;

birefnet_model birefnet_load_model(char const* filepath, backend const&);

image_data birefnet_compute(birefnet_model&, image_view image, backend const&);

//

struct birefnet_params {
    int image_size = 1024;
    swin_params encoder;
};

using birefnet_buffers = std::array<tensor_data, swin_params::n_layers + 2>;

birefnet_params birefnet_detect_params(model_ref);
birefnet_buffers birefnet_precompute(model_ref, birefnet_params const&);

image_data birefnet_process_input(image_view, birefnet_params const&);
image_data birefnet_process_output(
    std::span<float const> output_data, i32x2 target_extent, birefnet_params const&);

tensor birefnet_predict(model_ref, tensor image, birefnet_params const&);

//
// MI-GAN - image inpainting

struct migan_model;

migan_model migan_load_model(char const* filepath, backend const&);

image_data migan_compute(migan_model&, image_view image, image_view mask, backend const&);

//

struct migan_params {
    int resolution = 256;
    bool invert_mask = false;
};

migan_params migan_detect_params(model_ref m);

image_data migan_process_input(image_view image, image_view mask, migan_params const&);
image_data migan_process_output(std::span<float const> data, i32x2 extent, migan_params const&);

tensor migan_generate(model_ref, tensor image, migan_params const&);

//
// ESRGAN - image super-resolution

struct esrgan_model;

esrgan_model esrgan_load_model(char const* filepath, backend const&);

image_data esrgan_compute(esrgan_model&, image_view image, backend const&);

//

struct esrgan_params {
    int scale = 4;
    int n_blocks = 23;
};

esrgan_params esrgan_detect_params(model_ref);

tensor esrgan_generate(model_ref, tensor image, esrgan_params const&);

//
// Implementation

// internal
struct sam_model {
    model_weights weights;
    sam_params params;

    compute_graph encoder;
    i32x2 image_extent{};
    tensor input_image = nullptr;
    tensor output_embed = nullptr;

    compute_graph decoder;
    tensor input_embed = nullptr;
    tensor input_prompt = nullptr;
    sam_prediction output = {};
    bool is_point_prompt = true;
};

// internal
struct birefnet_model {
    model_weights weights;
    birefnet_params params;

    compute_graph graph;
    tensor input = nullptr;
    tensor output = nullptr;
};

// internal
struct migan_model {
    model_weights weights;
    migan_params params;

    compute_graph graph;
    tensor input = nullptr;
    tensor output = nullptr;
};

// internal
struct esrgan_model {
    model_weights weights;
    esrgan_params params;

    compute_graph graph;
    i32x2 tile_size{};
    tensor input = nullptr;
    tensor output = nullptr;
};

} // namespace visp