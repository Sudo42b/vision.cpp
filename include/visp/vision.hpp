#pragma once

#include "visp/image.hpp"
#include "visp/ml.hpp"
#include "visp/util.hpp"

#include <array>
#include <span>

namespace visp {

//
// Mobile SAM - image segmentation with prompt (point or box)

struct sam_model;

struct box_2d {
    i32x2 top_left;
    i32x2 bottom_right;
};

// Loads a SAM model from GGUF file onto the backend device.
// * only supports MobileSAM (TinyViT) for now
VISP_API sam_model sam_load_model(char const* filepath, backend_device const&);

// Creates an image embedding from RGB input, required for subsequent `sam_compute` calls.
VISP_API void sam_encode(sam_model&, image_view image);

// Computes a segmentation mask (alpha image) for an object in the image.
// * takes either a point, ie. a pixel location with origin (0, 0) in the top left
// * or a bounding box which contains the object
VISP_API image_data sam_compute(sam_model&, i32x2 point);
VISP_API image_data sam_compute(sam_model&, box_2d box);

//

struct sam_params {
    int image_size = 1024;
    int mask_size = 256;
};

struct sam_prediction {
    tensor masks;
    tensor iou;
};

VISP_API image_data sam_process_input(image_view image, sam_params const&);
VISP_API f32x4 sam_process_point(i32x2 point, i32x2 image_extent, sam_params const&);
VISP_API f32x4 sam_process_box(box_2d box, i32x2 image_extent, sam_params const&);

VISP_API tensor sam_encode_image(model_ref, tensor image, sam_params const&);
VISP_API tensor sam_encode_points(model_ref, tensor coords);
VISP_API tensor sam_encode_box(model_ref, tensor coords);

VISP_API sam_prediction sam_predict_mask(model_ref m, tensor image_embed, tensor prompt_embed);

VISP_API image_data sam_process_mask(
    std::span<float const> mask_data, int mask_index, i32x2 target_extent, sam_params const&);

//
// BiRefNet - dichotomous image segmentation (background removal)

struct birefnet_model;

// Loads a BiRefNet model from GGUF file onto the backend device.
// * supports BiRefNet, BiRefNet_lite, BiRefNet_Matting variants at 1024px resolution
VISP_API birefnet_model birefnet_load_model(char const* filepath, backend_device const&);

// Takes RGB input and computes an alpha mask with foreground as 1.0 and background as 0.0.
VISP_API image_data birefnet_compute(birefnet_model&, image_view image);

//

struct birefnet_params {
    int image_size = 1024;
    swin_params encoder;
};

using birefnet_buffers = std::array<tensor_data, swin_params::n_layers + 2>;

VISP_API birefnet_params birefnet_detect_params(model_ref);
VISP_API birefnet_buffers birefnet_precompute(model_ref, birefnet_params const&);

VISP_API image_data birefnet_process_input(image_view, birefnet_params const&);
VISP_API image_data birefnet_process_output(
    std::span<float const> output_data, i32x2 target_extent, birefnet_params const&);

VISP_API tensor birefnet_predict(model_ref, tensor image, birefnet_params const&);

//
// MI-GAN - image inpainting

struct migan_model;

// Loads a MI-GAN model from GGUF file onto the backend device.
// * supports variants at 256px or 512px resolution
VISP_API migan_model migan_load_model(char const* filepath, backend_device const&);

// Fills pixels in the input image where the mask is 1.0 with new content.
VISP_API image_data migan_compute(migan_model&, image_view image, image_view mask);

//

struct migan_params {
    int resolution = 256;
    bool invert_mask = false;
};

VISP_API migan_params migan_detect_params(model_ref m);

VISP_API image_data migan_process_input(image_view image, image_view mask, migan_params const&);
VISP_API image_data migan_process_output(
    std::span<float const> data, i32x2 extent, migan_params const&);

VISP_API tensor migan_generate(model_ref, tensor image, migan_params const&);

//
// ESRGAN - image super-resolution

struct esrgan_model;

// Loads an ESRGAN model from GGUF file onto the backend device.
// * supports ESRGAN, RealESRGAN variants with flexible scale and number of blocks
// * currently does not spport RealESRGAN+ (plus) models or those which use pixel shuffle
VISP_API esrgan_model esrgan_load_model(char const* filepath, backend_device const&);

// Upscales the input image by the model's scale factor. Uses tiling for large inputs.
VISP_API image_data esrgan_compute(esrgan_model&, image_view image);

//

struct esrgan_params {
    int scale = 4;
    int n_blocks = 23;
};

VISP_API esrgan_params esrgan_detect_params(model_ref);

VISP_API tensor esrgan_generate(model_ref, tensor image, esrgan_params const&);

//
// Implementation

// internal
struct sam_model {
    backend_device const* backend = nullptr;
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
    backend_device const* backend = nullptr;
    model_weights weights;
    birefnet_params params;

    compute_graph graph;
    tensor input = nullptr;
    tensor output = nullptr;
};

// internal
struct migan_model {
    backend_device const* backend = nullptr;
    model_weights weights;
    migan_params params;

    compute_graph graph;
    tensor input = nullptr;
    tensor output = nullptr;
};

// internal
struct esrgan_model {
    backend_device const* backend = nullptr;
    model_weights weights;
    esrgan_params params;

    compute_graph graph;
    i32x2 tile_size{};
    tensor input = nullptr;
    tensor output = nullptr;
};

} // namespace visp