#include "migan.hpp"
#include "image-impl.hpp"
#include "math.hpp"
#include "nn.hpp"
#include "string.hpp"

#include <array>
#include <cmath>
#include <optional>

namespace visp {
namespace migan {

constexpr float sqrt2 = 1.4142135623f;

tensor lrelu_agc(model_ref m, tensor x, float alpha, float gain, float clamp) {
    x = ggml_leaky_relu(m, x, alpha, true);
    if (gain != 1) {
        x = ggml_scale_inplace(m, x, gain);
    }
    if (clamp != 0) {
        x = ggml_clamp(m, x, -clamp, clamp);
    }
    return named(m, x);
}

tensor downsample_2d(model_ref m, tensor x) {
    return conv_2d_depthwise(m["filter"], x, 2, 1);
}

tensor upsample_2d(model_ref m, tensor x) {
    tensor filter_const = m.weights("filter_const");
    filter_const = ggml_reshape_4d(m, filter_const, 1, filter_const->ne[0], filter_const->ne[1], 1);

    auto [c, w, h, b] = nelements(x);
    x = ggml_upscale_ext(m, x, int(c), int(w * 2), int(h * 2), int(b), GGML_SCALE_MODE_NEAREST);
    x = ggml_mul_inplace(m, x, filter_const);
    x = conv_2d_depthwise(m["filter"], x, 1, 2); // 4x4 filter
    x = slice(m, x, {}, {0, -1}, {0, -1}, {});   // remove padding from right and bottom
    x = ggml_cont(m, x); // required by subsequent ggml_scale for some reason
    return named(m, x);
}

tensor separable_conv_2d(model_ref m, tensor x, flags<conv> flags) {
    int pad = int(m["conv1"].weights("weight")->ne[2] / 2);
    x = conv_2d_depthwise(m["conv1"], x, 1, pad);
    if (flags & conv::activation) {
        x = lrelu_agc(m, x, 0.2f, sqrt2, 256);
    }

    if (flags & conv::downsample) {
        x = downsample_2d(m["downsample"], x);
    }
    x = conv_2d(m["conv2"], x);
    if (flags & conv::upsample) {
        x = upsample_2d(m["upsample"], x);
    }

    if (flags & conv::noise) {
        tensor noise = m.weights("noise_const");
        noise = ggml_mul_inplace(m, noise, m.weights("noise_strength"));
        noise = ggml_reshape_4d(m, noise, 1, noise->ne[0], noise->ne[1], 1);
        x = ggml_add_inplace(m, x, noise);
    }
    if (flags & conv::activation) {
        x = lrelu_agc(m, x, 0.2f, sqrt2, 256);
    }
    return named(m, x);
}

tensor from_rgb(model_ref m, tensor x) {
    x = conv_2d(m["fromrgb"], x);
    x = lrelu_agc(m, x, 0.2f, sqrt2, 256);
    return named(m, x);
}

std::pair<tensor, tensor> encoder_block(model_ref m, tensor x, conv flag) {
    tensor feat = separable_conv_2d(m["conv1"], x, conv::activation);
    x = separable_conv_2d(m["conv2"], feat, conv::activation | flag);
    return {x, feat};
}

using Features = std::array<tensor, 9>;

std::pair<tensor, Features> encode(model_ref m, tensor x, int res) {
    ASSERT(res == int(x->ne[1]));
    int n = log2(res) - 1;
    ASSERT((1 << (n + 1)) == res);

    x = from_rgb(m[format<tensor_name>("b{}", res)], x);
    Features feats{};
    for (int i = 0; i < n - 1; ++i) {
        model_ref block = m[format<tensor_name>("b{}", res >> i)];
        std::tie(x, feats[i]) = encoder_block(block, x, conv::downsample);
    }
    std::tie(x, feats[n - 1]) = encoder_block(m["b4"], x);
    return {x, feats};
}

std::pair<tensor, tensor> synthesis_block(
    model_ref m, tensor x, tensor feat, tensor img, conv up_flag, conv noise_flag) {
    x = separable_conv_2d(m["conv1"], x, conv::activation | noise_flag | up_flag);
    x = ggml_add_inplace(m, x, feat);
    x = separable_conv_2d(m["conv2"], x, conv::activation | noise_flag);

    if (img) {
        img = upsample_2d(m["upsample"], img);
    }
    tensor y = conv_2d(m["torgb"], x);
    img = img ? ggml_add_inplace(m, img, y) : y;

    return {x, img};
}

tensor synthesis(model_ref m, tensor x_in, Features feats, int res) {
    int n = log2(res) - 1;
    ASSERT((1 << (n + 1)) == res);

    auto [x, img] = synthesis_block(m["b4"], x_in, feats[n - 1], nullptr);
    for (int i = n - 2; i >= 0; --i) {
        model_ref block = m[format<tensor_name>("b{}", res >> i)];
        std::tie(x, img) = synthesis_block(block, x, feats[i], img, conv::upsample, conv::noise);
    }
    return img;
}

} // namespace migan

tensor generate(model_ref m, tensor image, migan_params const& p) {
    auto [x, feats] = migan::encode(m["encoder"], image, p.resolution);
    tensor result = migan::synthesis(m["synthesis"], x, feats, p.resolution);
    return mark_output(m, result, "output");
}

migan_params migan_params::detect(model_ref m) {
    if (m.find("encoder.b512.fromrgb.weight") != nullptr) {
        return migan_params{512};
    } else if (m.find("encoder.b256.fromrgb.weight") != nullptr) {
        return migan_params{256};
    } else {
        throw std::runtime_error("Failed to detect model parameters");
    }
}

image_data_t<f32x4> migan_preprocess(image_view image, image_view mask, migan_params const& p) {
    i32x2 res = {p.resolution, p.resolution};
    std::optional<image_data> resized_image;
    if (image.extent != res) {
        resized_image = image_resize(image, res);
        image = image_view(*resized_image);
    }
    std::optional<image_data> resized_mask;
    if (mask.extent != res) {
        resized_mask = image_resize(mask, res);
        mask = image_view(*resized_mask);
    }
    pixel_lookup rgb(image);
    pixel_lookup alpha(mask);
    const float scale = 2.0f / 255.0f;
    const uint8_t no_fill = p.invert_mask ? 0 : 255;
    image_data_t<f32x4> result_image = image_alloc<f32x4>(res);
    image_span<f32x4> result = result_image.span();

    for (int y = 0; y < res[1]; ++y) {
        for (int x = 0; x < res[0]; ++x) {
            float a = alpha.get(mask.data, x, y, 0) == no_fill ? 1.0f : 0.0f;
            result.set(
                x, y,
                {a - 0.5f, //
                 a * (rgb.get(image.data, x, y, 0) * scale - 1.0f),
                 a * (rgb.get(image.data, x, y, 1) * scale - 1.0f),
                 a * (rgb.get(image.data, x, y, 2) * scale - 1.0f)});
        }
    }
    return result_image;
}

image_data migan_postprocess(std::span<float> data, i32x2 extent, migan_params const& p) {
    int res = p.resolution;
    auto image = image_alloc(i32x2{res, res}, image_format::rgb);
    image_from_float(data, std::span(image.data.get(), n_bytes(image)), 0.5f, 0.5f);
    if (extent[0] != res || extent[1] != res) {
        return image_resize(image, extent);
    }
    return image;
}

} // namespace visp