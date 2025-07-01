#include "visp/image.hpp"
#include "image-impl.hpp"
#include "math.hpp"
#include "string.hpp"

#include <stb_image.h>
#include <stb_image_resize.h>
#include <stb_image_write.h>

#include <algorithm>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace visp {
using std::clamp;

//
// image view

int n_channels(image_format format) {
    switch (format) {
    case image_format::rgba:
    case image_format::bgra:
    case image_format::argb: return 4;
    case image_format::rgb: return 3;
    case image_format::alpha: return 1;
    default: ASSERT(false, "unknown image format"); return 0;
    }
}

image_view::image_view(i32x2 extent, image_format format, uint8_t const* data)
    : extent(extent), stride(extent[0] * n_channels(format)), format(format), data(data) {}

int n_channels(image_view const& img) {
    return n_channels(img.format);
}

int n_pixels(image_view const& img) {
    return img.extent[0] * img.extent[1];
}

size_t n_bytes(image_view const& img) {
    return size_t(n_pixels(img)) * n_channels(img.format);
}

//
// image data

image_data image_alloc(i32x2 extent, image_format format) {
    size_t size = extent[0] * extent[1] * n_channels(format);
    return image_data{extent, format, std::unique_ptr<uint8_t[]>(new uint8_t[size])};
}

image_data image_load(char const* filepath) {
    i32x2 extent = {0, 0};
    int channels = 0;
    uint8_t* pixels = stbi_load(filepath, &extent[0], &extent[1], &channels, 0);
    if (!pixels) {
        throw error("Failed to load image {}: {}", filepath, stbi_failure_reason());
    }
    return image_data(extent, image_format::rgba, std::unique_ptr<uint8_t[]>(pixels));
}

void image_save(image_view const& img, char const* filepath) {
    if (!(img.format == image_format::alpha || img.format == image_format::rgb ||
          img.format == image_format::rgba)) {
        throw error("Unsupported image format [{}]", int(img.format));
    }
    int comp = n_channels(img.format);
    if (!stbi_write_png(
            filepath, img.extent[0], img.extent[1], comp, img.data, img.extent[0] * comp)) {
        throw error("Failed to save image {}", filepath);
    }
}

//
// operations on uint8 images

image_data image_resize(image_view const& img, i32x2 target) {
    ASSERT(img.stride >= img.extent[0] * n_channels(img));

    image_data resized = image_alloc(target, img.format);
    int result = stbir_resize_uint8_generic(
        img.data, img.extent[0], img.extent[1], img.stride, resized.data.get(), target[0],
        target[1], 0, n_channels(img), STBIR_ALPHA_CHANNEL_NONE, /*output_stride*/ 0,
        STBIR_EDGE_CLAMP, STBIR_FILTER_DEFAULT, STBIR_COLORSPACE_SRGB, nullptr);

    if (result == 0) {
        throw error(
            "Failed to resize image {}x{} to {}x{}", img.extent[0], img.extent[1], target[0],
            target[1]);
    }
    return resized;
}

image_data image_resize_mask(image_view const& img, i32x2 target) {
    ASSERT(img.format == image_format::alpha);

    image_data resized = image_alloc(target, img.format);
    int result = stbir_resize_uint8_generic(
        img.data, img.extent[0], img.extent[1], img.stride, resized.data.get(), target[0],
        target[1], /*output_stride*/ 0, /*num_channels*/ 1, STBIR_ALPHA_CHANNEL_NONE, 0,
        STBIR_EDGE_CLAMP, STBIR_FILTER_BOX, STBIR_COLORSPACE_LINEAR, nullptr);

    if (result == 0) {
        throw error(
            "Failed to resize image {}x{} to {}x{}", img.extent[0], img.extent[1], target[0],
            target[1]);
    }
    return resized;
}

//
// pixel_lookup implementation

pixel_lookup::pixel_lookup(i32x2 extent, image_format format) {
    stride_c = n_channels(format);
    stride_x = extent[0] * stride_c;

    switch (format) {
    case image_format::bgra: channel_map = {2, 1, 0, 3}; break;
    case image_format::argb: channel_map = {1, 2, 3, 0}; break;
    case image_format::alpha: channel_map = {0, 0, 0, 0}; break;
    case image_format::rgb: channel_map = {0, 1, 2, 0}; break;
    default: channel_map = {0, 1, 2, 3}; break; // rgba
    }
}

pixel_lookup::pixel_lookup(image_view image) : pixel_lookup(image.extent, image.format) {}

//
// image conversion (uint8 <-> float32)

template <typename T>
void image_to_float(
    image_view const& img, image_span<T> dst, f32x4 offset, f32x4 scale, i32x2 tile_offset) {
    auto input = pixel_lookup(img);
    for (int y = 0; y < dst.extent[1]; ++y) {
        for (int x = 0; x < dst.extent[0]; ++x) {
            int x0 = std::min(x + tile_offset[0], img.extent[0] - 1);
            int y0 = std::min(y + tile_offset[1], img.extent[1] - 1);
            f32x4 v{
                float(input.get(img.data, x0, y0, 0)), float(input.get(img.data, x0, y0, 1)),
                float(input.get(img.data, x0, y0, 2)), float(input.get(img.data, x0, y0, 3))};
            v = (v / 255.f + offset) * scale;
            dst.set(x, y, v);
        }
    }
}

template void image_to_float(image_view const&, image_span<f32x4>, f32x4, f32x4, i32x2);
template void image_to_float(image_view const&, image_span<f32x3>, f32x4, f32x4, i32x2);
template void image_to_float(image_view const&, image_span<float>, f32x4, f32x4, i32x2);

void image_from_float(
    std::span<float const> src, std::span<uint8_t> dst, float scale, float offset) {

    ASSERT(src.size() == dst.size());
    for (size_t i = 0; i < src.size(); ++i) {
        float value = 255.0f * std::clamp(src[i] * scale + offset, 0.0f, 1.0f);
        dst[i] = uint8_t(value);
    }
}

//
// image algorithms

template <typename T>
void blur_impl(image_span<const T> src, image_span<T> dst, int radius) {
    i32x2 extent = src.extent;
    ASSERT(src.extent == dst.extent);
    ASSERT(radius > 0);
    ASSERT(radius <= extent[0] / 2 && radius <= extent[1] / 2);

    std::vector<T> temp(src.n_pixels());
    float weight = 1.0f / (2 * radius + 1);

    // Horizontal pass (src -> temp)
    for (int y = 0; y < extent[1]; ++y) {
        int row_offset = y * extent[0];
        T sum = float(radius) * src[row_offset];
        for (int x = 0; x <= radius; ++x) {
            sum = sum + src[row_offset + x];
        }

        temp[row_offset] = sum * weight;

        for (int x = 1; x < extent[0]; ++x) {
            int left_x = clamp(x - radius - 1, 0, extent[0] - 1);
            sum = sum - src[row_offset + left_x];

            int right_x = clamp(x + radius, 0, extent[0] - 1);
            sum = sum + src[row_offset + right_x];

            temp[row_offset + x] = sum * weight;
        }
    }

    // Vertical pass (temp -> dst)
    for (int x = 0; x < extent[0]; ++x) {
        T sum = float(radius) * temp[x];
        for (int y = 0; y <= radius; ++y) {
            sum = sum + temp[y * extent[0] + x];
        }

        dst[x] = sum * weight;

        for (int y = 1; y < extent[1]; ++y) {
            int top_y = clamp(y - radius - 1, 0, extent[1] - 1);
            sum = sum - temp[top_y * extent[0] + x];

            int bottom_y = clamp(y + radius, 0, extent[1] - 1);
            sum = sum + temp[bottom_y * extent[0] + x];

            dst[y * extent[0] + x] = sum * weight;
        }
    }
}

void image_blur(image_span<const float> src, image_span<float> dst, int radius) {
    blur_impl(src, dst, radius);
}
void image_blur(image_span<const f32x4> src, image_span<f32x4> dst, int radius) {
    blur_impl(src, dst, radius);
}

// Approximate Fast Foreground Colour Estimation
// https://ieeexplore.ieee.org/document/9506164
auto blur_fusion_foreground_estimator(
    image_span<const f32x4> img,
    image_span<const f32x4> fg,
    image_span<const f32x4> bg,
    image_span<const float> mask,
    int radius) {

    i32x2 extent = img.extent;
    ASSERT(fg.extent == extent && bg.extent == extent && mask.extent == extent);
    size_t n = img.n_pixels();

    auto per_pixel = [n](auto&& f) {
        for (size_t i = 0; i < n; ++i) {
            f(i);
        }
    };

    auto blurred_mask = image_alloc<float>(extent);
    image_blur(mask, blurred_mask, radius);

    auto fg_masked = image_alloc<f32x4>(extent);
    per_pixel([&](size_t i) { fg_masked[i] = fg[i] * mask[i]; });

    auto blurred_fg = image_alloc<f32x4>(extent);
    image_blur(fg_masked, blurred_fg, radius);
    per_pixel([&](size_t i) { blurred_fg[i] = blurred_fg[i] / (blurred_mask[i] + 1e-5f); });

    auto& bg_masked = fg_masked; // Reuse fg_masked for bg
    per_pixel([&](size_t i) { bg_masked[i] = bg[i] * (1.0f - mask[i]); });

    auto blurred_bg = image_alloc<f32x4>(extent);
    image_blur(bg_masked, blurred_bg, radius);
    per_pixel([&](size_t i) {
        blurred_bg[i] = blurred_bg[i] / ((1.0f - blurred_mask[i]) + 1e-5f);
        f32x4 f = blurred_fg[i] +
                  mask[i] * (img[i] - mask[i] * blurred_fg[i] - (1.0f - mask[i]) * blurred_bg[i]);
        blurred_fg[i] = clamp(f, 0.0f, 1.0f);
    });
    return std::pair{std::move(blurred_fg), std::move(blurred_bg)};
}

image_data_t<f32x4> image_estimate_foreground(
    image_span<const f32x4> img, image_span<const float> mask, int radius) {

    auto&& [fg, blur_bg] = blur_fusion_foreground_estimator(img, img, img, mask, radius);
    return blur_fusion_foreground_estimator(img, fg, blur_bg, mask, 3).first;
}

void image_alpha_composite(
    image_view const& fg, image_view const& bg, image_view const& mask, uint8_t* dst) {

    ASSERT(fg.extent == bg.extent && fg.extent == mask.extent);
    pixel_lookup a(fg);
    pixel_lookup b(bg);
    pixel_lookup alpha(mask);

    auto comp = [&](int x, int y, int c, float w0, float w1) {
        return uint8_t(a.get(fg.data, x, y, c) * w0 + b.get(bg.data, x, y, c) * w1);
    };

    for (int y = 0; y < fg.extent[1]; ++y) {
        for (int x = 0; x < fg.extent[0]; ++x) {
            int i = y * fg.extent[0] * 3 + x * 3;
            float w0 = alpha.get(mask.data, x, y, 0) / 255.0f;
            float w1 = 1.0f - w0;
            for (int c = 0; c < 3; ++c) {
                dst[i + c] = comp(x, y, c, w0, w1);
            }
        }
    }
}

float image_difference_rms(image_view const& img1, image_view const& img2) {
    ASSERT(img1.extent == img2.extent && img1.format == img2.format);

    float sum_sq_diff = 0.0f;
    size_t n = n_bytes(img1);
    for (size_t i = 0; i < n; ++i) {
        float p1 = float(img1.data[i]) / 255.0f;
        float p2 = float(img2.data[i]) / 255.0f;
        sum_sq_diff += sqr(p1 - p2);
    }
    return std::sqrt(sum_sq_diff / n);
}

template<typename T>
float image_difference_rms_impl(image_span<T const> img1, image_span<T const> img2) {
    ASSERT(img1.extent == img2.extent);

    float sum_sq_diff = 0.0f;
    size_t n = img1.n_pixels();
    for (size_t i = 0; i < n; ++i) {
        f32x4 diff = load_pixel(img1[i]) - load_pixel(img2[i]);
        sum_sq_diff += dot(diff, diff);
    }
    return std::sqrt(sum_sq_diff / n);
}

float image_difference_rms(image_span<float const> img1, image_span<float const> img2) {
    return image_difference_rms_impl(img1, img2);
}
float image_difference_rms(image_span<f32x3 const> img1, image_span<f32x3 const> img2) {
    return image_difference_rms_impl(img1, img2);
}
float image_difference_rms(image_span<f32x4 const> img1, image_span<f32x4 const> img2) {
    return image_difference_rms_impl(img1, img2);
}

//
// image tiling

tile_layout::tile_layout(i32x2 extent, int max_tile_size, int overlap, int align)
    : image_extent{extent[0], extent[1]}, overlap{overlap, overlap} {

    n_tiles = div_ceil(image_extent, max_tile_size);
    i32x2 img_extent_overlap = image_extent + (n_tiles - i32x2{1, 1}) * overlap;
    tile_size = div_ceil(img_extent_overlap, n_tiles);
    tile_size = div_ceil(tile_size, align) * align;
}

tile_layout tile_scale(tile_layout const& o, int scale) {
    tile_layout scaled;
    scaled.image_extent = o.image_extent * scale;
    scaled.overlap = o.overlap * scale;
    scaled.n_tiles = o.n_tiles;
    scaled.tile_size = o.tile_size * scale;
    return scaled;
}

i32x2 tile_layout::start(i32x2 coord, i32x2 pad) const {
    i32x2 offset = coord * (tile_size - overlap);
    return offset + i32x2{coord[0] == 0 ? 0 : pad[0], coord[1] == 0 ? 0 : pad[1]};
}

i32x2 tile_layout::end(i32x2 coord, i32x2 pad) const {
    i32x2 offset = start(coord) + tile_size;
    offset = offset -
             i32x2{
                 coord[0] == n_tiles[0] - 1 ? 0 : pad[0], coord[1] == n_tiles[1] - 1 ? 0 : pad[1]};
    return min(offset, image_extent);
}

i32x2 tile_layout::size(i32x2 coord) const {
    return end(coord) - start(coord);
}

int tile_layout::total() const {
    return n_tiles[0] * n_tiles[1];
}

i32x2 tile_layout::coord(int index) const {
    ASSERT(index >= 0 && index < total());
    return {index % n_tiles[0], index / n_tiles[0]};
}

void tile_merge(
    image_span<const f32x3> const& tile,
    image_span<f32x3>& dst,
    i32x2 tile_coord,
    tile_layout const& layout) {

    i32x2 beg = layout.start(tile_coord);
    i32x2 end = layout.end(tile_coord);
    i32x2 pad_beg = layout.start(tile_coord, layout.overlap);
    i32x2 pad_end = layout.end(tile_coord, layout.overlap);

    for (int y = beg[1]; y < end[1]; ++y) {
        for (int x = beg[0]; x < end[0]; ++x) {
            i32x2 idx = {x, y};

            float weight = 1.0f;
            i32x2 coverage = {0, 0};
            for (int i = 0; i < 2; ++i) {
                if (idx[i] < pad_beg[i]) {
                    weight *= float(layout.overlap[i] - (pad_beg[i] - idx[i]) + 1);
                    coverage[i] = layout.overlap[i];
                } else if (idx[i] >= pad_end[i]) {
                    weight *= float(layout.overlap[i] - (idx[i] - pad_end[i]));
                    coverage[i] = layout.overlap[i];
                }
            }
            float norm = float((coverage[0] + 1) * (coverage[1] + 1));
            float blend = weight > 0 ? weight / norm : 1.0f;

            dst.set(idx, dst.get(idx) + blend * tile.get(idx - beg));
        }
    }
}

} // namespace visp
