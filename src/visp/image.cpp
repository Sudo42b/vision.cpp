#include "visp/image.hpp"
#include "image-impl.hpp"
#include "util/math.hpp"
#include "util/string.hpp"

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

i32x4 get_channel_map(image_format format) {
    switch (format) {
        case image_format::bgra: return {2, 1, 0, 3};
        case image_format::argb: return {1, 2, 3, 0};
        case image_format::alpha: return {0, 0, 0, 0};
        case image_format::rgb: return {0, 1, 2, 0};
        default: return {0, 1, 2, 3}; // rgba
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
// image data (uint8)

image_data image_alloc(i32x2 extent, image_format format) {
    size_t size = extent[0] * extent[1] * n_channels(format);
    return image_data{extent, format, std::unique_ptr<uint8_t[]>(new uint8_t[size])};
}

image_format image_format_from_channels(int n_channels) {
    switch (n_channels) {
        case 1: return image_format::alpha;
        case 3: return image_format::rgb;
        case 4: return image_format::rgba;
        default: ASSERT(false, "Invalid number of channels");
    }
    return image_format::rgba;
}

image_data image_load(char const* filepath) {
    i32x2 extent = {0, 0};
    int channels = 0;
    uint8_t* pixels = stbi_load(filepath, &extent[0], &extent[1], &channels, 0);
    if (!pixels) {
        throw error("Failed to load image {}: {}", filepath, stbi_failure_reason());
    }
    image_format format = image_format_from_channels(channels);
    return image_data(extent, format, std::unique_ptr<uint8_t[]>(pixels));
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
        target[1], /*output stride*/ 0, n_channels(img), STBIR_ALPHA_CHANNEL_NONE, /*flags*/ 0,
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
// image span

image_span::image_span(i32x2 extent, int n_channels, float* data)
    : extent(extent), n_channels(n_channels), data(data) {}

span<float> image_span::elements() const {
    return span(data, extent[0] * extent[1] * n_channels);
}

image_cspan::image_cspan(i32x2 extent, int n_channels, float const* data)
    : extent(extent), n_channels(n_channels), data(data) {}

image_cspan::image_cspan(image_span const& other)
    : image_cspan(other.extent, other.n_channels, other.data) {}

span<float const> image_cspan::elements() const {
    return span(data, extent[0] * extent[1] * n_channels);
}

int n_pixels(image_cspan const& img) {
    return img.extent[0] * img.extent[1];
}

size_t n_bytes(image_cspan const& img) {
    return size_t(n_pixels(img)) * img.n_channels * sizeof(float);
}

//
// image data (float32)

image_data_f32 image_alloc_f32(i32x2 extent, int n_channels) {
    size_t size = size_t(extent[0]) * extent[1] * n_channels;
    float* ptr = new float[size]; // TODO should be 16-byte aligned
    return image_data_f32{extent, n_channels, std::unique_ptr<float[]>(ptr)};
}

//
// image conversion (uint8 <-> float32)

template <typename Src, typename Dst>
void convert(
    image_source<Src> src, image_target<Dst> dst, f32x4 offset, f32x4 scale, i32x2 tile_offset) {

    for (int y = 0; y < dst.extent[1]; ++y) {
        for (int x = 0; x < dst.extent[0]; ++x) {
            i32x2 c = {x, y};
            c = min(c + tile_offset, dst.extent - i32x2{1, 1});
            dst.store(c, (src.load(c) + offset) * scale);
        }
    }
}

void image_u8_to_f32(
    image_view const& src, image_span const& dst, f32x4 offset, f32x4 scale, i32x2 tile_offset) {

    switch (dst.n_channels) {
        case 1: convert<uint8_t, float>(src, dst, offset, scale, tile_offset); break;
        case 3:
            switch (n_channels(src)) {
                case 3: convert<u8x3, f32x3>(src, dst, offset, scale, tile_offset); break;
                case 4: convert<u8x4, f32x3>(src, dst, offset, scale, tile_offset); break;
            }
            break;
        case 4:
            switch (n_channels(src)) {
                case 3: convert<u8x3, f32x4>(src, dst, offset, scale, tile_offset); break;
                case 4: convert<u8x4, f32x4>(src, dst, offset, scale, tile_offset); break;
            }
            break;
        default: ASSERT(false, "Number of channels in source and destination or not compatible");
    }
}

void image_f32_to_u8(
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
void blur_impl(image_cspan src_img, image_span dst_img, int radius) {
    i32x2 extent = src_img.extent;
    ASSERT(src_img.extent == dst_img.extent);
    ASSERT(radius > 0);
    ASSERT(radius <= extent[0] / 2 && radius <= extent[1] / 2);

    T const* src = reinterpret_cast<T const*>(src_img.data);
    T* dst = reinterpret_cast<T*>(dst_img.data);
    std::vector<T> temp(n_pixels(src_img));
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

void image_blur(image_cspan src, image_span dst, int radius) {
    ASSERT(src.n_channels == dst.n_channels);
    ASSERT(src.n_channels == 1 || src.n_channels == 4);
    ASSERT(src.extent == dst.extent);
    ASSERT(radius > 0);

    if (src.n_channels == 1) {
        blur_impl<float>(src, dst, radius);
    } else if (src.n_channels == 4) {
        blur_impl<f32x4>(src, dst, radius);
    }
}

// Approximate Fast Foreground Colour Estimation
// https://ieeexplore.ieee.org/document/9506164
auto blur_fusion_foreground_estimator(
    image_cspan img_in, image_cspan fg_in, image_cspan bg_in, image_cspan mask_in, int radius) {

    i32x2 extent = img_in.extent;
    size_t n = n_pixels(img_in);
    ASSERT(fg_in.extent == extent && bg_in.extent == extent && mask_in.extent == extent);
    image_source<f32x4> img(img_in);
    image_source<f32x4> fg(fg_in);
    image_source<f32x4> bg(bg_in);
    image_source<float> mask(mask_in);

    auto blurred_mask_data = image_alloc_f32(extent, 1);
    auto blurred_mask = image_target<float>(blurred_mask_data);
    image_blur(mask_in, blurred_mask_data, radius);

    auto fg_masked_data = image_alloc_f32(extent, 4);
    auto fg_masked = image_target<f32x4>(fg_masked_data);
    for (size_t i = 0; i < n; ++i) {
        fg_masked[i] = fg[i] * mask[i];
    }

    auto blurred_fg_data = image_alloc_f32(extent, 4);
    auto blurred_fg = image_target<f32x4>(blurred_fg_data);
    image_blur(fg_masked_data, blurred_fg_data, radius);
    for (size_t i = 0; i < n; ++i) {
        blurred_fg[i] = blurred_fg[i] / (blurred_mask[i] + 1e-5f);
    }

    auto& bg_masked_data = fg_masked_data; // Reuse fg_masked memory for bg
    auto& bg_masked = fg_masked;
    for (size_t i = 0; i < n; ++i) {
        bg_masked[i] = bg[i] * (1.0f - mask[i]);
    }

    auto blurred_bg_data = image_alloc_f32(extent, 4);
    auto blurred_bg = image_target<f32x4>(blurred_bg_data);
    image_blur(bg_masked_data, blurred_bg_data, radius);
    for (size_t i = 0; i < n; ++i) {
        blurred_bg[i] = blurred_bg[i] / ((1.0f - blurred_mask[i]) + 1e-5f);
        f32x4 f = blurred_fg[i] +
            mask[i] * (img[i] - mask[i] * blurred_fg[i] - (1.0f - mask[i]) * blurred_bg[i]);
        f[3] = mask[i];
        blurred_fg[i] = clamp(f, 0.0f, 1.0f);
    }
    return std::pair{std::move(blurred_fg_data), std::move(blurred_bg_data)};
}

image_data_f32 image_estimate_foreground(image_cspan img, image_cspan mask, int radius) {
    ASSERT(img.extent == mask.extent);
    ASSERT(img.n_channels == 4 && mask.n_channels == 1);

    auto&& [fg, blur_bg] = blur_fusion_foreground_estimator(img, img, img, mask, radius);
    return blur_fusion_foreground_estimator(img, fg, blur_bg, mask, 3).first;
}

template <typename FG, typename BG>
void alpha_composite(
    image_source<FG> fg, image_source<BG> bg, image_source<uint8_t> alpha, image_target<u8x4> out) {

    for (int y = 0; y < fg.extent[1]; ++y) {
        for (int x = 0; x < fg.extent[0]; ++x) {
            i32x2 c = {x, y};
            float w = alpha.load(c)[0];
            f32x4 v = w * fg.load(c) + (1.0f - w) * bg.load(c);
            v[3] = 1.0f;
            out.store(c, v);
        }
    }
}

void image_alpha_composite(
    image_view const& fg, image_view const& bg, image_view const& mask, uint8_t* dst) {

    ASSERT(fg.extent == bg.extent && fg.extent == mask.extent);
    image_target<u8x4> out(fg.extent, (u8x4*)dst, fg.extent[0]);

    switch (n_channels(bg.format)) {
        case 3: alpha_composite<u8x4, u8x3>(fg, bg, mask, out); break;
        case 4: alpha_composite<u8x4, u8x4>(fg, bg, mask, out); break;
        default: ASSERT(false, "Unsupported number of channels in background image"); return;
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

template <typename T>
float image_difference_rms_impl(image_cspan img1, image_cspan img2) {
    image_source<T> a(img1);
    image_source<T> b(img2);

    float sum_sq_diff = 0.0f;
    size_t n = n_pixels(img1);
    for (size_t i = 0; i < n; ++i) {
        f32x4 diff = a.load(i) - b.load(i);
        sum_sq_diff += dot(diff, diff);
    }
    return std::sqrt(sum_sq_diff / n);
}

float image_difference_rms(image_cspan const& img1, image_cspan const& img2) {
    ASSERT(img1.extent == img2.extent);
    ASSERT(img1.n_channels == img2.n_channels);

    switch (img1.n_channels) {
        case 1: return image_difference_rms_impl<float>(img1, img2);
        case 3: return image_difference_rms_impl<f32x3>(img1, img2);
        case 4: return image_difference_rms_impl<f32x4>(img1, img2);
        default: ASSERT(false, "Invalid number of channels"); return 0.0f;
    }
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
        i32x2{coord[0] == n_tiles[0] - 1 ? 0 : pad[0], coord[1] == n_tiles[1] - 1 ? 0 : pad[1]};
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
    image_cspan const& tile_img,
    image_span const& dst_img,
    i32x2 tile_coord,
    tile_layout const& layout) {

    image_source<f32x3> tile(tile_img);
    image_target<f32x3> dst(dst_img);

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

            dst.store(idx, dst.load(idx) + blend * tile.load(idx - beg));
        }
    }
}

} // namespace visp
