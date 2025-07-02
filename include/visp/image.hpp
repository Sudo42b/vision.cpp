#pragma once

#include "visp/util.hpp"

#include <memory>
#include <span>

namespace visp {
using std::span;

//
// Image channel formats

enum class image_format {
    rgba,
    bgra,
    argb,
    rgb,
    alpha // single channel mask
};

int n_channels(image_format format);

//
// Image view - read-only, non-owning reference to uint8 image data

struct image_view {
    i32x2 extent{}; // width, height
    int stride = 0; // row stride
    image_format format = image_format::rgba;
    uint8_t const* data = nullptr;

    image_view() = default;
    image_view(i32x2 extent, image_format format, uint8_t const* data);
};

int n_channels(image_view const& img);
int n_pixels(image_view const& img);
size_t n_bytes(image_view const& img);

//
// Image data - storage for uint8 image data
// Can be used everywhere a view is expected

struct image_data {
    i32x2 extent{};
    image_format format = image_format::rgba;
    std::unique_ptr<uint8_t[]> data;

    image_view view() const { return image_view{extent, format, data.get()}; }
    operator image_view() const { return view(); }
};

// Allocate image data. Pixels are not initialized.
image_data image_alloc(i32x2 extent, image_format format);

// Load image from file (PNG, JPEG, etc.)
image_data image_load(char const* filepath);

// Save image to file (PNG, JPEG, etc.)
void image_save(image_view const& img, char const* filepath);

// Resize image to target size
image_data image_resize(image_view const&, i32x2 target);

// Resize mask to target size (linear interpolation)
image_data image_resize_mask(image_view const&, i32x2 target, uint8_t* output);

//
// Image span - reference to float32 image data

struct image_span {
    i32x2 extent = {};
    int n_channels = 0;
    float* data = nullptr;

    image_span() = default;

    image_span(i32x2 extent, int n_channels, float* data);

    image_span(i32x2 extent, span<float> data) : image_span(extent, 1, data.data()) {}
    image_span(i32x2 extent, span<f32x3> data) : image_span(extent, 3, &data[0][0]) {}
    image_span(i32x2 extent, span<f32x4> data) : image_span(extent, 4, &data[0][0]) {}

    span<float> elements() const;
};

struct image_cspan {
    i32x2 extent = {};
    int n_channels = 0;
    float const* data = nullptr;

    image_cspan() = default;

    image_cspan(i32x2 extent, int n_channels, float const* data);

    image_cspan(i32x2 extent, span<float const> data) : image_cspan(extent, 1, data.data()) {}
    image_cspan(i32x2 extent, span<f32x3 const> data) : image_cspan(extent, 3, &data[0][0]) {}
    image_cspan(i32x2 extent, span<f32x4 const> data) : image_cspan(extent, 4, &data[0][0]) {}

    image_cspan(image_span const& other);

    span<float const> elements() const;
};

int n_pixels(image_cspan const& img);
size_t n_bytes(image_cspan const& img);

//
// Image data for float32 images
// Can be used everywhere an image_span is expected

struct image_data_f32 {
    i32x2 extent = {};
    int n_channels = 0;
    std::unique_ptr<float[]> data;

    image_span as_span() { return {extent, n_channels, data.get()}; }
    image_cspan as_span() const { return {extent, n_channels, data.get()}; }

    operator image_span() { return as_span(); }
    operator image_cspan() const { return as_span(); }
};

// Allocate image data for float32 image. Memory is not initialized.
image_data_f32 image_alloc_f32(i32x2 extent, int n_channels);

//
// Image algorithms

// Convert uint8 image to float32 image
// * `src` can have any image format, output is converted to format of `dst`
// * applies computation `dst = (src + offset) * scale` to each pixel
// * starts reading `src` at `tile_offset`
// * always writes all of `dst` -- if `src` is smaller the output is padded (clamp to edge)
void image_u8_to_f32(
    image_view const& src,
    image_span const& dst,
    f32x4 offset = f32x4{0.f, 0.f, 0.f, 0.f},
    f32x4 scale = f32x4{1.f, 1.f, 1.f, 1.f},
    i32x2 tile_offset = {0, 0});

// Convert float32 image to uint8 image
// * applies computation `dst = src * scale + offset` to every channel/pixel
void image_f32_to_u8(span<float const> src, span<uint8_t> dst, float scale = 1, float offset = 0);

// Box filter with kernel size `2*radius + 1`
void image_blur(image_cspan src, image_span dst, int radius);

// Try to separate foreground and background contribution from pixels at the mask border
image_data_f32 image_estimate_foreground(image_cspan img, image_cspan mask, int radius = 30);

// Composite foreground and background images using alpha mask: `dst = fg * alpha + bg * (1-alpha)`
void image_alpha_composite(
    image_view const& fg, image_view const& bg, image_view const& mask, uint8_t* dst);

// Compute root-mean-square difference between two images
float image_difference_rms(image_view const&, image_view const&);
float image_difference_rms(image_cspan const&, image_cspan const&);

//
// Image tiling - helpers for processing large images in tiles

struct tile_layout {
    i32x2 image_extent;
    i32x2 overlap;
    i32x2 n_tiles;
    i32x2 tile_size;

    tile_layout() = default;

    tile_layout(i32x2 extent, int max_tile_size, int overlap, int align = 16);

    i32x2 start(i32x2 coord, i32x2 pad = {0, 0}) const;
    i32x2 end(i32x2 coord, i32x2 pad = {0, 0}) const;
    i32x2 size(i32x2 coord) const;

    int total() const;            // flat number of tiles
    i32x2 coord(int index) const; // flat index -> tile index
};

// Returns layout with same number of tiles within a larger image
tile_layout tile_scale(tile_layout const&, int scale);

void tile_merge(
    image_cspan const& tile, image_span const& dst, i32x2 tile_coord, tile_layout const& layout);

} // namespace visp