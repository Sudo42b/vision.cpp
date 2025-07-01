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

// Allocate image data, pixel data is not initialized
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
// Image span - reference to float32 image data for processing
// Always converts to/from 4-channel float for math

template <typename T> // float, f32x3, f32x4
struct image_span {
    using value_type = std::remove_cv_t<T>;
    using float_type = std::conditional_t<std::is_const_v<T>, float const, float>;

    constexpr static int n_channels = sizeof(T) / sizeof(float);

    i32x2 extent = {};
    size_t row_stride = 0;
    T* data = nullptr;

    image_span() = default;

    image_span(i32x2 extent, T* data) : extent(extent), row_stride(extent[0]), data(data) {}

    image_span(i32x2 extent, std::span<float> data)
        : image_span(extent, reinterpret_cast<T*>(data.data())) {
        ASSERT(data.size() == size_t(extent[0] * extent[1] * n_channels));
    }

    image_span(image_span<value_type> const& other)
        : extent(other.extent), row_stride(other.row_stride), data(other.data) {}

    T operator[](size_t i) const { return data[i]; }
    T& operator[](size_t i) { return data[i]; }

    f32x4 get(int x, int y) const;
    void set(int x, int y, f32x4 value);

    f32x4 get(i32x2 coord) const { return get(coord[0], coord[1]); }
    void set(i32x2 coord, f32x4 value) { set(coord[0], coord[1], value); }

    int n_pixels() const { return extent[0] * extent[1]; }
    int n_bytes() const { return n_pixels() * sizeof(T); }

    span<float_type> as_float() const;
};

//
// Image data for float32 images (1, 3, or 4 channels depending on T)
// Can be used everywhere an image_span<T> is expected

template <typename T>
struct image_data_t {
    i32x2 extent;
    std::unique_ptr<T[]> data;

    T operator[](size_t i) const { return data[i]; }
    T& operator[](size_t i) { return data[i]; }

    image_span<T> span() { return {extent, data.get()}; }
    image_span<const T> span() const { return {extent, data.get()}; }

    operator image_span<T>() { return span(); }
    operator image_span<const T>() const { return span(); }
};

template <typename T>
image_data_t<T> image_alloc(i32x2 extent) {
    return {extent, std::unique_ptr<T[]>(new T[extent[0] * extent[1]])};
}

//
// Image algorithms

// Convert uint8 image to float32 image
// - can be used with any image format, output is converted to T
// - applies computation `dst = (src + offset) * scale` to each pixel
// - starts reading `src` at `tile_offset`
// - always writes all of `dst`, if `src` is smaller the output is padded (clamp to edge)
template <typename T>
void image_to_float(
    image_view const& src,
    image_span<T> dst,
    f32x4 offset = f32x4(0),
    f32x4 scale = f32x4(1),
    i32x2 tile_offset = {0, 0});

// Convert float32 image to uint8 image
// - applies computation `dst = src * scale + offset` to every channel/pixel
void image_from_float(span<float const> src, span<uint8_t> dst, float scale = 1, float offset = 0);

// Box blur
void image_blur(image_span<float const> src, image_span<float> dst, int radius);
void image_blur(image_span<f32x4 const> src, image_span<f32x4> dst, int radius);

// Try to separate foreground and background contribution from pixels at the mask border
image_data_t<f32x4> image_estimate_foreground(
    image_span<f32x4 const> img, image_span<float const> mask, int radius = 30);

// Composite foreground and background images using alpha mask: `dst = fg * alpha + bg * (1-alpha)`
void image_alpha_composite(
    image_view const& fg, image_view const& bg, image_view const& mask, uint8_t* dst);

// Compute root-mean-square difference between two images
float image_difference_rms(image_view const&, image_view const&);
float image_difference_rms(image_span<float const>, image_span<float const>);
float image_difference_rms(image_span<f32x3 const>, image_span<f32x3 const>);
float image_difference_rms(image_span<f32x4 const>, image_span<f32x4 const>);

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
    image_span<f32x3 const> const& tile,
    image_span<f32x3>& dst,
    i32x2 tile_coord,
    tile_layout const& layout);

//
// Implementation
//

constexpr f32x4 load_pixel(f32x4 v) {
    return v;
}

constexpr f32x4 load_pixel(f32x3 v) {
    return f32x4{v[0], v[1], v[2], 1.0f};
}

constexpr f32x4 load_pixel(float v) {
    return f32x4{v};
}

constexpr void store_pixel(f32x3& v, f32x4 value) {
    v = {value[0], value[1], value[2]};
}

constexpr void store_pixel(f32x4& v, f32x4 value) {
    v = value;
}

constexpr void store_pixel(float& v, f32x4 value) {
    v = value[0];
}

template <typename T>
f32x4 image_span<T>::get(int x, int y) const {
    return load_pixel(data[y * row_stride + x]);
}

template <typename T>
void image_span<T>::set(int x, int y, f32x4 value) {
    store_pixel(data[y * row_stride + x], value);
}

template <typename T>
auto image_span<T>::as_float() const -> span<float_type> {
    return span(reinterpret_cast<float_type*>(data), n_pixels() * n_channels);
}

} // namespace visp