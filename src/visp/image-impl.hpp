#pragma once

#include "util/string.hpp"
#include "visp/image.hpp"

#include <array>

namespace visp {

struct pixel_lookup {
    int stride_x;
    int stride_c;
    std::array<int, 4> channel_map;

    pixel_lookup(image_view image);

    pixel_lookup(i32x2 extent, image_format format);

    uint8_t get(uint8_t const* pixels, int x, int y, int c) const {
        return pixels[y * stride_x + x * stride_c + channel_map[c]];
    }

    void set(uint8_t* pixels, int x, int y, int c, uint8_t value) {
        pixels[y * stride_x + x * stride_c + channel_map[c]] = value;
    }
};

inline f32x4 image_load(float const* img, size_t i) {
    return f32x4{img[i]};
}
inline f32x4 image_load(f32x3 const* img, size_t i) {
    return f32x4{img[i][0], img[i][1], img[i][2], 1.0f};
}
inline f32x4 image_load(f32x4 const* img, size_t i) {
    return img[i];
}

inline void image_store(float* img, size_t i, f32x4 value) {
    img[i] = value[0];
}
inline void image_store(f32x3* img, size_t i, f32x4 value) {
    img[i] = f32x3{value[0], value[1], value[2]};
}
inline void image_store(f32x4* img, size_t i, f32x4 value) {
    img[i] = value;
}

template <typename T>
struct image_source {
    i32x2 extent;
    T const* data;

    static constexpr int n_channels = sizeof(T) / sizeof(float);

    image_source(image_cspan img) : extent(img.extent), data(reinterpret_cast<T const*>(img.data)) {
        ASSERT(img.n_channels == sizeof(T) / sizeof(float));
    }

    T const& operator[](size_t i) const { return data[i]; }

    f32x4 load(size_t i) const { return image_load(data, i); }
    f32x4 load(i32x2 c) const { return image_load(data, c[1] * extent[0] + c[0]); }

    operator image_cspan() const {
        return {extent, span(data, extent[0] * extent[1] * n_channels)};
    }
};

template <typename T>
struct image_target {
    i32x2 extent;
    T* data;

    static constexpr int n_channels = sizeof(T) / sizeof(float);

    image_target(image_span img) : extent(img.extent), data(reinterpret_cast<T*>(img.data)) {
        ASSERT(img.n_channels == sizeof(T) / sizeof(float));
    }

    T& operator[](size_t i) const { return data[i]; }

    f32x4 load(size_t i) const { return image_load(data, i); }
    f32x4 load(i32x2 c) const { return image_load(data, c[1] * extent[0] + c[0]); }

    void store(size_t i, f32x4 value) const { image_store(data, i, value); }
    void store(i32x2 c, f32x4 value) const { image_store(data, c[1] * extent[0] + c[0], value); }

    operator image_span() const { return {extent, span(data, extent[0] * extent[1] * n_channels)}; }
    operator image_cspan() const {
        return {extent, span(data, extent[0] * extent[1] * n_channels)};
    }
};

} // namespace visp