#pragma once

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

} // namespace visp