#include "testing.hpp"
#include "visp/image-impl.hpp"
#include "visp/image.hpp"
#include "visp/util.hpp"

#include <array>
#include <filesystem>

namespace visp {

TEST_CASE(image_formats) {
    auto formats = std::array{
        image_format::rgba, image_format::bgra, image_format::argb, image_format::rgb,
        image_format::alpha};

    for (image_format format : formats) {
        image_data img = image_alloc(i32x2{8, 6}, format);
        CHECK(n_bytes(img) == size_t(8 * 6 * n_channels(format)));
        CHECK(n_bytes(img) <= size_t(8 * 6 * 4));
        CHECK(n_bytes(img) >= size_t(8 * 6 * 1));
    }
}

TEST_CASE(image_load) {
    image_data img = image_load((test_dir().input / "cat-and-hat.jpg").string().c_str());
    CHECK(img.extent == i32x2{512, 512});
    CHECK(img.format == image_format::rgb);
    CHECK(n_bytes(img) == 512 * 512 * 3);
}

TEST_CASE(image_save) {
    image_data img = image_alloc(i32x2{16, 16}, image_format::rgba);
    for (int i = 0; i < 16 * 16; ++i) {
        img.data.get()[i * 4 + 0] = 255;
        img.data.get()[i * 4 + 1] = uint8_t(i);
        img.data.get()[i * 4 + 2] = 0;
        img.data.get()[i * 4 + 3] = 255;
    }
    path filepath = (test_dir().results / "image-save.png");
    image_save(img, filepath.string().c_str());
    CHECK(exists(filepath));

    image_data result = image_load(filepath.string().c_str());
    CHECK_IMAGES_EQUAL(result, img);
}

TEST_CASE(image_resize) {
    image_data img = image_alloc(i32x2{8, 8}, image_format::rgba);
    for (int i = 0; i < 8 * 8; ++i) {
        img.data[i * 4 + 0] = uint8_t(255);
        img.data[i * 4 + 1] = uint8_t(4 * (i / 8));
        img.data[i * 4 + 2] = uint8_t(4 * (i % 8));
        img.data[i * 4 + 3] = uint8_t(255);
    }
    image_data result = image_resize(img, i32x2{4, 4});
    CHECK(result.extent == i32x2{4, 4});
    CHECK(result.format == image_format::rgba);
    for (int i = 0; i < 16; ++i) {
        CHECK(result.data[i * 4 + 0] == 255);
        CHECK(int(result.data[i * 4 + 1]) == 2 + 8 * (i / 4));
        CHECK(int(result.data[i * 4 + 2]) == 2 + 8 * (i % 4));
        CHECK(result.data[i * 4 + 3] == 255);
    }
}

TEST_CASE(image_alpha_composite) {
    std::array<uint8_t, 2 * 2 * 4> fg_data = {255, 0, 0,   255, 0,   255, 0, 255, //
                                              0,   0, 255, 255, 255, 255, 0, 255};
    image_view fg = {i32x2{2, 2}, image_format::rgba, fg_data.data()};

    std::array<uint8_t, 2 * 2 * 3> bg_data = {0,   0,   0,   128, 128, 128, //
                                              255, 255, 255, 64,  64,  64};
    image_view bg = {i32x2{2, 2}, image_format::rgb, bg_data.data()};

    std::array<uint8_t, 2 * 2> mask_data = {255, 128, 64, 0};
    image_view mask = {i32x2{2, 2}, image_format::alpha, mask_data.data()};

    std::array<uint8_t, 2 * 2 * 4> output_data{};
    image_alpha_composite(fg, bg, mask, output_data.data());

    std::array<uint8_t, 2 * 2 * 4> expected_output = {255, 0,   0,   255, 63, 191, 63, 255, //
                                                      191, 191, 255, 255, 64, 64,  64, 255};
    for (size_t i = 0; i < output_data.size(); ++i) {
        CHECK_EQUAL(output_data[i], expected_output[i]);
    }
}

TEST_CASE(image_blur) {
    constexpr i32x2 extent{6, 6};
    // clang-format off
    std::array<float, extent[0] * extent[1]> input_data = {
         1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,
         7.0f,  8.0f,  9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f,
        19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f,
        25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f,
        31.0f, 32.0f, 33.0f, 34.0f, 35.0f, 36.0f
    };    
    std::array<float, extent[0] * extent[1]> expected_data = {
         3.33334f,  4.0f,  5.0f,  6.0f,  7.0f,  7.66667f,
         7.33334f,  8.0f,  9.0f, 10.0f, 11.0f, 11.66667f,
        13.33334f, 14.0f, 15.0f, 16.0f, 17.0f, 17.66667f,
        19.33334f, 20.0f, 21.0f, 22.0f, 23.0f, 23.66667f,
        25.33334f, 26.0f, 27.0f, 28.0f, 29.0f, 29.66667f,
        29.33334f, 30.0f, 31.0f, 32.0f, 33.0f, 33.66667f
    };
    // clang-format on
    std::array<float, extent[0] * extent[1]> output_data{};

    auto input = image_cspan(extent, input_data);
    auto output = image_span(extent, output_data);
    image_blur(input, output, 1);

    auto expected = image_cspan(extent, expected_data);
    CHECK_IMAGES_EQUAL(output, expected);
}

TEST_CASE(tile_merge) {
    std::array<std::array<f32x3, 5 * 5>, 4> tiles;
    for (int t = 0; t < 4; ++t) {
        float v = float(t);
        std::fill(tiles[t].begin(), tiles[t].end(), f32x3{v, v, v});
    }
    std::array<f32x3, 8 * 8> dst{};
    auto dst_span = image_span({8, 8}, dst);
    auto const layout = tile_layout(i32x2{8, 8}, 6, 2, 1);
    tile_merge(image_cspan({5, 5}, tiles[0]), dst_span, {0, 0}, layout);
    tile_merge(image_cspan({5, 5}, tiles[1]), dst_span, {1, 0}, layout);
    tile_merge(image_cspan({5, 5}, tiles[2]), dst_span, {0, 1}, layout);
    tile_merge(image_cspan({5, 5}, tiles[3]), dst_span, {1, 1}, layout);

    float e00 = float(4 * 0 + 2 * 1 + 2 * 2 + 1 * 3) / 9.f;
    float e10 = float(2 * 0 + 4 * 1 + 1 * 2 + 2 * 3) / 9.f;
    float e01 = float(2 * 0 + 1 * 1 + 4 * 2 + 2 * 3) / 9.f;
    float e11 = float(1 * 0 + 2 * 1 + 2 * 2 + 4 * 3) / 9.f;
    // clang-format off
    auto expected_float = std::array<float, 8 * 8>{
        0.f    , 0.f    , 0.f    , 1.f/3.f, 2.f/3.f, 1.f    , 1.f    , 1.f,
        0.f    , 0.f    , 0.f    , 1.f/3.f, 2.f/3.f, 1.f    , 1.f    , 1.f,
        0.f    , 0.f    , 0.f    , 1.f/3.f, 2.f/3.f, 1.f    , 1.f    , 1.f,
        2.f/3.f, 2.f/3.f, 2.f/3.f, e00    , e10    , 5.f/3.f, 5.f/3.f, 5.f/3.f,
        4.f/3.f, 4.f/3.f, 4.f/3.f, e01    , e11    , 7.f/3.f, 7.f/3.f, 7.f/3.f,
        2.f    , 2.f    , 2.f    , 7.f/3.f, 8.f/3.f, 3.f    , 3.f    , 3.f,
        2.f    , 2.f    , 2.f    , 7.f/3.f, 8.f/3.f, 3.f    , 3.f    , 3.f,
        2.f    , 2.f    , 2.f    , 7.f/3.f, 8.f/3.f, 3.f    , 3.f    , 3.f
    };
    // clang-format on
    std::array<f32x3, 8 * 8> expected_rgb;
    for (int i = 0; i < 8 * 8; ++i) {
        expected_rgb[i] = f32x3{expected_float[i], expected_float[i], expected_float[i]};
    }
    auto expected = image_cspan({8, 8}, expected_rgb);
    CHECK_IMAGES_EQUAL(dst_span, expected);
}

TEST_CASE(tile_merge_blending) {
    std::array<f32x3, 22 * 19> dst{};
    auto dst_span = image_span({22, 19}, dst);

    auto layout = tile_layout(i32x2{22, 19}, 10, 3, 2);
    auto te = layout.tile_size;
    auto tile = std::vector<f32x3>(te[0] * te[1], f32x3{1.f, 1.f, 1.f});
    auto tile_span = image_cspan(te, tile);

    for (int y = 0; y < layout.n_tiles[1]; ++y) {
        for (int x = 0; x < layout.n_tiles[0]; ++x) {
            tile_merge(tile_span, dst_span, {x, y}, layout);
        }
    }
    for (float value : dst_span.elements()) {
        CHECK_EQUAL(value, 1.0f);
    }
}

} // namespace visp
