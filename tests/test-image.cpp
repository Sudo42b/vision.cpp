#include "testing.hpp"
#include "visp/image-impl.hpp"
#include "visp/image.hpp"
#include "visp/util.hpp"

#include <array>
#include <filesystem>

namespace visp {
using std::filesystem::path;

path const& test_dir() {
    static path const p = []() {
        path cur = std::filesystem::current_path();
        while (!exists(cur / "README.md")) {
            cur = cur.parent_path();
            if (cur.empty()) {
                throw std::runtime_error("root directory not found");
            }
        }
        return cur / "test";
    }();
    return p;
}

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
    image_data img = image_load((test_dir() / "input" / "cat_and_hat.png").string().c_str());
    CHECK(img.extent == i32x2{512, 512});
    CHECK(img.format == image_format::rgba);
    CHECK(n_bytes(img) == 512 * 512 * 4);
}

TEST_CASE(image_save) {
    image_data img = image_alloc(i32x2{16, 16}, image_format::rgba);
    for (int i = 0; i < 16 * 16; ++i) {
        img.data.get()[i * 4 + 0] = 255;
        img.data.get()[i * 4 + 1] = uint8_t(i);
        img.data.get()[i * 4 + 2] = 0;
        img.data.get()[i * 4 + 3] = 255;
    }
    path filepath = (test_dir() / "result" / "test_image_save.png");
    image_save(img, filepath.string().c_str());
    CHECK(exists(filepath));

    image_data result = image_load(filepath.string().c_str());
    CHECK(result.extent == i32x2{16, 16});
    CHECK(result.format == image_format::rgba);
    for (int i = 0; i < 16 * 16; ++i) {
        CHECK(result.data[i * 4 + 0] == 255);
        CHECK(result.data[i * 4 + 1] == i);
        CHECK(result.data[i * 4 + 2] == 0);
        CHECK(result.data[i * 4 + 3] == 255);
    }
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

TEST_CASE(tile_merge) {
    std::array<std::array<f32x3, 5 * 5>, 4> tiles;
    for (int t = 0; t < 4; ++t) {
        float v = float(t);
        std::fill(tiles[t].begin(), tiles[t].end(), f32x3{v, v, v});
    }
    std::array<f32x3, 8 * 8> dst{};
    auto dst_span = image_span<f32x3>(i32x2{8, 8}, dst.data());
    auto const layout = tile_layout(i32x2{8, 8}, 6, 2, 1);
    tile_merge({i32x2{5, 5}, tiles[0].data()}, dst_span, {0, 0}, layout);
    tile_merge({i32x2{5, 5}, tiles[1].data()}, dst_span, {1, 0}, layout);
    tile_merge({i32x2{5, 5}, tiles[2].data()}, dst_span, {0, 1}, layout);
    tile_merge({i32x2{5, 5}, tiles[3].data()}, dst_span, {1, 1}, layout);

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
    std::array<f32x3, 8 * 8> expected;
    for (int i = 0; i < 8 * 8; ++i) {
        expected[i] = f32x3{expected_float[i], expected_float[i], expected_float[i]};
    }
    for (int i = 0; i < int(dst.size()); ++i) {
        CHECK(std::abs(dst[i][0] - expected[i][0]) < 0.0001f);
        CHECK(std::abs(dst[i][1] - expected[i][1]) < 0.0001f);
        CHECK(std::abs(dst[i][2] - expected[i][2]) < 0.0001f);
    }
}

TEST_CASE(tile_merge_blending) {
    std::array<f32x3, 22 * 19> dst{};
    auto dst_span = image_span<f32x3>(i32x2{22, 19}, dst.data());

    auto layout = tile_layout(i32x2{22, 19}, 10, 3, 2);
    auto te = layout.tile_size;
    auto tile = std::vector<f32x3>(te[0] * te[1], f32x3{1.f, 1.f, 1.f});
    auto tile_span = image_span<f32x3>(te, tile.data());

    for (int y = 0; y < layout.n_tiles[1]; ++y) {
        for (int x = 0; x < layout.n_tiles[0]; ++x) {
            tile_merge(tile_span, dst_span, {x, y}, layout);
        }
    }
    for (int y = 0; y < dst_span.extent[1]; ++y) {
        for (int x = 0; x < dst_span.extent[0]; ++x) {
            CHECK(dst_span.get(x, y) == f32x4{1.f, 1.f, 1.f, 1.f});
        }
    }
}

} // namespace visp
