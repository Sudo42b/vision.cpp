#include <dlimgedit/dlimgedit.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <filesystem>

namespace dlimgedit {
using Path = std::filesystem::path;

Path const &test_dir() {
    static Path const path = []() {
        Path cur = std::filesystem::current_path();
        while (!exists(cur / "README.md")) {
            cur = cur.parent_path();
            if (cur.empty()) {
                throw std::runtime_error("root directory not found");
            }
        }
        return cur / "test";
    }();
    return path;
}

TEST_CASE("Image formats", "[image]") {
    auto channels =
        GENERATE(Channels::mask, Channels::rgb, Channels::rgba, Channels::bgra, Channels::argb);
    auto const img = Image(Extent{8, 6}, channels);
    CHECK(img.size() == 8 * 6 * count(channels));
    CHECK(img.size() <= 8 * 6 * 4);
    CHECK(img.size() >= 8 * 6 * 1);
}

TEST_CASE("Image can be loaded from file", "[image]") {
    auto const img = Image::load((test_dir() / "input" / "cat_and_hat.png").string());
    REQUIRE(img.extent().width == 512);
    REQUIRE(img.extent().height == 512);
    REQUIRE(img.channels() == Channels::rgba);
    REQUIRE(img.size() == 512 * 512 * 4);
}

TEST_CASE("Image can be saved to file", "[image]") {
    auto img = Image(Extent{16, 16}, Channels::rgba);
    for (int i = 0; i < 16 * 16; ++i) {
        img.pixels()[i * 4 + 0] = 255;
        img.pixels()[i * 4 + 1] = i;
        img.pixels()[i * 4 + 2] = 0;
        img.pixels()[i * 4 + 3] = 255;
    }
    auto filepath = test_dir() / "result" / "test_image_save.png";
    Image::save(img, filepath.string());
    REQUIRE(exists(filepath));

    auto const result = Image::load(filepath.string());
    REQUIRE(result.extent().width == 16);
    REQUIRE(result.extent().height == 16);
    REQUIRE(result.channels() == Channels::rgba);
    for (int i = 0; i < 16 * 16; ++i) {
        REQUIRE(result.pixels()[i * 4 + 0] == 255);
        REQUIRE(result.pixels()[i * 4 + 1] == i);
        REQUIRE(result.pixels()[i * 4 + 2] == 0);
        REQUIRE(result.pixels()[i * 4 + 3] == 255);
    }
}

} // namespace dlimgedit
