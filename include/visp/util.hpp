#pragma once

#include <array>
#include <string_view>

namespace visp {

template <size_t N>
struct fixed_string {
    std::array<char, N> data{};
    size_t length = 0;

    constexpr fixed_string() {}

    fixed_string(char const* str) {
        auto view = std::string_view(str);
        length = std::min(view.size(), N - 1);
        std::copy(view.begin(), view.begin() + length, data.begin());
    }

    char const* c_str() const {
        return data.data();
    }

    std::string_view view() const {
        return {data.data(), length};
    }

    explicit operator bool() const {
        return length > 0;
    }
};

} // namespace visp
