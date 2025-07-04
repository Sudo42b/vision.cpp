#pragma once

#include <exception>
#include <string_view>

namespace visp {

//
// Fixed string - fixed length, does not allocate, truncates if too long

template <size_t N>
struct fixed_string {
    char data[N] = {0};
    size_t length = 0;

    constexpr fixed_string() {}

    fixed_string(char const* str) {
        auto view = std::string_view(str);
        length = std::min(view.size(), N - 1);
        std::copy(view.begin(), view.begin() + length, data);
    }

    template <size_t M>
    constexpr fixed_string(char const (&str)[M]) {
        static_assert(M <= N, "String literal is too long for fixed_string");
        length = M - 1;
        std::copy(str, str + length, data);
    }

    char const* c_str() const { return data; }

    std::string_view view() const { return {data, length}; }

    explicit operator bool() const { return length > 0; }
};

//
// Exception type used in the library for recoverable errors

struct exception : std::exception {
    fixed_string<128> message;

    explicit exception(char const* msg) : message(msg) {}
    explicit exception(fixed_string<128> msg) : message(msg) {}

    char const* what() const noexcept override { return message.c_str(); }
};

//
// Simple vector types (fixed-size arrays)

template <typename T, int Dim, size_t Align = alignof(T)>
struct vec_t {
    using value_type = T;
    static constexpr int dim = Dim;

    alignas(Align) T v[Dim];

    constexpr T& operator[](size_t i) { return v[i]; }
    constexpr T const& operator[](size_t i) const { return v[i]; }

    constexpr auto operator<=>(vec_t const&) const = default;
};

using u8x3 = vec_t<uint8_t, 3>;
using u8x4 = vec_t<uint8_t, 4>;

using i32x2 = vec_t<int32_t, 2>;
using i32x4 = vec_t<int32_t, 4>;

using i64x2 = vec_t<int64_t, 2>;
using i64x4 = vec_t<int64_t, 4>;

using f32x2 = vec_t<float, 2, 8>;
using f32x3 = vec_t<float, 3>;
using f32x4 = vec_t<float, 4, 16>;

//
// Flags - extends enums with bit-wise operations to be used as bitmask

template <typename E>
struct flags {
    using enum_type = E;

    uint32_t value = 0;

    constexpr flags() = default;
    constexpr flags(E value) : value(uint32_t(value)) {}
    explicit constexpr flags(uint32_t value) : value(value) {}

    flags& operator|=(E other) {
        value |= other;
        return *this;
    }

    friend constexpr bool operator&(flags<E> lhs, E rhs) {
        return (lhs.value & uint32_t(rhs)) != 0;
    }

    friend constexpr flags<E> operator|(flags<E> lhs, E rhs) {
        return flags<E>(lhs.value | uint32_t(rhs));
    }
};

} // namespace visp
