#pragma once

#include "util/string.h"

#include <span>
#include <vector>

namespace visp {
using std::byte;
using std::span;

//
// Array / range helpers

template <typename T, size_t N>
bool contains(std::span<const T, N> r, T const& value) {
    return std::find(r.begin(), r.end(), value) != r.end();
}

template <typename T>
constexpr span<byte> as_bytes(span<T> data) {
    static_assert(std::is_trivially_copyable_v<T>, "type must be trivially copyable");
    return span<byte>(reinterpret_cast<byte*>(data.data()), data.size() * sizeof(T));
}

template <typename T>
constexpr span<byte const> as_bytes(span<T const> data) {
    static_assert(std::is_trivially_copyable_v<T>, "type must be trivially copyable");
    return span<byte const>(reinterpret_cast<byte const*>(data.data()), data.size() * sizeof(T));
}

template <typename T>
constexpr span<byte const> as_bytes(std::vector<T> const& data) {
    return as_bytes(span(data));
}

template <typename T>
constexpr span<T> cast(span<byte> bytes) {
    static_assert(std::is_trivially_copyable_v<T>, "type must be trivially copyable");
    ASSERT(bytes.size() % sizeof(T) == 0);
    return span<T>(reinterpret_cast<T*>(bytes.data()), bytes.size() / sizeof(T));
}

} // namespace visp