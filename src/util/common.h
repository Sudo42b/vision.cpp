#pragma once

#include "util/string.h"

#include <span>

namespace visp {
using std::byte;
using std::span;

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
constexpr span<T> cast(span<byte> bytes) {
    static_assert(std::is_trivially_copyable_v<T>, "type must be trivially copyable");
    ASSERT(bytes.size() % sizeof(T) == 0);
    return span<T>(reinterpret_cast<T*>(bytes.data()), bytes.size() / sizeof(T));
}

} // namespace visp