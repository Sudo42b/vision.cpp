#include "visp/arch_registry.h"

#include <algorithm>

namespace visp {

namespace {

// ⚠️ **함수 지역 static 이어야 한다.** 파일 스코프 전역으로 두면 다른 TU 의 `arch_registrar`
//    전역 생성자가 이 벡터보다 **먼저** 돌 수 있다(정적 초기화 순서 미정의) — 아직 생성되지
//    않은 벡터에 push_back 하게 된다. 함수 지역 static 은 첫 호출 때 확실히 만들어진다.
std::vector<arch_entry>& registry() {
    static std::vector<arch_entry> r;
    return r;
}

} // namespace

arch_registrar::arch_registrar(std::string_view name, arch_forward_fn forward, arch_task task) {
    registry().push_back(arch_entry{name, forward, std::move(task)});
}

arch_entry const* arch_find(std::string_view name) {
    auto& r = registry();
    auto it = std::find_if(r.begin(), r.end(), [&](arch_entry const& e) { return e.name == name; });
    return it == r.end() ? nullptr : &*it;
}

std::span<arch_entry const> arch_all() {
    return std::span<arch_entry const>(registry());
}

} // namespace visp
