#include "testing.hpp"

#include <string_view>

int main(int argc, char** argv) {
    auto& registry = visp::test_registry_instance();

    int passed = 0;
    int failed = 0;
    int errors = 0;

    std::string_view filter;
    if (argc > 1) {
        filter = argv[1];
    }

    for (auto& test : registry.tests) {
        if (!filter.empty() && test.name != filter) {
            continue;
        }
        try {
            test.func();
            ++passed;
            std::printf("Test '%s' passed.\n", test.name);
        } catch (const visp::test_failure& e) {
            ++failed;
            std::fprintf(
                stderr, "Test '%s' failed: %s (%s:%d)\n", test.name, e.condition, e.file, e.line);
        } catch (const std::exception& e) {
            ++errors;
            std::fprintf(stderr, "Test '%s' threw an exception: %s\n", test.name, e.what());
        }
    }

    return (failed > 0 || errors > 0) ? 1 : 0;
}

namespace visp {

test_registry& test_registry_instance() {
    static test_registry registry;
    return registry;
}

test_registration::test_registration(char const* name, test_function f) {
    test_registry_instance().tests.push_back({name, f});
}

} // namespace visp
