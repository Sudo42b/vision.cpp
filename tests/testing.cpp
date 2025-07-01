#include "testing.hpp"

#include <chrono>
#include <filesystem>
#include <string_view>

using std::chrono::steady_clock;

int main(int argc, char** argv) {
    auto& registry = visp::test_registry_instance();

    int passed = 0;
    int failed = 0;
    int errors = 0;

    std::string_view filter;
    bool verbose = false;
    for (int i = 1; i < argc; ++i) {
        std::string_view arg(argv[i]);
        if (arg == "-v" || arg == "--verbose") {
            verbose = true;
        } else {
            filter = arg;
        }
    }

    auto time_start = steady_clock::now();

    for (auto& test : registry.tests) {
        if (!filter.empty() && test.name != filter) {
            continue;
        }
        try {
            test.func();
            ++passed;
            if (verbose) {
                printf("%s \033[32mPASSED\033[0m\n", test.name);
            }
        } catch (const visp::test_failure& e) {
            ++failed;
            printf("%s \033[31mFAILED\033[0m\n", test.name);
            printf("  \033[90m%s:%d:\033[0m Assertion failed\n", e.file, e.line);
            printf("  \033[93m%s\033[0m\n", e.condition);
            if (e.eval) {
                printf("  \033[93m%s\033[0m\n", e.eval.c_str());
            }
        } catch (const std::exception& e) {
            ++errors;
            printf("%s \033[31mERROR\033[0m\n", test.name);
            printf("  \033[90m%s:%d:\033[0m Unhandled exception\n", test.file, test.line);
            printf("  \033[93m%s\033[0m\n", e.what());
        }
    }

    auto time_end = steady_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();

    char const* color = (failed > 0 || errors > 0) ? "\033[31m" : "\033[32m";
    printf("%s----------------------------------------------------------------------\n", color);
    if (failed > 0) {
        printf("\033[31m%d failed, ", failed);
    }
    if (errors > 0) {
        printf("\033[31m%d errors, ", errors);
    }
    printf("\033[92m%d passed %sin %lldms\033[0m\n", passed, color, duration);

    return (failed > 0 || errors > 0) ? 1 : 0;
}

namespace visp {

test_registry& test_registry_instance() {
    static test_registry registry;
    return registry;
}

test_registration::test_registration(
    char const* name, test_function f, char const* file, int line) {
    test_registry_instance().tests.push_back({name, f, file, line});
}

test_directories const& test_dir() {
    static test_directories const result = []() {
        path cur = std::filesystem::current_path();
        while (!exists(cur / "README.md")) {
            cur = cur.parent_path();
            if (cur.empty()) {
                throw std::runtime_error("root directory not found");
            }
        }
        test_directories dirs{
            .root = cur,
            .test = cur / "tests",
            .input = cur / "tests" / "input",
            .results = cur / "tests" / "results",
            .reference = cur / "tests" / "reference"};
        if (!exists(dirs.results)) {
            create_directories(dirs.results);
        }
        return dirs;
    }();
    return result;
}

float tolerance = 1e-5f;
float& test_tolerance_value() {
    return tolerance;
}

test_failure test_failure_image_mismatch(
    char const* file, int line, char const* condition, float rms) {
    test_failure result(file, line, condition);
    format(
        result.eval, "-> rmse {:.5f} > {:.5f} tolerance", rms,
        test_tolerance_value());
    return result;
}

} // namespace visp
