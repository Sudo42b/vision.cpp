#pragma once

#include "util/string.hpp"

#include <filesystem>
#include <vector>

namespace visp {

struct test_failure {
    char const* file;
    int line;
    char const* condition;

    test_failure(char const* file, int line, char const* condition)
        : file(file), line(line), condition(condition) {}
};

using test_function = void (*)();

struct test_case {
    char const* name;
    test_function func;
    char const* file;
    int line;
};

struct test_registry {
    std::vector<test_case> tests;
};

test_registry& test_registry_instance();

struct test_registration {
    test_registration(char const* name, test_function f, char const* file, int line);
};

using std::filesystem::path;

struct test_directories {
    path root;
    path test;
    path input;
    path results;
    path reference;
};

test_directories const& test_dir();

} // namespace visp

#define TEST_CASE(name)                                                                            \
    void test_func_##name();                                                                       \
    const visp::test_registration test_reg_##name(#name, test_func_##name, __FILE__, __LINE__);    \
    void test_func_##name()

#define CHECK(...)                                                                                 \
    if (!(__VA_ARGS__)) {                                                                          \
        throw visp::test_failure(__FILE__, __LINE__, #__VA_ARGS__);                                \
    }