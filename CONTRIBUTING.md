## Development

* If you are making a bigger change, consider creating an issue first. A draft PR is also good.
* For implementing a new model architecture, take a look at [Model Implementation Guide](docs/model-implementation-guide.md)

> [!TIP]
> Pass `-D VISP_DEV=ON` to cmake to enable debug symbols and asserts that break into the debugger.

### Overview

* `include/visp/vision.hpp` - the main header has a fair amount of documentation
* `src/visp` - the main vision.cpp library
  * `src/visp/image.cpp` - image processing functionality
  * `src/visp/ml.cpp` - machine learning infrastructure on top of ggml
  * `src/visp/nn.cpp` - common neural network building blocks
  * `src/visp/arch/*` - model architecture implementations
* `src/cli` - the command-line interface
* `src/util` - utility shared between sub-projects
* `depend/ggml` - fork of the ggml tensor library

### Creating a Pull Request

* Make sure existing tests pass by running `ctest -C Release` in the build directory
* Add tests for new code where it makes sense
* Run `clang-format` on modified files

## Coding Guidelines

General rule: do as the rest of the code does. If in doubt, run `clang-format`.

Avoid introducing additional files and dependencies.

### Language

* **Use plain structs and functions** - keep it simple.
* Wrap resources with invariants into small single-purpose RAII objects
  * `std::unique_ptr` often does the job
  * avoid manual free/delete/close/...
* Use `ASSERT` generously. Throw a `visp::exception` only for exceptional errors which are user-recoverable.
* Avoid macros, class hierarchies, template meta programming, get/set boilerplate etc.
  * ... nothing is banned, but there better be a good reason.
* Avoid heap allocations if possible, make use of `std::array`, `fixed_string`, etc.

### Naming

Keep things concise and to the point.

* Use `snake_case` for types, functions, everything except macros and template type parameters.
* Names used across the project follow a `<group>_<verb>_*` convention, eg. all image-related types and functions start with `image_`.
* Use functions rather than constructors for complex initialization, especially if it involves memory allocation or IO.
* Prefer free functions. Member functions are okay for small property-like methods.
