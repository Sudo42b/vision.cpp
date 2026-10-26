# GTX NPU backend for ggml.
#
# It lives here rather than inside depend/llama so that depend/llama can point at an
# upstream tag without local patches. ggml's own registry does not know this backend,
# so visp registers it at startup (load_ggml_backends in src/visp/ml.cpp).

message(STATUS "Using GTX NPU backend")

# ggml_add_backend_library 는 GGML_VERSION(_MAJOR) 로 so 버전을 붙인다. 그 변수는 ggml 폴더
# 안에서만 정의되므로 ggml 타깃에 붙은 값을 가져온다.
get_target_property(GGML_VERSION ggml VERSION)
get_target_property(GGML_VERSION_MAJOR ggml SOVERSION)

ggml_add_backend_library(ggml-gtx
                         src/ggml-gtx.cpp
                         src/pcie_irq_host.cpp
                         )

target_include_directories(ggml-gtx
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_SOURCE_DIR}/depend/llama/ggml/src) # ggml-impl.h, ggml-backend-impl.h

# Consumers of ggml (visioncpp, vision-cli) see this and register the backend.
target_compile_definitions(ggml-gtx PUBLIC GGML_USE_GTX)

# GTX backend requires Windows (for PCIe driver) or Linux (for /dev/gtx*)
if (WIN32)
    # Windows: SetupAPI for device enumeration, could also need custom driver lib
    target_link_libraries(ggml-gtx PRIVATE setupapi)
    target_compile_definitions(ggml-gtx PRIVATE GGML_GTX_WINDOWS)
elseif (UNIX AND NOT APPLE)
    # Linux: mmap for BAR access
    target_compile_definitions(ggml-gtx PRIVATE GGML_GTX_LINUX)
else()
    message(WARNING "GTX backend may not work on this platform: ${CMAKE_SYSTEM_NAME}")
endif()

# Shared library export definitions
if (BUILD_SHARED_LIBS)
    target_compile_definitions(ggml-gtx PRIVATE GGML_GTX_BUILD)
    target_compile_definitions(ggml-gtx PUBLIC  GGML_GTX_SHARED)
endif()
