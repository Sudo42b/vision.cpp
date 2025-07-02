cmake_minimum_required(VERSION 3.22)

project(vision.cpp VERSION 0.1.0 LANGUAGES CXX)

option(VISP_VULKAN "Enable Vulkan support" OFF)
option(VISP_DEV "Enable development mode" OFF)
option(VISP_TESTS "Build tests" ${PROJECT_IS_TOP_LEVEL})
option(VISP_ASAN "Enable AddressSanitizer" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INSTALL_RPATH $ORIGIN)

if(PROJECT_IS_TOP_LEVEL)
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()

if(VISP_DEV OR ${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  set(VISP_ASSERT "VISP_ASSERT_BREAK")
else()
  set(VISP_ASSERT "VISP_ASSERT_DISABLE")
endif()

if(VISP_ASAN)
  if(MSVC)
    add_compile_options(/fsanitize=address)
    add_link_options(clang_rt.asan_dynamic-x86_64.lib clang_rt.asan_dynamic_runtime_thunk-x86_64.lib)
  else()
    add_compile_options(-fsanitize=address)
    add_link_options(-fsanitize=address)
  endif()
endif()

if(MSVC)  
  add_compile_options(/Zi /utf-8)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
  add_link_options(/DEBUG) # Enable debug symbols also in release builds
endif()

add_subdirectory(depend/stb)

set(GGML_VULKAN ${VISP_VULKAN})
set(GGML_LLAMAFILE ON)
add_subdirectory(depend/ggml)

add_subdirectory(src/visp)
add_subdirectory(src/cli)

if(VISP_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# install(TARGETS visioncpp
#   EXPORT visioncppTargets
#   RUNTIME DESTINATION bin
#   LIBRARY DESTINATION lib
#   ARCHIVE DESTINATION lib)
# install(DIRECTORY src/include/ DESTINATION include)
# install(FILES README.md LICENSE DESTINATION .)

# include(CMakePackageConfigHelpers)
# write_basic_package_version_file(
#   visioncppConfigVersion.cmake
#   VERSION ${PACKAGE_VERSION}
#   COMPATIBILITY AnyNewerVersion
# )
# install(EXPORT visioncppTargets
#   FILE visioncppConfig.cmake
#   CONFIGURATIONS Release
#   DESTINATION lib/cmake/visioncpp
# )

# if(WIN32)
#   set(CPACK_GENERATOR "ZIP")
#   set(CPACK_PACKAGE_FILE_NAME visioncpp-windows-x64-${PROJECT_VERSION})
# else()
#   set(CPACK_GENERATOR "TGZ")
#   set(CPACK_PACKAGE_FILE_NAME visioncpp-linux-x64-${PROJECT_VERSION})
# endif()
# set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY OFF)
# include(CPack)
