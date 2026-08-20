add_library(visioncpp)

# arch/*.cpp 는 **자동으로 모은다.** 예전엔 손으로 나열했는데, g2c 생성물을 붙일 때마다
# 한 줄 추가를 빠뜨리면 그 파일이 통째로 빌드에서 빠진 채 아무도 모른다
# (`src/cli/inference_yolov9t.cpp` 가 실제로 그렇게 죽은 코드가 됐다).
# ⚠️ CONFIGURE_DEPENDS 가 있어야 파일을 추가했을 때 CMake 가 다시 돈다.
file(GLOB VISP_ARCH_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/arch/*.cpp")

target_sources(visioncpp PRIVATE
  ${VISP_ARCH_SOURCES}
  arch_registry.cpp
  draw.cpp
  c-api.cpp
  image.cpp
  ml.cpp
  nn.cpp
  platform.cpp
  postproc.cpp
  tracker.cpp
  vision.cpp
)
target_compile_features(visioncpp PUBLIC cxx_std_20)
target_compile_definitions(visioncpp PRIVATE VISP_API_EXPORT ${VISP_ASSERT} ${VISP_DEFINITIONS})
target_compile_options(visioncpp PRIVATE ${VISP_WARNINGS} ${VISP_COMP_OPTIONS})
target_include_directories(visioncpp PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  PRIVATE ..
)
target_link_options(visioncpp PRIVATE ${VISP_LINK_OPTIONS})
target_link_libraries(visioncpp
  PUBLIC ggml
  PRIVATE stb ${VISP_FMT_LINK}
)
set_target_properties(visioncpp PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
)

if (MSVC AND VISP_TESTS AND NOT VISP_PACKAGE)
  set_target_properties(visioncpp PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()
