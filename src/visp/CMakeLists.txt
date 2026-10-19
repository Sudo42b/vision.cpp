add_library(visioncpp)

# arch/*.cpp 는 **자동으로 모은다.** 예전엔 손으로 나열했는데, g2c 생성물을 붙일 때마다
# 한 줄 추가를 빠뜨리면 그 파일이 통째로 빌드에서 빠진 채 아무도 모른다.
# ⚠️ CONFIGURE_DEPENDS 가 있어야 파일을 추가했을 때 CMake 가 다시 돈다.
file(GLOB VISP_ARCH_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/arch/*.cpp")

# VISP_ARCHS 를 주면 그 이름만 빌드한다. 안 주면 위에서 모은 arch/ 전체 — 기본 동작은 그대로다.
#   cmake -S . -B build -DVISP_ARCHS="ResNet;FamAnn"
# ⚠️ 없는 이름은 여기서 멈춘다. 조용히 빠지면 「등록했는데 vision-cli 에 안 보인다」로
#    나타나고, 원인이 이 목록이라 찾기 어렵다.
if(VISP_ARCHS)
  set(_visp_selected "")
  foreach(_visp_arch IN LISTS VISP_ARCHS)
    set(_visp_arch_src "${CMAKE_CURRENT_SOURCE_DIR}/arch/${_visp_arch}.cpp")
    if(NOT EXISTS "${_visp_arch_src}")
      message(FATAL_ERROR "VISP_ARCHS: arch/${_visp_arch}.cpp 가 없다")
    endif()
    list(APPEND _visp_selected "${_visp_arch_src}")
  endforeach()
  set(VISP_ARCH_SOURCES ${_visp_selected})
  message(STATUS "visp: arch 선택 빌드 — ${VISP_ARCHS}")
endif()

target_sources(visioncpp PRIVATE
  ${VISP_ARCH_SOURCES}
  arch_registry.cpp
  runtime_graph.cpp
  mmlab/config.cpp
  mmlab/modules.cpp
  mmlab/backbones.cpp
  mmlab/heads.cpp
  mmlab/transformers.cpp
  mmlab/model.cpp
  draw.cpp
  c-api.cpp
  image.cpp
  ml.cpp
  nn.cpp
  platform.cpp
  mmdet_head.cpp
  mmdet_run.cpp
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
  PRIVATE .. ${PROJECT_SOURCE_DIR}/depend/llama/vendor
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
