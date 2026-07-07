add_library(visioncpp)

# arch/*.cpp 는 glob(CONFIGURE_DEPENDS) — g2c 가 생성한 부품/모델 arch 를 파일만 넣으면
# 자동 빌드·(self-register)등록. 새 부품 추가 시 이 목록을 손댈 필요 없음.
file(GLOB VISP_ARCH_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/arch/*.cpp)

target_sources(visioncpp PRIVATE
  ${VISP_ARCH_SOURCES}
  c-api.cpp
  image.cpp
  ml.cpp
  nn.cpp
  postproc.cpp
  platform.cpp
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
