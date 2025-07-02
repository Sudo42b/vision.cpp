add_library(visioncpp SHARED)

target_sources(visioncpp PRIVATE
  birefnet.cpp
  esrgan.cpp
  image.cpp
  migan.cpp
  ml.cpp
  mobile-sam.cpp
  nn.cpp
)
target_compile_definitions(visioncpp PRIVATE VISP_EXPORTS ${VISP_ASSERT})
target_include_directories(visioncpp PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  PRIVATE ..
)
target_link_libraries(visioncpp
  PRIVATE stb ggml
)
set_target_properties(visioncpp PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
)

if (MSVC)
  target_compile_options(visioncpp PRIVATE /W4 /WX)

  if (VISP_TESTS)
    set_target_properties(visioncpp PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
  endif()
endif()
