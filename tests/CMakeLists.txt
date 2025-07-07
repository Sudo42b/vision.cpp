# Main library C++ tests

add_executable(test-vision)
target_sources(test-vision PRIVATE
  testing.cpp
  test-image.cpp
)
target_include_directories(test-vision PRIVATE . ../src)
target_compile_options(visioncpp PRIVATE ${VISP_WARNINGS})
target_link_libraries(test-vision PRIVATE visioncpp)
add_test(NAME vision COMMAND test-vision)

#
# Model tests (image comparisons)
#
# * these require models to be downloaded and placed in the `models` directory

add_executable(test-models)
target_sources(test-models PRIVATE
  testing.cpp
  test-models.cpp
)
target_include_directories(test-models PRIVATE . ../src)
target_compile_options(visioncpp PRIVATE ${VISP_WARNINGS})
target_link_libraries(test-models PRIVATE visioncpp)
add_test(NAME models COMMAND test-models)

#
# Workbench library for Python tests

add_library(vision-workbench workbench.cpp)
target_include_directories(vision-workbench PRIVATE ../src)
target_link_libraries(vision-workbench PRIVATE 
  visioncpp ggml
)

# Benchmark application for profiling

# add_executable(vision-prof benchmark.cpp)
# target_include_directories(vision-prof PRIVATE ../src)
# target_link_libraries(vision-prof PRIVATE 
#   visioncpp ggml
# )
