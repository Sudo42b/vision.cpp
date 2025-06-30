# Main library C++ tests

add_executable(test-unit)
target_sources(test-unit PRIVATE
  testing.cpp
  test-image.cpp
)
target_include_directories(test-unit PRIVATE . ../src)
target_link_libraries(test-unit PRIVATE
  visioncpp ggml
)

if(MSVC)
  target_compile_options(test-unit PRIVATE /W4 /WX)  
endif()

add_test(NAME unit COMMAND test-unit)

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
