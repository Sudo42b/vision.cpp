# Main library C++ tests

add_executable(test-vision)
target_sources(test-vision PRIVATE
  testing.cpp
  test-image.cpp
)
target_include_directories(test-vision PRIVATE . ../src)
target_link_libraries(test-vision PRIVATE
  visioncpp ggml
)

if(MSVC)
  target_compile_options(test-vision PRIVATE /W4 /WX)  
endif()

add_test(NAME vision COMMAND test-vision)

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
