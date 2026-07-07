# Flow C 데모 — C++ 자족 검출기 (Python 없음)
add_executable(detect_standalone detect_standalone.cpp)
target_include_directories(detect_standalone PRIVATE ../../src ../../include)
target_link_libraries(detect_standalone PRIVATE ggml visioncpp ${VISP_FMT_LINK})
