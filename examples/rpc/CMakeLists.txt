cmake_minimum_required(VERSION 3.16)

add_executable(rpc_server_example rpc_server_example.cpp)
target_link_libraries(rpc_server_example PRIVATE ggml::ggml)

add_executable(rpc_client_example rpc_client_example.cpp)
target_link_libraries(rpc_client_example PRIVATE ggml::ggml)


