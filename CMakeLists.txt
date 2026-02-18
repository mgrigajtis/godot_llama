cmake_minimum_required(VERSION 3.20)
project(godot_llama LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(godot_llama_ide OBJECT
    src/register_types.cpp
    src/llama_model.cpp
    src/llama_sampler.cpp
    src/llama_context.cpp
    src/llama_async_worker.cpp
)

target_include_directories(godot_llama_ide PRIVATE
    src
    godot-cpp/include
    godot-cpp/gen/include
)
