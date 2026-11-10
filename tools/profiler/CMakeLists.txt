cmake_minimum_required(VERSION 3.21)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(umm-profiler LANGUAGES CXX)
endif()
get_filename_component(UMM_PROFILER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
add_library(umm-profiler-api "${UMM_PROFILER_ROOT}/src/profiler.cpp")
target_compile_features(umm-profiler-api PUBLIC cxx_std_17)
target_include_directories(umm-profiler-api PUBLIC
    "${UMM_PROFILER_ROOT}/include"
    "${UMM_PROFILER_ROOT}/third_party/stable-diffusion.cpp/thirdparty")
add_executable(umm-profiler main.cpp)
target_link_libraries(umm-profiler PRIVATE umm-profiler-api)
