add_library(umm-profiler-api
    profiler.cpp
    profiler-run.cpp
    profiler-annotations.cpp
    profiler-hierarchy.cpp)
target_compile_features(umm-profiler-api PUBLIC cxx_std_17)
target_include_directories(umm-profiler-api
    PUBLIC "${PROJECT_SOURCE_DIR}/third_party/stable-diffusion.cpp/thirdparty"
    PRIVATE "${PROJECT_SOURCE_DIR}/src"
            "${PROJECT_SOURCE_DIR}/third_party/llama.cpp/vendor/hash/sha256")
target_link_libraries(umm-profiler-api PRIVATE umm-runtime ggml vendor::hash)
