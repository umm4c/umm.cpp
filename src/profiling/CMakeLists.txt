add_library(umm-profiler-api
    request-and-result.cpp
    workload-run.cpp
    graph-capture.cpp
    rpc-timing-batch.cpp
    timing-annotations.cpp
    hierarchy-timing.cpp)
target_compile_features(umm-profiler-api PUBLIC cxx_std_17)
target_include_directories(umm-profiler-api
    PUBLIC "${PROJECT_SOURCE_DIR}/third_party/stable-diffusion.cpp/thirdparty"
    PRIVATE "${PROJECT_SOURCE_DIR}/src"
            "${PROJECT_SOURCE_DIR}/third_party/llama.cpp/vendor/hash/sha256")
target_link_libraries(umm-profiler-api PRIVATE umm-runtime umm-scheduling ggml vendor::hash)
