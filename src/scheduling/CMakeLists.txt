add_library(umm-scheduling STATIC
    execution-policy.cpp
    backend-devices.cpp
    backend-environment.cpp
    placement.cpp
    policies/local-310p.cpp
    policies/remote-910b.cpp
    policies/random.cpp
    policies/runtime-placement.cpp
    policies/profiler-index.cpp
    policies/adaptive.cpp
    weights/weight-replicas.cpp)
target_include_directories(umm-scheduling PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(umm-scheduling PUBLIC ggml)
target_include_directories(umm-scheduling PRIVATE
    "${PROJECT_SOURCE_DIR}/third_party/stable-diffusion.cpp/thirdparty"
    "${PROJECT_SOURCE_DIR}/third_party/llama.cpp/vendor/hash/sha256"
    "${PROJECT_BINARY_DIR}/generated")
target_link_libraries(umm-scheduling PRIVATE vendor::hash)
