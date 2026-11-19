add_library(umm-scheduling STATIC
    execution-policy.cpp
    backend-devices.cpp
    backend-environment.cpp
    placement.cpp
    policies/local-310p.cpp
    policies/remote-910b.cpp
    policies/random.cpp
    weights/weight-replicas.cpp)
target_include_directories(umm-scheduling PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(umm-scheduling PUBLIC ggml)
