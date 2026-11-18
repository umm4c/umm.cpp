add_library(umm-scheduling STATIC
    execution-policy.cpp
    policies/local-310p.cpp
    policies/remote-910b.cpp
    policies/random.cpp
    weight-replicas.cpp)
target_include_directories(umm-scheduling PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(umm-scheduling PUBLIC ggml)
