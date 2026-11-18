add_subdirectory(profiler-cli)

if(UMM_BUILD_BACKEND_SMOKE)
    add_executable(umm-random-model-run random-model-run.cpp)
    target_include_directories(umm-random-model-run PRIVATE
        "${PROJECT_SOURCE_DIR}/src"
        "${PROJECT_SOURCE_DIR}/third_party/stable-diffusion.cpp/thirdparty")
    target_link_libraries(umm-random-model-run PRIVATE umm llama.cpp ggml)
endif()
