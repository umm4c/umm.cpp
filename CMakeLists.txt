cmake_minimum_required(VERSION 3.21)
project(umm.cpp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(BUILD_SHARED_LIBS OFF)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

# Shared engine configuration -------------------------------------------------
# Both engines must see exactly the same ggml tensor ABI.
add_compile_definitions(GGML_MAX_NAME=160)
set(LLAMA_BUILD_COMMON OFF)
set(LLAMA_BUILD_TESTS OFF)
set(LLAMA_BUILD_TOOLS OFF)
set(LLAMA_BUILD_EXAMPLES OFF)
set(SD_BUILD_EXAMPLES OFF)
set(SD_WEBP OFF)
set(SD_WEBM OFF)
set(SD_GGML_QUANT_EXTENSIONS OFF)

# Third-party engines ---------------------------------------------------------
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp" third_party/llama.cpp)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/tools/mtmd" third_party/mtmd)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/third_party/stable-diffusion.cpp" third_party/stable-diffusion.cpp)
add_library(sd.cpp ALIAS stable-diffusion)
target_include_directories(stable-diffusion PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp")

add_subdirectory(src/scheduling)

# UMM llama.cpp integration ---------------------------------------------------
add_library(llama.cpp src/adapters/llama-cpp-adapter.cpp src/inference/model-registry.cpp)
target_include_directories(llama.cpp PUBLIC include PRIVATE src "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/src")
target_link_libraries(llama.cpp PUBLIC llama PRIVATE umm-scheduling)

# UMM runtime -----------------------------------------------------------------
add_library(umm
    src/inference/session.cpp
    src/adapters/diffusion-bridge.cpp
    src/inference/model-workflow.cpp
    src/inference/model-package.cpp
    src/adapters/sd-cpp-adapter.cpp
)
target_include_directories(umm PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include" PRIVATE src)
target_link_libraries(umm PRIVATE llama.cpp sd.cpp umm-scheduling)
target_link_libraries(umm PRIVATE mtmd)

# Command-line application ----------------------------------------------------
add_executable(umm-cli src/apps/main.cpp src/apps/cli.cpp src/runtime/image-io.cpp)
target_include_directories(umm-cli PRIVATE src
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/stable-diffusion.cpp/thirdparty"
)
target_link_libraries(umm-cli PRIVATE umm umm-scheduling)

# Persistent, structured interface for serial benchmark replay.
find_package(Threads REQUIRED)
add_library(umm-runtime STATIC src/runtime/request-runtime.cpp src/runtime/runtime-service.cpp
    src/runtime/system-monitor.cpp src/runtime/runtime-http.cpp src/runtime/image-io.cpp)
target_include_directories(umm-runtime PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/stable-diffusion.cpp/thirdparty"
    "${CMAKE_CURRENT_BINARY_DIR}/generated"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/vendor/hash/sha256"
    PRIVATE src)
target_link_libraries(umm-runtime PUBLIC umm ggml Threads::Threads PRIVATE vendor::hash umm-scheduling)
add_executable(umm-session src/apps/bench-session.cpp)
add_executable(umm-server src/apps/server.cpp)
target_include_directories(umm-server PRIVATE src)
target_link_libraries(umm-server PRIVATE umm-runtime)

add_custom_target(umm-build-identity
    COMMAND "${CMAKE_COMMAND}" -DUMM_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
            -DUMM_IDENTITY_HEADER=${CMAKE_CURRENT_BINARY_DIR}/generated/build-identity.h
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/build-identity.cmake"
    BYPRODUCTS "${CMAKE_CURRENT_BINARY_DIR}/generated/build-identity.h"
    VERBATIM)
add_dependencies(umm-runtime umm-build-identity)
target_include_directories(umm-session PRIVATE src
    "${CMAKE_CURRENT_BINARY_DIR}/generated"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/stable-diffusion.cpp/thirdparty"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/ggml/include"
)
target_link_libraries(umm-session PRIVATE umm-runtime)

option(UMM_BUILD_BACKEND_SMOKE "Build the static backend integration probe" OFF)

# Profiling reuses the runtime. Tools and tests consume the production targets.
add_subdirectory(src/profiling)
add_subdirectory(tools)
add_subdirectory(tests)
