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

# UMM llama.cpp integration ---------------------------------------------------
add_library(llama.cpp src/llama-cpp-adapter.cpp src/model-registry.cpp)
target_include_directories(llama.cpp PUBLIC include PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/src")
target_link_libraries(llama.cpp PUBLIC llama)

# UMM runtime -----------------------------------------------------------------
add_library(umm
    src/session.cpp
    src/diffusion-bridge.cpp
    src/model-workflow.cpp
    src/model-package.cpp
    src/sd-cpp-adapter.cpp
)
target_include_directories(umm PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(umm PRIVATE llama.cpp sd.cpp)
target_link_libraries(umm PRIVATE mtmd)

# Command-line application ----------------------------------------------------
add_executable(umm-cli src/main.cpp src/cli.cpp src/image-io.cpp)
target_include_directories(umm-cli PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/stable-diffusion.cpp/thirdparty"
)
target_link_libraries(umm-cli PRIVATE umm)

# Persistent, structured interface for serial benchmark replay.
find_package(Threads REQUIRED)
add_library(umm-runtime STATIC src/request-runtime.cpp src/runtime-service.cpp
    src/system-monitor.cpp src/runtime-http.cpp src/image-io.cpp)
target_include_directories(umm-runtime PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/stable-diffusion.cpp/thirdparty"
    "${CMAKE_CURRENT_BINARY_DIR}/generated"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/vendor/hash/sha256")
target_link_libraries(umm-runtime PUBLIC umm ggml Threads::Threads PRIVATE vendor::hash)
add_executable(umm-session src/bench-session.cpp)
add_executable(umm-server src/server.cpp)
target_link_libraries(umm-server PRIVATE umm-runtime)

add_custom_target(umm-build-identity
    COMMAND "${CMAKE_COMMAND}" -DUMM_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
            -DUMM_IDENTITY_HEADER=${CMAKE_CURRENT_BINARY_DIR}/generated/build-identity.h
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/build-identity.cmake"
    BYPRODUCTS "${CMAKE_CURRENT_BINARY_DIR}/generated/build-identity.h"
    VERBATIM)
add_dependencies(umm-runtime umm-build-identity)
target_include_directories(umm-session PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/generated"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/stable-diffusion.cpp/thirdparty"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/ggml/include"
)
target_link_libraries(umm-session PRIVATE umm-runtime)
