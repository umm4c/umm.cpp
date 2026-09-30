#pragma once

#include "model-registry.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace umm {

struct image_input {
    int width;
    int height;
    std::vector<uint8_t> rgb;
};

struct image_options {
    int width = 0;   // Zero selects the model default.
    int height = 0;
    int steps = 50;
    float guidance = 4.0f;
    float image_guidance = 1.5f;
    float flow_shift = 3.0f;
    int64_t seed = 42;
    bool think = false;
    int max_think_tokens = 1024;
};

struct image_result {
    int width;
    int height;
    std::vector<uint8_t> rgb;
    std::string reasoning;
    std::vector<int32_t> reasoning_tokens;
    std::vector<int32_t> prefix_tokens;
};

class session {
public:
    // Accept a model package directory, or an understanding GGUF and optional generation checkpoint.
    session(const std::string & model,
            const std::string & generation_model = "",
            const std::string & understanding_backend = "",
            const std::string & generation_backend = "",
            const std::string & generation_max_vram = "");
    ~session();
    session(const session &) = delete;
    session & operator=(const session &) = delete;

    bool supports(model_capability capability) const;
    std::string text(const std::string & prompt, int max_tokens = 256);
    std::string understand(const image_input & image, const std::string & prompt,
                           int max_tokens = 256, bool think = false);
    image_result image(const std::string & prompt, const image_options & options = {});
    image_result edit(const image_input & image, const std::string & prompt, const image_options & options = {});

private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

}
