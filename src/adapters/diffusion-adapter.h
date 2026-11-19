#pragma once

#include "kv-prefix.h"
#include "session.h"

#include <memory>
#include <string>
#include <vector>

struct sd_ctx_t;

namespace umm {

enum class conditioning_slot {
    conditional,
    without_text,
    without_image,
};

class diffusion_adapter {
public:
    diffusion_adapter(const std::string & model, const std::string & vae, const std::string & llm,
                      const std::string & backend, const std::string & max_vram,
                      request_metrics * metrics);
    static void install_log_callback();

    void set_prefix(conditioning_slot slot, const kv_prefix & prefix);
    kv_prefix append_image_to_prefix(conditioning_slot slot, const image_input & image, int64_t seed);
    std::vector<float> encode_u1_image(const image_input & image);
    image_result generate(const std::string & prompt, const image_options & options, float image_guidance,
                          image_result result, request_metrics * metrics = nullptr);

private:
    std::unique_ptr<sd_ctx_t, void (*)(sd_ctx_t *)> context_;
};

}
