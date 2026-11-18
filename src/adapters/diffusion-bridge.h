#pragma once

#include "stable-diffusion.h"

#include <cstdint>

namespace umm {

enum class conditioning_slot {
    conditional,
    without_text,
    without_image,
};

bool set_kv_prefix(sd_ctx_t * context, conditioning_slot slot, const sd_kv_prefix_t & prefix);
bool encode_image_prefix(sd_ctx_t * context, conditioning_slot slot, const sd_image_t & image,
                         int64_t seed, sd_kv_prefix_t & output);

}
