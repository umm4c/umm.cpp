#include "diffusion-bridge.h"

namespace umm {

namespace {

int slot_id(conditioning_slot slot) {
    switch (slot) {
        case conditioning_slot::conditional: return 0;
        case conditioning_slot::without_text: return 1;
        case conditioning_slot::without_image: return 2;
    }
    return -1;
}

}

bool set_kv_prefix(sd_ctx_t * context, conditioning_slot slot, const sd_kv_prefix_t & prefix) {
    return sd_set_kv_prefix_slot(context, slot_id(slot), &prefix);
}

bool encode_image_prefix(sd_ctx_t * context, conditioning_slot slot, const sd_image_t & image,
                         int64_t seed, sd_kv_prefix_t & output) {
    return sd_encode_image_prefix(context, slot_id(slot), &image, seed, &output);
}

}
