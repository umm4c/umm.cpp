#include "vision-encoder.h"

#include "inference/image-input.h"
#include "scheduling/execution-policy.h"

#include "clip.h"
#include "clip-impl.h"
#include "ggml-backend.h"
#include "mtmd-image.h"

#include <stdexcept>

namespace umm {

image_input prepare_bagel_image(const image_input & image) {
    validate_rgb_image(image);
    clip_image_u8 input;
    input.set_size({image.width, image.height}, false);
    input.cpy_buf(image.rgb);
    const auto prepared = mtmd_image_preprocessor_bagel::prepare_image(input);
    return {prepared.get_size().width, prepared.get_size().height, prepared.get_ro_buf()};
}

bagel_vision_encoder::bagel_vision_encoder(const std::string & path, const std::string & backend)
    : context_(nullptr, clip_free) {
    clip_context_params params{};
    params.use_gpu = true;
    if (!backend.empty()) {
        params.device = ggml_backend_dev_by_name(backend.c_str());
        if (!params.device) {
            throw std::runtime_error("Vision backend was not found: " + backend);
        }
    }
    params.flash_attn_type = backend_uses_cann(backend) ? CLIP_FLASH_ATTN_TYPE_DISABLED : CLIP_FLASH_ATTN_TYPE_ENABLED;
    const auto loaded = clip_init(path.c_str(), params);
    context_.reset(loaded.ctx_v);
    clip_free(loaded.ctx_a);
    clip_free(loaded.ctx_gen_a);
    if (!context_ || clip_n_mmproj_embd(context_.get()) != 3584) {
        throw std::runtime_error("Could not load BAGEL vision projector");
    }
}

std::vector<float> bagel_vision_encoder::encode(const image_input & image) {
    validate_rgb_image(image);
    clip_image_u8 input;
    input.set_size({image.width, image.height}, false);
    input.cpy_buf(image.rgb);
    const auto images = mtmd_image_preprocessor_bagel(context_.get()).preprocess(input);
    if (images.entries.size() != 1)
        throw std::runtime_error("Expected one image embedding group");
    std::vector<float> embeddings(size_t(clip_n_output_tokens(context_.get(), &images.entries.front())) *
                                  clip_n_mmproj_embd(context_.get()));
    if (!clip_image_encode(context_.get(), 8, &images.entries.front(), embeddings)) {
        throw std::runtime_error("Image encoding failed");
    }
    return embeddings;
}

}
