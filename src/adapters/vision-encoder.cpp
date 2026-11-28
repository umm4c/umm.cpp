#include "vision-encoder.h"

#include "inference/image-input.h"
#include "scheduling/backend-devices.h"

#include "clip.h"
#include "clip-impl.h"
#include "ggml-backend.h"
#include "mtmd-image.h"

#include <algorithm>
#include <stdexcept>

namespace umm {
namespace {

using clip_context_ptr = std::unique_ptr<clip_ctx, void (*)(clip_ctx *)>;

clip_context_ptr load_vision_projector(const std::string & path, const std::string & backend, bool metadata_only) {
    clip_context_params params{};
    params.use_gpu = !metadata_only;
    params.no_alloc = metadata_only;
    if (params.use_gpu && !backend.empty()) {
        params.device = ggml_backend_dev_by_name(backend.c_str());
        if (!params.device) {
            throw std::runtime_error("Vision backend was not found: " + backend);
        }
    }
    params.flash_attn_type = backend_uses_cann(backend) ? CLIP_FLASH_ATTN_TYPE_DISABLED : CLIP_FLASH_ATTN_TYPE_ENABLED;
    const auto loaded = clip_init(path.c_str(), params);
    clip_context_ptr context(loaded.ctx_v, clip_free);
    clip_free(loaded.ctx_a);
    clip_free(loaded.ctx_gen_a);
    if (!context || clip_n_mmproj_embd(context.get()) != 3584) {
        throw std::runtime_error("Could not load BAGEL vision projector");
    }
    return context;
}

}

image_input prepare_bagel_image(const image_input & image) {
    validate_rgb_image(image);
    clip_image_u8 input;
    input.set_size({image.width, image.height}, false);
    input.cpy_buf(image.rgb);
    const auto prepared = mtmd_image_preprocessor_bagel::prepare_image(input);
    return {prepared.get_size().width, prepared.get_size().height, prepared.get_ro_buf()};
}

bagel_vision_encoder::bagel_vision_encoder(const std::string & path, const std::string & backend)
    : path_(path), backend_(backend), context_(nullptr, clip_free) {}

std::vector<float> bagel_vision_encoder::encode(const image_input & image, request_metrics * metrics,
                                               std::optional<size_t> max_tokens) {
    validate_rgb_image(image);
    const auto start = metrics_clock::now();
    double load_ms = 0;
    auto load_weights = [&] {
        const auto load_start = metrics_clock::now();
        context_ = load_vision_projector(path_, backend_, false);
        load_ms = metrics_elapsed_ms(load_start);
        if (metrics)
            metrics->vision_model_load_ms = load_ms;
    };
    if (!context_ && !max_tokens)
        load_weights();

    // On a cold request, get the real projector's token count without reading weights or initializing CANN.
    clip_context_ptr metadata(nullptr, clip_free);
    if (!context_)
        metadata = load_vision_projector(path_, "", true);
    const auto * projector = context_ ? context_.get() : metadata.get();
    clip_image_u8 input;
    input.set_size({image.width, image.height}, false);
    input.cpy_buf(image.rgb);
    const auto images = mtmd_image_preprocessor_bagel(projector).preprocess(input);
    if (images.entries.size() != 1)
        throw std::runtime_error("Expected one image embedding group");
    const auto tokens = size_t(clip_n_output_tokens(projector, &images.entries.front()));
    if (max_tokens && tokens > *max_tokens) {
        throw std::invalid_argument("BAGEL image needs " + std::to_string(tokens) +
                                    " vision tokens; context and microbatch allow " + std::to_string(*max_tokens));
    }
    metadata.reset();
    if (!context_)
        load_weights();
    std::vector<float> embeddings(tokens * clip_n_mmproj_embd(context_.get()));
    if (!clip_image_encode(context_.get(), 8, &images.entries.front(), embeddings)) {
        throw std::runtime_error("Image encoding failed");
    }
    if (metrics)
        metrics->vision_encode_ms = std::max(0.0, metrics->vision_encode_ms) + metrics_elapsed_ms(start) - load_ms;
    return embeddings;
}

}
