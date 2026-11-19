#include "diffusion-adapter.h"

#include "inference/image-input.h"
#include "inference/models/u1-rules.h"
#include "scheduling/execution-policy.h"
#include "stable-diffusion.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace umm {
namespace {

thread_local request_metrics * active_log_metrics = nullptr;

struct log_metric_scope {
    request_metrics * previous = active_log_metrics;
    explicit log_metric_scope(request_metrics * metrics) { active_log_metrics = metrics; }
    ~log_metric_scope() { active_log_metrics = previous; }
};

void log_stable_diffusion(enum sd_log_level_t, const char * message, void *) {
    std::fputs(message, stderr);
    if (active_log_metrics) {
        const char * marker = std::strstr(message, "decode_first_stage completed, taking ");
        double seconds = 0;
        if (marker && std::sscanf(marker, "decode_first_stage completed, taking %lfs", &seconds) == 1 && seconds >= 0) {
            if (active_log_metrics->vae_decode_ms < 0)
                active_log_metrics->vae_decode_ms = 0;
            active_log_metrics->vae_decode_ms += seconds * 1000;
        }
    }
}

struct progress_data {
    request_metrics * metrics;
    int expected_steps;
};

void diffusion_progress(int step, int steps, float seconds, void * data) {
    auto * progress = static_cast<progress_data *>(data);
    const int next_step = static_cast<int>(progress->metrics->diffusion_step_ms.size()) + 1;
    if (steps == progress->expected_steps && step == next_step && std::isfinite(seconds) && seconds >= 0) {
        progress->metrics->diffusion_step_ms.push_back(seconds * 1000.0);
        if (step == steps)
            sd_set_progress_callback(nullptr, nullptr);
    }
}

struct progress_scope {
    explicit progress_scope(request_metrics * metrics, int steps) : data{metrics, steps}, enabled(metrics != nullptr) {
        if (enabled)
            sd_set_progress_callback(diffusion_progress, &data);
    }
    ~progress_scope() {
        if (enabled)
            sd_set_progress_callback(nullptr, nullptr);
    }
    progress_data data;
    bool enabled;
};

int slot_id(conditioning_slot slot) {
    switch (slot) {
    case conditioning_slot::conditional:
        return 0;
    case conditioning_slot::without_text:
        return 1;
    case conditioning_slot::without_image:
        return 2;
    }
    return -1;
}

std::vector<int32_t> image_prefix_positions(const sd_kv_prefix_t & prefix) {
    if (prefix.positions) {
        return {prefix.positions, prefix.positions + prefix.token_count};
    }

    std::vector<int32_t> positions(prefix.token_count);
    for (size_t i = 0; i < positions.size(); ++i) {
        positions[i] = int32_t(i);
    }
    return positions;
}

}

void diffusion_adapter::install_log_callback() {
    sd_set_log_callback(log_stable_diffusion, nullptr);
}

diffusion_adapter::diffusion_adapter(const std::string & model, const std::string & vae, const std::string & llm,
                                     const std::string & backend, const std::string & max_vram,
                                     request_metrics * metrics)
    : context_(nullptr, free_sd_ctx) {
    if (model.empty()) {
        throw std::runtime_error("Provide the generation checkpoint for image output");
    }

    sd_ctx_params_t params;
    sd_ctx_params_init(&params);
    params.model_path = model.c_str();
    params.vae_path = vae.empty() ? nullptr : vae.c_str();
    params.n_threads = 8;
    params.enable_mmap = true;
    const bool multi_device = backend.find('&') != std::string::npos;
    const bool cann = backend_uses_cann(backend) || backend.find("CANN") != std::string::npos ||
                      backend.find("cann") != std::string::npos;
    const bool disable_flash = cann || std::getenv("UMM_DISABLE_FLASH_ATTN") != nullptr;
    params.flash_attn = params.diffusion_flash_attn = !disable_flash;
    params.backend = backend.empty() ? nullptr : backend.c_str();
    params.params_backend = backend.empty() || multi_device ? nullptr : backend.c_str();
    params.split_mode = multi_device ? "layer" : nullptr;
    params.max_vram = max_vram.empty() ? nullptr : max_vram.c_str();
    params.external_kv_prefix = true;

    params.llm_path = llm.empty() ? nullptr : llm.c_str();

    const auto start = metrics_clock::now();
    context_.reset(new_sd_ctx(&params));
    if (!context_) {
        throw std::runtime_error("Could not load generation checkpoint");
    }
    if (metrics)
        metrics->generation_model_load_ms = metrics_elapsed_ms(start);
}

void diffusion_adapter::set_prefix(conditioning_slot slot, const kv_prefix & prefix) {
    const sd_kv_prefix_t data{prefix.tokens.data(), prefix.tokens.size(), prefix.keys.data(),
                              prefix.values.data(), prefix.keys.size(),   prefix.positions.data()};
    if (!sd_set_kv_prefix_slot(context_.get(), slot_id(slot), &data)) {
        throw std::runtime_error("Could not transfer understanding prefix to generation");
    }
}

kv_prefix diffusion_adapter::append_image_to_prefix(conditioning_slot slot, const image_input & image, int64_t seed) {
    const sd_image_t input{uint32_t(image.width), uint32_t(image.height), 3, const_cast<uint8_t *>(image.rgb.data())};
    sd_kv_prefix_t prefix{};
    if (!sd_encode_image_prefix(context_.get(), slot_id(slot), &input, seed, &prefix)) {
        throw std::runtime_error("Could not encode image into generation prefix");
    }
    if (!prefix.token_ids || !prefix.token_count || !prefix.keys || !prefix.values || !prefix.layer_count) {
        throw std::runtime_error("Image prefix is incomplete");
    }
    return kv_prefix::copy_to_host({prefix.token_ids, prefix.token_ids + prefix.token_count},
                                   image_prefix_positions(prefix), {prefix.keys, prefix.keys + prefix.layer_count},
                                   {prefix.values, prefix.values + prefix.layer_count});
}

std::vector<float> diffusion_adapter::encode_u1_image(const image_input & image) {
    if (!context_) {
        throw std::invalid_argument("A loaded U1 image engine is required");
    }
    validate_rgb_image(image);
    if (image.width % u1_rules::image_stride || image.height % u1_rules::image_stride) {
        throw std::invalid_argument("U1 image dimensions must be divisible by 32 after preprocessing");
    }
    sd_image_t input{static_cast<uint32_t>(image.width), static_cast<uint32_t>(image.height), 3,
                     const_cast<uint8_t *>(image.rgb.data())};
    float * values = nullptr;
    size_t token_count = 0;
    size_t embedding_dim = 0;
    const bool encoded = sd_encode_sensenova_u1_image(context_.get(), &input, &values, &token_count, &embedding_dim);
    const std::unique_ptr<float, decltype(&sd_free_buffer)> buffer(values, sd_free_buffer);
    if (!encoded || !buffer || token_count == 0 || embedding_dim == 0 ||
        token_count > std::vector<float>().max_size() / embedding_dim) {
        throw std::runtime_error("SenseNova U1 image encoding failed");
    }
    return {buffer.get(), buffer.get() + token_count * embedding_dim};
}

image_result diffusion_adapter::generate(const std::string & prompt, const image_options & options,
                                         float image_guidance, image_result result, request_metrics * metrics) {
    log_metric_scope log_scope(metrics);
    sd_img_gen_params_t params;
    sd_img_gen_params_init(&params);
    params.prompt = prompt.c_str();
    params.negative_prompt = "";
    params.width = options.width;
    params.height = options.height;
    params.seed = options.seed;
    params.batch_count = 1;
    params.sample_params.sample_method = EULER_SAMPLE_METHOD;
    params.sample_params.scheduler = sd_get_default_scheduler(context_.get(), EULER_SAMPLE_METHOD);
    params.sample_params.sample_steps = options.steps;
    params.sample_params.guidance.txt_cfg = options.guidance;
    if (image_guidance > 0)
        params.sample_params.guidance.img_cfg = image_guidance;
    params.sample_params.flow_shift = options.flow_shift;
    params.vae_tiling_params.enabled = options.vae_tiling;
    params.vae_tiling_params.tile_size_x = options.vae_tile_size;
    params.vae_tiling_params.tile_size_y = options.vae_tile_size;
    params.vae_tiling_params.target_overlap = options.vae_tile_overlap;

    sd_image_t * images = nullptr;
    int count = 0;
    progress_scope progress(metrics, options.steps);
    const bool success = generate_image(context_.get(), &params, &images, &count);
    const auto release = [count](sd_image_t * data) { free_sd_images(data, count); };
    const std::unique_ptr<sd_image_t, decltype(release)> owned_images(images, release);
    if (metrics && !metrics->diffusion_step_ms.empty()) {
        metrics->diffusion_steps_ms =
            std::accumulate(metrics->diffusion_step_ms.begin(), metrics->diffusion_step_ms.end(), 0.0);
    }
    if (!success || count != 1 || !images || !images[0].data || images[0].channel != 3 ||
        images[0].width != uint32_t(result.width) || images[0].height != uint32_t(result.height)) {
        throw std::runtime_error("Image generation failed");
    }
    result.rgb.assign(images[0].data, images[0].data + size_t(result.width) * result.height * 3);
    return result;
}

}
