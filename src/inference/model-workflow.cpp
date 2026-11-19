#include "model-workflow.h"

#include "adapters/llama-cpp-adapter.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace umm {

std::unique_ptr<model_workflow> make_u1_workflow();
std::unique_ptr<model_workflow> make_bagel_workflow();

namespace workflow_detail {

void begin_text_decode(workflow_context & context) {
    auto * metrics = context.metrics;
    if (!metrics) return;
    context.language_model.synchronize();
    const auto now = metrics_clock::now();
    double prefill = std::chrono::duration<double, std::milli>(now - metrics->inference_start).count();
    for (double excluded : {metrics->vision_encode_ms, metrics->vision_model_load_ms,
                            metrics->generation_model_load_ms}) {
        if (excluded >= 0) prefill -= excluded;
    }
    metrics->text_prefill_ms = std::max(0.0, prefill);
    metrics->input_tokens = context.language_model.tokens().size();
    metrics->output_tokens = 0;
    metrics->decode_start = now;
}

void note_text_token(workflow_context & context, bool end) {
    auto * metrics = context.metrics;
    if (!metrics) return;
    if (end) { metrics->finish_reason = "stop"; return; }
    if (metrics->ttft_ms < 0) metrics->ttft_ms = metrics_elapsed_ms(metrics->request_start);
    ++metrics->output_tokens;
}

void finish_text_decode(workflow_context & context) {
    if (context.metrics) {
        context.language_model.synchronize();
        context.metrics->text_decode_ms = metrics_elapsed_ms(context.metrics->decode_start);
    }
}

void finish_conditioning(workflow_context & context, size_t input_tokens) {
    auto * metrics = context.metrics;
    if (!metrics) return;
    double duration = metrics_elapsed_ms(metrics->inference_start);
    for (double excluded : {metrics->generation_model_load_ms, metrics->vision_model_load_ms,
                            metrics->vision_encode_ms}) {
        if (excluded >= 0) duration -= excluded;
    }
    metrics->conditioning_ms = std::max(0.0, duration);
    metrics->input_tokens = input_tokens;
}

// Shared image-request validation. Model workflows supply defaults and then
// use the normalized values for both text conditioning and diffusion.
image_options normalize_image_options(model_family family, const image_options & requested) {
    const auto & descriptor = model_descriptor_for(family);
    auto options = requested;
    if (options.width == 0) options.width = descriptor.default_width;
    if (options.height == 0) options.height = descriptor.default_height;
    if (options.width < descriptor.image_stride || options.height < descriptor.image_stride ||
        options.width % descriptor.image_stride || options.height % descriptor.image_stride ||
        (descriptor.max_width && options.width > descriptor.max_width) ||
        (descriptor.max_height && options.height > descriptor.max_height) ||
        options.steps < 1 || !std::isfinite(options.guidance) || options.guidance < 1 ||
        !std::isfinite(options.flow_shift) || options.flow_shift <= 0 || options.max_think_tokens < 1 ||
        options.vae_tile_size < 0 || !std::isfinite(options.vae_tile_overlap) ||
        options.vae_tile_overlap < 0 || options.vae_tile_overlap >= 1) {
        throw std::invalid_argument("Use valid model dimensions, positive steps/shift, and guidance >= 1");
    }
    if (model_supports(family, model_capability::image_guidance) &&
        (!std::isfinite(options.image_guidance) || options.image_guidance < 1)) {
        throw std::invalid_argument("Image guidance must be finite and at least 1");
    }
    return options;
}

}

model_workflow::~model_workflow() = default;

interleave_result model_workflow::interleave(workflow_context &, const std::string &,
                                            const image_options &, int, int) {
    throw std::invalid_argument("Interleaved generation is not supported by the selected model");
}

std::string model_workflow::understand(workflow_context &, const image_input &, const std::string &, int, bool) {
    throw std::invalid_argument("Image understanding is not supported by the selected model");
}

std::string model_workflow::understand_sequence(workflow_context &,
                                               const std::vector<understanding_segment> &,
                                               int) {
    throw std::invalid_argument("Ordered multimodal understanding is not supported by the selected model");
}

std::unique_ptr<model_workflow> create_model_workflow(model_family family) {
    switch (family) {
        case model_family::sensenova_u1: return make_u1_workflow();
        case model_family::bagel: return make_bagel_workflow();
        case model_family::unknown: break;
    }
    throw std::invalid_argument("Unsupported model family");
}

}
