#include "case-protocol.h"
#include "request-execution.h"
#include "image-io.h"

#include <cmath>
#include <string_view>
#include <utility>

namespace umm::runtime {
namespace {

json available(double value) {
    return value < 0 ? json(nullptr) : json(value);
}
json available(int64_t value) {
    return value < 0 ? json(nullptr) : json(value);
}

json timings(const umm::request_metrics & metrics, metrics_clock::time_point start) {
    return {{"end_to_end", metrics_elapsed_ms(start)},
            {"image_decode", available(metrics.image_decode_ms)},
            {"input_preprocess", available(metrics.input_preprocess_ms)},
            {"generation_model_load", available(metrics.generation_model_load_ms)},
            {"vision_model_load", available(metrics.vision_model_load_ms)},
            {"vision_encode", available(metrics.vision_encode_ms)},
            {"image_encoder", available(metrics.image_encoder_ms)},
            {"image_prefill", available(metrics.image_prefill_ms)},
            {"text_prefill_compute", available(metrics.text_prefill_compute_ms)},
            {"mixed_prefill", available(metrics.mixed_prefill_ms)},
            {"text_prefill", available(metrics.text_prefill_ms)},
            {"text_decode", available(metrics.text_decode_ms)},
            {"ttft", available(metrics.ttft_ms)},
            {"conditioning", available(metrics.conditioning_ms)},
            {"diffusion_steps", available(metrics.diffusion_steps_ms)},
            {"diffusion_step", metrics.diffusion_step_ms.empty() ? json(nullptr) : json(metrics.diffusion_step_ms)},
            {"vae_decode", available(metrics.vae_decode_ms)},
            {"image_write", available(metrics.image_write_ms)}};
}

json prefill_metrics(const umm::request_metrics & metrics) {
    if (metrics.prefill_batches.empty())
        return nullptr;
    json batches = json::array();
    int64_t outputs = 0;
    for (const auto & batch : metrics.prefill_batches) {
        batches.push_back({{"kind", batch.kind},
                           {"tokens", batch.tokens},
                           {"cached_tokens", batch.cached_tokens},
                           {"outputs", batch.outputs},
                           {"elapsed_ms", batch.elapsed_ms}});
        outputs += batch.outputs;
    }
    return {{"call_count", batches.size()},
            {"logits_rows", outputs},
            {"attention_mode_switches", metrics.attention_mode_switches},
            {"input_hash_algorithm", "fnv1a64-token-position"},
            {"input_hash", std::to_string(metrics.prefill_input_hash)},
            {"batches", batches}};
}

json token_counts(const umm::request_metrics & metrics) {
    return {{"input", available(metrics.input_tokens)},
            {"vision", metrics.vision_encode_ms < 0 ? json(nullptr) : json(metrics.vision_tokens)},
            {"output", available(metrics.output_tokens)}};
}

image_options generation_options(const json & params, const options & args) {
    image_options generation;
    generation.width = required<int>(params, "width");
    generation.height = required<int>(params, "height");
    generation.seed = required<int64_t>(params, "seed");
    if (generation.width <= 0 || generation.height <= 0)
        throw std::invalid_argument("width and height must be positive");
    return parse_generation_options(args, generation);
}

image_input decode_data_url(const json & image) {
    return read_image_data_url(required<std::string>(image, "data_url"));
}

void configure_inline_generation(const json &, const json & params, image_options & generation) {
    for (const char * field : {"steps", "cfg", "shift"}) {
        if (!params.contains(field))
            continue;
        if (!params[field].is_number())
            throw std::invalid_argument("Invalid generation override");
        const double value = params[field].get<double>();
        if (!std::isfinite(value) || value <= 0)
            throw std::invalid_argument("Generation override must be positive and finite");
        if (std::string_view(field) == "steps") {
            if (!params[field].is_number_integer() || value > 1000)
                throw std::invalid_argument("Invalid step count");
            generation.steps = int(value);
        } else if (std::string_view(field) == "cfg") {
            generation.guidance = float(value);
        } else {
            generation.flow_shift = float(value);
        }
    }
}

json deliver_inline(const json &, const image_result & image, request_metrics &, double & output_encode_ms) {
    const auto start = metrics_clock::now();
    json output = {
        {"type", "image"}, {"b64_json", png_base64(image)}, {"width", image.width}, {"height", image.height}};
    output_encode_ms = metrics_elapsed_ms(start);
    return output;
}

}

json run_protocol_case(session & engine, const json & input, const options & args, const case_format & format) {
    const auto start = metrics_clock::now();
    inference_result result;
    std::string case_id;
    try {
        case_id = required<std::string>(input, "case_id");
        const auto task = required<std::string>(input, "task_class");
        const auto & content = input.at("content");
        const auto & params = input.at("params");
        if (!content.is_array() || !params.is_object())
            throw std::invalid_argument("content must be an array and params an object");

        inference_request request;
        request.content.reserve(content.size());
        for (const auto & part : content) {
            const auto type = required<std::string>(part, "type");
            if (type == "text") {
                request.content.emplace_back(required<std::string>(part, "value"));
            } else if (type == "image") {
                request.content.emplace_back(
                    image_reference{[decode = format.decode, part = &part] { return decode(*part); }});
            } else {
                throw std::invalid_argument("Unknown content type: " + type);
            }
        }
        validate_request(request, task);
        if (request.task == inference_task::generate_image) {
            request.generation = generation_options(params, args);
            format.configure_generation(input, params, request.generation);
        } else {
            request.max_tokens = required<int>(params, "max_new_tokens");
        }

        execute_request(engine, request, result);
        double engine_ms = -1, output_encode_ms = -1;
        json output;
        if (const auto * image = std::get_if<image_result>(&result.output)) {
            engine_ms = metrics_elapsed_ms(start);
            output = format.deliver_image(input, *image, result.metrics, output_encode_ms);
        } else {
            output = {{"type", "text"}, {"text", std::move(std::get<std::string>(result.output))}};
        }
        auto timing = timings(result.metrics, start);
        timing["engine"] = engine_ms >= 0 ? engine_ms : metrics_elapsed_ms(start);
        timing["output_encode"] = available(output_encode_ms);
        return {{"type", "result"},
                {"case_id", case_id},
                {"status", "ok"},
                {"finish_reason", result.metrics.finish_reason},
                {"output", std::move(output)},
                {"timings_ms", timing},
                {"token_counts", token_counts(result.metrics)},
                {"prefill", prefill_metrics(result.metrics)}};
    } catch (const unsupported_input_format & error) {
        result.error = inference_error{"unsupported_input_format", error.what()};
    } catch (const std::exception & error) {
        result.error = inference_error{
            dynamic_cast<const std::invalid_argument *>(&error) ? "invalid_request" : "request_failed", error.what()};
    }
    const auto & error = result.error.value();
    json failure = {{"type", "result"},
                    {"case_id", case_id},
                    {"status", "error"},
                    {"error", {{"code", error.code}, {"message", error.message}}},
                    {"timings_ms", timings(result.metrics, start)},
                    {"token_counts", token_counts(result.metrics)}};
    if (std::string_view(error.code) != "unsupported_input_format")
        failure["prefill"] = prefill_metrics(result.metrics);
    return failure;
}

json run_inline_case(session & engine, const json & request, const options & args) {
    return run_protocol_case(engine, request, args, {decode_data_url, configure_inline_generation, deliver_inline});
}

}
