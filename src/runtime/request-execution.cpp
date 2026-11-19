#include "request-execution.h"

namespace umm::runtime {
namespace {

image_input decode_image(const image_reference & image, request_metrics & metrics) {
    const auto start = metrics_clock::now();
    auto decoded = image.decode();
    metrics.image_decode_ms += metrics_elapsed_ms(start);
    return decoded;
}

}

void validate_request(inference_request & request, const std::string & task) {
    size_t image_count = 0;
    for (const auto & part : request.content) {
        if (const auto * text = std::get_if<std::string>(&part)) {
            request.prompt += *text;
        } else {
            ++image_count;
        }
    }
    if (request.prompt.empty())
        throw std::invalid_argument("Text prompt is empty");
    if (task == "mixed_understanding") {
        if (image_count != 1 || !std::holds_alternative<image_reference>(request.content.front()))
            throw unsupported_input_format("Exactly one leading image is supported");
        request.task = inference_task::single_image;
    } else if (task == "text_generation") {
        if (image_count)
            throw std::invalid_argument("Text generation expects text only");
        request.task = inference_task::text;
    } else if (task == "image_generation") {
        if (image_count)
            throw unsupported_input_format("Image generation expects text only");
        request.task = inference_task::generate_image;
    } else if (task == "multimodal_prefill") {
        if (image_count == 0)
            throw std::invalid_argument("Multimodal prefill needs at least one image");
        request.task = inference_task::mixed;
    } else {
        throw std::invalid_argument("Unknown task_class: " + task);
    }
}

void execute_request(session & engine, const inference_request & request, inference_result & result) {
    auto & metrics = result.metrics;
    switch (request.task) {
    case inference_task::text:
        result.output = engine.text(request.prompt, request.max_tokens, &metrics);
        break;
    case inference_task::single_image: {
        if (request.max_tokens <= 0)
            throw std::invalid_argument("max_new_tokens must be positive");
        const auto start = metrics_clock::now();
        const auto image = std::get<image_reference>(request.content.front()).decode();
        metrics.image_decode_ms = metrics_elapsed_ms(start);
        result.output = engine.understand(image, request.prompt, request.max_tokens, false, &metrics);
        break;
    }
    case inference_task::mixed: {
        if (request.max_tokens <= 0)
            throw std::invalid_argument("max_new_tokens must be positive");
        std::vector<understanding_segment> ordered;
        ordered.reserve(request.content.size());
        metrics.image_decode_ms = 0;
        for (const auto & part : request.content) {
            if (const auto * text = std::get_if<std::string>(&part)) {
                ordered.emplace_back(*text);
            } else {
                ordered.emplace_back(decode_image(std::get<image_reference>(part), metrics));
            }
        }
        result.output = engine.understand_sequence(ordered, request.max_tokens, &metrics);
        break;
    }
    case inference_task::generate_image:
        result.output = engine.image(request.prompt, request.generation, &metrics);
        break;
    }
}

}
