#include "inference/model-workflow.h"
#include "inference/profile-context.h"

#include "adapters/diffusion-adapter.h"
#include "adapters/llama-cpp-adapter.h"

#include "inference/image-input.h"
#include "u1-rules.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace umm {
namespace {

using namespace workflow_detail;

int round_to_factor(int value, int factor) {
    return std::max(factor, ((value + factor / 2) / factor) * factor);
}

image_input resize_rgb_image(const image_input & image, int width, int height) {
    if (image.width == width && image.height == height) {
        return image;
    }
    image_input result{width, height, std::vector<uint8_t>(size_t(width) * height * 3)};
    for (int y = 0; y < height; ++y) {
        const float source_y =
            height == 1 ? 0.f
                        : static_cast<float>(y) * static_cast<float>(image.height - 1) / static_cast<float>(height - 1);
        const int y0 = static_cast<int>(std::floor(source_y));
        const int y1 = std::min(y0 + 1, image.height - 1);
        const float fy = source_y - static_cast<float>(y0);
        for (int x = 0; x < width; ++x) {
            const float source_x = width == 1 ? 0.f
                                              : static_cast<float>(x) * static_cast<float>(image.width - 1) /
                                                    static_cast<float>(width - 1);
            const int x0 = static_cast<int>(std::floor(source_x));
            const int x1 = std::min(x0 + 1, image.width - 1);
            const float fx = source_x - static_cast<float>(x0);
            for (int c = 0; c < 3; ++c) {
                const auto sample = [&](int sx, int sy) {
                    return static_cast<float>(image.rgb[(size_t(sy) * image.width + sx) * 3 + c]);
                };
                const float top = sample(x0, y0) * (1.f - fx) + sample(x1, y0) * fx;
                const float bottom = sample(x0, y1) * (1.f - fx) + sample(x1, y1) * fx;
                const float value = top * (1.f - fy) + bottom * fy;
                result.rgb[(size_t(y) * width + x) * 3 + c] =
                    static_cast<uint8_t>(std::clamp(std::lround(value), 0L, 255L));
            }
        }
    }
    return result;
}

image_input prepare_u1_input(const image_input & image) {
    validate_rgb_image(image);
    constexpr int factor = u1_rules::image_stride;
    constexpr int64_t min_pixels = 256LL * 256LL;
    constexpr int64_t max_pixels = 2048LL * 2048LL;
    if (std::max(image.width, image.height) / static_cast<double>(std::min(image.width, image.height)) > 200.0) {
        throw std::invalid_argument("U1 image aspect ratio must be below 200");
    }

    int height = round_to_factor(image.height, factor);
    int width = round_to_factor(image.width, factor);
    const int64_t rounded_area = int64_t(height) * width;
    const int64_t original_area = int64_t(image.height) * image.width;
    if (rounded_area > max_pixels) {
        const double scale = std::sqrt(static_cast<double>(original_area) / max_pixels);
        height = std::max(factor, static_cast<int>(std::floor(image.height / scale / factor)) * factor);
        width = std::max(factor, static_cast<int>(std::floor(image.width / scale / factor)) * factor);
    } else if (rounded_area < min_pixels) {
        const double scale = std::sqrt(static_cast<double>(min_pixels) / original_area);
        height = std::max(factor, static_cast<int>(std::ceil(image.height * scale / factor)) * factor);
        width = std::max(factor, static_cast<int>(std::ceil(image.width * scale / factor)) * factor);
    }
    return resize_rgb_image(image, width, height);
}

// SenseNova U1 prompt grammar ------------------------------------------------

const std::string u1_generation_system_prompt =
    "You are an image generation and editing assistant that accurately understands and executes user intent.\n\n"
    "You support two modes:\n\n1. Think Mode:\nIf the task requires reasoning, you MUST start with a "
    "<think></think> block. Put all reasoning inside the block using plain text. DO NOT include any image tags. "
    "Keep it reasonable and directly useful for producing the final image.\n\n2. Non-Think Mode:\nIf no reasoning "
    "is needed, directly produce the final image.\n\nTask Types:\n\nA. Text-to-Image Generation:\n- Generate a "
    "high-quality image based on the user's description.\n- Ensure visual clarity, semantic consistency, and "
    "completeness.\n- DO NOT introduce elements that contradict or override the user's intent.\n\nB. Image Editing:\n"
    "- Use the provided image(s) as input or reference for modification or transformation.\n- The result can be an "
    "edited image or a new image based on the reference(s).\n- Preserve all unspecified attributes unless explicitly "
    "changed.\n\nGeneral Rules:\n- For any visible text in the image, follow the language specified for the rendered "
    "text in the user's description, not the language of the prompt. If no language is specified, use the user's input "
    "language.";

const std::string u1_interleave_system_prompt =
    "You are a multimodal assistant. Generate text and images together when the user asks for an illustrated answer. "
    "In think mode, put reasoning between <think> and </think>; images may appear in that block. "
    "In non-think mode, answer directly with text and images in their intended order. "
    "Finish with a concise answer in the user's language.";

std::string u1_chat_prompt(const std::string & user, const std::string & system = {}) {
    std::string result;
    if (!system.empty()) {
        result = "<|im_start|>system\n" + system + "<|im_end|>\n";
    }
    return result + "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n";
}

std::string u1_text_prompt(const std::string & user) {
    return u1_chat_prompt(user) + "<think>\n\n</think>\n\n";
}

std::string u1_image_prompt(const std::string & user, bool think) {
    return u1_chat_prompt(user, u1_generation_system_prompt) + (think ? "<think>\n" : "<think>\n\n</think>\n\n<img>");
}

std::string u1_unconditional_prompt() {
    return u1_chat_prompt("") + "<img>";
}

// U1 emits reasoning until </think>, then expects the image marker.
void decode_u1_reasoning(llama_cpp_adapter & engine, image_result & result, int max_tokens) {
    for (int i = 0; i < max_tokens; ++i) {
        const auto token = engine.greedy();
        const auto piece = engine.piece(token);
        if (engine.is_end(token) || piece == "</think>") {
            break;
        }
        result.reasoning += piece;
        result.reasoning_tokens.push_back(token);
        {
            profiling::phase_scope phase("understanding_decode");
            engine.append({token});
        }
    }
    engine.append(engine.tokenize("</think>\n\n<img>"));
}

struct u1_image_grid {
    int width;
    int height;

    size_t tokens() const { return size_t(width) * height; }
    size_t tokens_with_markers() const { return tokens() + 2; }
};

u1_image_grid image_grid(int width, int height) {
    return {width / u1_rules::image_stride, height / u1_rules::image_stride};
}

u1_prefill_segment encode_prefill_image(workflow_context & context, const image_input & image) {
    const auto grid = image_grid(image.width, image.height);
    auto & diffusion = context.diffusion();
    const auto start = metrics_clock::now();
    profiling::phase_scope phase("vision_encode");
    auto embeddings = diffusion.encode_u1_image(image);
    if (auto * metrics = context.metrics) {
        const auto duration = metrics_elapsed_ms(start);
        metrics->image_encoder_ms = std::max(0.0, metrics->image_encoder_ms) + duration;
        metrics->vision_encode_ms = std::max(0.0, metrics->vision_encode_ms) + duration;
        metrics->vision_tokens += grid.tokens_with_markers();
    }
    return {{}, std::move(embeddings), grid.width, grid.height};
}

void append_edit_image(workflow_context & context, const image_input & image) {
    const auto grid = image_grid(image.width, image.height);
    auto & diffusion = context.diffusion();
    const auto start = metrics_clock::now();
    profiling::phase_scope phase("vision_encode");
    const auto embeddings = diffusion.encode_u1_image(image);
    auto * metrics = context.metrics;
    if (metrics)
        metrics->image_encoder_ms = std::max(0.0, metrics->image_encoder_ms) + metrics_elapsed_ms(start);
    context.language_model.append_u1_image_embeddings(embeddings, grid.width, grid.height, true, metrics);
    if (metrics) {
        metrics->vision_encode_ms = std::max(0.0, metrics->vision_encode_ms) + metrics_elapsed_ms(start);
        metrics->vision_tokens += grid.tokens_with_markers();
    }
}

// U1 image-generation prompt fragments.
void append_u1_edit_prefix(llama_cpp_adapter & engine, workflow_context & context, const image_input & image,
                           const std::string & prompt) {
    engine.append(
        engine.tokenize("<|im_start|>system\n" + u1_generation_system_prompt + "<|im_end|>\n<|im_start|>user\n"));
    append_edit_image(context, image);
    engine.append(engine.tokenize("\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"));
}

void append_u1_image_only_prefix(llama_cpp_adapter & engine, workflow_context & context, const image_input & image) {
    engine.append(
        engine.tokenize("<|im_start|>system\n" + u1_generation_system_prompt + "<|im_end|>\n<|im_start|>user\n"));
    append_edit_image(context, image);
    engine.append(engine.tokenize("\n<|im_end|>\n<|im_start|>assistant\n"));
}

class u1_workflow final : public model_workflow {
public:
    model_family family() const override { return model_family::sensenova_u1; }

    image_input prepare_image(const image_input & image) const override { return prepare_u1_input(image); }

    std::string text_prompt(const std::string & prompt) const override { return u1_text_prompt(prompt); }

    std::string understand(workflow_context & context, const image_input & image, const std::string & prompt,
                           int max_tokens, bool think) override {
        return understand_content(context, {image, "\n" + prompt}, max_tokens, think);
    }

    std::string understand_sequence(workflow_context & context, const std::vector<understanding_segment> & content,
                                    int max_tokens) override {
        return understand_content(context, content, max_tokens, false);
    }

    std::string understand_content(workflow_context & context, const std::vector<understanding_segment> & content,
                                   int max_tokens, bool think) {
        if (max_tokens < 1) {
            throw std::invalid_argument("max_tokens must be positive");
        }
        auto & engine = context.language_model;
        size_t required_tokens = engine
                                     .tokenize("<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n"
                                               "<think>\n\n</think>\n\n")
                                     .size() +
                                 size_t(max_tokens) * (think ? 2 : 1) + 16;
        for (const auto & segment : content) {
            if (const auto * image = std::get_if<image_input>(&segment)) {
                required_tokens += image_grid(image->width, image->height).tokens_with_markers();
            } else {
                required_tokens += engine.tokenize(std::get<std::string>(segment)).size();
            }
        }
        if (required_tokens > engine.context_capacity()) {
            throw std::invalid_argument("Ordered multimodal input needs " + std::to_string(required_tokens) +
                                        " tokens; context has " + std::to_string(engine.context_capacity()));
        }
        engine.reset();
        constexpr int images_per_prefill = 4;
        std::vector<u1_prefill_segment> pending_segments;
        int pending_images = 0;
        std::vector<llama_token> pending_tokens;
        auto flush = [&](bool final) {
            if (!pending_tokens.empty())
                pending_segments.push_back({std::move(pending_tokens), {}, 0, 0});
            engine.append_u1_mixed(pending_segments, final, context.metrics);
            pending_segments.clear();
            pending_images = 0;
            pending_tokens.clear();
        };
        auto append_tokens = [&](const std::vector<llama_token> & tokens, bool final = false) {
            pending_tokens.insert(pending_tokens.end(), tokens.begin(), tokens.end());
            if (final)
                flush(true);
        };
        append_tokens(engine.tokenize("<|im_start|>user\n"));
        std::string pending_text;
        for (const auto & segment : content) {
            if (const auto * image = std::get_if<image_input>(&segment)) {
                if (!pending_text.empty()) {
                    append_tokens(engine.tokenize(pending_text));
                    pending_text.clear();
                }
                if (pending_images == images_per_prefill)
                    flush(false);
                append_tokens(engine.tokenize(u1_rules::image_start));
                pending_segments.push_back({std::move(pending_tokens), {}, 0, 0});
                pending_tokens.clear();
                pending_segments.push_back(encode_prefill_image(context, *image));
                ++pending_images;
                append_tokens(engine.tokenize(u1_rules::image_end));
            } else {
                pending_text += std::get<std::string>(segment);
            }
        }
        append_tokens(engine.tokenize(pending_text + "<|im_end|>\n<|im_start|>assistant\n"));
        if (think) {
            append_tokens(engine.tokenize("<think>\n"), true);
            begin_text_decode(context);
            for (int i = 0; i < max_tokens; ++i) {
                const auto token = engine.greedy();
                const auto piece = engine.piece(token);
                note_text_token(context, engine.is_end(token));
                if (engine.is_end(token) || piece == "</think>") {
                    break;
                }
                {
                    profiling::phase_scope phase("understanding_decode");
                    engine.append({token});
                }
            }
            engine.append(engine.tokenize("</think>\n\n"));
        } else {
            append_tokens(engine.tokenize("<think>\n\n</think>\n\n"), true);
            begin_text_decode(context);
        }
        std::string result;
        if (context.metrics)
            context.metrics->finish_reason = "length";
        for (int i = 0; i < max_tokens; ++i) {
            const auto token = engine.greedy();
            note_text_token(context, engine.is_end(token));
            if (engine.is_end(token)) {
                break;
            }
            result += engine.piece(token);
            {
                profiling::phase_scope phase("understanding_decode");
                engine.append({token});
            }
        }
        finish_text_decode(context);
        return result;
    }

    image_result generate(workflow_context & context, const std::string & prompt, const image_options & requested,
                          const image_input * input) override {
        const auto options = normalize_image_options(family(), requested);
        auto & diffusion = context.diffusion();
        auto & engine = context.language_model;

        if (!input && options.guidance > 1) {
            engine.reset();
            engine.append(engine.tokenize(u1_unconditional_prompt()));
            context.transfer_prefix(conditioning_slot::without_text);
        }

        image_result result{options.width, options.height, {}, {}, {}, {}};
        if (input) {
            if (options.guidance > 1) {
                engine.reset();
                append_u1_image_only_prefix(engine, context, *input);
                if (options.think) {
                    engine.append(engine.tokenize("<think>\n"));
                    image_result image_only_reasoning{0, 0, {}, {}, {}, {}};
                    decode_u1_reasoning(engine, image_only_reasoning, options.max_think_tokens);
                } else {
                    engine.append(engine.tokenize(u1_rules::image_start));
                }
                context.transfer_prefix(conditioning_slot::without_text);
            }
            engine.reset();
            append_u1_edit_prefix(engine, context, *input, prompt);
            if (options.think) {
                engine.append(engine.tokenize("<think>\n"));
                decode_u1_reasoning(engine, result, options.max_think_tokens);
            } else {
                engine.append(engine.tokenize(u1_rules::image_start));
            }
        } else {
            engine.reset();
            engine.append(engine.tokenize(u1_image_prompt(prompt, options.think)));
            if (options.think) {
                decode_u1_reasoning(engine, result, options.max_think_tokens);
            }
        }
        result.prefix_tokens = engine.tokens();
        context.transfer_prefix(conditioning_slot::conditional);

        if (input && options.guidance > 1 && options.image_guidance > 1) {
            engine.reset();
            engine.append(engine.tokenize(u1_image_prompt(prompt, options.think)));
            if (options.think) {
                engine.append(std::vector<llama_token>(result.reasoning_tokens.begin(), result.reasoning_tokens.end()));
                engine.append(engine.tokenize("</think>\n\n<img>"));
            }
            context.transfer_prefix(conditioning_slot::without_image);
        }

        finish_conditioning(context, result.prefix_tokens.size());
        return diffusion.generate(prompt, options, input && options.guidance > 1 ? options.image_guidance : 0.f,
                                  std::move(result), context.metrics);
    }

    interleave_result interleave(workflow_context & context, const std::string & prompt,
                                 const image_options & requested, int max_text_tokens, int max_images) override {
        if (max_text_tokens < 1 || max_images < 1) {
            throw std::invalid_argument("max_text_tokens and max_images must be positive");
        }
        const auto options = normalize_image_options(family(), requested);
        if (options.guidance != 1.f) {
            throw std::invalid_argument("Interleaved generation currently requires --cfg 1");
        }
        auto & engine = context.language_model;
        engine.reset();
        engine.append(engine.tokenize(u1_chat_prompt(prompt, u1_interleave_system_prompt) +
                                      (options.think ? "<think>\n" : "<think>\n\n</think>\n\n")));
        const size_t image_tokens = image_grid(options.width, options.height).tokens();
        if (image_tokens > engine.available_u1_generated_image_tokens()) {
            throw std::invalid_argument("Image dimensions exceed the interleaved understanding context");
        }

        interleave_result result;
        int text_tokens = 0;
        while (text_tokens < max_text_tokens) {
            const auto token = engine.greedy();
            if (engine.is_end(token)) {
                break;
            }
            const auto piece = engine.piece(token);
            if (piece != u1_rules::image_start) {
                result.text += piece;
                {
                    profiling::phase_scope phase("understanding_decode");
                    engine.append({token});
                }
                ++text_tokens;
                continue;
            }
            if (static_cast<int>(result.images.size()) >= max_images) {
                break;
            }
            if (image_tokens > engine.available_u1_generated_image_tokens()) {
                if (result.images.empty()) {
                    throw std::runtime_error("No understanding context remains for a generated image");
                }
                break;
            }
            {
                profiling::phase_scope phase("understanding_decode");
                engine.append({token});
            }
            auto & diffusion = context.diffusion();
            context.transfer_prefix(conditioning_slot::conditional);

            auto image_options = options;
            image_options.seed += static_cast<int64_t>(result.images.size());
            image_result image{options.width, options.height, {}, {}, {}, {}};
            image = diffusion.generate(prompt, image_options, 0.f, std::move(image));
            const image_input generated{image.width, image.height, image.rgb};
            const auto embeddings = diffusion.encode_u1_image(generated);
            const auto grid = image_grid(image.width, image.height);
            engine.append_u1_generated_image_embeddings(embeddings, grid.width, grid.height);
            result.text += "<image>";
            result.images.push_back(std::move(image));
        }
        return result;
    }
};

}

std::unique_ptr<model_workflow> make_u1_workflow() {
    return std::make_unique<u1_workflow>();
}

}
