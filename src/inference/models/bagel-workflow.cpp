#include "inference/model-workflow.h"
#include "inference/profile-context.h"

#include "adapters/diffusion-adapter.h"
#include "adapters/llama-cpp-adapter.h"

#include "adapters/vision-encoder.h"

#include <stdexcept>
#include <utility>

namespace umm {
namespace {

using namespace workflow_detail;

// BAGEL emits reasoning until its image-end marker, then needs an explicit
// image-end token before the diffusion prefix is transferred.
void decode_bagel_reasoning(llama_cpp_adapter & engine, image_result & result, int max_tokens) {
    for (int i = 0; i < max_tokens; ++i) {
        const auto token = engine.greedy();
        const auto piece = engine.piece(token);
        if (piece == "<|im_end|>") {
            break;
        }
        result.reasoning += piece;
        result.reasoning_tokens.push_back(token);
        { profiling::phase_scope phase("understanding_decode"); engine.append({token}); }
    }
    engine.append(engine.tokenize("<|im_end|>"));
}

// BAGEL's understanding path can include <think>...</think>; only return the
// answer text when reasoning is enabled.
std::string decode_bagel_answer(workflow_context & context, int max_tokens, bool think) {
    auto & engine = context.language_model;
    std::string result;
    bool in_reasoning = false;
    if (context.metrics) context.metrics->finish_reason = "length";
    for (int i = 0; i < max_tokens; ++i) {
        const auto token = engine.greedy();
        const auto piece = engine.piece(token);
        note_text_token(context, engine.is_end(token));
        if (piece == "<think>") {
            in_reasoning = true;
        } else if (piece == "</think>") {
            in_reasoning = false;
        } else if (!think || !in_reasoning) {
            result += piece;
        }
        if (engine.is_end(token)) {
            break;
        }
        { profiling::phase_scope phase("understanding_decode"); engine.append({token}); }
    }
    return result;
}

// BAGEL prompt grammar ------------------------------------------------------

std::string bagel_reasoning_prompt(bool generate_image) {
    if (generate_image) {
        return "<|im_start|>You should first think about the planning process in the mind and then generate the image. \n"
               "The planning process is enclosed within <think> </think> tags, i.e. <think> planning process here </think> image here<|im_end|>";
    }
    return "<|im_start|>You should first think about the reasoning process in the mind and then provide the user with the answer. \n"
           "The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here<|im_end|>";
}

class bagel_workflow final : public model_workflow {
public:
    model_family family() const override { return model_family::bagel; }

    image_input prepare_image(const image_input & image) const override {
        // This common resize supplies both the VAE expert and the vision path.
        return prepare_bagel_image(image);
    }

    bool diffusion_uses_understanding_weights() const override { return true; }

    std::string text_prompt(const std::string & prompt) const override {
        return "<|im_start|>" + prompt + "<|im_end|><|im_start|>";
    }

    std::string understand(workflow_context & context, const image_input & image,
                            const std::string & prompt, int max_tokens, bool think) override {
        if (max_tokens < 1) {
            throw std::invalid_argument("max_tokens must be positive");
        }
        auto & engine = context.language_model;
        engine.reset();
        if (think) engine.append(engine.tokenize(bagel_reasoning_prompt(false)));
        context.append_bagel_image(image);
        engine.append(engine.tokenize("<|im_start|>" + prompt + "<|im_end|><|im_start|>"));
        begin_text_decode(context);
        auto answer = decode_bagel_answer(context, max_tokens, think);
        finish_text_decode(context);
        return answer;
    }

    image_result generate(workflow_context & context, const std::string & prompt,
                          const image_options & requested, const image_input * input) override {
        const auto options = normalize_image_options(family(), requested);
        auto & diffusion = context.diffusion();
        auto & engine = context.language_model;

        if (!input && options.guidance > 1) {
            engine.reset();
            if (options.think) engine.append(engine.tokenize(bagel_reasoning_prompt(true)));
            context.transfer_prefix(conditioning_slot::without_text);
        }

        engine.reset();
        if (options.think) engine.append(engine.tokenize(bagel_reasoning_prompt(true)));
        if (input) {
            context.append_latent_image(*input, options.seed);
            context.append_bagel_image(*input);
            if (options.guidance > 1) {
                context.transfer_prefix(conditioning_slot::without_text);
            }
        }
        engine.append(engine.tokenize("<|im_start|>" + prompt + "<|im_end|>"));
        if (options.think) engine.append(engine.tokenize("<|im_start|>"));

        image_result result{options.width, options.height, {}, {}, {}, {}};
        if (options.think) {
            decode_bagel_reasoning(engine, result, options.max_think_tokens);
        }
        result.prefix_tokens = engine.tokens();
        context.transfer_prefix(conditioning_slot::conditional);

        if (input && options.guidance > 1 && options.image_guidance > 1) {
            engine.reset();
            if (options.think) engine.append(engine.tokenize(bagel_reasoning_prompt(true)));
            engine.append(engine.tokenize("<|im_start|>" + prompt + "<|im_end|>"));
            if (options.think) {
                engine.append(engine.tokenize("<|im_start|>"));
                engine.append(std::vector<llama_token>(result.reasoning_tokens.begin(), result.reasoning_tokens.end()));
                engine.append(engine.tokenize("<|im_end|>"));
            }
            context.transfer_prefix(conditioning_slot::without_image);
        }

        const float image_guidance = input && options.guidance > 1 ? options.image_guidance : 1.f;
        finish_conditioning(context, result.prefix_tokens.size());
        return diffusion.generate(prompt, options,
                                      image_guidance, std::move(result), context.metrics);
    }
};

}

std::unique_ptr<model_workflow> make_bagel_workflow() {
    return std::make_unique<bagel_workflow>();
}

}
