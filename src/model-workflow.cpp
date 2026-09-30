#include "model-workflow.h"

#include "llama-cpp-adapter.h"
#include "sd-cpp-adapter.h"
#include "stable-diffusion.h"

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace umm {

namespace {

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
    return u1_chat_prompt(user, u1_generation_system_prompt) +
           (think ? "<think>\n" : "<think>\n\n</think>\n\n<img>");
}

std::string u1_unconditional_prompt() {
    return u1_chat_prompt("") + "<img>";
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
        !std::isfinite(options.flow_shift) || options.flow_shift <= 0 || options.max_think_tokens < 1) {
        throw std::invalid_argument("Use valid model dimensions, positive steps/shift, and guidance >= 1");
    }
    if (model_supports(family, model_capability::image_guidance) &&
        (!std::isfinite(options.image_guidance) || options.image_guidance < 1)) {
        throw std::invalid_argument("Image guidance must be finite and at least 1");
    }
    return options;
}

// Shared diffusion submission and output validation.
image_result render_generated_image(sd_ctx_t * image_engine, const std::string & prompt,
                                    const image_options & options, float image_guidance,
                                    image_result result) {

    sd_img_gen_params_t params;
    sd_img_gen_params_init(&params);
    params.prompt = prompt.c_str();
    params.negative_prompt = "";
    params.width = options.width;
    params.height = options.height;
    params.seed = options.seed;
    params.batch_count = 1;
    params.sample_params.sample_method = EULER_SAMPLE_METHOD;
    params.sample_params.scheduler = sd_get_default_scheduler(image_engine, EULER_SAMPLE_METHOD);
    params.sample_params.sample_steps = options.steps;
    params.sample_params.guidance.txt_cfg = options.guidance;
    if (image_guidance > 0) params.sample_params.guidance.img_cfg = image_guidance;
    params.sample_params.flow_shift = options.flow_shift;

    sd_image_t * images = nullptr;
    int count = 0;
    const bool success = generate_image(image_engine, &params, &images, &count);
    if (!success || count != 1 || !images || !images[0].data || images[0].channel != 3 ||
        images[0].width != uint32_t(result.width) || images[0].height != uint32_t(result.height)) {
        free_sd_images(images, count);
        throw std::runtime_error("Image generation failed");
    }
    result.rgb.assign(images[0].data, images[0].data + size_t(result.width) * result.height * 3);
    free_sd_images(images, count);
    return result;
}

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
        engine.append({token});
    }
    engine.append(engine.tokenize("<|im_end|>"));
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
        engine.append({token});
    }
    engine.append(engine.tokenize("\n\n<img>"));
}

// BAGEL's understanding path can include <think>...</think>; only return the
// answer text when reasoning is enabled.
std::string decode_bagel_answer(llama_cpp_adapter & engine, int max_tokens, bool think) {
    std::string result;
    bool in_reasoning = false;
    for (int i = 0; i < max_tokens; ++i) {
        const auto token = engine.greedy();
        const auto piece = engine.piece(token);
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
        engine.append({token});
    }
    return result;
}

// U1 image-generation prompt fragments.
void append_u1_edit_prefix(llama_cpp_adapter & engine, workflow_context & context,
                           const image_input & image, const std::string & prompt) {
    engine.append(engine.tokenize("<|im_start|>system\n" + u1_generation_system_prompt +
                                  "<|im_end|>\n<|im_start|>user\n"));
    context.append_vision_image(image);
    engine.append(engine.tokenize("\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"));
}

void append_u1_image_only_prefix(llama_cpp_adapter & engine, workflow_context & context,
                                 const image_input & image) {
    engine.append(engine.tokenize("<|im_start|>system\n" + u1_generation_system_prompt +
                                  "<|im_end|>\n<|im_start|>user\n"));
    context.append_vision_image(image);
    engine.append(engine.tokenize("\n<|im_end|>\n<|im_start|>assistant\n"));
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

// ---------------------------------------------------------------------------
// SenseNova U1 workflow

class u1_workflow final : public model_workflow {
public:
    model_family family() const override { return model_family::sensenova_u1; }

    std::string text_prompt(const std::string & prompt) const override {
        return u1_text_prompt(prompt);
    }

    std::string understand(workflow_context & context, const image_input & image,
                            const std::string & prompt, int max_tokens, bool think) override {
        if (max_tokens < 1) {
            throw std::invalid_argument("max_tokens must be positive");
        }
        auto & engine = context.language_model;
        engine.reset();
        engine.append(engine.tokenize("<|im_start|>user\n"));
        context.append_vision_image(image);
        engine.append(engine.tokenize("\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"));
        if (think) {
            engine.append(engine.tokenize("<think>\n"));
            for (int i = 0; i < max_tokens; ++i) {
                const auto token = engine.greedy();
                const auto piece = engine.piece(token);
                if (engine.is_end(token) || piece == "</think>") {
                    break;
                }
                engine.append({token});
            }
            engine.append(engine.tokenize("\n\n"));
        }
        std::string result;
        for (int i = 0; i < max_tokens; ++i) {
            const auto token = engine.greedy();
            if (engine.is_end(token)) {
                break;
            }
            result += engine.piece(token);
            engine.append({token});
        }
        return result;
    }

    image_result generate(workflow_context & context, const std::string & prompt,
                          const image_options & requested, const image_input * input) override {
        const auto options = normalize_image_options(family(), requested);
        context.load_image_engine();
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
                    engine.append(engine.tokenize("<img>"));
                }
                context.transfer_prefix(conditioning_slot::without_text);
            }
            engine.reset();
            append_u1_edit_prefix(engine, context, *input, prompt);
            if (options.think) {
                engine.append(engine.tokenize("<think>\n"));
                decode_u1_reasoning(engine, result, options.max_think_tokens);
            } else {
                engine.append(engine.tokenize("<img>"));
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

        return render_generated_image(
            context.image_engine(), prompt, options,
            input && options.guidance > 1 ? options.image_guidance : 0.f,
            std::move(result));
    }

    interleave_result interleave(workflow_context & context, const std::string & prompt,
                                const image_options & requested, int max_text_tokens,
                                int max_images) override {
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
        const auto & descriptor = model_descriptor_for(family());
        const size_t image_tokens = size_t(options.width / descriptor.image_stride) *
                                    size_t(options.height / descriptor.image_stride);
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
            if (piece != "<img>") {
                result.text += piece;
                engine.append({token});
                ++text_tokens;
                continue;
            }
            if (static_cast<int>(result.images.size()) >= max_images) {
                break;
            }
            if (image_tokens > engine.available_u1_generated_image_tokens()) {
                throw std::runtime_error("No understanding context remains for another generated image");
            }
            engine.append({token});
            context.load_image_engine();
            context.transfer_prefix(conditioning_slot::conditional);

            auto image_options = options;
            image_options.seed += static_cast<int64_t>(result.images.size());
            image_result image{options.width, options.height, {}, {}, {}, {}};
            image = render_generated_image(context.image_engine(), prompt, image_options, 0.f,
                                           std::move(image));
            const image_input generated{image.width, image.height, image.rgb};
            const auto embeddings = encode_u1_image_with_diffusion(context.image_engine(), generated);
            engine.append_u1_generated_image_embeddings(embeddings, image.width / 32, image.height / 32);
            result.text += "<image>";
            result.images.push_back(std::move(image));
        }
        return result;
    }
};

// ---------------------------------------------------------------------------
// BAGEL workflow

class bagel_workflow final : public model_workflow {
public:
    model_family family() const override { return model_family::bagel; }

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
        context.append_vision_image(image);
        engine.append(engine.tokenize("<|im_start|>" + prompt + "<|im_end|><|im_start|>"));
        return decode_bagel_answer(engine, max_tokens, think);
    }

    image_result generate(workflow_context & context, const std::string & prompt,
                          const image_options & requested, const image_input * input) override {
        const auto options = normalize_image_options(family(), requested);
        context.load_image_engine();
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
            context.append_vision_image(*input);
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
        return render_generated_image(context.image_engine(), prompt, options,
                                      image_guidance, std::move(result));
    }
};

}

// ---------------------------------------------------------------------------
// Workflow factory and default behavior

model_workflow::~model_workflow() = default;

interleave_result model_workflow::interleave(workflow_context &, const std::string &,
                                            const image_options &, int, int) {
    throw std::invalid_argument("Interleaved generation is not supported by the selected model");
}

std::string model_workflow::understand(workflow_context &, const image_input &, const std::string &, int, bool) {
    throw std::invalid_argument("Image understanding is not supported by the selected model");
}

std::unique_ptr<model_workflow> create_model_workflow(model_family family) {
    switch (family) {
        case model_family::sensenova_u1: return std::make_unique<u1_workflow>();
        case model_family::bagel: return std::make_unique<bagel_workflow>();
        case model_family::unknown: break;
    }
    throw std::invalid_argument("Unsupported model family");
}

}
