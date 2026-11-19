#pragma once

#include "model-registry.h"
#include "session.h"

#include <functional>
#include <memory>
#include <string>

namespace umm {

class llama_cpp_adapter;
class diffusion_adapter;
class bagel_vision_encoder;
enum class conditioning_slot;

struct workflow_context {
    // Shared runtime state plus model-independent operations supplied by
    // session::impl.
    // The llama.cpp adapter is the language-model side of the workflow.
    llama_cpp_adapter & language_model;
    request_metrics * metrics = nullptr;
    std::function<diffusion_adapter &()> diffusion;
    std::function<void(conditioning_slot)> transfer_prefix;
    std::function<bagel_vision_encoder &()> vision;
};

class model_workflow {
public:
    virtual ~model_workflow();
    virtual model_family family() const = 0;
    virtual image_input prepare_image(const image_input & image) const = 0;
    virtual bool diffusion_uses_understanding_weights() const { return false; }
    // Format a prompt-only request for the model's tokenizer/chat template.
    virtual std::string text_prompt(const std::string & prompt) const = 0;
    // Answer a question conditioned on an input image.
    virtual std::string understand(workflow_context & context, const image_input & image, const std::string & prompt,
                                   int max_tokens, bool think);
    virtual std::string understand_sequence(workflow_context & context,
                                            const std::vector<understanding_segment> & content, int max_tokens);
    // Generate an image, optionally conditioned on an input image for editing.
    virtual image_result generate(workflow_context & context, const std::string & prompt, const image_options & options,
                                  const image_input * input) = 0;
    virtual interleave_result interleave(workflow_context & context, const std::string & prompt,
                                         const image_options & options, int max_text_tokens, int max_images);
};

namespace workflow_detail {
void begin_text_decode(workflow_context & context);
void note_text_token(workflow_context & context, bool end);
void finish_text_decode(workflow_context & context);
void finish_conditioning(workflow_context & context, size_t input_tokens);
image_options normalize_image_options(model_family family, const image_options & requested);
}

// Construct the workflow matching the resolved model family.
std::unique_ptr<model_workflow> create_model_workflow(model_family family);

}
