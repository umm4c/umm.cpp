#pragma once

#include "diffusion-bridge.h"
#include "model-registry.h"
#include "session.h"

#include <functional>
#include <memory>
#include <string>

struct sd_ctx_t;

namespace umm {

class llama_cpp_adapter;

struct workflow_context {
    // Shared runtime state plus model-independent operations supplied by
    // session::impl.
    // The llama.cpp adapter is the language-model side of the workflow.
    llama_cpp_adapter & language_model;
    std::function<void()> load_image_engine;
    std::function<sd_ctx_t *()> image_engine;
    std::function<void(conditioning_slot)> transfer_prefix;
    std::function<void(const image_input &, int64_t)> append_latent_image;
    std::function<void(const image_input &)> append_vision_image;
};

class model_workflow {
public:
    virtual ~model_workflow();
    virtual model_family family() const = 0;
    // Format a prompt-only request for the model's tokenizer/chat template.
    virtual std::string text_prompt(const std::string & prompt) const = 0;
    // Answer a question conditioned on an input image.
    virtual std::string understand(workflow_context & context, const image_input & image,
                                    const std::string & prompt, int max_tokens, bool think);
    // Generate an image, optionally conditioned on an input image for editing.
    virtual image_result generate(workflow_context & context, const std::string & prompt,
                                  const image_options & options, const image_input * input) = 0;
    virtual interleave_result interleave(workflow_context & context, const std::string & prompt,
                                         const image_options & options, int max_text_tokens,
                                         int max_images);
};

// Construct the workflow matching the resolved model family.
std::unique_ptr<model_workflow> create_model_workflow(model_family family);

}
