#pragma once

#include "model-registry.h"
#include "llama.h"
#include "ggml-backend.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace umm {

// Non-owning views into llama.cpp's KV cache. The descriptor/storage handles
// keep the views alive until the next reset/decode or destruction.
struct prefix_view {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> descriptors{nullptr, ggml_free};
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> storage{nullptr, ggml_backend_buffer_free};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> host_descriptors{nullptr, ggml_free};
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> host_buffer{nullptr, ggml_backend_buffer_free};
    std::vector<ggml_tensor *> keys;
    std::vector<ggml_tensor *> values;
};

// Project-level adapter around llama.cpp. It owns the text model/context and
// exposes only the operations needed by UMM workflows.
class llama_cpp_adapter {
public:
    explicit llama_cpp_adapter(const std::string & model_path, int context_size = 0, int gpu_layers = 99,
                               bool full_precision = false, const std::string & backend = "",
                               int batch_size = 0, int microbatch_size = 0);
    std::vector<llama_token> tokenize(const std::string & text) const;
    std::string piece(llama_token token) const;
    bool is_end(llama_token token) const;
    void reset();
    void synchronize();
    void append(const std::vector<llama_token> & tokens);
    // BAGEL supplies a flat sequence of vision embeddings.
    void append_bagel_image_embeddings(const std::vector<float> & embeddings);
    // SenseNova U1 supplies spatial embeddings with a 2-D image grid.
    void append_u1_image_embeddings(const std::vector<float> & embeddings,
                                    int grid_width, int grid_height);
    // The <img> token is already present after an image-generation request.
    void append_u1_generated_image_embeddings(const std::vector<float> & embeddings,
                                              int grid_width, int grid_height);
    // Capacity for image embeddings before appending their start/end tokens.
    size_t available_u1_generated_image_tokens() const;
    llama_token greedy() const;
    const float * logits() const;
    int vocab_size() const;
    ggml_tensor * tensor(const char * name) const;
    const std::string & architecture() const { return architecture_; }
    model_family family() const { return family_; }
    prefix_view prefix() const;
    void import_prefix(const std::vector<llama_token> & tokens, const std::vector<llama_pos> & positions,
                       const std::vector<ggml_tensor *> & keys, const std::vector<ggml_tensor *> & values);
    const std::vector<llama_token> & tokens() const { return tokens_; }
    const std::vector<llama_pos> & positions() const { return positions_; }

private:
    std::string architecture_;
    model_family family_ = model_family::unknown;
    llama_model_kv_override overrides_[2]{};
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model_{nullptr, llama_model_free};
    std::unique_ptr<llama_context, decltype(&llama_free)> context_{nullptr, llama_free};
    std::vector<llama_token> tokens_;
    std::vector<llama_pos> positions_;
    llama_pos next_position_ = 0;
    bool has_logits_ = false;
    void append_u1_image_body(const std::vector<float> & embeddings,
                              int grid_width, int grid_height);
};

}
