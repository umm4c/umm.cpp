#include "llama-cpp-adapter.h"

// Keep the dependency on llama's cache layout in this adapter only.
#include "llama-context.h"
#include "llama-model.h"
#include "llama-kv-cache.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace umm {

// Model/context setup -------------------------------------------------------

llama_cpp_adapter::llama_cpp_adapter(const std::string & model_path, int context_size, int gpu_layers,
                                     bool full_precision, const std::string & backend) {
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = gpu_layers;
    ggml_backend_dev_t devices[2]{};
    if (!backend.empty()) {
        devices[0] = ggml_backend_dev_by_name(backend.c_str());
        if (!devices[0]) {
            throw std::runtime_error("Understanding backend was not found: " + backend);
        }
        mp.devices = devices;
        mp.split_mode = LLAMA_SPLIT_MODE_NONE;
        mp.main_gpu = 0;
    }
    overrides_[0].tag = LLAMA_KV_OVERRIDE_TYPE_BOOL;
    std::strcpy(overrides_[0].key, "sensenova_u1.full_precision");
    overrides_[0].val_bool = full_precision;
    mp.kv_overrides = overrides_;
    model_.reset(llama_model_load_from_file(model_path.c_str(), mp));
    if (!model_) {
        throw std::runtime_error("Could not load understanding GGUF");
    }
    char architecture[64]{};
    llama_model_meta_val_str(model_.get(), "general.architecture", architecture, sizeof(architecture));
    architecture_ = architecture;
    char branch[64]{};
    llama_model_meta_val_str(model_.get(), "bagel.branch", branch, sizeof(branch));
    if (architecture_ == "qwen3" && std::string(branch) == "understanding") {
        architecture_ = "bagel";
    }
    family_ = model_family_for(architecture_);
    if (family_ == model_family::unknown) {
        throw std::runtime_error("Expected a supported understanding GGUF");
    }
    const auto & descriptor = model_descriptor_for(family_);
    auto cp = llama_context_default_params();
    if (context_size < 0) throw std::invalid_argument("Context size must be nonnegative");
    cp.n_ctx = context_size ? context_size : descriptor.context_size;
    cp.n_batch = std::min<uint32_t>(cp.n_ctx, descriptor.batch_size);
    cp.n_ubatch = cp.n_batch;
    cp.n_seq_max = 1;
    cp.n_threads = 8;
    cp.n_threads_batch = 8;
    const bool cann = backend.rfind("CANN", 0) == 0 || backend.rfind("cann", 0) == 0;
    cp.flash_attn_type = full_precision || cann ? LLAMA_FLASH_ATTN_TYPE_DISABLED : LLAMA_FLASH_ATTN_TYPE_ENABLED;
    const auto cache_type = llama_model_ftype(model_.get()) == LLAMA_FTYPE_MOSTLY_BF16 ? GGML_TYPE_BF16 : GGML_TYPE_F16;
    cp.type_k = cp.type_v = full_precision ? GGML_TYPE_F32 : cache_type;
    context_.reset(llama_init_from_model(model_.get(), cp));
    if (!context_) {
        throw std::runtime_error("Could not create understanding context");
    }
}

// Token and text decoding ---------------------------------------------------

std::vector<llama_token> llama_cpp_adapter::tokenize(const std::string & text) const {
    const auto * vocab = llama_model_get_vocab(model_.get());
    int count = llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, false, true);
    if (count >= 0) {
        return {};
    }
    std::vector<llama_token> result(-count);
    count = llama_tokenize(vocab, text.data(), text.size(), result.data(), result.size(), false, true);
    if (count < 0) {
        throw std::runtime_error("Tokenization failed");
    }
    result.resize(count);
    return result;
}

std::string llama_cpp_adapter::piece(llama_token token) const {
    const auto * vocab = llama_model_get_vocab(model_.get());
    std::string result(32, '\0');
    int size = llama_token_to_piece(vocab, token, result.data(), result.size(), 0, true);
    if (size < 0) {
        result.resize(-size);
        size = llama_token_to_piece(vocab, token, result.data(), result.size(), 0, true);
    }
    if (size < 0) {
        throw std::runtime_error("Token decoding failed");
    }
    result.resize(size);
    return result;
}

bool llama_cpp_adapter::is_end(llama_token token) const {
    return llama_vocab_is_eog(llama_model_get_vocab(model_.get()), token);
}

void llama_cpp_adapter::reset() {
    llama_memory_clear(llama_get_memory(context_.get()), true);
    tokens_.clear();
    positions_.clear();
    next_position_ = 0;
    has_logits_ = false;
}

void llama_cpp_adapter::append(const std::vector<llama_token> & tokens) {
    if (tokens_.size() + tokens.size() > llama_n_ctx(context_.get())) {
        throw std::runtime_error("Understanding context is full");
    }
    if (tokens.empty()) {
        return;
    }
    const auto & descriptor = model_descriptor_for(family_);
    const auto batch_limit = std::min<uint32_t>(llama_n_batch(context_.get()), descriptor.batch_size);
    for (size_t offset = 0; offset < tokens.size();) {
        const int n = std::min<size_t>(batch_limit, tokens.size() - offset);
        std::vector<llama_pos> positions(4*n, 0);
        std::vector<int32_t> seq_counts(n, 1);
        llama_seq_id sequence = 0;
        std::vector<llama_seq_id *> seq_ids(n, &sequence);
        std::vector<int8_t> outputs(n, 0);
        for (int i = 0; i < n; ++i) {
            positions[i] = next_position_ + i;
        }
        outputs.back() = 1;
        llama_batch batch{};
        batch.n_tokens = n;
        batch.token = const_cast<llama_token *>(tokens.data() + offset);
        batch.pos = positions.data();
        batch.n_seq_id = seq_counts.data();
        batch.seq_id = seq_ids.data();
        batch.logits = outputs.data();
        if (llama_decode(context_.get(), batch) != 0) {
            throw std::runtime_error("Understanding decode failed");
        }
        tokens_.insert(tokens_.end(), tokens.begin() + offset, tokens.begin() + offset + n);
        positions_.insert(positions_.end(), positions.begin(), positions.begin() + n);
        next_position_ += n;
        has_logits_ = true;
        offset += n;
    }
}

// Image embedding insertion -------------------------------------------------

void llama_cpp_adapter::append_bagel_image_embeddings(const std::vector<float> & embeddings) {
    const int dim = llama_model_n_embd(model_.get());
    if (family_ != model_family::bagel || embeddings.empty() || embeddings.size() % dim) {
        throw std::invalid_argument("Expected BAGEL vision embeddings");
    }
    const size_t n = embeddings.size() / dim + 2;
    if (n > llama_n_ubatch(context_.get()) || tokens_.size() + n > llama_n_ctx(context_.get())) {
        throw std::runtime_error("Image group exceeds the understanding context or batch capacity");
    }
    std::vector<float> input(n * dim);
    std::copy(embeddings.begin(), embeddings.end(), input.begin() + dim);
    auto * table = tensor("token_embd.weight");
    const auto * traits = ggml_get_type_traits(table->type);
    const auto row_bytes = ggml_row_size(table->type, dim);
    std::vector<uint8_t> row(row_bytes);
    const auto boundary = tokenize("<|vision_start|><|vision_end|>");
    if (boundary.size() != 2) throw std::runtime_error("Missing BAGEL image boundary tokens");
    for (int i = 0; i < 2; ++i) {
        ggml_backend_tensor_get(table, row.data(), boundary[i] * row_bytes, row_bytes);
        float * dest = input.data() + (i ? n - 1 : 0) * dim;
        if (table->type == GGML_TYPE_F32) std::memcpy(dest, row.data(), row_bytes);
        else if (traits->to_float) traits->to_float(row.data(), dest, dim);
        else throw std::runtime_error("Unsupported token embedding type");
    }
    std::vector<llama_pos> positions(n, next_position_);
    std::vector<int32_t> seq_counts(n, 1);
    llama_seq_id sequence = 0;
    std::vector<llama_seq_id *> seq_ids(n, &sequence);
    std::vector<int8_t> outputs(n, 0);
    outputs.back() = 1;
    llama_batch batch{};
    batch.n_tokens = n;
    batch.embd = input.data();
    batch.pos = positions.data();
    batch.n_seq_id = seq_counts.data();
    batch.seq_id = seq_ids.data();
    batch.logits = outputs.data();
    llama_set_causal_attn(context_.get(), false);
    const int status = llama_decode(context_.get(), batch);
    llama_set_causal_attn(context_.get(), true);
    if (status) throw std::runtime_error("Image understanding decode failed");
    const auto offset = tokens_.size();
    tokens_.resize(offset + n, LLAMA_TOKEN_NULL);
    tokens_[offset] = boundary.front();
    tokens_.back() = boundary.back();
    positions_.insert(positions_.end(), n, next_position_++);
    has_logits_ = true;
}

void llama_cpp_adapter::append_u1_image_embeddings(const std::vector<float> & embeddings,
                                                   int grid_width, int grid_height) {
    const auto start = tokenize("<img>");
    if (start.size() != 1) {
        throw std::runtime_error("Missing SenseNova U1 image start token");
    }
    append(start);
    append_u1_image_body(embeddings, grid_width, grid_height);
}

void llama_cpp_adapter::append_u1_generated_image_embeddings(const std::vector<float> & embeddings,
                                                             int grid_width, int grid_height) {
    const auto start = tokenize("<img>");
    if (start.size() != 1 || tokens_.empty() || tokens_.back() != start.front()) {
        throw std::runtime_error("Generated image must follow an appended <img> token");
    }
    append_u1_image_body(embeddings, grid_width, grid_height);
}

void llama_cpp_adapter::append_u1_image_body(const std::vector<float> & embeddings,
                                             int grid_width, int grid_height) {
    if (family_ != model_family::sensenova_u1) {
        throw std::invalid_argument("Spatial image embeddings are only supported by SenseNova U1");
    }
    const int dim = llama_model_n_embd(model_.get());
    if (grid_width <= 0 || grid_height <= 0 || embeddings.empty() || embeddings.size() % dim ||
        size_t(grid_width) * grid_height != embeddings.size() / dim) {
        throw std::invalid_argument("Invalid SenseNova U1 image embeddings or grid");
    }
    const size_t image_tokens = embeddings.size() / dim;
    if (image_tokens > llama_n_ubatch(context_.get()) ||
        tokens_.size() + image_tokens + 1 > llama_n_ctx(context_.get())) {
        throw std::runtime_error("Image group exceeds the SenseNova U1 context or batch capacity");
    }
    const llama_pos temporal = next_position_;
    std::vector<llama_pos> positions(image_tokens * 4, 0);
    for (size_t i = 0; i < image_tokens; ++i) {
        positions[i] = temporal;
        positions[image_tokens + i] = static_cast<llama_pos>(i / grid_width);
        positions[2 * image_tokens + i] = static_cast<llama_pos>(i % grid_width);
    }
    std::vector<int32_t> seq_counts(image_tokens, 1);
    llama_seq_id sequence = 0;
    std::vector<llama_seq_id *> seq_ids(image_tokens, &sequence);
    std::vector<int8_t> outputs(image_tokens, 0);
    llama_batch batch{};
    batch.n_tokens = image_tokens;
    batch.embd = const_cast<float *>(embeddings.data());
    batch.pos = positions.data();
    batch.n_seq_id = seq_counts.data();
    batch.seq_id = seq_ids.data();
    batch.logits = outputs.data();
    llama_set_causal_attn(context_.get(), false);
    const int status = llama_decode(context_.get(), batch);
    llama_set_causal_attn(context_.get(), true);
    if (status) {
        throw std::runtime_error("SenseNova U1 image understanding decode failed");
    }

    tokens_.insert(tokens_.end(), image_tokens, LLAMA_TOKEN_NULL);
    positions_.insert(positions_.end(), image_tokens, temporal);
    next_position_ = temporal + 1;
    const auto end = tokenize("</img>");
    if (end.size() != 1) {
        throw std::runtime_error("Missing SenseNova U1 image end token");
    }
    append(end);
}

// Logits and model tensor access -------------------------------------------

const float * llama_cpp_adapter::logits() const {
    if (!has_logits_) {
        throw std::runtime_error("Prefill the context before reading logits");
    }
    return llama_get_logits_ith(context_.get(), -1);
}

int llama_cpp_adapter::vocab_size() const {
    return llama_vocab_n_tokens(llama_model_get_vocab(model_.get()));
}

ggml_tensor * llama_cpp_adapter::tensor(const char * name) const {
    auto * tensor = model_->get_tensor(name);
    if (!tensor) {
        throw std::runtime_error(std::string("Missing understanding tensor: ") + name);
    }
    return const_cast<ggml_tensor *>(tensor);
}

llama_token llama_cpp_adapter::greedy() const {
    const float * scores = logits();
    return std::max_element(scores, scores + vocab_size()) - scores;
}

// KV-prefix export/import ---------------------------------------------------

prefix_view llama_cpp_adapter::prefix() const {
    if (tokens_.empty()) {
        return {};
    }
    context_->synchronize();
    auto * cache = dynamic_cast<llama_kv_cache *>(context_->get_memory());
    if (!cache || tokens_.empty()) {
        throw std::runtime_error("No exportable attention prefix");
    }
    const auto & cells = cache->get_cells(0);
    for (size_t i = 0; i < tokens_.size(); ++i) {
        if (!cells.seq_has(i, 0) || cells.pos_get(i) != positions_[i]) {
            throw std::runtime_error("Prefix export requires an unshifted, contiguous sequence");
        }
    }
    const auto layers = cache->get_layer_ids();
    prefix_view result;
    const size_t max_nodes = 6*layers.size() + 4;
    result.descriptors.reset(ggml_init({ggml_tensor_overhead()*max_nodes + ggml_graph_overhead_custom(max_nodes, false), nullptr, true}));
    if (!result.descriptors) {
        throw std::runtime_error("Could not allocate prefix descriptors");
    }
    llama_kv_cache::slot_info slots{};
    slots.s0 = slots.s1 = 0;
    ggml_cgraph * packing = nullptr;
    ggml_backend_dev_t packing_device = nullptr;
    for (uint32_t layer : layers) {
        auto * key = cache->get_k(result.descriptors.get(), layer, tokens_.size(), slots);
        auto * value = cache->get_v(result.descriptors.get(), layer, tokens_.size(), slots);
        ggml_backend_view_init(key);
        ggml_backend_view_init(value);
        if (value->nb[1] > value->nb[2]) {
            value = ggml_permute(result.descriptors.get(), value, 2, 1, 0, 3);
            ggml_backend_view_init(value);
        }
        if (!ggml_are_same_shape(key, value) || !ggml_is_contiguous(key)) {
            throw std::runtime_error("Unsupported prefix cache layout");
        }
        if (!ggml_is_contiguous(value)) {
            auto device = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(value->buffer));
            if (!device && ggml_backend_buffer_is_host(value->buffer)) {
                device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
            }
            if (!device) {
                throw std::runtime_error("Could not identify prefix packing device");
            }
            if (!packing) {
                packing = ggml_new_graph_custom(result.descriptors.get(), max_nodes, false);
                packing_device = device;
            } else if (packing_device != device) {
                throw std::runtime_error("Packing transposed values requires one cache device");
            }
            value = ggml_cont(result.descriptors.get(), value);
            ggml_build_forward_expand(packing, value);
        }
        result.keys.push_back(key);
        result.values.push_back(value);
    }
    if (packing) {
        // Ordinary attention transposes V. Pack it once at the phase boundary.
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(
                ggml_backend_dev_init(packing_device, nullptr), ggml_backend_free);
        if (!backend) {
            throw std::runtime_error("Could not initialize prefix packing backend");
        }
        result.storage.reset(ggml_backend_alloc_ctx_tensors(result.descriptors.get(), backend.get()));
        if (!result.storage || ggml_backend_graph_compute(backend.get(), packing) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Could not pack prefix values");
        }
        ggml_backend_synchronize(backend.get());
    }

    const auto device_keys = std::move(result.keys);
    const auto device_values = std::move(result.values);
    result.host_storage.reserve(2*layers.size());
    result.keys.reserve(layers.size());
    result.values.reserve(layers.size());
    auto stage_to_host = [&](ggml_tensor * source) {
        auto * host = ggml_dup_tensor(result.descriptors.get(), source);
        result.host_storage.emplace_back(ggml_nbytes(source));
        auto & bytes = result.host_storage.back();
        ggml_backend_tensor_get(source, bytes.data(), 0, bytes.size());
        host->data = bytes.data();
        return host;
    };
    for (size_t i = 0; i < layers.size(); ++i) {
        result.keys.push_back(stage_to_host(device_keys[i]));
        result.values.push_back(stage_to_host(device_values[i]));
    }
    return result;
}

void llama_cpp_adapter::import_prefix(const std::vector<llama_token> & tokens, const std::vector<llama_pos> & positions,
                                  const std::vector<ggml_tensor *> & keys, const std::vector<ggml_tensor *> & values) {
    if (tokens.size() != positions.size() || tokens.empty() || positions.back() >= INT32_MAX) {
        throw std::invalid_argument("Invalid external prefix metadata");
    }
    auto * cache = dynamic_cast<llama_kv_cache *>(context_->get_memory());
    if (!cache) throw std::runtime_error("Understanding context cannot import attention prefixes");
    reset();
    if (!cache->import_prefix(context_.get(), 0, positions, keys, values)) {
        throw std::runtime_error("Could not import attention prefix");
    }
    tokens_ = tokens;
    positions_ = positions;
    next_position_ = positions.back() + 1;
}

}
