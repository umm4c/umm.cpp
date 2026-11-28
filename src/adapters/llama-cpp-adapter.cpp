#include "inference/profile-context.h"
#include "llama-cpp-adapter.h"
#include "llama-cpp-prefill.h"
#include "scheduling/backend-devices.h"

#include "llama-context.h"
#include "llama-model.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace umm {

using llama_prefill_detail::hash_prefill_input;
using llama_prefill_detail::record_prefill;

// Model/context setup -------------------------------------------------------

llama_cpp_adapter::llama_cpp_adapter(const std::string & model_path, int context_size, int gpu_layers,
                                     bool full_precision, const std::string & backend, int batch_size,
                                     int microbatch_size, const std::string & secondary_backend) {
    if (backend.empty() && !secondary_backend.empty())
        throw std::invalid_argument("Secondary backend requires a primary backend");
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = gpu_layers;
    ggml_backend_dev_t devices[3]{};
    std::vector<float> primary_weights(llama_max_devices(), 0.f);
    primary_weights[0] = 1.f;
    if (!backend.empty()) {
        devices[0] = ggml_backend_dev_by_name(backend.c_str());
        if (!devices[0]) {
            throw std::runtime_error("Understanding backend was not found: " + backend);
        }
        mp.devices = devices;
        mp.split_mode = LLAMA_SPLIT_MODE_NONE;
        mp.main_gpu = 0;
        if (!secondary_backend.empty()) {
            devices[1] = ggml_backend_dev_by_name(secondary_backend.c_str());
            if (!devices[1] || devices[1] == devices[0])
                throw std::invalid_argument("Secondary backend must be a distinct registered device");
            // Keep immutable weights and KV on the primary device; scheduler copies remote inputs.
            mp.split_mode = LLAMA_SPLIT_MODE_LAYER;
            mp.tensor_split = primary_weights.data();
        }
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
    if (context_size < 0)
        throw std::invalid_argument("Context size must be nonnegative");
    if (batch_size < 0 || microbatch_size < 0)
        throw std::invalid_argument("Batch sizes must be nonnegative");
    const bool compact_bagel =
        family_ == model_family::bagel && llama_model_ftype(model_.get()) != LLAMA_FTYPE_MOSTLY_F16;
    cp.n_ctx = context_size ? context_size : descriptor.context_size;
    cp.n_batch = std::min<uint32_t>(cp.n_ctx, batch_size ? batch_size : (compact_bagel ? 512 : descriptor.batch_size));
    cp.n_ubatch =
        std::min<uint32_t>(cp.n_batch, microbatch_size ? microbatch_size : (compact_bagel ? 256 : cp.n_batch));
    cp.kv_v_contiguous = family_ == model_family::sensenova_u1;
    cp.n_seq_max = 1;
    cp.n_threads = 8;
    cp.n_threads_batch = 8;
    const bool cann = backend_uses_cann(backend);
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
    if (family_ == model_family::bagel) {
        set_causal_attention(true);
    }
    placement_error_ = {};
    tokens_.clear();
    positions_.clear();
    next_position_ = 0;
    has_logits_ = false;
}

std::vector<ggml_tensor *> llama_cpp_adapter::weights() const {
    std::vector<ggml_tensor *> result;
    for (const auto & item : model_->tensors_by_name)
        result.push_back(item.second);
    return result;
}

ggml_backend_sched_t llama_cpp_adapter::scheduler() const {
    return context_->get_sched();
}

void llama_cpp_adapter::set_placement(graph_placement placement) {
    placement_ = std::move(placement);
    apply_placement_callback();
}

void llama_cpp_adapter::set_causal_attention(bool enabled) {
    if (context_->get_cparams().causal_attn != enabled) {
        llama_set_causal_attn(context_.get(), enabled);
        rebind_placement_ = true;
    }
}

void llama_cpp_adapter::apply_placement_callback() {
    if (!placement_) {
        ggml_backend_sched_set_placement_callback(scheduler(), nullptr, nullptr);
        return;
    }
    ggml_backend_sched_set_placement_callback(
        scheduler(),
        [](ggml_backend_sched_t sched, ggml_cgraph * graph, bool allocated, void * user) {
            auto & adapter = *static_cast<llama_cpp_adapter *>(user);
            try {
                adapter.placement_(sched, graph, allocated);
                return true;
            } catch (...) {
                adapter.placement_error_ = std::current_exception();
                return false;
            }
        },
        this);
}

int llama_cpp_adapter::decode_profiled(llama_batch batch, const char * phase) {
    placement_error_ = {};
    if (rebind_placement_ && placement_) {
        // Reserve before decode so the placement callback is installed on the replacement scheduler.
        context_->sched_reserve();
        apply_placement_callback();
        rebind_placement_ = false;
    }
    profiling::phase_scope scope(
        std::string(profiling::current_phase.phase) == "understanding_decode" ? "understanding_decode" : phase,
        batch.n_tokens, tokens_.size());
    const int status = llama_decode(context_.get(), batch);
    if (placement_error_) {
        std::rethrow_exception(std::exchange(placement_error_, {}));
    }
    return status;
}

void llama_cpp_adapter::synchronize() {
    llama_synchronize(context_.get());
}

void llama_cpp_adapter::append(const std::vector<llama_token> & tokens, bool need_logits, request_metrics * metrics) {
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
        std::vector<llama_pos> positions(4 * n, 0);
        std::vector<int32_t> seq_counts(n, 1);
        llama_seq_id sequence = 0;
        std::vector<llama_seq_id *> seq_ids(n, &sequence);
        std::vector<int8_t> outputs(n, 0);
        for (int i = 0; i < n; ++i) {
            positions[i] = next_position_ + i;
        }
        outputs.back() = need_logits && (family_ != model_family::sensenova_u1 || offset + n == tokens.size());
        llama_batch batch{};
        batch.n_tokens = n;
        batch.token = const_cast<llama_token *>(tokens.data() + offset);
        batch.pos = positions.data();
        batch.n_seq_id = seq_counts.data();
        batch.seq_id = seq_ids.data();
        batch.logits = outputs.data();
        if (metrics)
            synchronize();
        const auto start = metrics_clock::now();
        has_logits_ = false;
        if (decode_profiled(batch, "understanding_prefill") != 0) {
            throw std::runtime_error("Understanding decode failed");
        }
        if (metrics)
            synchronize();
        record_prefill(metrics, "text", tokens_.size(), n, outputs.back(), metrics_elapsed_ms(start));
        hash_prefill_input(metrics, batch);
        tokens_.insert(tokens_.end(), tokens.begin() + offset, tokens.begin() + offset + n);
        positions_.insert(positions_.end(), positions.begin(), positions.begin() + n);
        next_position_ += n;
        has_logits_ = outputs.back() != 0;
        offset += n;
    }
}

// Image embedding insertion -------------------------------------------------

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

}
