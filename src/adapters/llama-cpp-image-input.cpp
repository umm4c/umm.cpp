#include "llama-cpp-adapter.h"
#include "llama-cpp-prefill.h"

#include "inference/models/bagel-rules.h"
#include "inference/models/u1-rules.h"

#include "ggml-alloc.h"
#include "llama-context.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <stdexcept>

namespace umm {

using llama_prefill_detail::hash_prefill_input;
using llama_prefill_detail::record_prefill;

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
    const auto boundary = tokenize(bagel_rules::vision_boundaries);
    if (boundary.size() != 2)
        throw std::runtime_error("Missing BAGEL image boundary tokens");
    for (int i = 0; i < 2; ++i) {
        ggml_backend_tensor_get(table, row.data(), boundary[i] * row_bytes, row_bytes);
        float * dest = input.data() + (i ? n - 1 : 0) * dim;
        if (table->type == GGML_TYPE_F32)
            std::memcpy(dest, row.data(), row_bytes);
        else if (traits->to_float)
            traits->to_float(row.data(), dest, dim);
        else
            throw std::runtime_error("Unsupported token embedding type");
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
    const int status = decode_profiled(batch, "understanding_prefill");
    llama_set_causal_attn(context_.get(), true);
    if (status)
        throw std::runtime_error("Image understanding decode failed");
    const auto offset = tokens_.size();
    tokens_.resize(offset + n, LLAMA_TOKEN_NULL);
    tokens_[offset] = boundary.front();
    tokens_.back() = boundary.back();
    positions_.insert(positions_.end(), n, next_position_);
    next_position_ += bagel_rules::image_position_count;
    has_logits_ = true;
}

void llama_cpp_adapter::append_u1_image_embeddings(const std::vector<float> & embeddings, int grid_width,
                                                   int grid_height, bool need_logits, request_metrics * metrics) {
    const auto start = tokenize(u1_rules::image_start);
    if (start.size() != 1) {
        throw std::runtime_error("Missing SenseNova U1 image start token");
    }
    append(start, need_logits, metrics);
    append_u1_image_body(embeddings, grid_width, grid_height, true, need_logits, metrics);
}

void llama_cpp_adapter::append_u1_generated_image_embeddings(const std::vector<float> & embeddings, int grid_width,
                                                             int grid_height) {
    const auto start = tokenize(u1_rules::image_start);
    if (start.size() != 1 || tokens_.empty() || tokens_.back() != start.front()) {
        throw std::runtime_error("Generated image must follow an appended <img> token");
    }
    append_u1_image_body(embeddings, grid_width, grid_height);
}

size_t llama_cpp_adapter::available_u1_generated_image_tokens() const {
    if (family_ != model_family::sensenova_u1) {
        throw std::invalid_argument("Generated image embeddings require SenseNova U1");
    }
    const size_t context_size = llama_n_ctx(context_.get());
    if (tokens_.size() + 2 > context_size) {
        return 0;
    }
    return std::min<size_t>(llama_n_ubatch(context_.get()), context_size - tokens_.size() - 2);
}

void llama_cpp_adapter::append_u1_image_body(const std::vector<float> & embeddings, int grid_width, int grid_height,
                                             bool close_image, bool need_logits, request_metrics * metrics) {
    if (family_ != model_family::sensenova_u1) {
        throw std::invalid_argument("Spatial image embeddings are only supported by SenseNova U1");
    }
    const auto start_token = tokenize(u1_rules::image_start);
    if (start_token.size() != 1 || tokens_.empty() || tokens_.back() != start_token.front()) {
        throw std::runtime_error("Image embeddings must follow an appended <img> token");
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
        const auto axes = u1_rules::image_position(temporal, i, grid_width);
        for (size_t axis = 0; axis < axes.size(); ++axis) {
            positions[axis * image_tokens + i] = axes[axis];
        }
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
    if (metrics)
        synchronize();
    const auto start = metrics_clock::now();
    has_logits_ = false;
    const int status = decode_profiled(batch, "understanding_prefill");
    if (status) {
        throw std::runtime_error("SenseNova U1 image understanding decode failed");
    }
    if (metrics)
        synchronize();
    record_prefill(metrics, "image", tokens_.size(), image_tokens, 0, metrics_elapsed_ms(start));
    hash_prefill_input(metrics, batch);

    tokens_.insert(tokens_.end(), image_tokens, LLAMA_TOKEN_NULL);
    positions_.insert(positions_.end(), image_tokens, temporal);
    next_position_ = temporal + u1_rules::image_position_count;
    const auto end = tokenize(u1_rules::image_end);
    if (end.size() != 1) {
        throw std::runtime_error("Missing SenseNova U1 image end token");
    }
    if (close_image)
        append(end, need_logits, metrics);
}

// Logits and model tensor access -------------------------------------------

std::vector<float> llama_cpp_adapter::token_embeddings(const std::vector<llama_token> & tokens) {
    if (tokens.empty())
        return {};
    synchronize();
    auto * table = tensor("token_embd.weight");
    const auto device = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(table->buffer));
    auto * sched = context_->get_sched();
    ggml_backend_t backend = nullptr;
    for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
        auto * candidate = ggml_backend_sched_get_backend(sched, i);
        if (ggml_backend_get_device(candidate) == device)
            backend = candidate;
    }
    if (!backend)
        throw std::runtime_error("Token embedding backend is unavailable");
    if (std::getenv("GGML_SCHED_STRICT_ACCEL") && ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        throw std::runtime_error("Strict accelerator mode rejected CPU token embedding lookup");
    }
    ggml_init_params params{ggml_tensor_overhead() * 4 + ggml_graph_overhead_custom(4, false), nullptr, true};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), ggml_free);
    if (!ctx)
        throw std::runtime_error("Token embedding graph allocation failed");
    auto * ids = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, tokens.size());
    auto * rows = ggml_get_rows(ctx.get(), table, ids);
    if (!ggml_backend_supports_op(backend, rows)) {
        throw std::runtime_error("Token embedding lookup is unsupported on the weight backend");
    }
    auto * graph = ggml_new_graph_custom(ctx.get(), 4, false);
    ggml_build_forward_expand(graph, rows);
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors(ctx.get(), backend), ggml_backend_buffer_free);
    if (!buffer)
        throw std::runtime_error("Token embedding buffer allocation failed");
    ggml_backend_tensor_set(ids, tokens.data(), 0, tokens.size() * sizeof(llama_token));
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("Token embedding lookup failed");
    }
    std::vector<float> result(tokens.size() * table->ne[0]);
    ggml_backend_tensor_get(rows, result.data(), 0, result.size() * sizeof(float));
    return result;
}

void llama_cpp_adapter::append_u1_mixed(const std::vector<u1_prefill_segment> & segments, bool need_logits,
                                        request_metrics * metrics) {
    if (family_ != model_family::sensenova_u1) {
        throw std::invalid_argument("Mixed prefill requires SenseNova image-group attention");
    }
    const size_t dim = llama_model_n_embd(model_.get());
    const size_t limit = std::min(llama_n_batch(context_.get()), llama_n_ubatch(context_.get()));
    size_t total = 0;
    for (const auto & segment : segments) {
        if (segment.embeddings.empty()) {
            if (segment.grid_width || segment.grid_height)
                throw std::invalid_argument("Empty mixed prefill image");
            for (auto token : segment.tokens) {
                if (token < 0 || token >= vocab_size())
                    throw std::invalid_argument("Invalid mixed prefill token");
            }
            total += segment.tokens.size();
        } else {
            if (!segment.tokens.empty() || segment.grid_width <= 0 || segment.grid_height <= 0 ||
                segment.embeddings.size() != size_t(segment.grid_width) * segment.grid_height * dim ||
                segment.embeddings.size() / dim > limit) {
                throw std::invalid_argument("Invalid mixed prefill image group or batch capacity");
            }
            total += segment.embeddings.size() / dim;
        }
    }
    if (tokens_.size() + total > context_capacity())
        throw std::runtime_error("Understanding context is full");
    if (!total)
        return;
    std::vector<llama_token> ids;
    std::vector<std::array<llama_pos, 4>> axes;
    std::vector<float> input;
    size_t processed = 0;
    bool contains_image = false;
    auto flush = [&] {
        if (ids.empty())
            return;
        const auto start = metrics_clock::now();
        std::vector<llama_token> text_ids;
        for (auto id : ids)
            if (id != LLAMA_TOKEN_NULL)
                text_ids.push_back(id);
        const auto text_rows = token_embeddings(text_ids);
        size_t row = 0;
        for (size_t i = 0; i < ids.size(); ++i) {
            if (ids[i] != LLAMA_TOKEN_NULL) {
                std::copy_n(text_rows.data() + row++ * dim, dim, input.data() + i * dim);
            }
        }
        std::vector<llama_pos> positions(4 * ids.size());
        for (size_t i = 0; i < ids.size(); ++i) {
            for (int a = 0; a < 4; ++a)
                positions[a * ids.size() + i] = axes[i][a];
        }
        std::vector<int32_t> counts(ids.size(), 1);
        llama_seq_id seq = 0;
        std::vector<llama_seq_id *> sequences(ids.size(), &seq);
        std::vector<int8_t> outputs(ids.size(), 0);
        outputs.back() = need_logits && processed + ids.size() == total;
        llama_batch batch{};
        batch.n_tokens = ids.size();
        batch.embd = input.data();
        batch.pos = positions.data();
        batch.n_seq_id = counts.data();
        batch.seq_id = sequences.data();
        batch.logits = outputs.data();
        has_logits_ = false;
        if (decode_profiled(batch, "understanding_prefill") != 0)
            throw std::runtime_error("Mixed SenseNova prefill failed");
        if (metrics)
            synchronize();
        record_prefill(metrics, contains_image ? "mixed" : "text", tokens_.size(), ids.size(), outputs.back(),
                       metrics_elapsed_ms(start));
        batch.token = ids.data();
        hash_prefill_input(metrics, batch);
        tokens_.insert(tokens_.end(), ids.begin(), ids.end());
        positions_.insert(positions_.end(), positions.begin(), positions.begin() + ids.size());
        processed += ids.size();
        has_logits_ = outputs.back();
        ids.clear();
        axes.clear();
        input.clear();
        contains_image = false;
    };
    for (const auto & segment : segments) {
        if (segment.embeddings.empty()) {
            for (auto token : segment.tokens) {
                if (ids.size() == limit)
                    flush();
                ids.push_back(token);
                // SenseNova leaves text spatial axes at zero.
                axes.push_back(u1_rules::text_position(next_position_));
                input.resize(input.size() + dim, 0.0f);
                ++next_position_;
            }
        } else {
            const size_t n = segment.embeddings.size() / dim;
            if (ids.size() + n > limit)
                flush();
            const auto start_token = tokenize(u1_rules::image_start);
            const auto previous = ids.empty() ? (tokens_.empty() ? LLAMA_TOKEN_NULL : tokens_.back()) : ids.back();
            if (start_token.size() != 1 || previous != start_token.front()) {
                throw std::invalid_argument("Mixed image must follow <img>");
            }
            for (size_t i = 0; i < n; ++i) {
                ids.push_back(LLAMA_TOKEN_NULL);
                axes.push_back(u1_rules::image_position(next_position_, i, segment.grid_width));
            }
            input.insert(input.end(), segment.embeddings.begin(), segment.embeddings.end());
            next_position_ += u1_rules::image_position_count;
            contains_image = true;
        }
    }
    flush();
}

}
