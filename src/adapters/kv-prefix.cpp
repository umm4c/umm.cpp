#include "kv-prefix.h"

#include <stdexcept>
#include <utility>

namespace umm {

kv_prefix kv_prefix::copy_to_host(std::vector<int32_t> tokens, std::vector<int32_t> positions,
                                  const std::vector<ggml_tensor *> & keys,
                                  const std::vector<ggml_tensor *> & values) {
    if (keys.empty() || keys.size() != values.size()) {
        throw std::invalid_argument("Invalid prefix tensors");
    }
    if (tokens.size() != positions.size()) {
        throw std::invalid_argument("Invalid prefix metadata");
    }
    kv_prefix result;
    result.tokens = std::move(tokens);
    result.positions = std::move(positions);
    result.descriptors.reset(ggml_init({2*keys.size()*ggml_tensor_overhead(), nullptr, true}));
    if (!result.descriptors) {
        throw std::runtime_error("Could not allocate prefix staging descriptors");
    }
    result.keys.reserve(keys.size());
    result.values.reserve(values.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        if (!keys[i] || !values[i] || !keys[i]->buffer || !values[i]->buffer ||
            !ggml_is_contiguous(keys[i]) || !ggml_is_contiguous(values[i])) {
            throw std::runtime_error("External prefix tensors must be contiguous backend tensors");
        }
        result.keys.push_back(ggml_dup_tensor(result.descriptors.get(), keys[i]));
        result.values.push_back(ggml_dup_tensor(result.descriptors.get(), values[i]));
    }
    result.storage.reset(ggml_backend_alloc_ctx_tensors_from_buft(
        result.descriptors.get(), ggml_backend_cpu_buffer_type()));
    if (!result.storage) {
        throw std::runtime_error("Could not allocate prefix staging buffer");
    }
    for (size_t i = 0; i < keys.size(); ++i) {
        for (const auto pair : {std::pair{keys[i], result.keys[i]},
                                std::pair{values[i], result.values[i]}}) {
            std::vector<uint8_t> bytes(ggml_nbytes(pair.first));
            ggml_backend_tensor_get(pair.first, bytes.data(), 0, bytes.size());
            ggml_backend_tensor_set(pair.second, bytes.data(), 0, bytes.size());
        }
    }
    return result;
}

}
