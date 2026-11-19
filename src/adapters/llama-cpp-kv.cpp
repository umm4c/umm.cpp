#include "llama-cpp-adapter.h"

#include "ggml-alloc.h"
#include "llama-context.h"
#include "llama-kv-cache.h"

#include <stdexcept>

namespace umm {

// KV-prefix export/import ---------------------------------------------------

kv_prefix llama_cpp_adapter::export_prefix() const {
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
    // The device views and any packed V tensors live only through the host copy.
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(nullptr, ggml_backend_free);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> descriptors(nullptr, ggml_free);
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> storage(nullptr,
                                                                                      ggml_backend_buffer_free);
    std::vector<ggml_tensor *> device_keys;
    std::vector<ggml_tensor *> device_values;
    device_keys.reserve(layers.size());
    device_values.reserve(layers.size());
    const size_t max_nodes = 6 * layers.size() + 4;
    descriptors.reset(
        ggml_init({ggml_tensor_overhead() * max_nodes + ggml_graph_overhead_custom(max_nodes, false), nullptr, true}));
    if (!descriptors) {
        throw std::runtime_error("Could not allocate prefix descriptors");
    }
    llama_kv_cache::slot_info slots{};
    slots.s0 = slots.s1 = 0;
    ggml_cgraph * packing = nullptr;
    ggml_backend_dev_t packing_device = nullptr;
    for (uint32_t layer : layers) {
        auto * key = cache->get_k(descriptors.get(), layer, tokens_.size(), slots);
        auto * value = cache->get_v(descriptors.get(), layer, tokens_.size(), slots);
        ggml_backend_view_init(key);
        ggml_backend_view_init(value);
        if (value->nb[1] > value->nb[2]) {
            value = ggml_permute(descriptors.get(), value, 2, 1, 0, 3);
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
                packing = ggml_new_graph_custom(descriptors.get(), max_nodes, false);
                packing_device = device;
            } else if (packing_device != device) {
                throw std::runtime_error("Packing transposed values requires one cache device");
            }
            value = ggml_cont(descriptors.get(), value);
            ggml_build_forward_expand(packing, value);
        }
        device_keys.push_back(key);
        device_values.push_back(value);
    }
    if (packing) {
        // Ordinary attention transposes V. Pack it once at the phase boundary.
        backend.reset(ggml_backend_dev_init(packing_device, nullptr));
        if (!backend) {
            throw std::runtime_error("Could not initialize prefix packing backend");
        }
        storage.reset(ggml_backend_alloc_ctx_tensors(descriptors.get(), backend.get()));
        if (!storage || ggml_backend_graph_compute(backend.get(), packing) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Could not pack prefix values");
        }
        ggml_backend_synchronize(backend.get());
    }

    return kv_prefix::copy_to_host(tokens_, positions_, device_keys, device_values);
}

void llama_cpp_adapter::import_prefix(const kv_prefix & prefix) {
    const auto & tokens = prefix.tokens;
    const auto & positions = prefix.positions;
    if (tokens.size() != positions.size() || tokens.empty() || positions.back() >= INT32_MAX) {
        throw std::invalid_argument("Invalid external prefix metadata");
    }
    auto * cache = dynamic_cast<llama_kv_cache *>(context_->get_memory());
    if (!cache)
        throw std::runtime_error("Understanding context cannot import attention prefixes");
    reset();
    if (!cache->import_prefix(context_.get(), 0, positions, prefix.keys, prefix.values)) {
        throw std::runtime_error("Could not import attention prefix");
    }
    tokens_ = tokens;
    positions_ = positions;
    next_position_ = positions.back() + 1;
}

}
