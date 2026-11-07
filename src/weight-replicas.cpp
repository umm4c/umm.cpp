#include "weight-replicas.h"
#include "ggml-alloc.h"
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace umm {
namespace {
using context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
struct device_weights {
    ggml_backend_t backend;
    context tensors{nullptr, ggml_free};
    buffer storage{nullptr, ggml_backend_buffer_free};
    std::unordered_map<ggml_tensor *, ggml_tensor *> copies;
    std::unordered_map<ggml_tensor *, ggml_tensor *> views;
};
bool metadata(const ggml_tensor * t) {
    return t->op == GGML_OP_NONE || t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE ||
           t->op == GGML_OP_TRANSPOSE || t->op == GGML_OP_PERMUTE;
}
}
struct weight_replicas::impl {
    std::vector<device_weights> devices;
    std::vector<weight_residency> metrics;
    std::unordered_map<ggml_tensor *, ggml_tensor *> originals;
    context views{nullptr, ggml_free};

    ggml_tensor * resolve(ggml_tensor * tensor, device_weights & device) {
        auto root = tensor;
        size_t offset = 0;
        while (root->view_src) {
            offset += root->view_offs;
            root = root->view_src;
        }
        auto original = originals.find(root);
        if (original == originals.end()) return tensor;
        auto copy = device.copies.at(original->second);
        if (root == tensor) return copy;
        if (root == copy) return tensor;
        auto cached = device.views.find(tensor);
        if (cached != device.views.end()) return cached->second;
        if (offset > ggml_nbytes(copy) || ggml_nbytes(tensor) > ggml_nbytes(copy) - offset)
            throw std::runtime_error("Weight view exceeds its replica");
        auto view = ggml_view_4d(views.get(), copy, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
                                 tensor->nb[1], tensor->nb[2], tensor->nb[3], offset);
        std::memcpy(view->nb, tensor->nb, sizeof(view->nb));
        ggml_set_name(view, tensor->name);
        if (ggml_backend_view_init(view) != GGML_STATUS_SUCCESS) throw std::runtime_error("Cannot initialize replicated weight view");
        device.views.emplace(tensor, view);
        return view;
    }
};

weight_replicas::weight_replicas(const std::vector<ggml_tensor *> & weights, const std::vector<ggml_backend_t> & backends)
    : impl_(new impl) {
    if (weights.empty() || backends.empty()) throw std::invalid_argument("Weight replication requires weights and devices");
    std::vector<ggml_tensor *> roots;
    for (auto tensor : weights) {
        while (tensor->view_src) tensor = tensor->view_src;
        if (impl_->originals.emplace(tensor, tensor).second) {
            if (!tensor->buffer || ggml_backend_buffer_get_usage(tensor->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS || !ggml_is_contiguous(tensor))
                throw std::invalid_argument("Weight replication requires allocated contiguous model weights");
            roots.push_back(tensor);
        }
    }
    std::unordered_set<ggml_backend_dev_t> seen;
    for (auto backend : backends) {
        if (!backend || !seen.insert(ggml_backend_get_device(backend)).second) throw std::invalid_argument("Weight replica devices must be distinct");
        device_weights device;
        device.backend = backend;
        device.tensors.reset(ggml_init({(roots.size() + 1) * ggml_tensor_overhead(), nullptr, true}));
        if (!device.tensors) throw std::runtime_error("Cannot create weight replica descriptors");
        weight_residency metric{backend, roots.size(), 0, 0, 0};
        std::unordered_set<ggml_backend_buffer_t> reused_buffers;
        for (auto original : roots) {
            metric.logical_bytes += ggml_nbytes(original);
            auto owner = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(original->buffer));
            if (owner == ggml_backend_get_device(backend) && ggml_backend_supports_buft(backend, ggml_backend_buffer_get_type(original->buffer))) {
                device.copies.emplace(original, original);
                reused_buffers.insert(original->buffer);
            } else {
                auto copy = ggml_dup_tensor(device.tensors.get(), original);
                ggml_set_name(copy, original->name);
                device.copies.emplace(original, copy);
                impl_->originals.emplace(copy, original);
                metric.uploaded_bytes += ggml_nbytes(original);
            }
        }
        if (metric.uploaded_bytes) {
            device.storage.reset(ggml_backend_alloc_ctx_tensors(device.tensors.get(), backend));
            if (!device.storage) throw std::runtime_error("Cannot allocate persistent weight replicas");
            ggml_backend_buffer_set_usage(device.storage.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            metric.allocated_bytes += ggml_backend_buffer_get_size(device.storage.get());
            for (auto original : roots) {
                auto copy = device.copies.at(original);
                if (copy != original) ggml_backend_tensor_copy(original, copy);
            }
            ggml_backend_synchronize(backend);
        }
        for (auto memory : reused_buffers) metric.allocated_bytes += ggml_backend_buffer_get_size(memory);
        impl_->metrics.push_back(metric);
        impl_->devices.push_back(std::move(device));
    }
}
weight_replicas::~weight_replicas() = default;
const std::vector<weight_residency> & weight_replicas::residency() const { return impl_->metrics; }

size_t weight_replicas::bind(ggml_backend_sched_t sched, const random_plan & plan) {
    for (auto & device : impl_->devices) device.views.clear();
    impl_->views.reset(ggml_init({(plan.placements.size() * GGML_MAX_SRC + 1) * ggml_tensor_overhead(), nullptr, true}));
    if (!impl_->views) throw std::runtime_error("Cannot create weight view descriptors");
    size_t bindings = 0;
    for (const auto & placement : plan.placements) {
        auto node = placement.node;
        if (metadata(node)) continue;
        device_weights * target = nullptr;
        for (auto & device : impl_->devices) if (device.backend == placement.backend) target = &device;
        if (!target) throw std::runtime_error("No weight replicas for planned backend");
        for (auto & source : node->src) {
            if (!source) continue;
            auto replica = impl_->resolve(source, *target);
            bindings += replica != source;
            source = replica;
            auto root = source;
            while (root->view_src) root = root->view_src;
            if (impl_->originals.count(root)) {
                ggml_backend_sched_set_tensor_backend(sched, source, placement.backend);
                ggml_backend_sched_set_tensor_backend(sched, root, placement.backend);
            }
        }
        if (!ggml_backend_supports_op(placement.backend, node)) throw std::runtime_error("Backend does not support replicated weight layout");
    }
    return bindings;
}
}
