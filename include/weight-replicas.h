#pragma once

#include "random-plan.h"
#include <memory>

namespace umm {
struct weight_residency {
    ggml_backend_t backend;
    size_t tensor_count;
    size_t logical_bytes;
    size_t allocated_bytes;
    size_t uploaded_bytes;
};

// Immutable model weights only. The model and backends must outlive this object.
class weight_replicas {
public:
    weight_replicas(const std::vector<ggml_tensor *> & weights, const std::vector<ggml_backend_t> & backends);
    ~weight_replicas();
    const std::vector<weight_residency> & residency() const;
    // Call once per new graph, after the previous graph has stopped executing.
    size_t bind(ggml_backend_sched_t sched, const random_plan & plan);
private:
    struct impl;
    std::unique_ptr<impl> impl_;
};
}
