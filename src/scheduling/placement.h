#pragma once

#include "ggml-backend.h"

#include <functional>
#include <vector>

namespace umm {

struct node_placement {
    ggml_tensor * node;
    ggml_backend_t backend;
};

using graph_placement = std::function<void(ggml_backend_sched_t, ggml_cgraph *, bool allocated)>;

bool is_metadata_node(const ggml_tensor * node);
ggml_backend_t find_scheduler_backend(ggml_backend_sched_t sched, ggml_backend_dev_t device);
// Apply before allocation, then verify the actual compute-node placement afterward.
void apply_placements(ggml_backend_sched_t sched, const std::vector<node_placement> & placements);
void verify_placements(ggml_backend_sched_t sched, const std::vector<node_placement> & placements);

}
