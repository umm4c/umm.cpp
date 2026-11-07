#pragma once
#include "ggml-backend.h"
#include <cstdint>
#include <vector>

namespace umm {
struct random_placement {
    ggml_tensor * node;
    ggml_backend_t backend;
};
struct random_plan {
    uint32_t seed;
    std::vector<int> cut_nodes;
    bool reverse; // First segment uses the second backend.
    std::vector<random_placement> placements;
};
// Graph tensors must be unallocated except for inputs/weights and their views.
random_plan make_random_plan(ggml_cgraph * graph, ggml_backend_t first,
                             ggml_backend_t second, uint32_t seed, bool reverse, int cut_count = 1);
void assign_random_plan(ggml_backend_sched_t sched, const random_plan & plan);
void check_random_plan(ggml_backend_sched_t sched, const random_plan & plan);
}
