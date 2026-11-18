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
void assign_random_plan(ggml_backend_sched_t sched, const random_plan & plan);
void check_random_plan(ggml_backend_sched_t sched, const random_plan & plan);
}
