#pragma once
#include "policy.h"
#include "scheduling/placement.h"

#include <cstdint>
#include <vector>

namespace umm {
struct random_plan {
    uint32_t seed;
    std::vector<int> cut_nodes;
    bool reverse; // First segment uses the second backend.
    std::vector<node_placement> placements;
};

class random_policy final : public execution_policy {
public:
    understanding_schedule apply(execution_options & options) const override;

    // Graph tensors must be unallocated except for inputs, weights and their views.
    static random_plan plan(ggml_cgraph * graph, ggml_backend_t first, ggml_backend_t second, uint32_t seed,
                            bool reverse, int cut_count = 1);
};

void assign_random_plan(ggml_backend_sched_t sched, const random_plan & plan);
void check_random_plan(ggml_backend_sched_t sched, const random_plan & plan);
}
