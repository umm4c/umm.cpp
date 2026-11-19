#include "placement.h"

#include <stdexcept>

namespace umm {

bool is_metadata_node(const ggml_tensor * node) {
    return node->op == GGML_OP_NONE || node->op == GGML_OP_VIEW || node->op == GGML_OP_RESHAPE ||
           node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_PERMUTE;
}

ggml_backend_t find_scheduler_backend(ggml_backend_sched_t sched, ggml_backend_dev_t device) {
    ggml_backend_t result = nullptr;
    for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
        const auto backend = ggml_backend_sched_get_backend(sched, i);
        if (ggml_backend_get_device(backend) == device)
            result = backend;
    }
    return result;
}

void apply_placements(ggml_backend_sched_t sched, const std::vector<node_placement> & placements) {
    if (!sched || placements.empty())
        throw std::invalid_argument("Invalid graph placement");
    for (const auto & placement : placements)
        ggml_backend_sched_set_tensor_backend(sched, placement.node, placement.backend);
}

void verify_placements(ggml_backend_sched_t sched, const std::vector<node_placement> & placements) {
    for (const auto & placement : placements) {
        if (!is_metadata_node(placement.node) &&
            ggml_backend_sched_get_tensor_backend(sched, placement.node) != placement.backend) {
            throw std::runtime_error("Scheduler placement differs from the requested placement");
        }
    }
}

}
