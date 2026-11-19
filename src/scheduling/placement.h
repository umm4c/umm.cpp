#pragma once

#include "ggml-backend.h"

#include <functional>

namespace umm {

struct node_placement {
    ggml_tensor * node;
    ggml_backend_t backend;
};

using graph_placement = std::function<void(ggml_backend_sched_t, ggml_cgraph *, bool allocated)>;

}
