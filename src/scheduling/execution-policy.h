#pragma once

#include "ggml-backend.h"

#include <functional>
#include <map>
#include <string>

namespace umm {

using execution_options = std::map<std::string, std::string>;
using graph_placement = std::function<void(ggml_backend_sched_t, ggml_cgraph *, bool allocated)>;

struct understanding_schedule {
    std::string secondary_backend;
    graph_placement place;
};

understanding_schedule configure_execution_policy(execution_options & options);
std::string register_remote_cann(const std::string & endpoint);
bool backend_uses_cann(const std::string & name);

}
