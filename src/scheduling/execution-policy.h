#pragma once

#include "placement.h"

#include <cstdint>
#include <map>
#include <string>

namespace umm {

using execution_options = std::map<std::string, std::string>;

struct prepared_placement {
    graph_placement place;
    std::function<bool()> recover;
    std::function<bool(const char *, int64_t, int64_t)> replan;
};

struct understanding_schedule {
    std::string secondary_backend;
    graph_placement place;
    std::function<prepared_placement(ggml_backend_sched_t, const std::vector<ggml_tensor *> &, const std::string &)> prepare;
};

struct execution_configuration {
    understanding_schedule schedule;
    bool strict_acceleration = false;
    std::string required_device;
    std::string rpc_cann_device;
};

execution_configuration configure_execution_policy(execution_options & options);
// Apply the resolved layout and accelerator guards before loading the model.
void apply_execution_environment(const execution_options & options, const execution_configuration & configuration);

}
