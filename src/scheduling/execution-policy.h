#pragma once

#include "placement.h"

#include <map>
#include <string>

namespace umm {

using execution_options = std::map<std::string, std::string>;

struct understanding_schedule {
    std::string secondary_backend;
    graph_placement place;
};

understanding_schedule configure_execution_policy(execution_options & options);

}
