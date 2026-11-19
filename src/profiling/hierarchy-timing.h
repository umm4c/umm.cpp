#pragma once

#include "ggml-backend-timing.h"
#include "json.hpp"

#include <map>
#include <string>
#include <utility>
#include <vector>

namespace umm::profiling::detail {
class hierarchy_capture {
    struct range {
        int end;
        std::string label;
    };

    std::map<std::pair<std::string, std::string>, std::vector<range>> plans;
    const std::vector<range> * current = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_timing_finish_t finish_timing = nullptr;
    bool armed = false;

public:
    void begin(ggml_backend_t target, ggml_cgraph * graph, const std::string & graph_id, const std::string & level);
    nlohmann::json finish(ggml_backend_t target);
    void cancel();
};
}
