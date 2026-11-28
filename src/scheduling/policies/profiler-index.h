#pragma once

#include "runtime-placement.h"
#include "ggml.h"
#include "json.hpp"

#include <filesystem>
#include <map>
#include <string>

namespace umm::runtime_placement {

struct measured_costs {
    std::vector<timing> ranges;
    std::vector<group> fused_groups;
    std::array<std::vector<double>, 2> submission_ms;
    std::array<std::string, 2> submission_scope;
    std::string logical_id;
};

class profiler_index {
public:
    explicit profiler_index(const std::filesystem::path & directory);
    const nlohmann::json & identity() const { return identity_; }
    const nlohmann::json & devices() const { return devices_; }
    bool has_submission(const std::string & phase, int64_t tokens, int64_t cached) const;
    static std::string signature(ggml_cgraph * graph);
    const measured_costs * find(const std::string & structural, const std::string & phase, int64_t tokens,
                                int64_t cached) const;
    const measured_costs * find(ggml_cgraph * graph, const std::string & phase, int64_t tokens, int64_t cached) const;

private:
    using key = std::tuple<std::string, int64_t, int64_t, std::string>;
    nlohmann::json identity_, devices_;
    std::map<key, measured_costs> costs_;
};

}
