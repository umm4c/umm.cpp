#pragma once

#include "json.hpp"
#include <string>
#include <vector>

namespace umm::profiling {
struct device {
    std::string id;
    std::string transport; // local or rpc
    std::string backend;   // Backend device name, e.g. CANN0.
    std::string endpoint;  // Required for RPC; empty for local.
};

struct request {
    std::string model;
    std::vector<device> devices;
    std::string representative_workload;
};

struct result {
    nlohmann::json manifest;
    nlohmann::json graphs;
    nlohmann::json trace;
    nlohmann::json costs;
    nlohmann::json requests;
    nlohmann::json candidates, candidate_costs, snapshots;
    std::string report;
};

// Parse the v1 request fields; unknown fields are rejected.
request parse_request(const nlohmann::json & value);
// Runs structure, capture, and paired baseline/timing replays in a dedicated process.
result profile(const request & input);
// Writes the manifest, graph/event/cost/request and candidate/snapshot artifacts and report.
// The output directory must not already exist.
void write_result(const result & output, const std::string & directory);
}
