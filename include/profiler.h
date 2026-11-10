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
    std::string report;
};

// Parse the v1 request fields; unknown fields are rejected.
request parse_request(const nlohmann::json & value);
// Interface prototype: synthetic data only, with no model/device/workload access.
result profile(const request & input);
// Writes manifest.json, graphs.jsonl, trace.jsonl, costs.jsonl and report.md.
// The output directory must not already exist.
void write_result(const result & output, const std::string & directory);
}
