#include "profiler.h"
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>

namespace umm::profiling {
using json = nlohmann::json;
namespace {
void fields(const json & value, const std::set<std::string> & allowed) {
    if (!value.is_object()) throw std::invalid_argument("Expected a JSON object");
    for (auto it = value.begin(); it != value.end(); ++it) {
        if (!allowed.count(it.key())) throw std::invalid_argument("Unknown profiler field: " + it.key());
    }
}
void validate(const request & input) {
    if (input.model.empty() || input.representative_workload.empty() || input.devices.empty()) {
        throw std::invalid_argument("model, devices and representative_workload must be nonempty");
    }
    std::set<std::string> ids;
    for (const auto & d : input.devices) {
        if (d.id.empty() || !ids.insert(d.id).second || d.backend.empty()) {
            throw std::invalid_argument("Device IDs must be nonempty and unique; backend is required");
        }
        if (d.transport != "local" && d.transport != "rpc") throw std::invalid_argument("transport must be local or rpc");
        if ((d.transport == "rpc") == d.endpoint.empty()) throw std::invalid_argument("Only RPC devices require an endpoint");
    }
}
void write(const std::filesystem::path & path, const std::string & text) {
    std::ofstream stream(path);
    stream.exceptions(std::ios::failbit | std::ios::badbit);
    stream << text;
    stream.close();
}
}
request parse_request(const json & value) {
    fields(value, {"model", "devices", "representative_workload"});
    request input;
    input.model = value.at("model").get<std::string>();
    input.representative_workload = value.at("representative_workload").get<std::string>();
    if (!value.at("devices").is_array()) throw std::invalid_argument("devices must be an array");
    for (const auto & d : value.at("devices")) {
        fields(d, {"id", "transport", "backend", "endpoint"});
        input.devices.push_back({d.at("id").get<std::string>(), d.at("transport").get<std::string>(),
                                 d.at("backend").get<std::string>(), d.value("endpoint", std::string{})});
    }
    validate(input);
    return input;
}
result profile(const request & input) {
    validate(input);
    result out;
    json devices = json::array();
    for (const auto & d : input.devices) devices.push_back({{"id", d.id}, {"transport", d.transport}, {"backend", d.backend}, {"endpoint", d.endpoint}});
    out.manifest = {{"schema_version", "umm-profiler/v1"}, {"data_origin", "mock"},
        {"status", "mock"}, {"executed", false}, {"decision_usable", false},
        {"request", {{"model", input.model}, {"devices", devices}, {"representative_workload", input.representative_workload}}},
        {"identity", {{"model_hash", nullptr}, {"environment", nullptr}}},
        {"coverage", {{"workload_loaded", false}, {"device_probed", false}, {"correctness", "not_checked"}}},
        {"artifacts", {"graphs.jsonl", "trace.jsonl", "costs.jsonl", "report.md"}}};
    // This fixture is independent of the requested model and workload.
    out.graphs = json::array({{{"schema_version", "umm-profiler-graph/v1"}, {"data_origin", "mock"},
        {"graph_id", "mock-add-f32-4"}, {"phase", "synthetic"},
        {"shape_bucket", {{"kind", "synthetic"}, {"input_tokens", nullptr}, {"kv_tokens", nullptr}}},
        {"tensors", json::array({
            {{"id", "a"}, {"role", "activation"}, {"shape", {4}}, {"dtype", "f32"}, {"layout", "contiguous"}, {"bytes", 16}, {"alias_of", nullptr}},
            {{"id", "b"}, {"role", "activation"}, {"shape", {4}}, {"dtype", "f32"}, {"layout", "contiguous"}, {"bytes", 16}, {"alias_of", nullptr}},
            {{"id", "c"}, {"role", "activation"}, {"shape", {4}}, {"dtype", "f32"}, {"layout", "contiguous"}, {"bytes", 16}, {"alias_of", nullptr}}})},
        {"nodes", json::array({{{"id", "add"}, {"op", "ADD"}, {"op_params", json::object()},
            {"inputs", {"a", "b"}}, {"outputs", {"c"}}, {"dependencies", json::array()},
            {"writes_inputs", false}, {"persistent_state", json::array()}}})},
        {"cut_boundaries", json::array()}, {"legality", "not_checked"}}});
    out.trace = json::array();
    out.costs = json::array();
    for (const auto & d : input.devices) {
        out.trace.push_back({{"schema_version", "umm-profiler-event/v1"}, {"data_origin", "mock"},
            {"graph_id", "mock-add-f32-4"}, {"event_id", "mock-" + d.id}, {"node_ids", {"add"}},
            {"split_id", "mock-split"}, {"device_id", d.id}, {"kind", "compute"},
            {"start_ms", 0.0}, {"duration_ms", 1.0}, {"executed", false}});
        out.costs.push_back({{"schema_version", "umm-profiler-cost/v1"}, {"data_origin", "mock"},
            {"graph_id", "mock-add-f32-4"}, {"device_id", d.id}, {"scope", "graph"},
            {"state", "synthetic"}, {"sample_count", 0}, {"duration_ms", {{"mean", 1.0}, {"p50", 1.0}, {"p95", 1.0}}},
            {"startup_ms", nullptr}, {"memory_peak_bytes", nullptr}, {"residency", "unknown"},
            {"transfer", nullptr}, {"overlap", nullptr}, {"measurement_valid", false}});
    }
    out.report = "# Profiler interface prototype\n\nMOCK DATA - not usable for scheduling decisions.\n\n"
        "No model or workload was loaded, no device was contacted, and no computation was executed.\n"
        "The synthetic ADD graph and 1 ms values are interface fixtures, not measurements of the requested model.\n"
        "Sample count is zero. Null fields indicate unavailable measurements, not zero cost.\n";
    return out;
}
void write_result(const result & output, const std::string & directory) {
    const std::filesystem::path root(directory);
    if (!std::filesystem::create_directory(root)) throw std::invalid_argument("Output directory already exists");
    write(root / "manifest.json", output.manifest.dump(2) + "\n");
    for (const auto & item : {std::make_pair("graphs.jsonl", &output.graphs), std::make_pair("trace.jsonl", &output.trace), std::make_pair("costs.jsonl", &output.costs)}) {
        std::string lines;
        for (const auto & record : *item.second) lines += record.dump() + "\n";
        write(root / item.first, lines);
    }
    write(root / "report.md", output.report);
}
}
