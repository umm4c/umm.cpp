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
void write_result(const result & output, const std::string & directory) {
    const std::filesystem::path root(directory);
    if (!std::filesystem::create_directory(root)) throw std::invalid_argument("Output directory already exists");
    write(root / "manifest.json", output.manifest.dump(2) + "\n");
    for (const auto & item : {std::make_pair("graphs.jsonl", &output.graphs), std::make_pair("trace.jsonl", &output.trace), std::make_pair("costs.jsonl", &output.costs), std::make_pair("requests.jsonl", &output.requests), std::make_pair("candidates.jsonl", &output.candidates), std::make_pair("candidate-costs.jsonl", &output.candidate_costs), std::make_pair("snapshots.jsonl", &output.snapshots)}) {
        std::string lines;
        for (const auto & record : *item.second) lines += record.dump() + "\n";
        write(root / item.first, lines);
    }
    write(root / "report.md", output.report);
}
}
