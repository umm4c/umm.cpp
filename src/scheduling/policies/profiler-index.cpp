#include "profiler-index.h"
#include "hash/hash.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <optional>
#include <set>
#include <stdexcept>
#include <unordered_map>

namespace umm::runtime_placement {
namespace {

using json = nlohmann::json;

template <typename Callback> void read_lines(const std::filesystem::path & path, Callback callback) {
    std::ifstream file(path);
    if (!file)
        throw std::runtime_error("Cannot open planner measurement file: " + path.string());
    for (std::string line; std::getline(file, line);) {
        if (!line.empty())
            callback(json::parse(line));
    }
    if (!file.eof())
        throw std::runtime_error("Cannot read planner measurement file: " + path.string());
}

// Profiler device copies are named <backend>#<name>#<copy-index>. The
// normalized source graph retains dependency, view, shape and parameter data.
std::string normalize(std::string name) {
    const auto first = name.find('#');
    const auto last = name.rfind('#');
    if ((name.rfind("CANN", 0) == 0 || name.rfind("RPC", 0) == 0) && first < last && last + 1 < name.size() &&
        name.find_first_not_of("0123456789", last + 1) == std::string::npos) {
        return name.substr(first + 1, last - first - 1);
    }
    return name;
}

struct signature {
    std::string bytes;
    template <typename T> void add(T value) { bytes.append(reinterpret_cast<const char *>(&value), sizeof(value)); }
    void name(const std::string & value) {
        add(uint32_t(value.size()));
        bytes.append(value);
    }
    std::string digest() const { return hash_sha256_hex(bytes.data(), bytes.size()); }
};

int tensor_id(const json & encoded) {
    if (encoded.is_null())
        return -1;
    const auto & id = encoded.get_ref<const std::string &>();
    if (id.empty() || id.front() != 't')
        throw std::runtime_error("Invalid profiler tensor identity");
    return std::stoi(id.substr(1));
}

std::string fingerprint(const json & graph) {
    signature sig;
    const auto & nodes = graph.at("nodes");
    const auto & tensors = graph.at("tensors");
    sig.add(uint32_t(nodes.size()));
    sig.add(uint32_t(tensors.size()));
    for (const auto & node : nodes) {
        sig.name(node.at("op").get<std::string>());
        const auto & inputs = node.at("inputs");
        sig.add(uint32_t(inputs.size()));
        for (const auto & input : inputs)
            sig.add(tensor_id(input));
    }
    for (const auto & tensor : tensors) {
        sig.name(normalize(tensor.at("name").get<std::string>()));
        sig.name(tensor.at("dtype").get<std::string>());
        for (const auto & dim : tensor.at("shape"))
            sig.add(dim.get<int64_t>());
        for (const auto & stride : tensor.at("strides_bytes"))
            sig.add(stride.get<uint64_t>());
        for (const auto & byte : tensor.at("op_params_bytes"))
            sig.add(byte.get<uint8_t>());
        // Scheduler-created input copies carry allocator flags of their own.
        sig.add(tensor_id(tensor.at("alias_of")));
        sig.add(tensor.at("view_offset_bytes").get<uint64_t>());
    }
    return sig.digest();
}

std::string fingerprint(ggml_cgraph * graph) {
    signature sig;
    std::vector<ggml_tensor *> tensors;
    std::unordered_map<ggml_tensor *, int> ids;
    auto insert = [&](ggml_tensor * tensor) {
        if (tensor && ids.emplace(tensor, int(tensors.size())).second)
            tensors.push_back(tensor);
    };
    const int count = ggml_graph_n_nodes(graph);
    for (int i = 0; i < count; ++i)
        insert(ggml_graph_node(graph, i));
    for (size_t i = 0; i < tensors.size(); ++i) {
        auto * tensor = tensors[i];
        insert(tensor->view_src);
        for (auto * src : tensor->src)
            insert(src);
    }
    sig.add(uint32_t(count));
    sig.add(uint32_t(tensors.size()));
    for (int i = 0; i < count; ++i) {
        const auto * tensor = tensors[i];
        sig.name(ggml_op_name(tensor->op));
        sig.add(
            uint32_t(std::count_if(std::begin(tensor->src), std::end(tensor->src), [](auto * src) { return src; })));
        for (auto * src : tensor->src)
            if (src)
                sig.add(ids.at(src));
    }
    for (const auto * tensor : tensors) {
        sig.name(normalize(tensor->name));
        sig.name(ggml_type_name(tensor->type));
        for (auto dim : tensor->ne)
            sig.add(int64_t(dim));
        for (auto stride : tensor->nb)
            sig.add(uint64_t(stride));
        for (const auto byte : *reinterpret_cast<const uint8_t(*)[GGML_MAX_OP_PARAMS]>(tensor->op_params))
            sig.add(byte);
        sig.add(tensor->view_src ? ids.at(tensor->view_src) : -1);
        sig.add(uint64_t(tensor->view_offs));
    }
    return sig.digest();
}

struct graph_record {
    device at;
    std::string phase, kind, structural, logical_id;
    size_t count;
};

double median(std::vector<double> & data) {
    if (data.empty())
        return unknown;
    std::sort(data.begin(), data.end());
    return data[(data.size() - 1) / 2] / 2 + data[data.size() / 2] / 2;
}

}

profiler_index::profiler_index(const std::filesystem::path & directory) {
    std::ifstream manifest(directory / "manifest.json");
    json metadata;
    manifest >> metadata;
    if (metadata.value("data_origin", "") != "measured" || !metadata.value("executed", false)) {
        throw std::invalid_argument("Runtime placement requires a measured Profiler capture");
    }
    identity_ = metadata.at("identity");
    devices_ = metadata.at("request").at("devices");
    std::map<std::string, device> ranks;
    for (const auto & desc : devices_) {
        const auto transport = desc.at("transport").get<std::string>();
        if (transport == "local" || transport == "rpc") {
            ranks.emplace(desc.at("id").get<std::string>(), transport == "local" ? local : remote);
        }
    }
    if (ranks.size() != 2 || ranks.begin()->second == ranks.rbegin()->second) {
        throw std::invalid_argument("Profiler capture must have one local device and one RPC device");
    }
    std::unordered_map<std::string, graph_record> graphs;
    std::map<std::pair<device, std::string>, std::string> scheduler_graphs;
    auto id = [](const json & row, const char * field = "graph_id") { return row.at(field).get<std::string>(); };
    read_lines(directory / "graphs.jsonl", [&](const json & row) {
        if (row.value("schema_version", "") != "umm-profiler-graph/v3")
            throw std::runtime_error("Unsupported profiler graph format");
        const auto phase = id(row, "phase");
        if (phase != "understanding_prefill" && phase != "understanding_decode")
            return;
        graph_record record{
            ranks.at(id(row, "device_id")), phase, id(row, "kind"), fingerprint(row), id(row, "logical_graph_id"),
            row.at("nodes").size()};
        if (record.kind == "scheduler") {
            auto [it, inserted] =
                scheduler_graphs.emplace(std::make_pair(record.at, record.structural), record.logical_id);
            if (!inserted && it->second != record.logical_id)
                it->second.clear();
        }
        graphs.emplace(id(row), std::move(record));
    });
    using samples = std::array<std::vector<double>, 2>;
    using range = std::pair<size_t, size_t>;
    std::map<key, std::map<range, samples>> spans;
    std::map<key, std::set<range>> fused_groups;
    auto match = [&](const graph_record & graph, const json & submission) -> std::optional<key> {
        const auto & tokens = submission.at("tokens");
        const auto & cached = submission.at("cached_tokens");
        if (!tokens.is_number_integer() || !cached.is_number_integer())
            return {};
        auto scheduler = scheduler_graphs.find({graph.at, graph.structural});
        if (scheduler == scheduler_graphs.end() || scheduler->second.empty())
            return {};
        return key{graph.phase, tokens.get<int64_t>(), cached.get<int64_t>(), graph.structural};
    };
    read_lines(directory / "hierarchy.jsonl", [&](const json & row) {
        auto found = graphs.find(id(row));
        if (found == graphs.end() || row.value("timing_scope", "") != "cann_stream_events" ||
            row.value("status", "") != "completed")
            return;
        const auto & graph = found->second;
        if (graph.kind != "backend" || ranks.at(id(row, "device_id")) != graph.at)
            return;
        auto matched = match(graph, row.at("submission"));
        if (!matched)
            return;
        for (const auto & span : row.at("spans")) {
            const auto & bounds = span.at("node_range");
            const auto begin = bounds.at(0).get<size_t>();
            const auto end = bounds.at(1).get<size_t>();
            if (begin >= end || end > graph.count)
                throw std::runtime_error("Profiler span outside its physical graph");
            if (span.at("elapsed_ms").is_null())
                continue;
            const auto ms = span.at("elapsed_ms").get<double>();
            if (std::isfinite(ms) && ms >= 0) {
                spans[*matched][{begin, end}][graph.at].push_back(ms);
                if (span.at("labels").size() > 1)
                    fused_groups[*matched].emplace(begin, end);
            }
        }
    });
    for (auto & [key, ranges] : spans) {
        auto & costs = costs_[key];
        const auto local_graph = scheduler_graphs.find({local, std::get<3>(key)});
        const auto remote_graph = scheduler_graphs.find({remote, std::get<3>(key)});
        if (local_graph == scheduler_graphs.end() || remote_graph == scheduler_graphs.end() ||
            local_graph->second.empty() || local_graph->second != remote_graph->second)
            continue;
        costs.logical_id = local_graph->second;
        if (const auto groups = fused_groups.find(key); groups != fused_groups.end()) {
            for (const auto & [begin, end] : groups->second)
                costs.fused_groups.push_back({begin, end});
        }
        for (auto & [bounds, values] : ranges) {
            costs.ranges.push_back({bounds.first, bounds.second, {median(values[local]), median(values[remote])}});
        }
    }
    read_lines(directory / "timings.jsonl", [&](const json & row) {
        for (const auto & sample : row.at("timings")) {
            auto found = graphs.find(id(sample));
            if (found == graphs.end())
                continue;
            const auto & graph = found->second;
            if (ranks.at(id(sample, "device_id")) != graph.at || id(row, "logical_graph_id") != graph.logical_id)
                continue;
            auto matched = match(graph, row.at("submission"));
            if (!matched)
                continue;
            auto costs = costs_.find(*matched);
            if (costs == costs_.end())
                continue;
            const auto & bounds = sample.at("node_range");
            if (bounds.at(0) != 0 || bounds.at(1) != graph.count)
                continue;
            const auto scope = sample.at("timing_scope").get<std::string>();
            if (scope != (graph.at == local ? "synchronized_host_wall" : "rpc_worker_synchronized_host_wall"))
                continue;
            const double ms = sample.at("median_ms").get<double>();
            if (!std::isfinite(ms) || ms < 0)
                continue;
            costs->second.submission_scope[graph.at] = scope;
            costs->second.submission_ms[graph.at].push_back(ms);
        }
    });
}

bool profiler_index::has_submission(const std::string & phase, int64_t tokens, int64_t cached) const {
    const auto found = costs_.lower_bound({phase, tokens, cached, {}});
    return found != costs_.end() && std::get<0>(found->first) == phase && std::get<1>(found->first) == tokens &&
           std::get<2>(found->first) == cached;
}

std::string profiler_index::signature(ggml_cgraph * graph) {
    return fingerprint(graph);
}

const measured_costs * profiler_index::find(const std::string & structural, const std::string & phase, int64_t tokens,
                                            int64_t cached) const {
    const auto found = costs_.find({phase, tokens, cached, structural});
    if (found == costs_.end() || found->second.ranges.empty())
        return nullptr;
    return &found->second;
}

const measured_costs * profiler_index::find(ggml_cgraph * graph, const std::string & phase, int64_t tokens,
                                            int64_t cached) const {
    return find(signature(graph), phase, tokens, cached);
}

}
