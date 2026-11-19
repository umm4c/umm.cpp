#include "timing-annotations.h"
#include "hash-and-statistics.h"

#include <cmath>
#include <map>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace umm::profiling::detail {
using json = nlohmann::json;

json annotate_timings(const json & graphs, const json & trace) {
    using graph_key = std::pair<std::string, std::string>; // Device and physical graph IDs.
    std::map<graph_key, const json *> graph_index;
    for (const auto & graph : graphs) {
        graph_key key{graph.at("device_id").get<std::string>(), graph.at("graph_id").get<std::string>()};
        if (!graph_index.emplace(std::move(key), &graph).second) {
            throw std::runtime_error("Duplicate graph in profiler annotations");
        }
    }

    using input_key = std::tuple<std::string, std::string, json, json>;
    using measurement_key = std::tuple<std::string, std::string, std::string>;
    struct measurement {
        size_t nodes = 0;
        std::vector<double> samples;
    };
    std::map<input_key, std::map<measurement_key, measurement>> grouped;
    for (const auto & event : trace) {
        if (event.at("pass") != "timing" || event.at("warmup") || event.at("event_type") != "completed" ||
            event.at("kind") != "backend") {
            continue;
        }
        const auto device = event.at("device_id").get<std::string>();
        const auto physical_id = event.at("graph_id").get<std::string>();
        const auto & graph = *graph_index.at({device, physical_id});
        const auto count = graph.at("nodes").size();
        const auto scope = event.at("timing_scope").get<std::string>();
        const auto duration = event.at("duration_ms").get<double>();
        if (graph.at("kind") != "backend" || graph.at("phase") != event.at("phase") ||
            !event.at("executed").get<bool>() || event.at("status") != 0 || !std::isfinite(duration) || duration < 0 ||
            (scope != "synchronized_host_wall" && scope != "rpc_worker_synchronized_host_wall") ||
            (scope == "rpc_worker_synchronized_host_wall" && event.at("worker_nodes") != count)) {
            throw std::runtime_error("Invalid completed backend timing in profiler annotations");
        }
        const auto & submission = event.at("submission");
        auto & samples =
            grouped[{graph.at("logical_graph_id").get<std::string>(), event.at("case_id").get<std::string>(),
                     submission.at("tokens"), submission.at("cached_tokens")}];
        auto & record = samples[{device, physical_id, scope}];
        record.nodes = count;
        record.samples.push_back(duration);
    }

    json result = json::array();
    for (auto & [input, measurements] : grouped) {
        const auto & [logical_id, case_id, tokens, cached_tokens] = input;
        json timings = json::array();
        for (auto & [source, measurement] : measurements) {
            const auto & [device, physical_id, scope] = source;
            const auto summary = sample_statistics(std::move(measurement.samples));
            timings.push_back({
                {"device_id", device},
                {"graph_id", physical_id},
                {"timing_scope", scope},
                {"node_range", {0, measurement.nodes}},
                {"median_ms", summary.at("p50")},
                {"sample_count", summary.at("sample_count")},
            });
        }
        result.push_back({
            {"schema_version", "umm-profiler-timing/v1"},
            {"logical_graph_id", logical_id},
            {"case_id", case_id},
            {"submission", {{"tokens", tokens}, {"cached_tokens", cached_tokens}}},
            {"timings", std::move(timings)},
        });
    }
    return result;
}
} // namespace umm::profiling::detail
