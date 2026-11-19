#include "graph-capture.h"
#include "inference/profile-context.h"

#include <stdexcept>
#include <utility>

namespace umm::profiling::detail {

using json = nlohmann::json;
using clock = std::chrono::steady_clock;

namespace {

double elapsed(clock::time_point start) {
    return std::chrono::duration<double, std::milli>(clock::now() - start).count();
}

}

bool capture::observe_fine(ggml_backend_t backend, ggml_backend_sched_t sched, ggml_cgraph * graph, bool completed) {
    if (!completed) {
        const auto & binding = bindings.at(case_id).at(execution_cursor++);
        if (binding.phase != current_phase.phase) {
            throw std::runtime_error("Execution phase differs from frozen structure");
        }
        binding.contract->check(backend, sched, graph);
        if (backend && (binding.phase == "understanding_prefill" || binding.phase == "understanding_decode")) {
            hierarchy.begin(backend, graph, binding.id, pass);
            stack.push_back({binding.id, {}, {}, clock::now(), 0});
        }
    } else if (backend && !stack.empty()) {
        auto sample = hierarchy.finish(backend);
        if (sample.is_null()) {
            throw std::runtime_error("Unmatched fine timing graph");
        }
        const auto tokens = current_phase.tokens;
        const auto cached = current_phase.cached_tokens;
        sample.update({
            {"schema_version", "umm-profiler-hierarchy/v1"},
            {"graph_id", stack.back().graph_id},
            {"device_id", device_id},
            {"case_id", case_id},
            {"replay", replay},
            {"level", pass},
            {"submission",
             {{"tokens", tokens < 0 ? json(nullptr) : json(tokens)},
              {"cached_tokens", cached < 0 ? json(nullptr) : json(cached)}}},
        });
        output.hierarchy.push_back(std::move(sample));
        stack.pop_back();
    }
    return true;
}

bool capture::observe(ggml_backend_t backend, ggml_backend_sched_t sched, ggml_cgraph * graph, bool completed,
                      ggml_status status) {
    if (fine()) {
        if (status != GGML_STATUS_SUCCESS)
            throw std::runtime_error("Fine timing graph execution failed");
        return observe_fine(backend, sched, graph, completed);
    }
    if (!completed) {
        auto start = clock::now();
        std::string fingerprint;
        if (pass == "structure") {
            auto contract = std::make_unique<graph_contract>(backend, sched, graph);
            const auto & descriptor = contract->description();
            fingerprint = descriptor.signature(current_phase.phase, requested_backend, true);
            if (seen.insert(fingerprint).second) {
                auto description = descriptor.describe(current_phase.phase, current_phase.tokens,
                                                       current_phase.cached_tokens, requested_backend);
                description["graph_id"] = fingerprint;
                description["logical_graph_id"] = descriptor.signature(current_phase.phase, requested_backend, false);
                description["device_id"] = device_id;
                output.graphs.push_back(std::move(description));
            }
            bindings[case_id].push_back({fingerprint, current_phase.phase, std::move(contract), sched != nullptr});
        } else {
            auto & sequence = bindings.at(case_id);
            if (execution_cursor >= sequence.size())
                throw std::runtime_error("Execution has more graphs than frozen structure");
            auto & registered = sequence[execution_cursor];
            if (registered.phase != current_phase.phase)
                throw std::runtime_error("Execution phase differs from frozen structure");
            registered.contract->check(backend, sched, graph);
            fingerprint = registered.id;
        }
        ++execution_cursor;
        const auto parent = stack.empty() ? std::string{} : stack.back().event_id;
        stack.push_back(
            {fingerprint, device_id + "-e" + std::to_string(++event_number), parent, clock::now(), elapsed(start)});
    } else {
        if (stack.empty())
            throw std::runtime_error("Unmatched graph event");
        auto f = stack.back();
        stack.pop_back();
        const double duration = elapsed(f.start);
        output.trace.push_back(
            {{"schema_version", "umm-profiler-event/v3"},
             {"data_origin", "measured"},
             {"graph_id", f.graph_id},
             {"event_id", f.event_id},
             {"parent_event_id", f.parent.empty() ? json(nullptr) : json(f.parent)},
             {"device_id", device_id},
             {"case_id", case_id},
             {"replay", replay},
             {"pass", pass},
             {"warmup", warmup},
             {"kind", sched ? "scheduler" : "backend"},
             {"phase", current_phase.phase},
             {"submission", {{"tokens", current_phase.tokens}, {"cached_tokens", current_phase.cached_tokens}}},
             {"start_ms", std::chrono::duration<double, std::milli>(f.start - origin).count()},
             {"duration_ms", duration},
             {"export_before_ms", f.export_ms},
             {"status", int(status)},
             {"timing_scope", remote_timing ? "client_submission_host_wall" : "synchronized_host_wall"},
             {"event_type", remote_timing ? "submitted" : "completed"},
             {"clock_domain", "client:" + device_id},
             {"executed", true},
             {"kernel_ms", nullptr}});
        if (status != GGML_STATUS_SUCCESS)
            throw std::runtime_error("Graph execution failed");
        if (!warmup && !remote_timing)
            samples[f.graph_id].push_back(duration);
    }
    return true;
}

bool capture::callback(ggml_backend_t backend, ggml_backend_sched_t sched, ggml_cgraph * graph, bool completed,
                       ggml_status status, void * user) noexcept {
    auto & self = *static_cast<capture *>(user);
    try {
        return self.observe(backend, sched, graph, completed, status);
    } catch (const std::exception & e) {
        if (self.error.empty())
            self.error = e.what();
        return false;
    } catch (...) {
        self.error = "Graph capture failed";
        return false;
    }
}

observer_scope::observer_scope(capture & c) : collector(c), submission(c.remote_timing || (c.remote && c.fine())) {
    if (submission)
        ggml_backend_set_graph_submission_observer(capture::callback, &c);
    else
        ggml_backend_set_graph_observer(capture::callback, &c);
}

observer_scope::~observer_scope() {
    if (submission)
        ggml_backend_set_graph_submission_observer(nullptr, nullptr);
    else
        ggml_backend_set_graph_observer(nullptr, nullptr);
    collector.hierarchy.cancel();
}

}
