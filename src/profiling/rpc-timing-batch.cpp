#include "rpc-timing-batch.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace umm::profiling::detail {

#ifdef GGML_USE_RPC
uint32_t rpc_batch::drain() {
    uint32_t count = 0;
    auto window = active;
    active = 0;
    if (!ggml_backend_rpc_profile_end(backend, window, records.data(), uint32_t(records.size()), &count))
        throw std::runtime_error("RPC profile records missing, overflowed or out of order");
    return count;
}
#endif

rpc_batch::rpc_batch(const std::string & name) {
#ifdef GGML_USE_RPC
    backend = ggml_backend_init_by_name(name.c_str(), nullptr);
    if (!backend)
        throw std::runtime_error("Cannot initialize RPC profiling backend");
    try {
        start(1);
        if (drain() != 0)
            throw std::runtime_error("RPC profiling probe returned unexpected graphs");
    } catch (...) {
        ggml_backend_free(backend);
        backend = nullptr;
        throw;
    }
#else
    (void)name;
    throw std::runtime_error("RPC profiling requires GGML_RPC=ON");
#endif
}

rpc_batch::~rpc_batch() {
#ifdef GGML_USE_RPC
    if (active) {
        uint32_t count = 0;
        ggml_backend_rpc_profile_end(backend, active, records.data(), uint32_t(records.size()), &count);
    }
    if (backend)
        ggml_backend_free(backend);
#endif
}

void rpc_batch::start(size_t expected) {
#ifdef GGML_USE_RPC
    if (expected > GGML_RPC_PROFILE_MAX_RECORDS)
        throw std::runtime_error("RPC request exceeds profiling record capacity");
    records.resize(std::max(size_t(64), expected));
    auto window = ++sequence;
    if (!ggml_backend_rpc_profile_begin(backend, window, uint32_t(records.size())))
        throw std::runtime_error(
            "RPC worker does not support batch timing or a window is already active; requires protocol 6.1");
    active = window;
#else
    (void)expected;
#endif
}

void rpc_batch::finish(nlohmann::json & trace, size_t trace_start, const std::string & device_id,
                       std::map<std::string, std::vector<double>> & samples) {
#ifdef GGML_USE_RPC
    const auto window = active;
    const auto count = drain();
    std::vector<size_t> submissions;
    const auto end = trace.size();
    for (size_t i = trace_start; i < end; ++i)
        if (trace[i].at("kind") == "backend")
            submissions.push_back(i);
    if (count != submissions.size())
        throw std::runtime_error("RPC records do not match frozen graph submissions");
    for (size_t i = 0; i < count; ++i) {
        const auto & record = records[i];
        auto event = trace[submissions[i]];
        const auto submitted_id = event.at("event_id").get<std::string>();
        event["event_id"] = submitted_id + "-worker";
        event["submission_event_id"] = submitted_id;
        event["parent_event_id"] = nullptr;
        event["clock_domain"] = "worker:" + device_id + ":" + std::to_string(window);
        event["window_id"] = window;
        event["event_type"] = "completed";
        event["timing_scope"] = "rpc_worker_synchronized_host_wall";
        event["start_ms"] = (double(record.start_ns) + double(record.prepare_ns)) / 1e6;
        event["duration_ms"] = double(record.compute_ns) / 1e6;
        event["export_before_ms"] = 0;
        event["worker_prepare_ms"] = double(record.prepare_ns) / 1e6;
        event["worker_sequence"] = record.sequence;
        event["worker_device"] = record.device;
        event["worker_nodes"] = record.n_nodes;
        event["worker_reused"] = bool(record.reused);
        event["status"] = record.status;
        samples[event.at("graph_id").get<std::string>()].push_back(double(record.compute_ns) / 1e6);
        trace.push_back(std::move(event));
    }
#else
    (void)trace;
    (void)device_id;
    (void)samples;
    (void)trace_start;
#endif
}

}
