#pragma once

#include "profiler.h"
#include "graph-contract.h"
#include "hierarchy-timing.h"

#include <chrono>
#include <map>
#include <memory>
#include <set>

namespace umm::profiling::detail {

struct capture {
    result & output;
    std::string device_id, case_id, requested_backend, error;
    std::string pass = "structure";
    int replay = 0;
    bool warmup = false;
    bool remote_timing = false;
    bool remote = false;
    detail::hierarchy_capture hierarchy;
    std::chrono::steady_clock::time_point origin = std::chrono::steady_clock::now();
    struct frame {
        std::string graph_id, event_id, parent;
        std::chrono::steady_clock::time_point start;
        double export_ms;
    };
    std::vector<frame> stack;
    std::set<std::string> seen;
    struct binding {
        std::string id, phase;
        std::unique_ptr<graph_contract> contract;
        bool scheduler = false;
    };
    std::map<std::string, std::vector<binding>> bindings;
    size_t execution_cursor = 0;
    size_t event_number = 0;
    std::map<std::string, std::vector<double>> samples;
    bool fine() const { return pass == "layer" || pass == "block" || pass == "operator"; }
    bool observe_fine(ggml_backend_t backend, ggml_backend_sched_t sched, ggml_cgraph * graph, bool completed);
    bool observe(ggml_backend_t backend, ggml_backend_sched_t sched, ggml_cgraph * graph, bool completed,
                 ggml_status status);
    static bool callback(ggml_backend_t backend, ggml_backend_sched_t sched, ggml_cgraph * graph, bool completed,
                         ggml_status status, void * user) noexcept;
};

struct observer_scope {
    capture & collector;
    bool submission;

    explicit observer_scope(capture & collector);
    ~observer_scope();
    observer_scope(const observer_scope &) = delete;
    observer_scope & operator=(const observer_scope &) = delete;
};

}
