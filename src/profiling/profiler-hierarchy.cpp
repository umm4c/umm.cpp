#include "profiler-hierarchy.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace umm::profiling::detail {
using json = nlohmann::json;

void hierarchy_capture::begin(ggml_backend_t target, ggml_cgraph * graph, const std::string & graph_id,
                              const std::string & level) {
    const auto [it, inserted] = plans.try_emplace(std::make_pair(graph_id, level));
    auto & plan = it->second;
    if (inserted) {
        const int count = ggml_graph_n_nodes(graph);
        if (level == "operator") {
            for (int i = 0; i < count; ++i) {
                const auto * node = ggml_graph_node(graph, i);
                const auto op = node->op;
                const bool metadata = op == GGML_OP_NONE || op == GGML_OP_VIEW || op == GGML_OP_RESHAPE ||
                    op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
                if (!metadata && (node->flags & GGML_TENSOR_FLAG_COMPUTE) && !ggml_is_empty(node)) {
                    plan.push_back({i + 1, ggml_op_name(op)});
                }
            }
            if (!plan.empty()) {
                plan.back().end = count;
            }
        } else {
            int first_norm = -1;
            int attention_end = -1;
            int layer = 0;
            auto matches = [&](int i, const char * prefix) {
                return std::string(ggml_graph_node(graph, i)->name) == prefix + std::to_string(layer);
            };
            for (int i = 0; i < count; ++i) {
                if (matches(i, "attn_norm-")) {
                    if (first_norm != -1 || i == 0 ||
                        ggml_graph_node(graph, i)->src[0] != ggml_graph_node(graph, i - 1) ||
                        ggml_graph_node(graph, i - 1)->op != GGML_OP_RMS_NORM) {
                        plan.clear();
                        break;
                    }
                    first_norm = i - 1;
                } else if (matches(i, "ffn_inp-")) {
                    if (first_norm < 0 || attention_end != -1) {
                        plan.clear();
                        break;
                    }
                    attention_end = i + 1;
                } else if (matches(i, "l_out-")) {
                    if (attention_end < 0 || first_norm < 0 || i + 1 <= attention_end ||
                        (!plan.empty() && first_norm != plan.back().end)) {
                        plan.clear();
                        break;
                    }
                    if (layer == 0 && first_norm > 0) {
                        plan.push_back({first_norm, "preamble"});
                    }
                    const std::string name = "layer." + std::to_string(layer);
                    if (level == "block") {
                        plan.push_back({attention_end, name + ".attention"});
                    }
                    plan.push_back({i + 1, level == "block" ? name + ".ffn" : name});
                    ++layer;
                    first_norm = attention_end = -1;
                }
            }
            if (!layer || first_norm != -1) {
                plan.clear();
            }
            if (!plan.empty() && plan.back().end < count) {
                plan.push_back({count, "epilogue"});
            }
        }
    }
    backend = target;
    current = &plan;
    auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(target));
    if (plan.empty() || !reg) {
        return;
    }
    auto start = reinterpret_cast<ggml_backend_timing_begin_t>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_timing_begin_v1"));
    finish_timing = reinterpret_cast<ggml_backend_timing_finish_t>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_timing_finish_v1"));
    if (start && finish_timing) {
        std::vector<int> ends;
        ends.reserve(plan.size());
        for (const auto & part : plan) {
            ends.push_back(part.end);
        }
        armed = start(target, ends.data(), ends.size());
    }
}

json hierarchy_capture::finish(ggml_backend_t target) {
    if (target != backend || !current) {
        return nullptr;
    }
    const auto & plan = *current;
    backend = nullptr;
    current = nullptr;
    if (!armed) {
        return {{"status", "unavailable"}, {"reason", plan.empty() ? "unrecognized_boundaries" : "timing_backend_unavailable"}};
    }
    armed = false;
    std::vector<ggml_backend_timing_sample> intervals(plan.size());
    const size_t count = finish_timing(target, intervals.data(), intervals.size());
    if (!count) {
        return {{"status", "unavailable"}, {"reason", "no_completed_stream_timing"}};
    }
    json spans = json::array();
    bool partial = false;
    int begin = 0;
    size_t next = 0;
    for (size_t i = 0; i < count; ++i) {
        const auto & sample = intervals[i];
        if (sample.begin != begin || sample.end <= begin || sample.end > plan.back().end) {
            throw std::runtime_error("Invalid backend timing range");
        }
        json labels = json::array();
        while (next < plan.size() && (next == 0 || plan[next - 1].end < sample.end)) {
            labels.push_back(plan[next].label);
            if (plan[next].end > sample.end) {
                break;
            }
            ++next;
        }
        const bool valid = std::isfinite(sample.elapsed_ms) && sample.elapsed_ms >= 0;
        json span = {{"node_range", {begin, sample.end}}, {"labels", std::move(labels)},
                     {"elapsed_ms", valid ? json(sample.elapsed_ms) : json(nullptr)}};
        if (!valid) {
            partial = true;
            span["reason"] = "invalid_stream_elapsed_time";
        }
        spans.push_back(std::move(span));
        begin = sample.end;
    }
    if (begin != plan.back().end) {
        throw std::runtime_error("Incomplete backend timing range");
    }
    return {{"status", partial ? "partial" : "completed"}, {"timing_scope", "cann_stream_events"},
            {"spans", std::move(spans)}};
}

void hierarchy_capture::cancel() {
    if (armed) {
        finish_timing(backend, nullptr, 0);
    }
    armed = false;
    backend = nullptr;
    current = nullptr;
}
}
