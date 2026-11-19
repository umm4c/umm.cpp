#pragma once
#include "ggml-backend.h"
#include "profiler-utils.h"
#include "json.hpp"
#include <array>
#include <cstring>
#include <stdexcept>
#include <unordered_map>

namespace umm::profiling {
// Immutable structure-pass metadata shared by JSON, identity and replay checks.
// Tensor addresses are used only while collecting; device handles remain stable.
struct graph_description {
    struct tensor_record {
        ggml_type type;
        ggml_op op;
        int flags;
        std::array<int64_t, GGML_MAX_DIMS> ne;
        std::array<size_t, GGML_MAX_DIMS> nb;
        std::array<int32_t, GGML_MAX_OP_PARAMS / sizeof(int32_t)> params;
        std::array<int, GGML_MAX_SRC> src;
        int view = -1, parent = -1, slot = -1;
        size_t offset = 0, bytes = 0;
        std::string name, buffer;
        int usage = -1;
        bool contiguous = false;
    };
    int n_nodes;
    bool scheduler;
    std::vector<tensor_record> records;
    std::vector<ggml_backend_dev_t> placement;
    std::vector<std::string> device_names, backend_names;
    std::string direct_backend;
    nlohmann::json splits = nlohmann::json::array();
    static const char * buffer_name(const ggml_tensor * t) {
        return t->buffer ? ggml_backend_buft_name(ggml_backend_buffer_get_type(t->buffer)) : "";
    }
    static int usage(const ggml_tensor * t) { return t->buffer ? int(ggml_backend_buffer_get_usage(t->buffer)) : -1; }
    static ggml_backend_t target(ggml_backend_t backend, ggml_backend_sched_t sched, ggml_tensor * t) {
        return backend ? backend : ggml_backend_sched_get_tensor_backend(sched, t);
    }
    static ggml_backend_dev_t device(ggml_backend_t backend, ggml_backend_sched_t sched, ggml_tensor * t) {
        auto b = target(backend, sched, t);
        return b ? ggml_backend_get_device(b) : nullptr;
    }

    graph_description(ggml_backend_t backend, ggml_backend_sched_t sched, ggml_cgraph * graph)
        : n_nodes(ggml_graph_n_nodes(graph)), scheduler(sched != nullptr) {
        std::unordered_map<const ggml_tensor *, int> ids;
        struct reference {
            int parent, slot;
        };
        std::vector<reference> references;
        std::vector<ggml_tensor *> ordered;
        auto add = [&](ggml_tensor * t, int parent, int slot) {
            if (t && !ids.count(t)) {
                ids[t] = int(ordered.size());
                ordered.push_back(t);
                references.push_back({parent, slot});
            }
        };
        if (backend)
            direct_backend = ggml_backend_name(backend);
        for (int i = 0; i < n_nodes; ++i) {
            auto t = ggml_graph_node(graph, i);
            if (ids.count(t))
                throw std::runtime_error("Duplicate graph node");
            add(t, -1, -1);
            auto b = target(backend, sched, t);
            auto dev = b ? ggml_backend_get_device(b) : nullptr;
            placement.push_back(dev);
            device_names.emplace_back(dev ? ggml_backend_dev_name(dev) : "");
            backend_names.emplace_back(b ? ggml_backend_name(b) : "");
        }
        for (size_t i = 0; i < ordered.size(); ++i) {
            auto t = ordered[i];
            add(t->view_src, int(i), -1);
            for (int k = 0; k < GGML_MAX_SRC; ++k)
                add(t->src[k], int(i), k);
        }
        records.reserve(ordered.size());
        for (size_t i = 0; i < ordered.size(); ++i) {
            auto t = ordered[i];
            tensor_record r{};
            r.type = t->type;
            r.op = t->op;
            r.flags = t->flags;
            r.offset = t->view_offs;
            r.name = t->name;
            r.buffer = buffer_name(t);
            r.usage = usage(t);
            r.bytes = ggml_nbytes(t);
            r.contiguous = ggml_is_contiguous(t);
            std::memcpy(r.ne.data(), t->ne, sizeof(t->ne));
            std::memcpy(r.nb.data(), t->nb, sizeof(t->nb));
            std::memcpy(r.params.data(), t->op_params, GGML_MAX_OP_PARAMS);
            for (int k = 0; k < GGML_MAX_SRC; ++k)
                r.src[k] = t->src[k] ? ids.at(t->src[k]) : -1;
            r.view = t->view_src ? ids.at(t->view_src) : -1;
            r.parent = references[i].parent;
            r.slot = references[i].slot;
            records.push_back(std::move(r));
        }
        if (sched)
            for (int i = 0; i < ggml_backend_sched_get_n_splits(sched); ++i) {
                ggml_backend_sched_split_info info{};
                if (!ggml_backend_sched_get_split_info(sched, i, &info))
                    throw std::runtime_error("Cannot inspect scheduler split");
                splits.push_back({{"id", i},
                                  {"node_start", info.node_start},
                                  {"node_end_exclusive", info.node_end},
                                  {"device", ggml_backend_dev_name(ggml_backend_get_device(info.backend))},
                                  {"weight_input_bytes", info.weight_input_bytes}});
            }
    }
    int storage(int index) const {
        while (records[index].view >= 0)
            index = records[index].view;
        return index;
    }
    std::string signature(const char * phase, const std::string & requested_backend, bool with_placement) const {
        std::string data;
        auto append = [&](const auto & value) { data.append(reinterpret_cast<const char *>(&value), sizeof(value)); };
        auto text = [&](const std::string & value) {
            size_t size = value.size();
            append(size);
            data.append(value);
        };
        text(phase);
        append(scheduler);
        append(n_nodes);
        std::string named_backend = direct_backend;
        if (named_backend.empty())
            for (int i = 0; i < n_nodes; ++i)
                if (device_names[i] == requested_backend) {
                    named_backend = backend_names[i];
                    break;
                }
        const std::string prefix = (named_backend.empty() ? requested_backend : named_backend) + "#";
        for (size_t i = 0; i < records.size(); ++i) {
            const auto & r = records[i];
            text(!with_placement && r.name.compare(0, prefix.size(), prefix) == 0 ? r.name.substr(prefix.size())
                                                                                  : r.name);
            append(r.op);
            append(r.type);
            append(r.flags);
            data.append(reinterpret_cast<const char *>(r.ne.data()), sizeof(int64_t) * GGML_MAX_DIMS);
            data.append(reinterpret_cast<const char *>(r.nb.data()), sizeof(size_t) * GGML_MAX_DIMS);
            data.append(reinterpret_cast<const char *>(r.params.data()), GGML_MAX_OP_PARAMS);
            append(r.offset);
            append(r.view);
            for (auto src : r.src)
                append(src);
            if (with_placement)
                text(records[storage(int(i))].buffer);
        }
        if (with_placement)
            text(requested_backend);
        return detail::hash(data);
    }
    nlohmann::json describe(const char * phase, int64_t tokens, int64_t cached, const std::string & required) const {
        using json = nlohmann::json;
        json tensors = json::array(), nodes = json::array();
        auto id = [](int index) { return "t" + std::to_string(index); };
        for (size_t i = 0; i < records.size(); ++i) {
            const auto & r = records[i];
            const auto root = storage(int(i));
            const auto & base = records[root];
            const auto role = base.usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS ? "weight"
                              : r.flags & GGML_TENSOR_FLAG_INPUT              ? "input"
                              : i < size_t(n_nodes)                           ? "activation"
                                                                              : "unknown";
            json params = json::array();
            auto bytes = reinterpret_cast<const uint8_t *>(r.params.data());
            for (size_t k = 0; k < GGML_MAX_OP_PARAMS; ++k)
                params.push_back(bytes[k]);
            tensors.push_back({{"id", id(int(i))},
                               {"name", r.name},
                               {"shape", r.ne},
                               {"strides_bytes", r.nb},
                               {"dtype", ggml_type_name(r.type)},
                               {"bytes", r.bytes},
                               {"role", role},
                               {"layout", r.contiguous ? "contiguous" : "strided"},
                               {"layout_scope", "logical_ggml"},
                               {"backend_layout", nullptr},
                               {"flags", r.flags},
                               {"alias_of", r.view >= 0 ? json(id(r.view)) : json(nullptr)},
                               {"view_offset_bytes", r.offset},
                               {"storage_id", id(root)},
                               {"op_params_bytes", params},
                               {"buffer_type", base.usage < 0 ? json(nullptr) : json(base.buffer)}});
        }
        for (int i = 0; i < n_nodes; ++i) {
            const auto & r = records[i];
            json inputs = json::array(), deps = json::array();
            for (auto src : r.src)
                if (src >= 0) {
                    inputs.push_back(id(src));
                    if (src < n_nodes)
                        deps.push_back(src);
                }
            const bool metadata =
                r.op == GGML_OP_NONE || r.op == GGML_OP_VIEW || r.op == GGML_OP_RESHAPE || r.op == GGML_OP_PERMUTE;
            if (!metadata && device_names[i] != required)
                throw std::runtime_error("Unexpected compute device in profile graph");
            nodes.push_back({{"id", i},
                             {"tensor_id", id(i)},
                             {"op", ggml_op_name(r.op)},
                             {"inputs", inputs},
                             {"dependencies", deps},
                             {"outputs", {id(i)}},
                             {"actual_device", device_names[i].empty() ? json(nullptr) : json(device_names[i])},
                             {"may_write_inputs", nullptr},
                             {"persistent_state", nullptr}});
        }
        return {{"schema_version", "umm-profiler-graph/v3"},
                {"data_origin", "measured"},
                {"phase", phase},
                {"example_submission",
                 {{"tokens", tokens < 0 ? json(nullptr) : json(tokens)},
                  {"cached_tokens", cached < 0 ? json(nullptr) : json(cached)}}},
                {"kind", scheduler ? "scheduler" : "backend"},
                {"nodes", nodes},
                {"tensors", tensors},
                {"actual_splits", splits},
                {"cut_boundaries", json::array()},
                {"legality", "not_analyzed"}};
    }
};
}
