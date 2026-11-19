#pragma once
#include "graph-description.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace umm::profiling {
// An address-independent execution contract. Construction belongs to the structure
// pass. Checking uses preallocated storage: no JSON, SHA or pointer-ID map rebuild.
class graph_contract {
    graph_description description_;
    int n_nodes;
    bool scheduler;
    std::vector<ggml_tensor *> current, unique;
    static const char * buffer_name(const ggml_tensor * t) { return graph_description::buffer_name(t); }
    static int usage(const ggml_tensor * t) { return graph_description::usage(t); }
    static ggml_backend_dev_t device(ggml_backend_t b, ggml_backend_sched_t s, ggml_tensor * t) {
        return graph_description::device(b, s, t);
    }
    [[noreturn]] static void mismatch() {
        throw std::runtime_error("Graph differs from frozen structure contract; collect structure again");
    }

public:
    graph_contract(ggml_backend_t backend, ggml_backend_sched_t sched, ggml_cgraph * graph)
        : description_(backend, sched, graph), n_nodes(description_.n_nodes), scheduler(description_.scheduler),
          current(description_.records.size(), nullptr) {
        size_t capacity = 1;
        while (capacity < current.size() * 2)
            capacity *= 2;
        unique.resize(capacity, nullptr);
    }
    const graph_description & description() const { return description_; }
    void check(ggml_backend_t backend, ggml_backend_sched_t sched, ggml_cgraph * graph) {
        const auto & records = description_.records;
        const auto & placement = description_.placement;
        if ((sched != nullptr) != scheduler || ggml_graph_n_nodes(graph) != n_nodes)
            mismatch();
        for (int i = 0; i < n_nodes; ++i)
            current[i] = ggml_graph_node(graph, i);
        for (size_t i = n_nodes; i < records.size(); ++i) {
            const auto & r = records[i];
            auto parent = current[r.parent];
            if (!parent)
                mismatch();
            current[i] = r.slot < 0 ? parent->view_src : parent->src[r.slot];
        }
        std::fill(unique.begin(), unique.end(), nullptr);
        for (auto t : current) {
            if (!t)
                mismatch();
            size_t h = reinterpret_cast<uintptr_t>(t) >> 4;
            h ^= h >> 17;
            h *= size_t(0x9e3779b97f4a7c15ULL);
            size_t slot = h & (unique.size() - 1);
            while (unique[slot] && unique[slot] != t)
                slot = (slot + 1) & (unique.size() - 1);
            if (unique[slot])
                mismatch();
            unique[slot] = t;
        }
        for (size_t i = 0; i < records.size(); ++i) {
            auto t = current[i];
            const auto & r = records[i];
            if (!t || t->type != r.type || t->op != r.op || t->flags != r.flags || t->view_offs != r.offset ||
                r.name != t->name || std::memcmp(r.ne.data(), t->ne, sizeof(t->ne)) ||
                std::memcmp(r.nb.data(), t->nb, sizeof(t->nb)) ||
                std::memcmp(r.params.data(), t->op_params, GGML_MAX_OP_PARAMS) || r.buffer != buffer_name(t) ||
                r.usage != usage(t))
                mismatch();
            if (t->view_src != (r.view < 0 ? nullptr : current[r.view]))
                mismatch();
            for (int k = 0; k < GGML_MAX_SRC; ++k)
                if (t->src[k] != (r.src[k] < 0 ? nullptr : current[r.src[k]]))
                    mismatch();
            if (i < size_t(n_nodes) && device(backend, sched, t) != placement[i])
                mismatch();
        }
    }
};
} // namespace umm::profiling
