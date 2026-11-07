#include "random-plan.h"
#include <algorithm>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace umm {
namespace {
bool metadata(const ggml_tensor * t) {
    return t->op == GGML_OP_NONE || t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE ||
           t->op == GGML_OP_TRANSPOSE || t->op == GGML_OP_PERMUTE;
}
ggml_backend_t owner(ggml_tensor * t, const std::unordered_map<ggml_tensor *, ggml_backend_t> & assigned,
                     ggml_backend_t a, ggml_backend_t b) {
    while (t->view_src) t = t->view_src;
    const auto it = assigned.find(t);
    if (it != assigned.end()) return it->second;
    if (!t->buffer) return nullptr;
    auto device = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(t->buffer));
    if (device == ggml_backend_get_device(a)) return a;
    if (device == ggml_backend_get_device(b)) return b;
    return nullptr;
}
bool candidate(ggml_cgraph * graph, ggml_backend_t a, ggml_backend_t b, const std::vector<int> & cuts, random_plan & plan,
               std::unordered_map<ggml_tensor *, bool> & support_a, std::unordered_map<ggml_tensor *, bool> & support_b) {
    std::unordered_map<ggml_tensor *, ggml_backend_t> assigned;
    plan.placements.clear();
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        auto node = ggml_graph_node(graph, i);
        const size_t partition = std::upper_bound(cuts.begin(), cuts.end(), i) - cuts.begin();
        auto backend = partition % 2 == 0 ? a : b;
        if (metadata(node)) {
            if (node->buffer && !node->view_src) {
                backend = owner(node, assigned, a, b);
                if (!backend) continue;
            }
            if (node->view_src) {
                backend = owner(node, assigned, a, b);
                if (!backend) continue; // Host-only metadata remains with its allocated storage.
            }
        } else {
            // Allocated outputs and aliases cannot change their storage owner.
            if ((node->buffer || node->view_src) && owner(node, assigned, a, b) != backend) return false;
            auto & support = backend == a ? support_a : support_b;
            auto it = support.find(node);
            if (it == support.end()) it = support.emplace(node, ggml_backend_supports_op(backend, node)).first;
            if (!it->second) return false;
        }
        assigned[node] = backend;
        plan.placements.push_back({node, backend});
    }
    return true;
}
}
random_plan make_random_plan(ggml_cgraph * graph, ggml_backend_t first,
                                   ggml_backend_t second, uint32_t seed, bool reverse, int cut_count) {
    if (!graph || !first || !second || ggml_backend_get_device(first) == ggml_backend_get_device(second)) {
        throw std::invalid_argument("Random partition requires a graph and two distinct devices");
    }
    std::vector<int> cuts;
    bool seen_compute = false;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        if (!metadata(ggml_graph_node(graph, i))) {
            if (seen_compute) cuts.push_back(i);
            seen_compute = true;
        }
    }
    if (cut_count < 1 || size_t(cut_count) > cuts.size()) {
        throw std::invalid_argument("Cut count must be between 1 and compute-node-count minus one");
    }
    std::mt19937 generator(seed);
    // Pick the starting device as well as the cuts. reverse tests the opposite assignment.
    const bool start_remote = (generator() & 1U) != 0;
    const bool swapped = start_remote != reverse;
    random_plan plan{seed, {}, swapped, {}};
    auto a = swapped ? second : first;
    auto b = swapped ? first : second;
    std::unordered_map<ggml_tensor *, bool> support_a, support_b;
    std::shuffle(cuts.begin(), cuts.end(), generator);
    if (cut_count == 1) {
        for (int cut : cuts) {
            if (candidate(graph, a, b, {cut}, plan, support_a, support_b)) {
                plan.cut_nodes = {cut};
                return plan;
            }
        }
    } else {
        for (int attempt = 0; attempt < 64; ++attempt) {
            std::vector<int> selected(cuts.begin(), cuts.begin() + cut_count);
            std::sort(selected.begin(), selected.end());
            if (candidate(graph, a, b, selected, plan, support_a, support_b)) {
                plan.cut_nodes = std::move(selected);
                return plan;
            }
            std::shuffle(cuts.begin(), cuts.end(), generator);
        }
    }
    throw std::runtime_error("No valid random partition found in the candidate search budget");
}
void assign_random_plan(ggml_backend_sched_t sched, const random_plan & plan) {
    if (!sched || plan.cut_nodes.empty() || plan.placements.empty()) throw std::invalid_argument("Invalid random plan");
    for (const auto & p : plan.placements) ggml_backend_sched_set_tensor_backend(sched, p.node, p.backend);
}
void check_random_plan(ggml_backend_sched_t sched, const random_plan & plan) {
    for (const auto & p : plan.placements) {
        if (!metadata(p.node) && ggml_backend_sched_get_tensor_backend(sched, p.node) != p.backend) {
            throw std::runtime_error("Scheduler placement differs from the random plan");
        }
    }
}
}
