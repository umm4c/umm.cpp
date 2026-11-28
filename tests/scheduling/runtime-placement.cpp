#include "scheduling/policies/runtime-placement.h"

#include <iostream>
#include <stdexcept>

using namespace umm::runtime_placement;

namespace {

void check(bool condition, const char * message) {
    if (!condition)
        throw std::runtime_error(message);
}

problem two_units() {
    problem graph;
    graph.units.resize(2);
    graph.current = {local, local};
    graph.capacity = {16, 16};
    graph.timings = {{0, 1, {6, 3}}, {1, 2, {6, 3}}};
    graph.values = {
        {-1, {0}, 1, 0, 4, {2, 2}},
        {0, {1}, 0, 0, 4, {2, 2}},
        {1, {}, 0, 1, 4, {2, 2}},
    };
    graph.switch_ms = 0;
    return graph;
}

void search_examples() {
    auto graph = two_units();
    check(score(graph, {local, local}).total_ms == 12, "local chain score");
    check(score(graph, {remote, local}).total_ms == 13, "first node alone");
    check(score(graph, {local, remote}).total_ms == 13, "second node alone");
    check(score(graph, {remote, remote}).total_ms == 10, "joint migration wins");

    for (const auto method : {algorithm::local_search, algorithm::min_cut, algorithm::beam_search}) {
        graph.switch_ms = 3;
        graph.repetitions = 1;
        check(choose(graph, method).placement == graph.current, "single-use preparation costs");
        graph.repetitions = 4;
        const auto winning = choose(graph, method);
        check(winning.placement == assignment{remote, remote} && winning.cost.total_ms == 43,
              "four executions amortize preparation");
        budget expired;
        expired.deadline = clock::now();
        const auto retained = choose(graph, method, expired);
        check(retained.placement == graph.current && retained.evaluated == 0 && retained.reason == "deadline",
              "expired decision preserves the complete current plan");
    }
}

void accumulate_small_savings() {
    problem graph;
    graph.units.resize(7);
    graph.units.front().allowed = graph.units.back().allowed = 1;
    graph.current.assign(graph.units.size(), local);
    graph.capacity = {16, 16};
    graph.switch_ms = 4.5;
    for (size_t i = 0; i < graph.units.size(); ++i) {
        graph.timings.push_back({i, i + 1, {2, 1}});
    }
    const auto result = choose(graph, algorithm::local_search);
    check(result.placement == assignment{local, remote, remote, remote, remote, remote, local} &&
              result.cost.total_ms == 13.5,
          "local search accumulates small gains before charging preparation");
}

void scoring_examples() {
    auto graph = two_units();
    graph.timings.push_back({0, 2, {11, 5}});
    check(score(graph, graph.current).compute_ms == 11, "coarse timing replaces child timings");
    check(score(graph, {remote, local}).compute_ms == 9, "mixed plan uses complete nonoverlapping child timings");
    graph.together = {{0, 2}};
    check(score(graph, {remote, local}).reason == "required group split", "fused group stays together");

    graph = {};
    graph.units.resize(3);
    graph.current = {local, local, local};
    graph.capacity = {32, 32};
    graph.switch_ms = 0;
    graph.timings = {{0, 1, {1, 1}}, {1, 2, {1, 1}}, {2, 3, {1, 1}}};
    graph.values = {{0, {1, 2}, 0, 0, 8, {7, 2}}};
    const assignment plan{local, remote, remote};
    auto cost = score(graph, plan);
    check(cost.transfer_ms == 2 && cost.handoffs.size() == 1, "two readers share one uploaded version");
    graph.values.push_back({1, {2}, 0, 1, 8, {7, 2}});
    cost = score(graph, plan);
    check(cost.transfer_ms == 9 && cost.handoffs.size() == 2, "a newer version owes a distinct directional writeback");
    graph.capacity[remote] = 8;
    check(score(graph, plan).reason == "capacity exceeded", "overlapping versions exceed memory");
}

}

int main() {
    try {
        search_examples();
        scoring_examples();
        accumulate_small_savings();
        std::cout << "Runtime placement regression passed\n";
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
