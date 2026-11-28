#include "runtime-placement.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <queue>
#include <utility>

namespace umm::runtime_placement {
namespace {

bool known(double time) {
    return std::isfinite(time) && time >= 0;
}
unsigned bit(device at) {
    return 1u << at;
}

struct graph_index {
    const problem & input;
    std::vector<std::vector<size_t>> reads, writes, releases;
    std::vector<std::vector<const timing *>> timings;
    bool valid = true;
    std::vector<size_t> linked;
    std::vector<unsigned> allowed;

    explicit graph_index(const problem & data, clock::time_point deadline)
        : input(data), reads(data.units.size() + 1), writes(reads.size()), releases(reads.size()),
          timings(reads.size()), linked(data.units.size()), allowed(data.units.size(), 3) {
        const auto count = data.units.size();
        for (size_t i = 0; i < data.values.size(); ++i) {
            if (clock::now() >= deadline) {
                valid = false;
                return;
            }
            const auto & v = data.values[i];
            if (v.producer < -1 || v.producer >= int(count) || v.initial > 3 || v.terminal > 3 ||
                (v.producer < 0) != (v.initial != 0)) {
                valid = false;
                return;
            }
            size_t last = v.producer < 0 ? 0 : size_t(v.producer);
            if (v.producer >= 0)
                writes[size_t(v.producer)].push_back(i);
            for (auto reader : v.readers) {
                if (reader >= count || (v.producer >= 0 && reader <= size_t(v.producer))) {
                    valid = false;
                    return;
                }
                reads[reader].push_back(i);
                last = std::max(last, reader);
            }
            releases[v.terminal ? count : last].push_back(i);
        }
        for (const auto & t : data.timings) {
            if (clock::now() >= deadline) {
                valid = false;
                return;
            }
            if (t.begin >= t.end || t.end > count) {
                valid = false;
                return;
            }
            timings[t.begin].push_back(&t);
        }
        for (size_t i = 0; i < count; ++i)
            linked[i] = i;
        auto join = [&](size_t a, size_t b) {
            while (linked[a] != a)
                a = linked[a];
            while (linked[b] != b)
                b = linked[b];
            linked[std::max(a, b)] = std::min(a, b);
        };
        for (const auto & pair : data.equal) {
            if (pair[0] >= count || pair[1] >= count) {
                valid = false;
                return;
            }
            join(pair[0], pair[1]);
        }
        for (const auto & g : data.together) {
            if (g.begin >= g.end || g.end > count) {
                valid = false;
                return;
            }
            for (size_t i = g.begin + 1; i < g.end; ++i)
                join(g.begin, i);
        }
        for (size_t i = 0; i < count; ++i)
            linked[i] = linked[linked[i]];
        for (size_t i = 0; i < count; ++i)
            allowed[linked[i]] &= data.units[i].allowed;
        for (size_t i = 0; i < count; ++i)
            allowed[i] = allowed[linked[i]];
        for (auto & ranges : timings) {
            std::stable_sort(ranges.begin(), ranges.end(),
                             [](const auto * a, const auto * b) { return a->end > b->end; });
        }
    }
};

struct frontier {
    std::vector<uint8_t> copies;
    std::array<size_t, 2> used{};
    std::array<size_t, 2> peak{};
    std::array<bool, 2> active{};
    double transfer_ms = 0;
    double wait_ms = 0;
    device previous = 2;
    const char * reason = nullptr;

    explicit frontier(const graph_index & graph) : copies(graph.input.values.size()) {
        for (size_t i = 0; i < copies.size() && !reason; ++i) {
            const auto & v = graph.input.values[i];
            copies[i] = v.initial;
            if (!v.persistent) {
                for (device d : {local, remote}) {
                    if (v.initial & bit(d))
                        reserve(graph, d, v.bytes);
                }
            }
        }
    }

    bool reserve(const graph_index & graph, device at, size_t bytes) {
        if (reason)
            return false;
        if (bytes > graph.input.capacity[at] - used[at]) {
            reason = "capacity exceeded";
            return false;
        }
        used[at] += bytes;
        peak[at] = std::max(peak[at], used[at]);
        return true;
    }

    bool need(const graph_index & graph, size_t i, device at, size_t before, std::vector<handoff> * transfers) {
        const auto & v = graph.input.values[i];
        if (copies[i] & bit(at))
            return true;
        if (!copies[i] || !known(v.transfer_ms[at])) {
            reason = "missing source or path calibration";
            return false;
        }
        if (!reserve(graph, at, v.bytes))
            return false;
        copies[i] |= bit(at);
        transfer_ms += v.transfer_ms[at];
        if (transfers)
            transfers->push_back({i, before, device(1 - at), at});
        return true;
    }

    bool step(const graph_index & graph, size_t i, device at, std::vector<handoff> * transfers) {
        if (reason)
            return false;
        if (i < graph.input.units.size()) {
            const auto & u = graph.input.units[i];
            if (at > remote || !(graph.allowed[i] & bit(at))) {
                reason = "unsupported device";
                return false;
            }
            if (u.executes && at != previous) {
                if (!known(graph.input.submission_ms[at])) {
                    reason = "missing submission overhead";
                    return false;
                }
                wait_ms += graph.input.submission_ms[at];
                previous = at;
            }
            if (u.executes && !active[at]) {
                if (!known(graph.input.wait_ms[at])) {
                    reason = "missing device wait estimate";
                    return false;
                }
                wait_ms += graph.input.wait_ms[at];
                active[at] = true;
            }
            for (auto id : graph.reads[i]) {
                if (!need(graph, id, at, i, transfers))
                    return false;
            }
            const auto workspace = u.workspace[at];
            if (!reserve(graph, at, workspace))
                return false;
            for (auto id : graph.writes[i]) {
                if (!reserve(graph, at, graph.input.values[id].bytes))
                    return false;
                copies[id] = bit(at);
            }
            used[at] -= workspace;
        } else {
            for (auto id : graph.releases[i]) {
                for (device d : {local, remote}) {
                    if ((graph.input.values[id].terminal & bit(d)) && !need(graph, id, d, i, transfers))
                        return false;
                }
            }
        }
        for (auto id : graph.releases[i]) {
            const auto & v = graph.input.values[id];
            for (device d : {local, remote}) {
                if ((copies[id] & bit(d)) && !(v.persistent && (v.initial & bit(d))))
                    used[d] -= v.bytes;
            }
            copies[id] = 0;
        }
        return true;
    }
};

evaluation assess(const graph_index & graph, const assignment & plan, clock::time_point deadline) {
    const auto & input = graph.input;
    evaluation result;
    const auto count = input.units.size();
    auto fail = [&](const char * reason) {
        result.reason = reason;
        return result;
    };
    if (clock::now() >= deadline)
        return fail("deadline");
    if (!graph.valid || plan.size() != count || !input.repetitions)
        return fail("invalid placement problem");
    for (size_t i = 0; i < count; ++i) {
        if (plan[i] != plan[graph.linked[i]])
            return fail("required group split");
    }
    frontier state(graph);
    for (size_t i = 0; i < count; ++i) {
        if (clock::now() >= deadline)
            return fail("deadline");
        if (!state.step(graph, i, plan[i], &result.handoffs))
            return fail(state.reason);
    }
    if (!state.step(graph, count, local, &result.handoffs))
        return fail(state.reason);
    result.transfer_ms = state.transfer_ms;
    result.wait_ms = state.wait_ms;
    result.peak = state.peak;
    // Prefer the coarsest prefix whose remainder has a complete cover.
    std::vector<const timing *> cover(count + 1);
    size_t run_end = count;
    for (size_t i = count; i-- > 0;) {
        if (clock::now() >= deadline)
            return fail("deadline");
        if (i + 1 < count && plan[i] != plan[i + 1])
            run_end = i + 1;
        for (const auto * range : graph.timings[i]) {
            if (range->end <= run_end && (range->end == count || cover[range->end]) && known(range->ms[plan[i]])) {
                cover[i] = range;
                break;
            }
        }
    }
    for (size_t i = 0; i < count;) {
        if (!cover[i])
            return fail("missing comparable timing coverage");
        result.coverage.push_back(*cover[i]);
        result.compute_ms += cover[i]->ms[plan[i]];
        i = cover[i]->end;
    }
    if (plan != input.current) {
        if (!known(input.switch_ms))
            return fail("missing switching cost");
        result.preparation_ms = input.switch_ms;
    }
    const double total =
        result.preparation_ms + input.repetitions * (result.compute_ms + result.transfer_ms + result.wait_ms);
    if (!known(total))
        return fail("non-finite cost");
    result.total_ms = total;
    return result;
}

struct search {
    const graph_index & graph;
    const budget & limits;
    decision best;
    size_t actions = 0;
    const double baseline;
    std::string rejected;

    search(const graph_index & indexed, const budget & allowed)
        : graph(indexed), limits(allowed), baseline(initialize()) {}

    double initialize() {
        best.placement = graph.input.current;
        best.cost = assess(graph, best.placement, limits.deadline);
        return best.cost.total_ms;
    }

    bool running() const { return actions < limits.actions && clock::now() < limits.deadline; }

    double evaluate(const assignment & plan) {
        if (!running())
            return unknown;
        ++actions;
        ++best.evaluated;
        auto cost = assess(graph, plan, limits.deadline);
        if (!known(cost.total_ms)) {
            if (rejected.empty())
                rejected = cost.reason;
            return unknown;
        }
        const double total = cost.total_ms;
        const double execution = total - cost.preparation_ms;
        if (baseline - total > graph.input.margin_ms && total < best.cost.total_ms) {
            best.placement = plan;
            best.cost = std::move(cost);
        }
        // A local step can improve execution before a sequence of steps
        // has recouped the one-time cost of switching the complete plan.
        return execution;
    }
};

void improve_locally(search & run) {
    const auto & graph = run.graph;
    const auto count = graph.input.units.size();
    std::vector<group> moves{{0, count}};
    for (const auto & g : graph.input.together)
        moves.push_back(g);
    for (const auto & ranges : graph.timings) {
        for (auto * t : ranges)
            if (t->end - t->begin > 1)
                moves.push_back({t->begin, t->end});
    }
    for (size_t i = 0; i < count; ++i) {
        if (i + 1 < count)
            moves.push_back({i, i + 2});
        moves.push_back({i, i + 1});
    }
    std::sort(moves.begin(), moves.end(), [](const auto & a, const auto & b) {
        if (a.end - a.begin != b.end - b.begin)
            return a.end - a.begin > b.end - b.begin;
        return a.begin < b.begin;
    });
    moves.erase(std::unique(moves.begin(), moves.end(),
                            [](const auto & a, const auto & b) { return a.begin == b.begin && a.end == b.end; }),
                moves.end());
    auto move = [&](assignment & candidate, const group & g, device at) {
        for (size_t i = g.begin; i < g.end; ++i)
            candidate[graph.linked[i]] = at;
        for (size_t i = 0; i < candidate.size(); ++i)
            candidate[i] = candidate[graph.linked[i]];
    };
    std::vector<group> seeds;
    for (size_t i = 0; i < std::min(run.limits.starts, moves.size()); ++i) {
        seeds.push_back(moves[i * moves.size() / std::min(run.limits.starts, moves.size())]);
    }
    for (size_t start = 0; start < run.limits.starts && run.running(); ++start) {
        auto current = graph.input.current;
        double current_ms = run.baseline;
        if (start && start - 1 < seeds.size()) {
            move(current, seeds[start - 1], remote);
            current_ms = run.evaluate(current);
        }
        const auto remaining = (run.limits.actions - run.actions) / (run.limits.starts - start);
        const auto limit = run.actions + std::max(size_t{1}, remaining);
        for (bool changed = true; changed && run.running() && run.actions < limit;) {
            changed = false;
            for (const auto & g : moves) {
                if (!run.running() || run.actions >= limit)
                    break;
                for (device at : {remote, local}) {
                    if (std::any_of(graph.allowed.begin() + g.begin, graph.allowed.begin() + g.end,
                                    [at](unsigned mask) { return !(mask & bit(at)); }))
                        continue;
                    auto candidate = current;
                    move(candidate, g, at);
                    if (candidate == current || !run.running() || run.actions >= limit)
                        continue;
                    const double cost = run.evaluate(candidate);
                    if (known(cost) && (!known(current_ms) || cost < current_ms)) {
                        current = std::move(candidate);
                        current_ms = cost;
                        changed = true;
                    }
                }
            }
        }
    }
}

std::vector<timing> additive_cover(const graph_index & graph, clock::time_point deadline) {
    const auto count = graph.input.units.size();
    std::vector<int> segments(count + 1, -1);
    std::vector<const timing *> cover(count + 1);
    segments[count] = 0;
    for (size_t i = count; i-- > 0;) {
        if (clock::now() >= deadline)
            return {};
        for (auto * range : graph.timings[i]) {
            if (segments[range->end] < 0 || segments[range->end] + 1 <= segments[i])
                continue;
            uint8_t allowed = 3;
            for (size_t j = i; j < range->end; ++j)
                allowed &= graph.allowed[j];
            for (device d : {local, remote})
                if (!known(range->ms[d]))
                    allowed &= ~bit(d);
            if (allowed) {
                segments[i] = segments[range->end] + 1;
                cover[i] = range;
            }
        }
    }
    std::vector<timing> result;
    for (size_t i = 0; i < count;) {
        if (!cover[i])
            return {};
        result.push_back(*cover[i]);
        i = cover[i]->end;
    }
    return result;
}

class flow_graph {
    struct edge {
        size_t to, reverse;
        double remaining;
    };
    std::vector<std::vector<edge>> edges;
    std::vector<int> levels;
    std::vector<size_t> next;
    clock::time_point deadline;

    double augment(size_t v, size_t sink, double limit) {
        if (v == sink)
            return limit;
        if (clock::now() >= deadline)
            return 0;
        for (auto & i = next[v]; i < edges[v].size(); ++i) {
            auto & e = edges[v][i];
            if (e.remaining <= 0 || levels[e.to] != levels[v] + 1)
                continue;
            const double pushed = augment(e.to, sink, std::min(limit, e.remaining));
            if (pushed > 0) {
                e.remaining -= pushed;
                edges[e.to][e.reverse].remaining += pushed;
                return pushed;
            }
        }
        return 0;
    }

    void set_levels(size_t source) {
        std::fill(levels.begin(), levels.end(), -1);
        levels[source] = 0;
        std::queue<size_t> pending;
        pending.push(source);
        while (!pending.empty() && clock::now() < deadline) {
            const auto v = pending.front();
            pending.pop();
            for (const auto & e : edges[v]) {
                if (e.remaining > 0 && levels[e.to] == -1) {
                    levels[e.to] = levels[v] + 1;
                    pending.push(e.to);
                }
            }
        }
    }

public:
    flow_graph(size_t vertices, clock::time_point stop)
        : edges(vertices), levels(vertices), next(vertices), deadline(stop) {}

    void connect(size_t a, size_t b, double cost) {
        if (a == b)
            return;
        const auto index = edges[a].size();
        edges[a].push_back({b, edges[b].size(), cost});
        edges[b].push_back({a, index, 0});
    }

    bool solve(size_t source, size_t sink) {
        while (clock::now() < deadline) {
            set_levels(source);
            if (levels[sink] == -1)
                return clock::now() < deadline;
            std::fill(next.begin(), next.end(), 0);
            while (augment(source, sink, std::numeric_limits<double>::max()) > 0) {
            }
        }
        return false;
    }

    bool on_source_side(size_t v) const { return levels[v] >= 0; }
};

void cut(search & run) {
    const auto & data = run.graph.input;
    const auto count = data.units.size();
    auto cover = additive_cover(run.graph, run.limits.deadline);
    if (cover.empty())
        return;
    double finite = 0;
    for (const auto & range : cover) {
        for (double t : range.ms)
            if (known(t))
                finite += t;
    }
    for (const auto & v : data.values) {
        for (double t : v.transfer_ms)
            if (known(t))
                finite += t;
    }
    for (device d : {local, remote}) {
        if (known(data.wait_ms[d]))
            finite += data.wait_ms[d];
        if (known(data.submission_ms[d]))
            finite += count * data.submission_ms[d];
    }
    const double hard = 2 * finite + 1;
    if (!std::isfinite(hard))
        return;
    const auto source = count;
    const auto sink = count + 1;
    const auto local_gate = count + 2;
    const auto remote_gate = count + 3;
    flow_graph network(count + 4 + 2 * data.values.size(), run.limits.deadline);
    auto capacity = [hard](double cost) { return known(cost) ? cost : hard; };
    network.connect(source, remote_gate, capacity(data.wait_ms[remote]));
    network.connect(local_gate, sink, capacity(data.wait_ms[local]));
    size_t last = count;
    for (size_t i = 0; i < count; ++i) {
        if (!data.units[i].executes)
            continue;
        network.connect(remote_gate, i, hard);
        network.connect(i, local_gate, hard);
        if (last == count) {
            network.connect(source, i, capacity(data.submission_ms[remote]));
            network.connect(i, sink, capacity(data.submission_ms[local]));
        } else {
            network.connect(last, i, capacity(data.submission_ms[remote]));
            network.connect(i, last, capacity(data.submission_ms[local]));
        }
        last = i;
    }
    auto tie = [&](size_t a, size_t b) {
        network.connect(a, b, hard);
        network.connect(b, a, hard);
    };
    for (const auto & range : cover) {
        network.connect(source, range.begin, known(range.ms[remote]) ? range.ms[remote] : hard);
        network.connect(range.begin, sink, known(range.ms[local]) ? range.ms[local] : hard);
        for (size_t i = range.begin + 1; i < range.end; ++i)
            tie(range.begin, i);
    }
    for (size_t i = 0; i < count; ++i) {
        if (!(data.units[i].allowed & bit(remote)))
            network.connect(source, i, hard);
        if (!(data.units[i].allowed & bit(local)))
            network.connect(i, sink, hard);
    }
    for (size_t i = 0; i < count; ++i)
        tie(run.graph.linked[i], i);
    for (size_t i = 0; i < data.values.size() && run.running(); ++i) {
        const auto & v = data.values[i];
        if (v.initial == 3)
            continue;
        const auto producer = v.producer >= 0 ? size_t(v.producer) : v.initial & bit(local) ? source : sink;
        const auto upload = count + 4 + 2 * i;
        const auto download = upload + 1;
        network.connect(producer, upload, known(v.transfer_ms[remote]) ? v.transfer_ms[remote] : hard);
        network.connect(download, producer, known(v.transfer_ms[local]) ? v.transfer_ms[local] : hard);
        auto attach = [&](size_t reader) {
            network.connect(upload, reader, hard);
            network.connect(reader, download, hard);
        };
        for (auto reader : v.readers)
            attach(reader);
        if (v.terminal & bit(local))
            attach(source);
        if (v.terminal & bit(remote))
            attach(sink);
    }
    if (!run.running() || !network.solve(source, sink))
        return;
    assignment plan(count);
    for (size_t i = 0; i < count; ++i)
        plan[i] = network.on_source_side(i) ? local : remote;
    run.evaluate(plan);
}

void beam(search & run) {
    struct history {
        std::shared_ptr<const history> previous;
        group range;
        device at;
    };
    struct partial {
        frontier state;
        std::shared_ptr<const history> tail;
        double compute = 0;
        double rank = 0;
    };
    const auto & graph = run.graph;
    const auto cover = additive_cover(graph, run.limits.deadline);
    if (cover.empty() || !run.limits.beam_width)
        return;
    const size_t budget_per_path = 2 * cover.size() + 1;
    auto search_width = [&](size_t width) {
        std::vector<partial> beam{{frontier(graph), nullptr}};
        for (const auto & range : cover) {
            std::vector<partial> following;
            for (const auto & candidate : beam) {
                for (device at : {local, remote}) {
                    if (!run.running())
                        return;
                    if (!known(range.ms[at]))
                        continue;
                    ++run.actions;
                    auto next = candidate;
                    for (size_t i = range.begin; i < range.end && !next.state.reason; ++i) {
                        if (clock::now() >= run.limits.deadline)
                            return;
                        const auto linked = graph.linked[i];
                        if (linked < range.begin) {
                            auto prior = candidate.tail;
                            while (prior && prior->range.begin > linked)
                                prior = prior->previous;
                            if (!prior || prior->at != at) {
                                next.state.reason = "required group split";
                                break;
                            }
                        }
                        next.state.step(graph, i, at, nullptr);
                    }
                    if (next.state.reason)
                        continue;
                    if (range.end == graph.input.units.size() && !next.state.step(graph, range.end, local, nullptr))
                        continue;
                    next.compute += range.ms[at];
                    next.rank = next.compute + next.state.transfer_ms + next.state.wait_ms;
                    next.tail = std::make_shared<history>(history{candidate.tail, {range.begin, range.end}, at});
                    following.push_back(std::move(next));
                }
            }
            std::stable_sort(following.begin(), following.end(),
                             [](const auto & a, const auto & b) { return a.rank < b.rank; });
            if (following.size() > width)
                following.erase(following.begin() + width, following.end());
            beam = std::move(following);
            if (beam.empty())
                return;
        }
        for (const auto & candidate : beam) {
            if (!run.running())
                break;
            assignment plan(graph.input.units.size());
            for (auto h = candidate.tail; h; h = h->previous) {
                std::fill(plan.begin() + h->range.begin, plan.begin() + h->range.end, h->at);
            }
            run.evaluate(plan);
        }
    };
    auto available_width = [&] {
        return std::min(run.limits.beam_width, (run.limits.actions - run.actions) / budget_per_path);
    };
    // Complete a narrow pass before spending the remaining time on alternatives.
    if (available_width())
        search_width(1);
    if (run.running() && available_width() > 1)
        search_width(available_width());
}

}

evaluation score(const problem & input, const assignment & placement, clock::time_point deadline) {
    if (clock::now() >= deadline) {
        evaluation result;
        result.reason = "deadline";
        return result;
    }
    return assess(graph_index(input, deadline), placement, deadline);
}

decision choose(const problem & input, algorithm method, const budget & limits) {
    const auto started = clock::now();
    if (started >= limits.deadline) {
        decision result;
        result.placement = input.current;
        result.reason = result.cost.reason = "deadline";
        return result;
    }
    const graph_index graph(input, limits.deadline);
    search run(graph, limits);
    if (known(run.baseline) && known(input.margin_ms) && !input.units.empty() && run.running()) {
        switch (method) {
        case algorithm::local_search:
            improve_locally(run);
            break;
        case algorithm::min_cut:
            cut(run);
            break;
        case algorithm::beam_search:
            beam(run);
            break;
        }
    }
    run.best.planning_ms = std::chrono::duration<double, std::milli>(clock::now() - started).count();
    if (known(run.baseline))
        run.best.savings_ms = run.baseline - run.best.cost.total_ms;
    if (run.best.placement != input.current)
        run.best.reason = "improved";
    else if (!known(run.baseline))
        run.best.reason = run.best.cost.reason;
    else if (clock::now() >= limits.deadline)
        run.best.reason = "deadline";
    else if (!run.rejected.empty())
        run.best.reason = run.rejected;
    else
        run.best.reason = "no verified improvement";
    return std::move(run.best);
}

}
