#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace umm::runtime_placement {

using clock = std::chrono::steady_clock;
using device = uint8_t;
using assignment = std::vector<device>;
constexpr device local = 0;
constexpr device remote = 1;
constexpr double unknown = std::numeric_limits<double>::quiet_NaN();

enum class algorithm { local_search, min_cut, beam_search };

struct unit {
    uint8_t allowed = 3;
    std::array<size_t, 2> workspace{};
    bool executes = true;
};

// Each entry represents one independently transferable storage range and version.
struct value {
    int producer = -1;
    std::vector<size_t> readers;
    uint8_t initial = 1;
    uint8_t terminal = 0;
    size_t bytes = 0;
    std::array<double, 2> transfer_ms{unknown, unknown}; // Indexed by destination.
    bool persistent = false; // Initially resident memory is deducted from the available budget.
};

struct timing {
    size_t begin;
    size_t end;
    std::array<double, 2> ms;
};

struct group {
    size_t begin;
    size_t end;
};

struct problem {
    std::vector<unit> units;
    std::vector<value> values;
    std::vector<timing> timings; // Same comparable time basis, half-open unit ranges.
    std::vector<group> together;
    std::vector<std::array<size_t, 2>> equal;
    std::array<size_t, 2> capacity{0, 0};
    std::array<double, 2> wait_ms{};
    std::array<double, 2> submission_ms{};
    assignment current;
    size_t repetitions = 1;
    double switch_ms = unknown;
    double margin_ms = 0;
};

struct handoff {
    size_t value_index;
    size_t before_unit;
    device source;
    device destination;
};

struct evaluation {
    double total_ms = unknown;
    double compute_ms = 0;
    double transfer_ms = 0;
    double wait_ms = 0;
    double preparation_ms = 0;
    std::array<size_t, 2> peak{};
    std::vector<timing> coverage;
    std::vector<handoff> handoffs;
    std::string reason;
};

struct budget {
    clock::time_point deadline = clock::time_point::max();
    size_t actions = 16384;
    size_t starts = 4;
    size_t beam_width = 8;
};

struct decision {
    assignment placement;
    evaluation cost;
    double savings_ms = 0;
    double planning_ms = 0;
    size_t evaluated = 0;
    std::string reason;
};

evaluation score(const problem & input, const assignment & placement,
                 clock::time_point deadline = clock::time_point::max());
decision choose(const problem & input, algorithm method, const budget & limits = {});

}
