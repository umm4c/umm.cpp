#pragma once
#include <cstdint>

namespace umm::profiling {
struct phase_context {
    const char * phase = "unclassified";
    int64_t tokens = -1;
    int64_t cached_tokens = -1;
    bool single_ubatch = false;
    double rebuild_ms = 0;
};
inline thread_local phase_context current_phase;
struct phase_scope {
    phase_context previous = current_phase;
    explicit phase_scope(const char * phase, int64_t tokens = -1, int64_t cached = -1, bool single_ubatch = false) {
        current_phase = {phase, tokens, cached, single_ubatch};
    }
    ~phase_scope() { current_phase = previous; }
    phase_scope(const phase_scope &) = delete;
    phase_scope & operator=(const phase_scope &) = delete;
};
}
