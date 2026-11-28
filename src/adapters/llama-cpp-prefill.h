#pragma once

#include "llama.h"
#include "request-metrics.h"

#include <algorithm>
#include <cstring>

namespace umm::llama_prefill_detail {
inline void hash_prefill_input(request_metrics * metrics, const llama_batch & batch, int position_axes = 4) {
    if (!metrics)
        return;
    for (int i = 0; i < batch.n_tokens; ++i) {
        const int32_t values[] = {batch.token ? batch.token[i] : LLAMA_TOKEN_NULL, batch.pos[i],
                                  position_axes > 1 ? batch.pos[batch.n_tokens + i] : 0,
                                  position_axes > 2 ? batch.pos[2 * batch.n_tokens + i] : 0,
                                  position_axes > 3 ? batch.pos[3 * batch.n_tokens + i] : 0};
        for (int32_t value : values) {
            for (unsigned shift = 0; shift < 32; shift += 8) {
                metrics->prefill_input_hash ^= (uint32_t(value) >> shift) & 255;
                metrics->prefill_input_hash *= 1099511628211ULL;
            }
        }
    }
}

inline void record_prefill(request_metrics * metrics, const char * kind, size_t cached_tokens, int tokens, int outputs,
                           double elapsed) {
    if (!metrics)
        return;
    metrics->prefill_batches.push_back({kind, tokens, int64_t(cached_tokens), outputs, elapsed});
    auto & duration =
        std::strcmp(kind, "mixed") == 0
            ? metrics->mixed_prefill_ms
            : (std::strcmp(kind, "image") == 0 ? metrics->image_prefill_ms : metrics->text_prefill_compute_ms);
    duration = std::max(0.0, duration) + elapsed;
}
} // namespace
