#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace umm {

using metrics_clock = std::chrono::steady_clock;

inline double metrics_elapsed_ms(metrics_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(metrics_clock::now() - start).count();
}

// Negative values mean that the corresponding phase was not observed.
struct prefill_batch_metrics {
    std::string kind;
    int64_t tokens;
    int64_t cached_tokens;
    int64_t outputs;
    double elapsed_ms;
};

struct request_metrics {
    metrics_clock::time_point request_start = metrics_clock::now();
    metrics_clock::time_point inference_start{};
    metrics_clock::time_point decode_start{};
    double image_decode_ms = -1;
    double input_preprocess_ms = -1;
    double generation_model_load_ms = -1;
    double vision_model_load_ms = -1;
    double vision_encode_ms = -1;
    double image_encoder_ms = -1;
    double image_prefill_ms = -1;
    double text_prefill_compute_ms = -1;
    double mixed_prefill_ms = -1;
    double text_prefill_ms = -1;
    double text_decode_ms = -1;
    double ttft_ms = -1;
    double conditioning_ms = -1;
    double diffusion_steps_ms = -1;
    double vae_decode_ms = -1;
    double image_write_ms = -1;
    int64_t input_tokens = -1;
    int64_t vision_tokens = 0;
    int64_t output_tokens = -1;
    std::vector<double> diffusion_step_ms;
    std::vector<prefill_batch_metrics> prefill_batches;
    int64_t attention_mode_switches = 0;
    // FNV-1a over token IDs and all four position axes, independent of batching.
    uint64_t prefill_input_hash = 14695981039346656037ULL;
};

} // namespace umm
