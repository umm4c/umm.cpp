#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct options {
    int dims = 64, context = 512, mode = GGML_ROPE_TYPE_NEOX;
    float base = 10000, scale = 1, ext = 0, attention = 1, fast = 32, slow = 1;
    bool factors = false, alternate_sections = false;
};

void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

// The reference executes each RoPE in isolation; coefficients cannot be reused there.
void run(ggml_backend_t backend, const char * name, options first, options second,
         int tokens, bool mutate, bool separate_graphs, bool position_view) {
    using context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
    using buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
    context ctx(ggml_init({4 * 1024 * 1024, nullptr, true}), ggml_free);
    const bool multi = (first.mode & GGML_ROPE_TYPE_MROPE) || (second.mode & GGML_ROPE_TYPE_MROPE);
    const int position_count = multi ? tokens * 4 : tokens;
    auto storage = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, position_count * (position_view ? 2 : 1));
    auto positions = position_view ? ggml_view_1d(ctx.get(), storage, position_count, position_count * sizeof(int32_t)) : storage;
    // Deliberately leave INPUT unset: scheduler-created input copies can have flags=0.
    auto replacement = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, position_count);
    auto alias = ggml_view_1d(ctx.get(), storage, position_count, position_view ? position_count * sizeof(int32_t) : 0);
    auto copy = ggml_cpy(ctx.get(), replacement, alias);
    ggml_tensor * inputs[2];
    ggml_tensor * outputs[2];
    std::vector<float> values[2];
    const options params[] = {first, second};
    for (int i = 0; i < 2; ++i) {
        const auto & p = params[i];
        inputs[i] = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, p.dims, 2, tokens);
        values[i].resize(p.dims * 2 * tokens);
        for (size_t j = 0; j < values[i].size(); ++j) values[i][j] = float(int((j * 7 + i) % 31) - 15) / 32;
        ggml_tensor * factors = nullptr;
        if (p.factors) {
            factors = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, p.dims / 2);
        }
        if (p.mode & GGML_ROPE_TYPE_MROPE) {
            int sections[] = {p.dims / 4, p.dims / 8, p.dims / 8, 0};
            if (p.alternate_sections) std::swap(sections[0], sections[1]);
            outputs[i] = ggml_rope_multi(ctx.get(), inputs[i], positions, factors, p.dims, sections, p.mode,
                                        p.context, p.base, p.scale, p.ext, p.attention, p.fast, p.slow);
        } else {
            outputs[i] = ggml_rope_ext(ctx.get(), inputs[i], positions, factors, p.dims, p.mode,
                                      p.context, p.base, p.scale, p.ext, p.attention, p.fast, p.slow);
        }
    }
    auto single0 = ggml_new_graph_custom(ctx.get(), 64, false);
    auto single1 = ggml_new_graph_custom(ctx.get(), 64, false);
    auto update = ggml_new_graph_custom(ctx.get(), 64, false);
    auto combined = ggml_new_graph_custom(ctx.get(), 64, false);
    ggml_build_forward_expand(single0, outputs[0]);
    ggml_build_forward_expand(single1, outputs[1]);
    ggml_build_forward_expand(update, copy);
    ggml_build_forward_expand(combined, outputs[0]);
    if (mutate) ggml_build_forward_expand(combined, copy);
    ggml_build_forward_expand(combined, outputs[1]);
    buffer memory(ggml_backend_alloc_ctx_tensors(ctx.get(), backend), ggml_backend_buffer_free);
    require(bool(memory), "allocation failed");
    for (int i = 0; i < 2; ++i) if (params[i].factors) {
        std::vector<float> factors(params[i].dims / 2);
        for (size_t j = 0; j < factors.size(); ++j) factors[j] = 1 + float((j + i * 3) % 7) / 8;
        ggml_backend_tensor_set(outputs[i]->src[2], factors.data(), 0, factors.size() * sizeof(float));
    }
    std::vector<int32_t> before(position_count), after(position_count), packed(position_count * (position_view ? 2 : 1));
    for (int i = 0; i < position_count; ++i) {
        before[i] = 3 + i * 7;
        after[i] = 101 + i * 19;
    }
    auto set_positions = [&](const std::vector<int32_t> & value) {
        std::copy(value.begin(), value.end(), packed.begin() + (position_view ? position_count : 0));
        ggml_backend_tensor_set(storage, packed.data(), 0, packed.size() * sizeof(int32_t));
    };
    std::vector<float> expected[2];
    for (int i = 0; i < 2; ++i) {
        set_positions(i && mutate ? after : before);
        ggml_backend_tensor_set(inputs[i], values[i].data(), 0, ggml_nbytes(inputs[i]));
        require(ggml_backend_graph_compute(backend, i ? single1 : single0) == GGML_STATUS_SUCCESS, "reference failed");
        expected[i].resize(values[i].size());
        ggml_backend_tensor_get(outputs[i], expected[i].data(), 0, ggml_nbytes(outputs[i]));
    }
    set_positions(before);
    ggml_backend_tensor_set(replacement, after.data(), 0, after.size() * sizeof(int32_t));
    for (int i = 0; i < 2; ++i) ggml_backend_tensor_set(inputs[i], values[i].data(), 0, ggml_nbytes(inputs[i]));
    if (separate_graphs) {
        require(ggml_backend_graph_compute_async(backend, single0) == GGML_STATUS_SUCCESS, "first async graph failed");
        if (mutate) require(ggml_backend_graph_compute_async(backend, update) == GGML_STATUS_SUCCESS, "async update failed");
        require(ggml_backend_graph_compute_async(backend, single1) == GGML_STATUS_SUCCESS, "second async graph failed");
    } else {
        require(ggml_backend_graph_compute_async(backend, combined) == GGML_STATUS_SUCCESS, "combined graph failed");
    }
    ggml_backend_synchronize(backend);
    for (int i = 0; i < 2; ++i) {
        std::vector<float> actual(values[i].size());
        ggml_backend_tensor_get(outputs[i], actual.data(), 0, ggml_nbytes(outputs[i]));
        for (size_t j = 0; j < actual.size(); ++j) {
            require(std::isfinite(actual[j]), "nonfinite output");
            require(std::abs(actual[j] - expected[i][j]) <= 1e-5f, "RoPE differs from isolated reference");
        }
    }
    std::printf("PASS %s tokens=%d mutation=%d async_graphs=%d view=%d\n", name, tokens, mutate, separate_graphs, position_view);
}
}

int main() {
    try {
        ggml_backend_load_all();
        using backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
        backend device(ggml_backend_init_by_name("CANN0", nullptr), ggml_backend_free);
        require(bool(device), "CANN0 unavailable");
        for (int tokens : {1, 3, 17}) {
            options base;
            run(device.get(), "same parameters", base, base, tokens, false, false, false);
            run(device.get(), "position view", base, base, tokens, false, false, true);
            run(device.get(), "same-graph direct position write", base, base, tokens, true, false, false);
            run(device.get(), "same-graph alias write", base, base, tokens, true, false, true);
            run(device.get(), "async graph position update", base, base, tokens, true, true, true);
            options narrow = base; narrow.dims = 32;
            run(device.get(), "different dimensions", base, narrow, tokens, false, false, true);
            options yarn = base; yarn.ext = 1; yarn.scale = .25f;
            options corrected = yarn; corrected.context = 8192;
            run(device.get(), "YaRN context key", yarn, corrected, tokens, false, false, true);
            corrected = yarn; corrected.fast = 8;
            run(device.get(), "YaRN beta-fast key", yarn, corrected, tokens, false, false, true);
            corrected = yarn; corrected.slow = 2;
            run(device.get(), "YaRN beta-slow key", yarn, corrected, tokens, false, false, true);
            options factored = base; factored.factors = true;
            run(device.get(), "frequency factors", factored, factored, tokens, false, false, true);
            options multi = base; multi.mode = GGML_ROPE_TYPE_MROPE;
            run(device.get(), "multi-axis positions", multi, multi, tokens, false, false, true);
            options rearranged = multi; rearranged.alternate_sections = true;
            run(device.get(), "multi-axis section key", multi, rearranged, tokens, false, false, true);
        }
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
