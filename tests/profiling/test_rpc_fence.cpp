// Optional integration check against an already running single-device RPC worker.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-rpc.h"
#include "ggml-cpu.h"
#include "json.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <vector>
namespace {
struct scheduler_check {
    ggml_tensor * output;
    float expected;
    int parents = 0, children = 0;
};
bool observe_scheduler(ggml_backend_t backend, ggml_backend_sched_t, ggml_cgraph *, bool completed, ggml_status status,
                       void * user) {
    auto & check = *static_cast<scheduler_check *>(user);
    if (status != GGML_STATUS_SUCCESS)
        return false;
    if (completed) {
        if (backend)
            ++check.children;
        else {
            ++check.parents;
            // Final output is host-resident, so checking it cannot fence RPC work.
            for (int64_t i = 0; i < ggml_nelements(check.output); ++i)
                if (static_cast<float *>(check.output->data)[i] != check.expected)
                    return false;
        }
    }
    return true;
}
void check_multi_split(ggml_backend_t remote) {
    auto cpu = ggml_backend_cpu_init();
    ggml_backend_t backends[] = {remote, cpu};
    auto ctx = ggml_init({1024 * 1024, nullptr, true});
    auto a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 16384), b = ggml_dup_tensor(ctx, a);
    auto first = ggml_add(ctx, a, b), middle = ggml_add(ctx, first, b), last = ggml_add(ctx, middle, a);
    auto graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(graph, last);
    auto sched = ggml_backend_sched_new(backends, nullptr, 2, 32, false, true);
    ggml_backend_sched_set_tensor_backend(sched, a, cpu);
    ggml_backend_sched_set_tensor_backend(sched, b, cpu);
    ggml_backend_sched_set_tensor_backend(sched, first, cpu);
    ggml_backend_sched_set_tensor_backend(sched, middle, remote);
    ggml_backend_sched_set_tensor_backend(sched, last, cpu);
    if (!ggml_backend_sched_alloc_graph(sched, graph))
        throw std::runtime_error("Multi-split allocation failed");
    if (ggml_backend_sched_get_tensor_backend(sched, first) != cpu ||
        ggml_backend_sched_get_tensor_backend(sched, middle) != remote ||
        ggml_backend_sched_get_tensor_backend(sched, last) != cpu || !ggml_backend_buffer_is_host(last->buffer))
        throw std::runtime_error("Multi-split placement mismatch");
    scheduler_check check{last, 0};
    std::vector<float> input(16384);
    std::array<ggml_rpc_profile_record, 64> records{};
    uint32_t count = 0;
    if (!ggml_backend_rpc_profile_begin(remote, 42, records.size()))
        throw std::runtime_error("Multi-split window start failed");
    for (int replay = 0; replay < 3; ++replay) {
        const float value = 1.25f + replay;
        std::fill(input.begin(), input.end(), value);
        check.expected = 4 * value;
        ggml_backend_tensor_set(a, input.data(), 0, ggml_nbytes(a));
        ggml_backend_tensor_set(b, input.data(), 0, ggml_nbytes(b));
        ggml_backend_set_graph_submission_observer(observe_scheduler, &check);
        const auto status = ggml_backend_sched_graph_compute_async(sched, graph);
        ggml_backend_set_graph_submission_observer(nullptr, nullptr);
        if (status != GGML_STATUS_SUCCESS)
            throw std::runtime_error("CPU/RPC/CPU completion or output mismatch");
    }
    if (check.parents != 3 || check.children != 9)
        throw std::runtime_error("Multi-split observation count mismatch");
    if (!ggml_backend_rpc_profile_end(remote, 42, records.data(), records.size(), &count) || count != 3)
        throw std::runtime_error("Multi-split worker record mismatch");
    if (records[0].reused || !records[1].reused || !records[2].reused)
        throw std::runtime_error("GRAPH/RECOMPUTE coverage mismatch");
    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    ggml_backend_free(cpu);
}
void check_large_uploads(ggml_backend_t backend) {
    const int64_t n = 10 * 1024 * 1024 / sizeof(float) + 1;
    auto ctx = ggml_init({1024 * 1024, nullptr, true});
    auto weights_ctx = ggml_init({1024 * 1024, nullptr, true});
    if (!ctx || !weights_ctx)
        throw std::runtime_error("Upload context allocation failed");
    auto base = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n + 16);
    auto input = ggml_view_1d(ctx, base, n, 16 * sizeof(float));
    auto weights = ggml_new_tensor_1d(weights_ctx, GGML_TYPE_F32, n);
    auto out = ggml_add(ctx, input, weights);
    auto graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(graph, out);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    auto weights_buffer = ggml_backend_alloc_ctx_tensors(weights_ctx, backend);
    if (!buffer || !weights_buffer)
        throw std::runtime_error("Upload buffer allocation failed");
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    ggml_backend_buffer_set_usage(weights_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::vector<float> data(n), reference(n, 1.0f), actual(n);
    for (int replay = 0; replay < 3; ++replay) {
        const float value = 1.25f + replay;
        std::fill(data.begin(), data.end(), value);
        if (replay % 2) {
            ggml_backend_tensor_set_async(backend, weights, reference.data(), 0, ggml_nbytes(weights));
            ggml_backend_tensor_set_async(backend, input, data.data(), 0, ggml_nbytes(input));
        } else {
            ggml_backend_tensor_set(weights, reference.data(), 0, ggml_nbytes(weights));
            ggml_backend_tensor_set(input, data.data(), 0, ggml_nbytes(input));
        }
        for (int patch = 0; patch < 2; ++patch) {
            if (patch) {
                const float changed = value + 1;
                ggml_backend_tensor_set(input, &changed, 3 * sizeof(float), sizeof(float));
            }
            if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS)
                throw std::runtime_error("Upload graph execution failed");
            ggml_backend_tensor_get(out, actual.data(), 0, ggml_nbytes(out));
            for (int64_t i = 0; i < n; ++i)
                if (actual[i] != value + 1 + (patch && i == 3 ? 1 : 0))
                    throw std::runtime_error("Large/view/partial upload or cached recompute mismatch");
        }
    }
    ggml_backend_buffer_free(weights_buffer);
    ggml_backend_buffer_free(buffer);
    ggml_free(weights_ctx);
    ggml_free(ctx);
}

}
int main(int argc, char ** argv) {
    if (argc != 2 && argc != 3)
        throw std::invalid_argument("Usage: umm-profiler-rpc-fence-test ENDPOINT [legacy]");
    auto backend = ggml_backend_rpc_init(argv[1], 0);
    if (!backend)
        throw std::runtime_error("RPC initialization failed");
    if (argc == 3) {
        if (std::string(argv[2]) != "legacy" || ggml_backend_rpc_profile_begin(backend, 1, 64))
            throw std::runtime_error("Legacy worker unexpectedly supports batch timing");
        ggml_backend_free(backend);
        std::puts("PASS: legacy worker rejected profiling before sending unknown commands");
        return 0;
    }
    auto ctx = ggml_init({1024 * 1024, nullptr, true});
    if (!ctx)
        throw std::runtime_error("Context allocation failed");
    auto a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 16384), b = ggml_dup_tensor(ctx, a), out = ggml_add(ctx, a, b);
    auto graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(graph, out);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer)
        throw std::runtime_error("Buffer allocation failed");
    std::vector<float> input(16384, 1.25f), output(16384);
    ggml_backend_tensor_set(a, input.data(), 0, ggml_nbytes(a));
    ggml_backend_tensor_set(b, input.data(), 0, ggml_nbytes(b));
    nlohmann::json records = nlohmann::json::array();
    std::array<unsigned char, 65536> scratch{};
    for (size_t size : {size_t(1), size_t(64), size_t(4096), size_t(65536)}) {
        nlohmann::json samples = nlohmann::json::array();
        for (int i = -2; i < 5; ++i) {
            auto start = std::chrono::steady_clock::now();
            if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS)
                throw std::runtime_error("Graph execution failed");
            ggml_backend_tensor_get(out, scratch.data(), 0, size);
            if (i >= 0)
                samples.push_back(
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
        }
        records.push_back({{"completion_read_bytes", size}, {"samples_ms", samples}});
    }
    ggml_backend_tensor_get(out, output.data(), 0, ggml_nbytes(out));
    for (auto v : output)
        if (v != 2.5f)
            throw std::runtime_error("RPC output mismatch");
    std::array<ggml_rpc_profile_record, 64> batch{};
    uint32_t count = 0;
    if (ggml_backend_rpc_profile_begin(backend, 1, 0) || !ggml_backend_rpc_profile_begin(backend, 1, batch.size()) ||
        ggml_backend_rpc_profile_begin(backend, 2, batch.size()))
        throw std::runtime_error("Window state validation failed");
    for (int i = 0; i < 3; ++i)
        if (ggml_backend_graph_compute_async(backend, graph) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("Batch compute failed");
    if (ggml_backend_rpc_profile_end(backend, 2, batch.data(), batch.size(), &count) ||
        !ggml_backend_rpc_profile_end(backend, 1, batch.data(), batch.size(), &count) || count != 3)
        throw std::runtime_error("Window completion/count failed");
    for (uint32_t i = 0; i < count; ++i)
        if (batch[i].sequence != i || batch[i].n_nodes != 1 || batch[i].compute_ns == 0 ||
            (i && batch[i].start_ns < batch[i - 1].start_ns + batch[i - 1].prepare_ns + batch[i - 1].compute_ns))
            throw std::runtime_error("Worker time/order mismatch");
    if (!ggml_backend_rpc_profile_begin(backend, 3, 1))
        throw std::runtime_error("Overflow window start failed");
    ggml_backend_graph_compute_async(backend, graph);
    ggml_backend_graph_compute_async(backend, graph);
    if (ggml_backend_rpc_profile_end(backend, 3, batch.data(), 1, &count))
        throw std::runtime_error("Overflow silently accepted");
    if (!ggml_backend_rpc_profile_begin(backend, 4, batch.size()) ||
        !ggml_backend_rpc_profile_end(backend, 4, batch.data(), batch.size(), &count) || count != 0)
        throw std::runtime_error("Empty window/reuse failed");
    check_multi_split(backend);
    check_large_uploads(backend);
    std::puts(
        nlohmann::json(
            {{"status", "passed"},
             {"large_uploads", "sync/async, weight reuse, changed compute views, partial writes and graph recompute"},
             {"records", records},
             {"batch", "ordered worker records, completion, nesting, mismatch, overflow, empty window and reuse"},
             {"multi_split", "CPU/RPC/CPU: three replays, exact host output at scheduler END"}})
            .dump(2)
            .c_str());
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}
