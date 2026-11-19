#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-backend-impl.h"
#include <array>
#include <cstdio>
#include <stdexcept>
#include <thread>

namespace {
void require_at(bool ok, int line) {
    if (!ok)
        throw std::runtime_error("Graph observer contract failed at line " + std::to_string(line));
}
#define require(ok) require_at((ok), __LINE__)
struct observations {
    int begin = 0, end = 0;
    bool abort = false;
};
bool observe(ggml_backend_t, ggml_backend_sched_t, ggml_cgraph *, bool completed, ggml_status status, void * user) {
    auto & state = *static_cast<observations *>(user);
    if (completed) {
        ++state.end;
        if (status != GGML_STATUS_SUCCESS)
            return false;
    } else
        ++state.begin;
    return !state.abort;
}

// Model RPC queue semantics: synchronize drains submissions; a read completes work.
struct simulated_rpc {
    ggml_backend_t backend;
    ggml_backend_device device;
    ggml_backend_reg reg{};
    ggml_backend_dev_t original_device, original_buffer_device;
    ggml_backend_buffer_type_t buft;
    decltype(ggml_backend_i::graph_compute) compute;
    decltype(ggml_backend_i::synchronize) synchronize;
    decltype(ggml_backend_buffer_i::get_tensor) get = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_cgraph * pending = nullptr;
    ggml_cgraph * callback_graph = nullptr;
    int reads = 0, completed = 0;
    bool abort_child = false;
    bool fail_compute = false;
    static simulated_rpc * active;
    explicit simulated_rpc(ggml_backend_t b)
        : backend(b), device(*b->device), original_device(b->device), buft(ggml_backend_get_default_buffer_type(b)),
          compute(b->iface.graph_compute), synchronize(b->iface.synchronize) {
        original_buffer_device = buft->device;
        reg.iface.get_name = [](ggml_backend_reg_t) { return "RPC"; };
        device.reg = &reg;
        b->device = &device;
        buft->device = &device;
        active = this;
        b->iface.synchronize = [](ggml_backend_t) {};
        b->iface.graph_compute = [](ggml_backend_t, ggml_cgraph * g) {
            require(!active->pending);
            active->pending = g;
            return active->fail_compute ? GGML_STATUS_FAILED : GGML_STATUS_SUCCESS;
        };
    }
    void instrument(ggml_backend_buffer_t b) {
        buffer = b;
        get = b->iface.get_tensor;
        b->iface.get_tensor = [](ggml_backend_buffer_t b, const ggml_tensor * t, void * data, size_t offset,
                                 size_t size) {
            auto & self = *active;
            ++self.reads;
            if (self.pending) {
                auto * graph = self.pending;
                self.pending = nullptr;
                require(self.compute(self.backend, graph) == GGML_STATUS_SUCCESS);
                ++self.completed;
            }
            self.get(b, t, data, offset, size);
        };
    }
    ~simulated_rpc() {
        if (buffer)
            buffer->iface.get_tensor = get;
        backend->iface.graph_compute = compute;
        backend->iface.synchronize = synchronize;
        backend->device = original_device;
        buft->device = original_buffer_device;
        active = nullptr;
    }
};
simulated_rpc * simulated_rpc::active = nullptr;
bool observe_rpc(ggml_backend_t backend, ggml_backend_sched_t, ggml_cgraph *, bool completed, ggml_status,
                 void * user) {
    auto & rpc = *static_cast<simulated_rpc *>(user);
    if (completed)
        require(rpc.pending == nullptr);
    return !(!completed && backend && rpc.abort_child);
}

void test_rpc_completion() {
    auto backend = ggml_backend_cpu_init();
    {
        simulated_rpc rpc(backend);
        auto ctx = ggml_init({1024 * 1024, nullptr, true});
        auto a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4), b = ggml_dup_tensor(ctx, a), sum = ggml_add(ctx, a, b);
        auto graph = ggml_new_graph_custom(ctx, 32, false);
        ggml_build_forward_expand(graph, sum);
        auto sched = ggml_backend_sched_new(&backend, nullptr, 1, 32, false, true);
        require(ggml_backend_sched_alloc_graph(sched, graph));
        rpc.instrument(sum->buffer);
        std::array<float, 4> input{1, 2, 3, 4};
        ggml_backend_tensor_set(a, input.data(), 0, sizeof(input));
        ggml_backend_tensor_set(b, input.data(), 0, sizeof(input));
        ggml_backend_set_graph_observer(observe_rpc, &rpc);
        require(ggml_backend_sched_graph_compute_async(sched, graph) == GGML_STATUS_SUCCESS);
        require(rpc.reads == 3 && rpc.completed == 1 && static_cast<float *>(sum->data)[3] == 8);
        rpc.reads = 0;
        require(ggml_backend_graph_compute_async(backend, graph) == GGML_STATUS_SUCCESS);
        require(rpc.reads == 2 && rpc.completed == 2);
        // An eval callback may submit work after the child completion fence.
        rpc.callback_graph = graph;
        ggml_backend_sched_set_eval_callback(
            sched,
            [](ggml_tensor *, bool ask, void * user) {
                if (!ask) {
                    auto & rpc = *static_cast<simulated_rpc *>(user);
                    require(rpc.backend->iface.graph_compute(rpc.backend, rpc.callback_graph) == GGML_STATUS_SUCCESS);
                }
                return true;
            },
            &rpc);
        rpc.reads = 0;
        require(ggml_backend_sched_graph_compute_async(sched, graph) == GGML_STATUS_SUCCESS);
        require(rpc.reads == 4 && rpc.pending == nullptr);
        ggml_backend_sched_set_eval_callback(sched, nullptr, nullptr);
        rpc.abort_child = true;
        rpc.reads = 0;
        require(ggml_backend_sched_graph_compute_async(sched, graph) == GGML_STATUS_ABORTED);
        require(rpc.reads == 3 && rpc.pending == nullptr);
        rpc.abort_child = false;
        rpc.fail_compute = true;
        rpc.reads = 0;
        require(ggml_backend_sched_graph_compute_async(sched, graph) == GGML_STATUS_FAILED);
        require(rpc.reads == 4 && rpc.pending == nullptr);
        ggml_backend_set_graph_observer(nullptr, nullptr);
        rpc.fail_compute = false;
        rpc.reads = 0;
        observations submitted;
        ggml_backend_tensor_set(a, input.data(), 0, sizeof(input));
        ggml_backend_tensor_set(b, input.data(), 0, sizeof(input));
        ggml_backend_set_graph_submission_observer(observe, &submitted);
        require(ggml_backend_graph_compute_async(backend, graph) == GGML_STATUS_SUCCESS);
        require(submitted.begin == 1 && submitted.end == 1 && rpc.pending && rpc.reads == 0);
        ggml_backend_set_graph_submission_observer(nullptr, nullptr);
        std::array<float, 4> values{};
        ggml_backend_tensor_get(sum, values.data(), 0, sizeof(values));
        require(!rpc.pending && values[3] == 8);
        rpc.reads = 0;
        submitted = {};
        ggml_backend_tensor_set(a, input.data(), 0, sizeof(input));
        ggml_backend_tensor_set(b, input.data(), 0, sizeof(input));
        ggml_backend_set_graph_submission_observer(observe, &submitted);
        require(ggml_backend_sched_graph_compute_async(sched, graph) == GGML_STATUS_SUCCESS);
        require(submitted.begin == 2 && submitted.end == 2 && rpc.pending && rpc.reads == 0);
        ggml_backend_set_graph_submission_observer(nullptr, nullptr);
        ggml_backend_tensor_get(sum, values.data(), 0, sizeof(values));
        require(!rpc.pending && values[3] == 8);
        rpc.buffer->iface.get_tensor = rpc.get;
        rpc.buffer = nullptr;
        ggml_backend_sched_free(sched);
        ggml_free(ctx);
    }
    ggml_backend_free(backend);
}
}
int main() {
    auto * backend = ggml_backend_cpu_init();
    require(backend != nullptr);
    auto * ctx = ggml_init({1024 * 1024, nullptr, true});
    require(ctx != nullptr);
    auto * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    auto * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    auto * sum = ggml_add(ctx, a, b);
    auto * graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(graph, sum);
    auto * memory = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(memory != nullptr);
    std::array<float, 4> input{1, 2, 3, 4}, output{};
    ggml_backend_tensor_set(a, input.data(), 0, sizeof(input));
    ggml_backend_tensor_set(b, input.data(), 0, sizeof(input));
    observations state;
    ggml_backend_set_graph_observer(observe, &state);
    require(ggml_backend_graph_compute_async(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(sum, output.data(), 0, sizeof(output));
    require(state.begin == 1 && state.end == 1 && output[3] == 8);
    std::thread worker([&] { require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS); });
    worker.join();
    require(state.begin == 1 && state.end == 1);
    state.abort = true;
    require(ggml_backend_graph_compute_async(backend, graph) == GGML_STATUS_ABORTED);
    require(state.begin == 2 && state.end == 1);
    ggml_backend_set_graph_observer(nullptr, nullptr);
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    require(state.begin == 2 && state.end == 1);
    ggml_backend_buffer_free(memory);
    ggml_free(ctx);
    // Scheduler observation includes a parent pair and its backend child pair.
    ctx = ggml_init({1024 * 1024, nullptr, true});
    a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    sum = ggml_add(ctx, a, b);
    graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(graph, sum);
    auto * sched = ggml_backend_sched_new(&backend, nullptr, 1, 32, false, true);
    require(sched != nullptr);
    require(ggml_backend_sched_alloc_graph(sched, graph));
    ggml_backend_tensor_set(a, input.data(), 0, sizeof(input));
    ggml_backend_tensor_set(b, input.data(), 0, sizeof(input));
    state = {};
    ggml_backend_set_graph_observer(observe, &state);
    require(ggml_backend_sched_graph_compute_async(sched, graph) == GGML_STATUS_SUCCESS);
    require(state.begin == 2 && state.end == 2);
    require(ggml_backend_sched_graph_compute_async(sched, graph) == GGML_STATUS_SUCCESS);
    require(state.begin == 4 && state.end == 4);
    ggml_backend_set_graph_observer(nullptr, nullptr);
    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    ggml_backend_free(backend);
    test_rpc_completion();
    std::puts(
        "PASS: direct/scheduler/reused graphs, synchronized output, thread isolation, abort, disable and RPC fence reuse");
}
