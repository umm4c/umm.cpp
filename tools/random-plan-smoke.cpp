#include "random-plan.h"
#include "weight-replicas.h"
#include "execution-policy.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
using context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
using scheduler = std::unique_ptr<ggml_backend_sched, decltype(&ggml_backend_sched_free)>;
bool compute(ggml_tensor * t) {
    return t->op != GGML_OP_NONE && t->op != GGML_OP_VIEW && t->op != GGML_OP_RESHAPE &&
           t->op != GGML_OP_TRANSPOSE && t->op != GGML_OP_PERMUTE;
}
float silu(float x) { return x / (1.f + std::exp(-x)); }
struct hook_state {
    const umm::random_plan * plan;
    umm::weight_replicas * replicas = nullptr;
    int before = 0;
    int after = 0;
};
bool placement_hook(ggml_backend_sched_t sched, ggml_cgraph *, bool allocated, void * user) {
    auto & state = *static_cast<hook_state *>(user);
    if (!allocated) { ++state.before; if (state.replicas) state.replicas->bind(sched, *state.plan); umm::assign_random_plan(sched, *state.plan); }
    else { ++state.after; umm::check_random_plan(sched, *state.plan); }
    return true;
}
std::vector<float> run(ggml_backend_t local, ggml_backend_t remote, ggml_backend_t cpu,
                       ggml_type type, int columns, uint32_t seed, bool reverse, int pure, int cuts = 1, bool resident = false) {
    const int n = 64;
    context inputs(ggml_init({1024*1024, nullptr, true}), ggml_free);
    if (!inputs) throw std::runtime_error("Input context failed");
    auto w = ggml_new_tensor_2d(inputs.get(), type, n, n+1);
    context mutable_inputs(ggml_init({1024*1024, nullptr, true}), ggml_free);
    auto x = ggml_new_tensor_2d(mutable_inputs.get(), GGML_TYPE_F32, n, columns);
    ggml_set_input(x);
    buffer input_buffer(ggml_backend_alloc_ctx_tensors(inputs.get(), local), ggml_backend_buffer_free);
    if (!input_buffer) throw std::runtime_error("Input allocation failed");
    ggml_backend_buffer_set_usage(input_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    buffer mutable_buffer(ggml_backend_alloc_ctx_tensors(mutable_inputs.get(), local), ggml_backend_buffer_free);
    if (!mutable_buffer) throw std::runtime_error("Mutable input allocation failed");
    std::vector<float> weights(n*(n+1)), values(n*columns);
    std::vector<ggml_fp16_t> packed(n*(n+1));
    for (int i=0; i<n*(n+1); ++i) {
        weights[i] = float(i % 19 - 9)/128;
        packed[i] = ggml_fp32_to_fp16(weights[i]);
        if (type == GGML_TYPE_F16) weights[i] = ggml_fp16_to_fp32(packed[i]);
    }
    ggml_backend_tensor_set(w, type == GGML_TYPE_F16 ? static_cast<void *>(packed.data()) : weights.data(), 0, ggml_nbytes(w));
    context ctx(ggml_init({2*1024*1024, nullptr, true}), ggml_free);
    if (!ctx) throw std::runtime_error("Graph context failed");
    auto weight_view = ggml_view_2d(ctx.get(), w, n, n, w->nb[1], w->nb[1]);
    auto z = ggml_mul_mat(ctx.get(), weight_view, x);
    auto h = ggml_silu(ctx.get(), z);
    auto view = ggml_reshape_2d(ctx.get(), h, n, columns);
    auto half = ggml_scale(ctx.get(), view, .5f);
    auto second_matmul = ggml_mul_mat(ctx.get(), weight_view, x);
    auto joined = ggml_add(ctx.get(), half, second_matmul);
    auto output = ggml_silu(ctx.get(), joined);
    ggml_tensor * observed[] = {z, h, half, joined, output};
    const char * names[] = {"matmul", "silu", "scale", "residual", "output"};
    for (int i=0; i<5; ++i) { ggml_set_name(observed[i], names[i]); ggml_set_output(observed[i]); }
    auto graph = ggml_new_graph_custom(ctx.get(), 64, false);
    ggml_build_forward_expand(graph, output);
    ggml_backend_t backends[] = {local, remote, cpu};
    scheduler sched(ggml_backend_sched_new(backends, nullptr, 3, 128, false, true), ggml_backend_sched_free);
    if (!sched) throw std::runtime_error("Scheduler creation failed");
    umm::random_plan plan{};
    std::unique_ptr<umm::weight_replicas> replicas;
    if (resident && pure < 0) replicas.reset(new umm::weight_replicas({w}, {local, remote}));
    hook_state hook{&plan, replicas.get()};
    if (pure < 0) {
        plan = umm::make_random_plan(graph, local, remote, seed, reverse, cuts);
        const auto repeat = umm::make_random_plan(graph, local, remote, seed, reverse, cuts);
        if (repeat.cut_nodes != plan.cut_nodes || repeat.reverse != plan.reverse) throw std::runtime_error("Seed is not reproducible");
        ggml_backend_sched_set_placement_callback(sched.get(), placement_hook, &hook);
    } else {
        for (int i=0; i<ggml_graph_n_nodes(graph); ++i) {
            auto node = ggml_graph_node(graph, i);
            if (compute(node)) ggml_backend_sched_set_tensor_backend(sched.get(), node, backends[pure]);
        }
    }
    // Explicit allocation is required: graph_compute would otherwise reset manual assignments.
    if (!ggml_backend_sched_alloc_graph(sched.get(), graph)) throw std::runtime_error("Graph allocation failed");
    if (pure < 0) umm::check_random_plan(sched.get(), plan);
    if (replicas) for (int i=0; i<ggml_backend_sched_get_n_splits(sched.get()); ++i) {
        ggml_backend_sched_split_info info{};
        if (!ggml_backend_sched_get_split_info(sched.get(), i, &info) || info.weight_input_bytes)
            throw std::runtime_error("Resident weight still requires a split input copy");
    }
    std::vector<float> result;
    for (int round=0; round<2; ++round) {
        for (int i=0; i<n*columns; ++i) values[i] = float((i+round*7)%23-11)/64;
        ggml_backend_tensor_set(x, values.data(), 0, ggml_nbytes(x));
        if (ggml_backend_sched_graph_compute(sched.get(), graph) != GGML_STATUS_SUCCESS) throw std::runtime_error("Graph compute failed");
        float error = 0;
        for (int stage=0; stage<5; ++stage) {
            std::vector<float> actual(n*columns);
            ggml_backend_tensor_get(observed[stage], actual.data(), 0, ggml_nbytes(observed[stage]));
            for (int col=0; col<columns; ++col) for (int row=0; row<n; ++row) {
                float expected=0;
                for (int k=0; k<n; ++k) expected += weights[(row+1)*n+k]*values[col*n+k];
                if (stage==1) expected=silu(expected);
                if (stage==2) expected=.5f*silu(expected);
                if (stage==3) expected+=.5f*silu(expected);
                if (stage==4) expected=silu(expected+.5f*silu(expected));
                const float value=actual[col*n+row];
                if (!std::isfinite(value)) throw std::runtime_error("Non-finite intermediate");
                error=std::max(error,std::abs(expected-value));
            }
            if (stage==4) result.insert(result.end(), actual.begin(), actual.end());
        }
        if (error > .001f) throw std::runtime_error("Intermediate differs from reference by more than 0.001");
        std::printf("PASS type=%s columns=%d seed=%u reverse=%d pure=%d cuts=%d first_cut=%d splits=%d round=%d max_abs=%g\n",
                    ggml_type_name(type),columns,seed,reverse,pure,pure < 0 ? cuts : 0,plan.cut_nodes.empty() ? -1 : plan.cut_nodes.front(),ggml_backend_sched_get_n_splits(sched.get()),round,error);
    }
    if (pure < 0) {
        if (hook.before != 1 || hook.after != 1) throw std::runtime_error("Placement hooks did not follow allocation/reuse lifecycle");
        for (const auto & p : plan.placements) if (compute(p.node)) {
            std::printf("PLACEMENT seed=%u reverse=%d cuts=%d node=%s op=%s device=%s\n",seed,reverse,cuts,p.node->name,ggml_op_name(p.node->op),ggml_backend_dev_name(ggml_backend_get_device(p.backend)));
        }
    }
    std::fflush(stdout);
    return result;
}
void rejected_aliases(ggml_backend_t a, ggml_backend_t b) {
    context ctx(ggml_init({1024*1024, nullptr, true}), ggml_free);
    auto x=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,64);
    buffer memory(ggml_backend_alloc_ctx_tensors(ctx.get(),a),ggml_backend_buffer_free);
    if (!memory) throw std::runtime_error("Alias test allocation failed");
    auto h=ggml_scale_inplace(ctx.get(),x,2);
    auto y=ggml_scale_inplace(ctx.get(),h,2);
    auto graph=ggml_new_graph_custom(ctx.get(),16,false);ggml_build_forward_expand(graph,y);
    for (bool reverse : {false,true}) {
        bool rejected=false;
        try { (void)umm::make_random_plan(graph,a,b,0,reverse); }
        catch (const std::runtime_error &) { rejected=true; }
        if (!rejected) throw std::runtime_error("Unsafe alias partition was accepted");
    }
    for (int cuts : {0,2}) {
        bool rejected=false;
        try { (void)umm::make_random_plan(graph,a,b,0,false,cuts); }
        catch (const std::invalid_argument &) { rejected=true; }
        if (!rejected) throw std::runtime_error("Invalid cut count was accepted");
    }
    bool rejected=false;
    try { (void)umm::make_random_plan(graph,a,a,0,false); }
    catch (const std::invalid_argument &) { rejected=true; }
    if (!rejected) throw std::runtime_error("Duplicate device was accepted");
    std::puts("PASS pinned in-place alias rejects both single-cut directions");
    std::puts("PASS invalid cut counts and duplicate devices rejected");
}
}
int main(int argc,char ** argv) {
    try {
        if (argc<2 || argc>5) throw std::invalid_argument("Usage: umm-random-plan-smoke ENDPOINT [LOCAL_DEVICE] [SEEDS] [resident]");
        const bool resident=argc==5 && std::string(argv[4])=="resident";
        const int seeds=argc>=4 ? std::stoi(argv[3]) : 4;
        if (seeds<1 || seeds>128) throw std::invalid_argument("SEEDS must be 1..128");
        ggml_backend_load_all();
        const auto remote_name=umm::register_remote_cann(argv[1]);
        using backend=std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)>;
        backend local(ggml_backend_init_by_name(argc>=3 ? argv[2] : "CANN0",nullptr),ggml_backend_free);
        backend remote(ggml_backend_init_by_name(remote_name.c_str(),nullptr),ggml_backend_free);
        backend cpu(ggml_backend_init_by_name("CPU",nullptr),ggml_backend_free);
        if (!local || !remote || !cpu) throw std::runtime_error("Both accelerator backends and scheduler CPU backend are required");
        if (std::string(ggml_backend_reg_name(ggml_backend_dev_backend_reg(ggml_backend_get_device(local.get()))))!="CANN") throw std::invalid_argument("Local device must be CANN");
        setenv("GGML_SCHED_STRICT_ACCEL","1",1);
        rejected_aliases(local.get(),remote.get());
        for (auto type : {GGML_TYPE_F32,GGML_TYPE_F16}) for (int columns : {1,16}) {
            const auto native=run(local.get(),remote.get(),cpu.get(),type,columns,0,false,0);
            const auto distant=run(local.get(),remote.get(),cpu.get(),type,columns,0,false,1);
            for (int cuts : {1,2,5}) for (int seed=0; seed<seeds; ++seed) for (bool reverse : {false,true}) {
                const auto mixed=run(local.get(),remote.get(),cpu.get(),type,columns,seed,reverse,-1,cuts,resident);
                for (size_t i=0; i<mixed.size(); ++i) {
                    if (std::abs(mixed[i]-native[i])>.001f || std::abs(mixed[i]-distant[i])>.001f) throw std::runtime_error("Mixed output differs from pure-device baseline");
                }
            }
        }
        std::puts("PASS single-cut, multi-cut and dense-node cooperative execution and pure-device comparisons");
        return 0;
    } catch (const std::exception & error) { std::fprintf(stderr,"Random plan smoke failed: %s\n",error.what());return 1; }
}
