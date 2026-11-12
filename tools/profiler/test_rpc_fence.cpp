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
struct scheduler_check { ggml_tensor * output; float expected; int parents=0,children=0; };
bool observe_scheduler(ggml_backend_t backend,ggml_backend_sched_t,ggml_cgraph *,bool completed,ggml_status status,void * user){
    auto & check=*static_cast<scheduler_check*>(user);
    if(status!=GGML_STATUS_SUCCESS)return false;
    if(completed){
        if(backend)++check.children;
        else {
            ++check.parents;
            // Final output is host-resident, so checking it cannot fence RPC work.
            for(int64_t i=0;i<ggml_nelements(check.output);++i)
                if(static_cast<float*>(check.output->data)[i]!=check.expected)return false;
        }
    }
    return true;
}
void check_multi_split(ggml_backend_t remote){
    auto cpu=ggml_backend_cpu_init();
    ggml_backend_t backends[]={remote,cpu};
    auto ctx=ggml_init({1024*1024,nullptr,true});
    auto a=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,16384),b=ggml_dup_tensor(ctx,a);
    auto first=ggml_add(ctx,a,b),middle=ggml_add(ctx,first,b),last=ggml_add(ctx,middle,a);
    auto graph=ggml_new_graph_custom(ctx,32,false);ggml_build_forward_expand(graph,last);
    auto sched=ggml_backend_sched_new(backends,nullptr,2,32,false,true);
    ggml_backend_sched_set_tensor_backend(sched,a,cpu);ggml_backend_sched_set_tensor_backend(sched,b,cpu);
    ggml_backend_sched_set_tensor_backend(sched,first,cpu);ggml_backend_sched_set_tensor_backend(sched,middle,remote);ggml_backend_sched_set_tensor_backend(sched,last,cpu);
    if(!ggml_backend_sched_alloc_graph(sched,graph))throw std::runtime_error("Multi-split allocation failed");
    if(ggml_backend_sched_get_tensor_backend(sched,first)!=cpu || ggml_backend_sched_get_tensor_backend(sched,middle)!=remote ||
       ggml_backend_sched_get_tensor_backend(sched,last)!=cpu || !ggml_backend_buffer_is_host(last->buffer))
        throw std::runtime_error("Multi-split placement mismatch");
    scheduler_check check{last,0};std::vector<float> input(16384);
    for(int replay=0;replay<3;++replay){
        const float value=1.25f+replay;std::fill(input.begin(),input.end(),value);check.expected=4*value;
        ggml_backend_tensor_set(a,input.data(),0,ggml_nbytes(a));ggml_backend_tensor_set(b,input.data(),0,ggml_nbytes(b));
        ggml_backend_set_graph_observer(observe_scheduler,&check);
        const auto status=ggml_backend_sched_graph_compute_async(sched,graph);
        ggml_backend_set_graph_observer(nullptr,nullptr);
        if(status!=GGML_STATUS_SUCCESS)throw std::runtime_error("CPU/RPC/CPU completion or output mismatch");
    }
    if(check.parents!=3 || check.children!=9)throw std::runtime_error("Multi-split observation count mismatch");
    ggml_backend_sched_free(sched);ggml_free(ctx);ggml_backend_free(cpu);
}
}
int main(int argc,char ** argv){
    if(argc!=2)throw std::invalid_argument("Usage: umm-profiler-rpc-fence-test ENDPOINT");
    auto backend=ggml_backend_rpc_init(argv[1],0);if(!backend)throw std::runtime_error("RPC initialization failed");
    auto ctx=ggml_init({1024*1024,nullptr,true});if(!ctx)throw std::runtime_error("Context allocation failed");
    auto a=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,16384),b=ggml_dup_tensor(ctx,a),out=ggml_add(ctx,a,b);
    auto graph=ggml_new_graph_custom(ctx,32,false);ggml_build_forward_expand(graph,out);
    auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);if(!buffer)throw std::runtime_error("Buffer allocation failed");
    std::vector<float> input(16384,1.25f),output(16384);ggml_backend_tensor_set(a,input.data(),0,ggml_nbytes(a));ggml_backend_tensor_set(b,input.data(),0,ggml_nbytes(b));
    nlohmann::json records=nlohmann::json::array();std::array<unsigned char,65536> scratch{};
    for(size_t size:{size_t(1),size_t(64),size_t(4096),size_t(65536)}){
        nlohmann::json samples=nlohmann::json::array();
        for(int i=-2;i<5;++i){auto start=std::chrono::steady_clock::now();
            if(ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("Graph execution failed");
            ggml_backend_tensor_get(out,scratch.data(),0,size);
            if(i>=0)samples.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
        }
        records.push_back({{"completion_read_bytes",size},{"samples_ms",samples}});
    }
    ggml_backend_tensor_get(out,output.data(),0,ggml_nbytes(out));for(auto v:output)if(v!=2.5f)throw std::runtime_error("RPC output mismatch");
    check_multi_split(backend);
    std::puts(nlohmann::json({{"status","passed"},{"records",records},{"multi_split","CPU/RPC/CPU: three replays, exact host output at scheduler END"}}).dump(2).c_str());
    ggml_backend_buffer_free(buffer);ggml_free(ctx);ggml_backend_free(backend);
}
