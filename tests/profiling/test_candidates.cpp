#include "profiling/profiler-candidates.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include <cstdio>
#include <stdexcept>
#include <vector>
using umm::profiling::candidate_catalog;
void require(bool ok){if(!ok)throw std::runtime_error("Candidate contract failed");}
struct watch {candidate_catalog * catalog;};
bool observer(ggml_backend_t,ggml_backend_sched_t sched,ggml_cgraph * graph,bool done,ggml_status,void * user){
    if(sched){auto & c=*static_cast<watch*>(user)->catalog;if(!done)c.begin_capture(sched,graph,"cpu-fixture");else c.end_capture(sched);}return true;
}
int main(){
    auto backend=ggml_backend_cpu_init();require(backend);ggml_backend_cpu_set_n_threads(backend,2);
    auto weights=ggml_init({1024*1024,nullptr,true});auto ctx=ggml_init({1024*1024,nullptr,true});require(weights&&ctx);
    auto gate=ggml_new_tensor_2d(weights,GGML_TYPE_F32,4,8),up=ggml_dup_tensor(weights,gate),down=ggml_new_tensor_2d(weights,GGML_TYPE_F32,8,4);
    ggml_set_name(gate,"blk.0.ffn_gate.weight");ggml_set_name(up,"blk.0.ffn_up.weight");ggml_set_name(down,"blk.0.ffn_down.weight");
    auto wb=ggml_backend_alloc_ctx_tensors(weights,backend);require(wb);ggml_backend_buffer_set_usage(wb,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::vector<float> values(32);for(size_t i=0;i<values.size();++i)values[i]=float(int(i%7)-3)*.1f;
    for(auto t:{gate,up,down})ggml_backend_tensor_set(t,values.data(),0,ggml_nbytes(t));
    auto leaf=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,4,2);ggml_set_input(leaf);
    auto x=ggml_scale(ctx,leaf,.5f),a=ggml_mul_mat(ctx,gate,x),b=ggml_mul_mat(ctx,up,x),glu=ggml_swiglu_split(ctx,a,b),out=ggml_mul_mat(ctx,down,glu);
    ggml_set_name(out,"ffn_out-0");ggml_set_output(out);
    auto graph=ggml_new_graph_custom(ctx,64,false);ggml_build_forward_expand(graph,out);
    auto sched=ggml_backend_sched_new(&backend,nullptr,1,64,false,true);require(sched);require(ggml_backend_sched_alloc_graph(sched,graph));
    std::vector<float> input(8,.25f);ggml_backend_tensor_set(leaf,input.data(),0,input.size()*sizeof(float));
    candidate_catalog catalog("test-model");catalog.start_device("cpu");catalog.discover(graph,"graph");catalog.select();
    require(catalog.records().size()==1 && catalog.records()[0]["selected"]==true);
    watch state{&catalog};ggml_backend_set_graph_observer(observer,&state);
    require(ggml_backend_sched_graph_compute(sched,graph)==GGML_STATUS_SUCCESS);ggml_backend_set_graph_observer(nullptr,nullptr);
    require(catalog.snapshots().size()==1);catalog.measure(backend,backend,"cpu");
    auto costs=catalog.measurements();require(costs.size()==1 && costs[0]["status"]=="completed" && costs[0]["accuracy"]["byte_identical"]==true);
    require(costs[0]["compute_ms"]["sample_count"]==5 && costs[0]["boundary_and_compute_ms"]["sample_count"]==5);
    // A consumer outside the FFN invalidates closure, even if its result is otherwise harmless.
    auto other=ggml_scale(ctx,a,2.f);ggml_build_forward_expand(graph,other);
    candidate_catalog rejected("test-model");rejected.start_device("cpu");rejected.discover(graph,"graph");rejected.select();
    require(rejected.records().size()==1 && rejected.records()[0]["rejection_reason"]=="intermediate_has_external_consumer" && rejected.records()[0]["selected"]==false);
    ggml_backend_sched_free(sched);ggml_backend_buffer_free(wb);ggml_free(ctx);ggml_free(weights);ggml_backend_free(backend);
    std::puts("PASS: closed FFN selection, execution-boundary snapshot, exact replay, boundary timing and external-consumer rejection");
}
