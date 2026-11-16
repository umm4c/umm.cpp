#include "profiler-graph-contract.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include <cstdio>
#include <stdexcept>
#include <cmath>
struct fixture {
 ggml_context * ctx;ggml_backend_buffer_t buffer;ggml_tensor * a,*b,*out;ggml_cgraph * graph;
 fixture(ggml_backend_t backend){ctx=ggml_init({1024*1024,nullptr,true});a=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,8);b=ggml_dup_tensor(ctx,a);out=ggml_add(ctx,a,b);ggml_set_name(a,"a");ggml_set_name(b,"b");ggml_set_name(out,"out");graph=ggml_new_graph_custom(ctx,32,false);ggml_build_forward_expand(graph,out);buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);}
 ~fixture(){ggml_backend_buffer_free(buffer);ggml_free(ctx);}
};
void require(bool ok){if(!ok)throw std::runtime_error("Contract regression failed");}
template<class F>void rejects(F fn){bool rejected=false;try{fn();}catch(const std::runtime_error &){rejected=true;}require(rejected);}
int main(){
 require(umm::profiling::detail::hash("")=="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
 require(umm::profiling::detail::base64({})=="");require(umm::profiling::detail::base64({0,255})=="AP8=");
 require(umm::profiling::detail::base64({'f','o','o'})=="Zm9v");
 auto stats=umm::profiling::detail::sample_statistics({4,1,2,3});require(stats["p50"]==2.5 && std::abs(stats["p95"].get<double>()-3.85)<1e-12);
 auto backend=ggml_backend_cpu_init();
 {
 fixture original(backend);umm::profiling::graph_contract contract(backend,nullptr,original.graph);
 const auto description=contract.description().describe("test",1,0,"CPU");
 require(description["nodes"].size()==1 && description["tensors"].size()==3);
 contract.check(backend,nullptr,original.graph);
 fixture rebuilt(backend);require(rebuilt.a!=original.a);contract.check(backend,nullptr,rebuilt.graph);
 float values[8]={1,2,3,4,5,6,7,8};ggml_backend_tensor_set(rebuilt.a,values,0,sizeof(values));contract.check(backend,nullptr,rebuilt.graph);
 rebuilt.a->ne[0]=4;rejects([&]{contract.check(backend,nullptr,rebuilt.graph);});rebuilt.a->ne[0]=8;
 rebuilt.out->op_params[1]=1;rejects([&]{contract.check(backend,nullptr,rebuilt.graph);});rebuilt.out->op_params[1]=0;
 rebuilt.out->src[1]=rebuilt.a;rejects([&]{contract.check(backend,nullptr,rebuilt.graph);});rebuilt.out->src[1]=rebuilt.b;
 rebuilt.out->view_src=rebuilt.a;rejects([&]{contract.check(backend,nullptr,rebuilt.graph);});rebuilt.out->view_src=nullptr;
 rebuilt.b->type=GGML_TYPE_F16;rejects([&]{contract.check(backend,nullptr,rebuilt.graph);});rebuilt.b->type=GGML_TYPE_F32;
 ggml_set_name(rebuilt.b,"changed");rejects([&]{contract.check(backend,nullptr,rebuilt.graph);});ggml_set_name(rebuilt.b,"b");
 ggml_backend_buffer_set_usage(rebuilt.buffer,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);rejects([&]{contract.check(backend,nullptr,rebuilt.graph);});ggml_backend_buffer_set_usage(rebuilt.buffer,GGML_BACKEND_BUFFER_USAGE_ANY);
 contract.check(backend,nullptr,rebuilt.graph);
 // Equal names and shapes must not hide newly collapsed tensor identities.
 ggml_set_name(rebuilt.a,"same");ggml_set_name(rebuilt.b,"same");umm::profiling::graph_contract equal_names(backend,nullptr,rebuilt.graph);
 rebuilt.out->src[1]=rebuilt.a;rejects([&]{equal_names.check(backend,nullptr,rebuilt.graph);});rebuilt.out->src[1]=rebuilt.b;
 equal_names.check(backend,nullptr,rebuilt.graph);
 }
 ggml_backend_free(backend);std::puts("PASS: rebuilt graph addresses and changed input values accepted; shape, params, topology, alias, dtype, name and buffer role changes rejected");
}
