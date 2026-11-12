#include "profiler-candidates.h"
#include "profiler-utils.h"
#include "ggml-alloc.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>

namespace umm::profiling {
using json=nlohmann::json;
namespace {
using detail::clock;using detail::elapsed;
constexpr int warmups=2, repetitions=5;
using params=std::array<int32_t,GGML_MAX_OP_PARAMS/sizeof(int32_t)>;
struct pattern { ggml_tensor * input=nullptr;std::array<ggml_tensor*,3> weights{};std::array<ggml_tensor*,4> nodes{};int index=0;std::string reason; };
bool matrix(const ggml_tensor * t){return t && !t->view_src && ggml_is_contiguous(t) && t->ne[2]==1 && t->ne[3]==1;}
bool weight(const ggml_tensor * t){return matrix(t) && t->op==GGML_OP_NONE && t->buffer && ggml_backend_buffer_get_usage(t->buffer)==GGML_BACKEND_BUFFER_USAGE_WEIGHTS;}
bool suffix(const std::string & value,const std::string & end){return value.size()>=end.size() && value.compare(value.size()-end.size(),end.size(),end)==0;}
std::vector<pattern> find_patterns(ggml_cgraph * graph) {
    std::vector<pattern> found;
    for(int i=0;i<ggml_graph_n_nodes(graph);++i){auto out=ggml_graph_node(graph,i);
        if(out->op!=GGML_OP_MUL_MAT || !out->src[0] || !suffix(out->src[0]->name,".ffn_down.weight"))continue;
        pattern p;p.index=i;p.nodes[3]=out;p.weights[2]=out->src[0];
        auto glu=out->src[1];
        if(!glu || glu->op!=GGML_OP_GLU || ggml_get_glu_op(glu)!=GGML_GLU_OP_SWIGLU || !glu->src[0] || !glu->src[1]){p.reason="unsupported_activation_topology";found.push_back(p);continue;}
        p.nodes[2]=glu;p.nodes[0]=glu->src[0];p.nodes[1]=glu->src[1];
        auto a=p.nodes[0],b=p.nodes[1];
        if(a->op!=GGML_OP_MUL_MAT || b->op!=GGML_OP_MUL_MAT || a->src[1]!=b->src[1] || !a->src[1]){p.reason="unclosed_ffn_inputs";found.push_back(p);continue;}
        p.input=a->src[1];p.weights[0]=a->src[0];p.weights[1]=b->src[0];
        bool named=p.weights[0] && p.weights[1] &&
            ((suffix(p.weights[0]->name,".ffn_gate.weight") && suffix(p.weights[1]->name,".ffn_up.weight")) ||
             (suffix(p.weights[1]->name,".ffn_gate.weight") && suffix(p.weights[0]->name,".ffn_up.weight")));
        if(!named){p.reason="unrecognized_weight_identity";found.push_back(p);continue;}
        if(!matrix(p.input) || p.input->type!=GGML_TYPE_F32 || !matrix(out) || out->type!=GGML_TYPE_F32){p.reason="unsupported_boundary_layout";found.push_back(p);continue;}
        bool safe=true;
        for(auto w:p.weights)safe&=weight(w);
        for(auto node:p.nodes){safe&=matrix(node) && node->type==GGML_TYPE_F32;for(int k=2;k<GGML_MAX_SRC;++k)safe&=node->src[k]==nullptr;}
        if(!safe){p.reason="view_or_nonstandard_operand";found.push_back(p);continue;}
        std::set<ggml_tensor*> inside(p.nodes.begin(),p.nodes.end());
        for(int n=0;n<ggml_graph_n_nodes(graph);++n){auto node=ggml_graph_node(graph,n);if(inside.count(node))continue;
            for(auto src:node->src)if(src && inside.count(src) && src!=out)safe=false;
            if(node->view_src && inside.count(node->view_src) && node->view_src!=out)safe=false;
        }
        if(!safe)p.reason="intermediate_has_external_consumer";
        if(out->ne[0]!=p.input->ne[0] || out->ne[1]!=p.input->ne[1])p.reason="boundary_shape_mismatch";
        found.push_back(p);
    }
    return found;
}
json dimensions(const ggml_tensor * t){return {t->ne[0],t->ne[1],t->ne[2],t->ne[3]};}
struct context_owner {ggml_context * ptr=nullptr;~context_owner(){if(ptr)ggml_free(ptr);} };
struct buffer_owner {ggml_backend_buffer_t ptr=nullptr;~buffer_owner(){if(ptr)ggml_backend_buffer_free(ptr);} };
std::vector<uint8_t> bytes(ggml_tensor * t){std::vector<uint8_t> out(ggml_nbytes(t));ggml_backend_tensor_get(t,out.data(),0,out.size());return out;}
json stats(std::vector<double> x){std::sort(x.begin(),x.end());double sum=0;for(auto v:x)sum+=v;
    auto q=[&](double f){double n=f*(x.size()-1);size_t a=size_t(n),b=std::min(a+1,x.size()-1);return x[a]+(x[b]-x[a])*(n-a);};
    return {{"sample_count",x.size()},{"mean",sum/x.size()},{"p50",q(.5)},{"p95",q(.95)},{"min",x.front()},{"max",x.back()},{"samples",x}};
}
void completed(ggml_backend_t backend,ggml_tensor * output){ggml_backend_synchronize(backend);
    auto reg=ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    if(reg && std::string(ggml_backend_reg_name(reg)).rfind("RPC",0)==0){std::array<unsigned char,4096> bytes;ggml_backend_tensor_get(output,bytes.data(),0,std::min(bytes.size(),ggml_nbytes(output)));}
}
}
struct candidate_catalog::impl {
    struct entry {
        json record;
        std::array<params,4> operations{};
        std::array<int64_t,4> shape{};
        std::array<std::string,3> names;
        std::vector<uint8_t> input,reference;
        bool selected=false;
        std::set<std::string> origins;
    };
    std::string model_key,device,case_id;
    bool frozen=false;
    std::map<std::string,entry> entries;
    std::map<std::string,std::array<ggml_tensor*,3>> bindings;
    std::vector<std::pair<std::string,pattern>> active;
    json costs=json::array();
    size_t shape_number=0;
    std::map<std::string,size_t> shapes;
    std::string identity(const pattern & p) const {
        json key={{"model",model_key},{"pattern","ffn_swiglu_v1"},{"output",p.nodes[3]->name},{"shape",dimensions(p.nodes[3])}};
        json nodes=json::array();for(auto n:p.nodes)if(n){json op=json::array();for(size_t k=0;k<GGML_MAX_OP_PARAMS/sizeof(int32_t);++k)op.push_back(n->op_params[k]);nodes.push_back({{"op",int(n->op)},{"shape",dimensions(n)},{"type",int(n->type)},{"params",op}});}
        key["nodes"]=nodes;json weights=json::array();for(auto w:p.weights)if(w)weights.push_back({{"name",w->name},{"shape",dimensions(w)},{"type",int(w->type)}});key["weights"]=weights;
        return detail::hash(key.dump());
    }
    void capture_value(ggml_tensor * node){
        for(auto & item:active){auto & e=entries.at(item.first);auto & p=item.second;
            if(node==p.input && e.input.empty()){e.input=bytes(p.input);e.record["snapshot_source"]={{"device_id",device},{"case_id",case_id}};}
            if(node==p.nodes[3] && e.reference.empty())e.reference=bytes(p.nodes[3]);
        }
    }
    static bool eval(ggml_tensor * node,bool ask,void * user) noexcept {
        auto & self=*static_cast<impl*>(user);
        try {
            if(ask){for(const auto & item:self.active){const auto & e=self.entries.at(item.first);if((node==item.second.input && e.input.empty()) || (node==item.second.nodes[3] && e.reference.empty()))return true;}return false;}
            self.capture_value(node);return true;
        }catch(const std::exception & error){for(auto & item:self.active)self.entries.at(item.first).record["capture_error"]=error.what();return true;}
    }
};
candidate_catalog::candidate_catalog(std::string key):state(new impl){state->model_key=std::move(key);}
candidate_catalog::~candidate_catalog()=default;
void candidate_catalog::start_device(const std::string & id){state->device=id;state->bindings.clear();state->active.clear();}
void candidate_catalog::discover(ggml_cgraph * graph,const std::string & graph_id){
    for(const auto & p:find_patterns(graph)){
        const auto id=state->identity(p);auto & e=state->entries[id];
        if(e.record.is_null()){
            e.record={{"schema_version","umm-profiler-candidate/v1"},{"data_origin","measured"},{"candidate_id",id},{"pattern","ffn_swiglu_v1"},{"output",p.nodes[3]->name},{"graph_id",graph_id},
                {"isolated_replay_eligible",p.reason.empty()},{"dispatch_legality","not_validated"},{"rejection_reason",p.reason.empty()?json(nullptr):json(p.reason)},
                {"selection_rank",nullptr},{"selected",false},{"origins",json::array()}};
            if(p.reason.empty()){
                const auto D=p.input->ne[0],N=p.input->ne[1],F=p.nodes[0]->ne[0];const auto shape=dimensions(p.input).dump();
                if(!state->shapes.count(shape))state->shapes[shape]=state->shape_number++;
                e.record["shape_order"]=state->shapes[shape];e.record["input_shape"]=dimensions(p.input);e.record["output_shape"]=dimensions(p.nodes[3]);
                e.record["boundary_bytes"]=ggml_nbytes(p.input)+ggml_nbytes(p.nodes[3]);
                e.record["matmul_flops_estimate"]=6.0*D*N*F;e.record["compute_per_boundary_byte"]=6.0*D*N*F/e.record["boundary_bytes"].get<double>();
                json weights=json::array();size_t total=0;
                for(size_t i=0;i<3;++i){auto w=p.weights[i];e.names[i]=w->name;total+=ggml_nbytes(w);weights.push_back({{"name",w->name},{"dtype",ggml_type_name(w->type)},{"shape",dimensions(w)}});}
                e.record["weights"]=weights;e.record["weight_bytes"]=total;e.record["node_index"]=p.index;
                for(size_t i=0;i<4;++i){std::memcpy(e.operations[i].data(),p.nodes[i]->op_params,GGML_MAX_OP_PARAMS);e.shape[i]=p.input->ne[i];}
            }
        }
        if(e.origins.insert(state->device+graph_id).second){json indices=json::array();for(auto node:p.nodes)if(node)for(int i=0;i<ggml_graph_n_nodes(graph);++i)if(ggml_graph_node(graph,i)==node){indices.push_back(i);break;}
            e.record["origins"].push_back({{"device_id",state->device},{"graph_id",graph_id},{"node_indices",indices}});}
        if(p.reason.empty()){state->bindings[id]=p.weights;e.record["input_name"]=p.input->name;e.record["output_name"]=p.nodes[3]->name;}
    }
}
void candidate_catalog::select(){
    if(state->frozen)return;state->frozen=true;
    std::map<size_t,std::vector<std::string>> groups;
    for(const auto & item:state->entries)if(item.second.record["isolated_replay_eligible"].get<bool>())groups[item.second.record["shape_order"].get<size_t>()].push_back(item.first);
    size_t selected=0;
    for(auto & group:groups){auto & ids=group.second;
        std::sort(ids.begin(),ids.end(),[&](const std::string & a,const std::string & b){const json & x=state->entries.at(a).record;const json & y=state->entries.at(b).record;
            if(x["compute_per_boundary_byte"]!=y["compute_per_boundary_byte"])return x["compute_per_boundary_byte"].get<double>()>y["compute_per_boundary_byte"].get<double>();return x["node_index"].get<int>()<y["node_index"].get<int>();});
        for(size_t i=0;i<ids.size();++i)state->entries.at(ids[i]).record["selection_rank"]=i+1;
        if(selected++<4){auto & e=state->entries.at(ids[0]);e.selected=true;e.record["selected"]=true;e.record["selection_reason"]="best_structural_rank_per_input_shape_with_four_shape_budget";}
    }
}
void candidate_catalog::begin_capture(ggml_backend_sched_t sched,ggml_cgraph * graph,const std::string & case_id){
    state->active.clear();state->case_id=case_id;
    for(const auto & p:find_patterns(graph))if(p.reason.empty()){
        auto id=state->identity(p);auto it=state->entries.find(id);
        if(it!=state->entries.end() && it->second.selected && (it->second.input.empty() || it->second.reference.empty())){
            state->active.push_back({id,p});bool producer_present=false;
            for(int i=0;i<ggml_graph_n_nodes(graph);++i)producer_present|=ggml_graph_node(graph,i)==p.input;
            if(!producer_present || p.input->op==GGML_OP_NONE)state->capture_value(p.input);
        }
    }
    if(!state->active.empty())ggml_backend_sched_set_eval_callback(sched,impl::eval,state.get());
}
void candidate_catalog::end_capture(ggml_backend_sched_t sched){ggml_backend_sched_set_eval_callback(sched,nullptr,nullptr);state->active.clear();}

void candidate_catalog::measure(ggml_backend_t target,ggml_backend_t primary,const std::string & primary_id){
    for(auto & item:state->entries){auto & e=item.second;if(!e.selected)continue;
        json record={{"schema_version","umm-profiler-candidate-cost/v1"},{"candidate_id",item.first},{"device_id",state->device},{"data_origin","measured"},
            {"primary_device_id",primary?json(primary_id):json(nullptr)},{"status","unavailable"},{"decision_usable",false},
            {"weights","native_model_resident"},{"weight_upload_ms",nullptr},{"rpc_completion_read_bytes",4096},{"warmups",warmups},{"repetitions",repetitions}};
        auto reg=ggml_backend_dev_backend_reg(ggml_backend_get_device(target));
        if(!reg || std::string(ggml_backend_reg_name(reg)).rfind("RPC",0)!=0)record["rpc_completion_read_bytes"]=0;
        auto binding=state->bindings.find(item.first);
        if(binding==state->bindings.end() || e.input.empty() || e.reference.empty() || e.record.contains("capture_error")){
            record["reason"]="missing_native_weights_or_valid_snapshot";state->costs.push_back(record);continue;
        }
        const auto prepare=clock::now();
        context_owner ctx;ctx.ptr=ggml_init({ggml_tensor_overhead()*32+ggml_graph_overhead_custom(32,false),nullptr,true});
        if(!ctx.ptr)throw std::runtime_error("Cannot allocate candidate metadata");
        auto input=ggml_new_tensor(ctx.ptr,GGML_TYPE_F32,4,e.shape.data());
        auto a=ggml_mul_mat(ctx.ptr,binding->second[0],input),b=ggml_mul_mat(ctx.ptr,binding->second[1],input);
        auto glu=ggml_swiglu_split(ctx.ptr,a,b),out=ggml_mul_mat(ctx.ptr,binding->second[2],glu);
        std::array<ggml_tensor*,4> nodes={a,b,glu,out};bool supported=true;
        for(size_t i=0;i<nodes.size();++i){std::memcpy(nodes[i]->op_params,e.operations[i].data(),GGML_MAX_OP_PARAMS);supported &= ggml_backend_supports_op(target,nodes[i]);}
        if(!supported){record["reason"]="backend_rejects_candidate_operator";state->costs.push_back(record);continue;}
        auto graph=ggml_new_graph_custom(ctx.ptr,32,false);ggml_build_forward_expand(graph,out);
        buffer_owner buffer;buffer.ptr=ggml_backend_alloc_ctx_tensors(ctx.ptr,target);
        if(!buffer.ptr)throw std::runtime_error("Cannot allocate candidate tensors");
        ggml_backend_tensor_set(input,e.input.data(),0,e.input.size());ggml_backend_synchronize(target);
        record["prepare_ms"]=elapsed(prepare);
        auto compute=[&]{if(ggml_backend_graph_compute(target,graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("Candidate execution failed");completed(target,out);};
        std::vector<double> compute_times;
        for(int i=-warmups;i<repetitions;++i){auto start=clock::now();compute();if(i>=0)compute_times.push_back(elapsed(start));}
        record["compute_ms"]=stats(compute_times);
        auto actual=bytes(out);double max_abs=0,square=0;bool finite=true,within=true;
        if(actual.size()!=e.reference.size())throw std::runtime_error("Candidate reference size mismatch");
        for(size_t i=0;i<actual.size();i+=sizeof(float)){float x,y;std::memcpy(&x,actual.data()+i,sizeof(float));std::memcpy(&y,e.reference.data()+i,sizeof(float));
            finite &= std::isfinite(x)&&std::isfinite(y);double delta=std::abs(double(x)-y);max_abs=std::max(max_abs,delta);square+=delta*delta;within &= delta<=.05+.005*std::abs(double(y));}
        record["accuracy"]={{"all_finite",finite},{"within_tolerance",finite&&within},{"max_abs",finite?json(max_abs):json(nullptr)},
            {"rmse",finite?json(std::sqrt(square/(actual.size()/sizeof(float)))):json(nullptr)},{"byte_identical",actual==e.reference},{"atol",.05},{"rtol",.005}};
        if(primary){
            context_owner source;source.ptr=ggml_init({ggml_tensor_overhead()*2,nullptr,true});
            if(!source.ptr)throw std::runtime_error("Cannot allocate boundary metadata");
            auto src=ggml_new_tensor(source.ptr,GGML_TYPE_F32,4,e.shape.data());auto dst=ggml_dup_tensor(source.ptr,out);
            buffer_owner source_buffer;source_buffer.ptr=ggml_backend_alloc_ctx_tensors(source.ptr,primary);
            if(!source_buffer.ptr)throw std::runtime_error("Cannot allocate primary boundary tensors");
            ggml_backend_tensor_set(src,e.input.data(),0,e.input.size());ggml_backend_synchronize(primary);
            std::vector<double> copies,roundtrip;
            for(int mode=0;mode<2;++mode)for(int i=-warmups;i<repetitions;++i){auto start=clock::now();
                ggml_backend_tensor_copy(src,input);
                if(mode && ggml_backend_graph_compute(target,graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("Candidate roundtrip execution failed");
                // Reading/copying the full result is the RPC completion fence in the combined path.
                ggml_backend_tensor_copy(out,dst);ggml_backend_synchronize(primary);
                if(i>=0)(mode?roundtrip:copies).push_back(elapsed(start));}
            record["copy_only_roundtrip_ms"]=stats(copies);record["boundary_and_compute_ms"]=stats(roundtrip);
            record["boundary_path"]="ggml_tensor_copy_primary_to_target_and_back";
        }else{record["copy_only_roundtrip_ms"]=nullptr;record["boundary_and_compute_ms"]=nullptr;}
        record["status"]="completed";record["measurement_valid"]=finite&&within;record["timing_scope"]="synchronized_host_wall_with_rpc_completion_read";
        state->costs.push_back(std::move(record));
    }
}
json candidate_catalog::records() const {json out=json::array();for(const auto & item:state->entries){auto r=item.second.record;r["snapshot_available"]=!item.second.input.empty()&&!item.second.reference.empty();out.push_back(std::move(r));}return out;}
json candidate_catalog::measurements() const {return state->costs;}
json candidate_catalog::snapshots() const {json out=json::array();for(const auto & item:state->entries){const auto & e=item.second;if(e.input.empty()||e.reference.empty())continue;
    auto digest=[](const std::vector<uint8_t> & v){return detail::hash(std::string(reinterpret_cast<const char*>(v.data()),v.size()));};
    out.push_back({{"schema_version","umm-profiler-snapshot/v1"},{"data_origin","measured"},{"candidate_id",item.first},{"encoding","base64"},{"dtype","F32"},{"input_shape",e.record["input_shape"]},{"output_shape",e.record["output_shape"]},
        {"input",detail::base64(e.input)},{"reference",detail::base64(e.reference)},{"input_sha256",digest(e.input)},{"reference_sha256",digest(e.reference)},
        {"weights",e.record["weights"]},{"model_key",state->model_key},{"source",e.record["snapshot_source"]}});
}return out;}
}
