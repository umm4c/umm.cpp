#include "execution-policy.h"
#include "llama-cpp-adapter.h"
#include "random-plan.h"
#include "json.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unordered_set>

using json = nlohmann::json;
using clock_type = std::chrono::steady_clock;
namespace fs = std::filesystem;
namespace {
double ms(clock_type::time_point t) { return std::chrono::duration<double,std::milli>(clock_type::now()-t).count(); }
std::string device_name(ggml_backend_t backend) { return ggml_backend_dev_name(ggml_backend_get_device(backend)); }
bool compute(ggml_tensor * node) {
    return node->op != GGML_OP_NONE && node->op != GGML_OP_VIEW && node->op != GGML_OP_RESHAPE &&
           node->op != GGML_OP_PERMUTE && node->op != GGML_OP_TRANSPOSE;
}
void write(const fs::path & p,const json & data) { std::ofstream f(p);f << data.dump(2) << '\n';if (!f) throw std::runtime_error("Cannot write "+p.string()); }
struct placement_state {
    std::ofstream graphs;
    std::string mode, request, error;
    uint32_t seed = 0;
    int cuts = 1;
    bool flip = false;
    int graph_id = 0;
    umm::random_plan plan{};
    double planning_ms = 0;
    clock_type::time_point allocation_started;
};
bool place(ggml_backend_sched_t sched,ggml_cgraph * graph,bool allocated,void * user) {
    auto & state=*static_cast<placement_state *>(user);
    try {
        if (!allocated) {
            state.allocation_started=clock_type::now();
            ++state.graph_id;
            if (state.mode=="random") {
                if (ggml_backend_sched_get_n_backends(sched)<3) throw std::runtime_error("Mixed model requires two accelerator backends");
                auto start=clock_type::now();
                state.plan=umm::make_random_plan(graph,ggml_backend_sched_get_backend(sched,0),
                    ggml_backend_sched_get_backend(sched,1),state.seed,state.flip,state.cuts);
                state.planning_ms=ms(start);
                umm::assign_random_plan(sched,state.plan);
            }
            return true;
        }
        if (state.mode=="random") umm::check_random_plan(sched,state.plan);
        json nodes=json::array(),splits=json::array(),boundaries=json::array();
        for (int i=0;i<ggml_graph_n_nodes(graph);++i) {
            auto t=ggml_graph_node(graph,i);auto backend=ggml_backend_sched_get_tensor_backend(sched,t);
            if (!backend) throw std::runtime_error("Unassigned model node");
            if (compute(t) && ggml_backend_dev_type(ggml_backend_get_device(backend))==GGML_BACKEND_DEVICE_TYPE_CPU)
                throw std::runtime_error("CPU model compute is forbidden");
            nodes.push_back({{"index",i},{"name",t->name},{"op",ggml_op_name(t->op)},
                {"dtype",ggml_type_name(t->type)},{"shape",{t->ne[0],t->ne[1],t->ne[2],t->ne[3]}},
                {"compute",compute(t)},{"device",device_name(backend)}});
        }
        for (int i=0;i<ggml_backend_sched_get_n_splits(sched);++i) {
            ggml_backend_sched_split_info info{};
            if (!ggml_backend_sched_get_split_info(sched,i,&info)) throw std::runtime_error("Cannot inspect split");
            splits.push_back({{"index",i},{"start",info.node_start},{"end_exclusive",info.node_end},{"device",device_name(info.backend)}});
        }
        if (state.mode=="random") {
            for (int cut:state.plan.cut_nodes) {
                auto t=ggml_graph_node(graph,cut);
                boundaries.push_back({{"index",cut},{"name",t->name},{"op",ggml_op_name(t->op)},
                    {"dtype",ggml_type_name(t->type)},{"shape",{t->ne[0],t->ne[1],t->ne[2],t->ne[3]}}});
            }
            bool first=false,second=false;
            for (const auto & p:state.plan.placements) if (compute(p.node)) {
                first|=p.backend==ggml_backend_sched_get_backend(sched,0);
                second|=p.backend==ggml_backend_sched_get_backend(sched,1);
            }
            if (!first || !second) throw std::runtime_error("Random plan did not use both devices");
        }
        json record={{"graph_id",state.graph_id},{"request_id",state.request},{"mode",state.mode},
            {"seed",state.seed},{"flip_devices",state.flip},{"planning_ms",state.planning_ms},
            {"allocation_and_planning_ms",ms(state.allocation_started)},{"planned_boundaries",boundaries},
            {"actual_splits",splits},{"nodes",nodes}};
        state.graphs << record.dump() << '\n';state.graphs.flush();
        if (!state.graphs) throw std::runtime_error("Cannot persist graph placement");
        return true;
    } catch (const std::exception & e) {
        state.error=e.what();
        state.graphs << json({{"graph_id",state.graph_id},{"request_id",state.request},{"status","failed"},{"error",state.error}}).dump() << '\n';state.graphs.flush();
        std::cerr << "Placement failed: " << state.error << '\n';return false;
    }
}
std::vector<float> read_logits(const fs::path & p,int vocab) {
    std::ifstream f(p,std::ios::binary|std::ios::ate);
    if (!f || f.tellg()!=std::streamoff(vocab*sizeof(float))) throw std::runtime_error("Invalid reference logits: "+p.string());
    f.seekg(0);std::vector<float> values(vocab);f.read(reinterpret_cast<char *>(values.data()),vocab*sizeof(float));
    if (!f) throw std::runtime_error("Cannot read reference logits");return values;
}
}
int main(int argc,char ** argv) {
    fs::path out;json summary;bool owns_output=false;
    try {
        if (argc!=3) throw std::invalid_argument("Usage: umm-random-model-run CONFIG.json NEW_OUTPUT_DIR");
        json config;{std::ifstream f(argv[1]);f >> config;}
        const std::string mode=config.at("mode");
        if (mode!="local" && mode!="remote" && mode!="random") throw std::invalid_argument("Mode must be local, remote or random");
        const int steps=config.value("max_tokens",8),nctx=config.value("n_ctx",512),batch=config.value("n_batch",128);
        if (steps<1 || steps>64 || nctx<64 || batch<1 || batch>nctx) throw std::invalid_argument("Invalid context, batch or output length");
        std::unordered_set<std::string> request_ids;
        const auto & inputs=config.at("requests");
        if (!inputs.is_array() || inputs.empty()) throw std::invalid_argument("Requests must be a non-empty array");
        for (const auto & request:inputs) {
            const auto id=request.at("id").get<std::string>();
            if (id.empty() || id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")!=std::string::npos || !request_ids.insert(id).second)
                throw std::invalid_argument("Request IDs must be unique simple filename components");
            (void)request.at("prompt").get<std::string>();
        }
        out=fs::absolute(argv[2]);if (fs::exists(out)) throw std::invalid_argument("Output directory already exists");fs::create_directories(out);owns_output=true;
        summary={{"status","running"},{"mode",mode},{"config",config},{"requests",json::array()}};write(out/"summary.json",summary);
        setenv("GGML_CANN_Q8_NZ","0",1);setenv("GGML_SCHED_STRICT_ACCEL","1",1);setenv("GGML_RPC_NO_RDMA","1",1);
        ggml_backend_load_all();
        const std::string local=config.value("local_device","CANN0");
        std::string remote;
        if (mode!="local") remote=umm::register_remote_cann(config.at("endpoint"));
        const std::string primary=mode=="remote" ? remote : local;
        placement_state state;state.mode=mode;state.seed=config.value("seed",0u);state.cuts=config.value("cuts",1);state.flip=config.value("flip_devices",false);
        state.graphs.open(out/"graphs.jsonl");if (!state.graphs) throw std::runtime_error("Cannot open graph log");
        const auto loading=clock_type::now();
        umm::llama_cpp_adapter model(config.at("model"),nctx,99,false,primary,batch,batch,true,true,mode=="random" ? remote : "");
        summary["load_ms"]=ms(loading);summary["vocab_size"]=model.vocab_size();write(out/"summary.json",summary);
        ggml_backend_sched_set_placement_callback(model.scheduler(),place,&state);
        std::ofstream forwards(out/"forwards.jsonl"),requests(out/"requests.jsonl");
        if (!forwards || !requests) throw std::runtime_error("Cannot open timing logs");
        const fs::path reference=config.value("reference_dir",std::string());
        for (const auto & request:config.at("requests")) {
            state.request=request.at("id");
            model.reset();
            const std::string prompt=request.at("prompt");
            const auto tokens=model.tokenize("<|im_start|>user\n"+prompt+"<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
            if (tokens.size()>size_t(batch) || tokens.size()+steps>model.context_capacity()) throw std::invalid_argument("Prompt exceeds single-batch probe capacity");
            json ref;
            if (!reference.empty()) {std::ifstream f(reference/(state.request+".json"));f >> ref;if (ref.at("input_tokens").get<std::vector<llama_token>>()!=tokens) throw std::runtime_error("Reference prompt tokens differ");}
            const int count=reference.empty() ? steps : int(ref.at("reference_tokens").size());
            if (count<1 || count>steps || tokens.size()+size_t(count)>model.context_capacity())
                throw std::invalid_argument("Reference continuation exceeds configured output capacity or is empty");
            if (!reference.empty()) for (const auto & token:ref.at("reference_tokens")) {
                const auto id=token.get<llama_token>();
                if (id<0 || id>=model.vocab_size()) throw std::invalid_argument("Reference token is outside vocabulary");
            }
            json predicted=json::array(),forced=json::array();std::string output;
            int matches=0,within=0;double request_max=0,request_squared=0,model_ms=0;size_t compared=0;
            const auto started=clock_type::now();
            for (int step=0;step<count;++step) {
                const int previous=state.graph_id;auto start=clock_type::now();
                model.append(step==0 ? tokens : std::vector<llama_token>{forced.at(step-1).get<llama_token>()});
                model.synchronize();const double elapsed=ms(start);model_ms+=elapsed;
                const float * logits=model.logits();const int vocab=model.vocab_size();
                const auto prediction=model.greedy();predicted.push_back(prediction);output+=model.piece(prediction);
                const auto file=state.request+"-"+std::to_string(step)+".f32";
                {std::ofstream f(out/file,std::ios::binary);f.write(reinterpret_cast<const char *>(logits),vocab*sizeof(float));if (!f) throw std::runtime_error("Cannot persist logits");}
                json accuracy=nullptr;
                if (!reference.empty()) {
                    const auto expected=read_logits(reference/file,vocab);double max_abs=0,squared=0,dot=0,norm_a=0,norm_b=0;size_t passed=0;
                    const double atol=config.value("logit_atol",.05),rtol=config.value("logit_rtol",.005);
                    for (int i=0;i<vocab;++i) {
                        if (!std::isfinite(logits[i]) || !std::isfinite(expected[i])) throw std::runtime_error("Non-finite logits");
                        const double d=double(logits[i])-expected[i];max_abs=std::max(max_abs,std::abs(d));squared+=d*d;
                        dot+=double(logits[i])*expected[i];norm_a+=double(logits[i])*logits[i];norm_b+=double(expected[i])*expected[i];
                        passed+=std::abs(d)<=atol+rtol*std::abs(expected[i]);
                    }
                    const auto expected_token=ref.at("reference_tokens").at(step).get<llama_token>();
                    matches+=prediction==expected_token;within+=passed==size_t(vocab);request_max=std::max(request_max,max_abs);request_squared+=squared;compared+=vocab;
                    accuracy={{"reference_token",expected_token},{"top1_match",prediction==expected_token},{"max_abs_error",max_abs},
                        {"rmse",std::sqrt(squared/vocab)},{"cosine_similarity",dot/std::sqrt(norm_a*norm_b)},
                        {"within_atol_rtol_count",passed},{"vocab_size",vocab},{"atol",atol},{"rtol",rtol}};
                    forced.push_back(expected_token);
                } else {
                    for (int i=0;i<vocab;++i) if (!std::isfinite(logits[i])) throw std::runtime_error("Non-finite baseline logits");
                    forced.push_back(prediction);
                }
                forwards << json({{"request_id",state.request},{"step",step},{"phase",step==0 ? "prefill" : "decode"},
                    {"input_tokens_this_forward",step==0 ? tokens.size() : 1},{"model_ms",elapsed},{"active_graph_id",state.graph_id},
                    {"new_graph_count",state.graph_id-previous},{"graph_reused",state.graph_id==previous},{"prediction",prediction},
                    {"fed_reference_token",forced.back()},{"accuracy",accuracy},{"logits_file",file}}).dump() << '\n';forwards.flush();
                if (reference.empty() && model.is_end(prediction)) break;
            }
            json result={{"request_id",state.request},{"prompt",prompt},{"mode",mode},{"seed",state.seed},{"cuts",state.cuts},
                {"flip_devices",state.flip},{"input_tokens",tokens},{"reference_tokens",forced},{"predicted_tokens",predicted},
                {"predicted_text",output},{"finish_reason",model.is_end(forced.back().get<llama_token>()) ? "stop" : "length"},{"teacher_forced",!reference.empty()},{"model_ms",model_ms},{"request_wall_ms",ms(started)},
                {"top1_matches",reference.empty() ? json(nullptr) : json(matches)},
                {"compared_steps",reference.empty() ? 0 : int(predicted.size())},{"all_logits_within_tolerance_steps",reference.empty() ? json(nullptr) : json(within)},
                {"max_abs_logit_error",reference.empty() ? json(nullptr) : json(request_max)},
                {"logit_rmse",compared ? json(std::sqrt(request_squared/compared)) : json(nullptr)}};
            write(out/(state.request+".json"),result);requests << result.dump() << '\n';requests.flush();
            summary["requests"].push_back(result);write(out/"summary.json",summary);
            std::cout << "MODEL_REQUEST " << state.request << " ms=" << model_ms << " steps=" << predicted.size() << " top1_matches=" << matches << std::endl;
        }
        ggml_backend_sched_set_placement_callback(model.scheduler(),nullptr,nullptr);
        summary["status"]="completed";summary["graph_allocations"]=state.graph_id;write(out/"summary.json",summary);
        return 0;
    } catch (const std::exception & e) {
        if (owns_output) {summary["status"]="failed";summary["error"]=e.what();write(out/"summary.json",summary);}
        std::cerr << "Random model run failed: " << e.what() << '\n';return 1;
    }
}
