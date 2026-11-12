#include "profiler.h"
#include "profile-context.h"
#include "profiler-candidates.h"
#include "profiler-graph-contract.h"
#include <memory>
#include "runtime-support.h"
#include "execution-policy.h"
#include "model-package.h"
#include "ggml-backend.h"
#ifdef GGML_USE_RPC
#include "ggml-rpc.h"
#endif
extern "C" {
#include "sha256.h"
}
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>

namespace umm::profiling {
using json = nlohmann::json;
using clock = std::chrono::steady_clock;
namespace {
double elapsed(clock::time_point start) { return std::chrono::duration<double, std::milli>(clock::now()-start).count(); }
using detail::hash;
std::string read(const std::filesystem::path & path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::invalid_argument("Cannot read " + path.string());
    std::ostringstream out; out << file.rdbuf();
    if (file.bad()) throw std::runtime_error("Read failed: " + path.string());
    return out.str();
}
std::string file_hash(const std::filesystem::path & path) {
    std::ifstream file(path,std::ios::binary);
    if(!file)throw std::invalid_argument("Cannot hash "+path.string());
    sha256_t state;sha256_init(&state);std::array<unsigned char,1024*1024> buffer;
    while(file){file.read(reinterpret_cast<char*>(buffer.data()),buffer.size());sha256_update(&state,buffer.data(),size_t(file.gcount()));}
    if(!file.eof())throw std::runtime_error("Hash read failed");
    unsigned char digest[32];sha256_final(&state,digest);std::ostringstream out;
    for(auto byte:digest)out<<std::hex<<std::setw(2)<<std::setfill('0')<<int(byte);
    return out.str();
}
struct environment {
    std::map<std::string, std::pair<bool,std::string>> old;
    void set(const std::string & key, const std::string & value) {
        if (!old.count(key)) { const char * p=std::getenv(key.c_str()); old[key]={p!=nullptr,p?p:""}; }
        if (setenv(key.c_str(),value.c_str(),1)) throw std::runtime_error("Cannot configure environment");
    }
    ~environment() { for (const auto & item:old) { if (item.second.first) setenv(item.first.c_str(),item.second.second.c_str(),1); else unsetenv(item.first.c_str()); } }
};
struct capture {
    result & output;
    std::string device_id, case_id, requested_backend, error;
    candidate_catalog * catalog=nullptr;
    std::string pass="structure";
    int replay=0;
    bool warmup=false;
    bool remote_timing=false;
    clock::time_point origin=clock::now();
    struct frame {std::string graph_id,event_id,parent;clock::time_point start;double export_ms;};
    std::vector<frame> stack;
    std::set<std::string> seen;
    struct binding {std::string id,phase;std::unique_ptr<graph_contract> contract;bool scheduler=false;};
    std::map<std::string,std::vector<binding>> bindings;
    size_t execution_cursor=0;
    size_t event_number=0;
    std::map<std::string,std::vector<double>> samples;
    bool observe(ggml_backend_t backend,ggml_backend_sched_t sched,ggml_cgraph * graph,bool completed,ggml_status status) {
        if(pass=="capture") {
            if(sched){if(!completed)catalog->begin_capture(sched,graph,case_id);else catalog->end_capture(sched);}
            return true;
        }
        if(!completed) {
            auto start=clock::now();std::string fingerprint;
            if(pass=="structure"){
                auto contract=std::make_unique<graph_contract>(backend,sched,graph);
                const auto & descriptor=contract->description();
                fingerprint=descriptor.signature(current_phase.phase,requested_backend,true);
                if(seen.insert(fingerprint).second){auto description=descriptor.describe(current_phase.phase,current_phase.tokens,current_phase.cached_tokens,requested_backend);description["graph_id"]=fingerprint;description["logical_graph_id"]=descriptor.signature(current_phase.phase,requested_backend,false);description["device_id"]=device_id;output.graphs.push_back(std::move(description));}
                if(sched)catalog->discover(graph,fingerprint);
                bindings[case_id].push_back({fingerprint,current_phase.phase,std::move(contract),sched!=nullptr});
            }else{
                auto & sequence=bindings.at(case_id);
                if(execution_cursor>=sequence.size())throw std::runtime_error("Execution has more graphs than frozen structure");
                auto & registered=sequence[execution_cursor];
                if(registered.phase!=current_phase.phase)throw std::runtime_error("Execution phase differs from frozen structure");
                registered.contract->check(backend,sched,graph);fingerprint=registered.id;
            }
            ++execution_cursor;
            const auto parent=stack.empty()?std::string{}:stack.back().event_id;
            stack.push_back({fingerprint,device_id+"-e"+std::to_string(++event_number),parent,clock::now(),elapsed(start)});
        } else {
            if(stack.empty())throw std::runtime_error("Unmatched graph event");
            auto f=stack.back();stack.pop_back();const double duration=elapsed(f.start);
            output.trace.push_back({{"schema_version","umm-profiler-event/v3"},{"data_origin","measured"},
                {"graph_id",f.graph_id},{"event_id",f.event_id},{"parent_event_id",f.parent.empty()?json(nullptr):json(f.parent)},
                {"device_id",device_id},{"case_id",case_id},{"replay",replay},{"pass",pass},{"warmup",warmup},
                {"kind",sched?"scheduler":"backend"},{"phase",current_phase.phase},
                {"submission",{{"tokens",current_phase.tokens},{"cached_tokens",current_phase.cached_tokens}}},
                {"start_ms",std::chrono::duration<double,std::milli>(f.start-origin).count()},
                {"duration_ms",duration},{"export_before_ms",f.export_ms},{"status",int(status)},
                {"timing_scope",remote_timing?"client_submission_host_wall":"synchronized_host_wall"},
                {"event_type",remote_timing?"submitted":"completed"},{"clock_domain","client:"+device_id},
                {"executed",true},{"kernel_ms",nullptr}});
            if(status!=GGML_STATUS_SUCCESS)throw std::runtime_error("Graph execution failed");
            if(!warmup && !remote_timing)samples[f.graph_id].push_back(duration);
        }
        return true;
    }
    static bool callback(ggml_backend_t backend,ggml_backend_sched_t sched,ggml_cgraph * graph,bool completed,ggml_status status,void * user) noexcept {
        auto & self=*static_cast<capture*>(user);
        try{return self.observe(backend,sched,graph,completed,status);}catch(const std::exception & e){if(self.error.empty())self.error=e.what();return false;}catch(...){self.error="Graph capture failed";return false;}
    }
};
struct observer_scope {
    bool submission;
    explicit observer_scope(capture & c):submission(c.remote_timing){
        if(submission)ggml_backend_set_graph_submission_observer(capture::callback,&c);
        else ggml_backend_set_graph_observer(capture::callback,&c);
    }
    ~observer_scope(){if(submission)ggml_backend_set_graph_submission_observer(nullptr,nullptr);else ggml_backend_set_graph_observer(nullptr,nullptr);}
};

// One window per request; its final response fences the ordered worker stream.
class rpc_batch {
#ifdef GGML_USE_RPC
    ggml_backend_t backend=nullptr;
    uint64_t sequence=0,active=0;
    std::vector<ggml_rpc_profile_record> records;
    uint32_t drain(){uint32_t count=0;auto window=active;active=0;
        if(!ggml_backend_rpc_profile_end(backend,window,records.data(),uint32_t(records.size()),&count))throw std::runtime_error("RPC profile records missing, overflowed or out of order");
        return count;
    }
#endif
public:
    explicit rpc_batch(const std::string & name){
#ifdef GGML_USE_RPC
        backend=ggml_backend_init_by_name(name.c_str(),nullptr);
        if(!backend)throw std::runtime_error("Cannot initialize RPC profiling backend");
        try{start(1);if(drain()!=0)throw std::runtime_error("RPC profiling probe returned unexpected graphs");}
        catch(...){ggml_backend_free(backend);backend=nullptr;throw;}
#else
        (void)name;throw std::runtime_error("RPC profiling requires GGML_RPC=ON");
#endif
    }
    ~rpc_batch(){
#ifdef GGML_USE_RPC
        if(active){uint32_t count=0;ggml_backend_rpc_profile_end(backend,active,records.data(),uint32_t(records.size()),&count);}
        if(backend)ggml_backend_free(backend);
#endif
    }
    void start(size_t expected){
#ifdef GGML_USE_RPC
        if(expected>GGML_RPC_PROFILE_MAX_RECORDS)throw std::runtime_error("RPC request exceeds profiling record capacity");
        records.resize(std::max(size_t(64),expected));
        auto window=++sequence;
        if(!ggml_backend_rpc_profile_begin(backend,window,uint32_t(records.size())))throw std::runtime_error("RPC worker does not support batch timing or a window is already active; requires protocol 6.1");
        active=window;
#else
        (void)expected;
#endif
    }
    void finish(capture & c,size_t trace_start){
#ifdef GGML_USE_RPC
        const auto window=active;const auto count=drain();
        std::vector<size_t> submissions;
        const auto end=c.output.trace.size();
        for(size_t i=trace_start;i<end;++i)if(c.output.trace[i].at("kind")=="backend")submissions.push_back(i);
        if(count!=submissions.size())throw std::runtime_error("RPC records do not match frozen graph submissions");
        for(size_t i=0;i<count;++i){const auto & record=records[i];auto event=c.output.trace[submissions[i]];
            const auto submitted_id=event.at("event_id").get<std::string>();
            event["event_id"]=submitted_id+"-worker";event["submission_event_id"]=submitted_id;event["parent_event_id"]=nullptr;
            event["clock_domain"]="worker:"+c.device_id+":"+std::to_string(window);event["window_id"]=window;
            event["event_type"]="completed";event["timing_scope"]="rpc_worker_synchronized_host_wall";
            event["start_ms"]=(double(record.start_ns)+double(record.prepare_ns))/1e6;event["duration_ms"]=double(record.compute_ns)/1e6;
            event["export_before_ms"]=0;event["worker_prepare_ms"]=double(record.prepare_ns)/1e6;
            event["worker_sequence"]=record.sequence;event["worker_device"]=record.device;event["worker_nodes"]=record.n_nodes;event["worker_reused"]=bool(record.reused);
            event["status"]=record.status;c.samples[event.at("graph_id").get<std::string>()].push_back(double(record.compute_ns)/1e6);
            c.output.trace.push_back(std::move(event));
        }
#else
        (void)c;(void)trace_start;
#endif
    }
};
}
result profile(const request & input) {
    // Backend configuration and SD callbacks are process-global. Profiling owns this process.
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    json descriptions=json::array();for(const auto & d:input.devices)descriptions.push_back({{"id",d.id},{"transport",d.transport},{"backend",d.backend},{"endpoint",d.endpoint}});
    const json request_json={{"model",input.model},{"devices",descriptions},{"representative_workload",input.representative_workload}};
    parse_request(request_json);
    const auto model=std::filesystem::canonical(input.model);
    const auto workload=std::filesystem::canonical(input.representative_workload);
    auto package=resolve_model(model.string());
    if(package.family!=model_family::sensenova_u1)throw std::invalid_argument("Real profiler currently supports SenseNova U1 model packages");
    const auto raw_cases=read(workload/"cases.jsonl");
    const auto workload_manifest=json::parse(read(workload/"manifest.json"));
    if(workload_manifest.value("schema_version","")!="workload/v1")throw std::invalid_argument("Unsupported workload schema");
    std::istringstream lines(raw_cases);std::string line;std::vector<json> cases;json original_cases=json::array();std::set<std::string> ids;
    size_t max_images=0;bool generation_only=true;
    while(std::getline(lines,line)) {
        if(line.empty())continue;
        auto c=json::parse(line);original_cases.push_back(c);const auto id=c.at("case_id").get<std::string>();
        if(id.empty() || !ids.insert(id).second)throw std::invalid_argument("Duplicate or empty workload case ID");
        const auto task=c.at("task_class").get<std::string>();
        if(task!="text_generation" && task!="mixed_understanding" && task!="multimodal_prefill" && task!="image_generation")throw std::invalid_argument("Unsupported task class: "+task);
        generation_only&=task=="image_generation";size_t images=0;
        for(auto & item:c.at("content"))if(item.at("type")=="image") {
            ++images;auto path=std::filesystem::canonical(workload/item.at("path").get<std::string>());
            const auto bytes=read(path); // PNG/JPEG content is checked by the regular UMM request path.
            const auto digest=hash(bytes);
            if(item.contains("sha256") && item.at("sha256")!=digest)throw std::invalid_argument("Workload image hash mismatch");
            const auto mime=bytes.size()>=8 && bytes.compare(0,8,std::string("\x89PNG\r\n\x1a\n",8))==0?"image/png":"image/jpeg";
            item["data_url"]=std::string("data:")+mime+";base64,"+::base64::encode(bytes);item.erase("path");item.erase("sha256");
        }
        max_images=std::max(max_images,images);cases.push_back(std::move(c));
    }
    if(cases.empty() || workload_manifest.at("case_count").get<size_t>()!=cases.size())throw std::invalid_argument("Workload case count mismatch");
    if(workload_manifest.contains("cases_sha256") && workload_manifest["cases_sha256"]!=hash(raw_cases))throw std::invalid_argument("Workload cases hash mismatch");
    result out;out.graphs=json::array();out.trace=json::array();out.costs=json::array();out.requests=json::array();
    json identities=json::array();bool stable=true;
    const auto model_manifest=read(model/"model.json");
    const auto model_info=json::parse(model_manifest);
    const auto lora=model_info.value("merged_lora",json::object());
    const bool fast_lora=lora.is_object() && lora.value("file",std::string{}).find("8step-V2")!=std::string::npos;
    json components=json::object();for(const auto & comp:package.components){std::cerr<<"profiler: hashing model component "<<comp.first<<std::endl;components[comp.first]=file_hash(comp.second);}
    // The current 310P NZ implementation latches its environment on first use.
    // Keep one setting for all local CANN workers in this process; RPC worker settings remain remote.
    environment env;env.set("GGML_CANN_Q8_NZ","1");env.set("GGML_CANN_SATURATION","1");env.set("GGML_SCHED_STRICT_ACCEL","1");
    candidate_catalog catalog(hash(components.dump()));
    auto devices=input.devices;std::stable_partition(devices.begin(),devices.end(),[](const auto & d){return d.transport=="local";});
    struct backend_owner{ggml_backend_t ptr=nullptr;~backend_owner(){if(ptr)ggml_backend_free(ptr);}} primary;
    std::string primary_id;
    json overhead=json::array();
    for(const auto & d:devices) {
        capture c{out};c.device_id=d.id;c.catalog=&catalog;catalog.start_device(d.id);
        env.set("GGML_CANN_Q8_NZ",d.transport=="local"?"1":"0");
        std::string backend=d.backend;
        if(d.transport=="rpc") {
            if(d.backend!="CANN0")throw std::invalid_argument("RPC profiling requires a single-card CANN0 worker");
            env.set("GGML_RPC_NO_RDMA","1");backend=register_remote_cann(d.endpoint);
        } else {
            ggml_backend_load_all();auto dev=ggml_backend_dev_by_name(backend.c_str());
            if(!dev)throw std::invalid_argument("Device is unavailable: "+backend);
            if(std::string(ggml_backend_dev_description(dev)).find("310P")==std::string::npos)throw std::invalid_argument("Local profiling currently requires Ascend310P");
        }
        env.set("GGML_REQUIRED_DEVICE",backend);c.requested_backend=backend;
        std::unique_ptr<rpc_batch> batch;
        if(d.transport=="rpc")batch=std::make_unique<rpc_batch>(backend);
        if(!primary.ptr && d.transport=="local"){primary.ptr=ggml_backend_init_by_name(backend.c_str(),nullptr);if(!primary.ptr)throw std::runtime_error("Cannot initialize primary backend");primary_id=d.id;}
        runtime::options options={{"--model",model.string()},{"--understanding-backend",backend},{"--generation-backend",backend},
            {"--vision-backend",backend},{"--generation-max-vram",backend+"=40"},{"--steps",fast_lora?"8":"50"},{"--cfg",fast_lora?"1":"4"},{"--shift","3"},
            {"--n-ctx",generation_only?"2048":max_images>4?"32768":"8192"},
            {"--n-batch",generation_only?"512":max_images>4?"1152":"4096"},
            {"--n-ubatch",generation_only?"256":max_images>4?"1152":"4096"},
            {"--cann-q8-nz",d.transport=="local"?"1":"0"}};
        const auto init=clock::now();
        session engine(model.string(),"",backend,backend,backend+"=40",backend,
            std::stoi(options["--n-ctx"]),std::stoi(options["--n-batch"]),std::stoi(options["--n-ubatch"]));
        const auto startup=elapsed(init);
        identities.push_back({{"device_id",d.id},{"resolved_options",options},{"runtime",runtime::runtime_identity(model.string(),options)},
            {"startup_ms",startup},{"remote_configuration_verified",false}});
        std::map<std::string,std::string> expected;
        std::map<std::string,std::vector<double>> baseline_ms,timed_ms;
        const std::array<std::string,6> passes={"structure","capture","baseline","timing","baseline","timing"};
        for(size_t replay=0;replay<passes.size();++replay) {
            c.replay=int(replay);c.pass=passes[replay];c.warmup=c.pass!="timing";
            std::cerr<<"profiler: device="<<d.id<<" pass="<<c.pass<<" replay="<<replay<<std::endl;
            for(const auto & item:cases) {
                c.case_id=item.at("case_id").get<std::string>();c.execution_cursor=0;
                c.remote_timing=d.transport=="rpc" && c.pass=="timing";
                const auto phase=item.at("task_class")=="image_generation"?"generation":"request";
                phase_scope scope(phase);const auto trace_start=out.trace.size();
                json response;const auto request_start=clock::now();
                if(c.remote_timing){size_t expected_graphs=0;for(const auto & binding:c.bindings.at(c.case_id))expected_graphs+=!binding.scheduler;batch->start(expected_graphs);}
                if(c.pass=="baseline")response=runtime::run_case(engine,item,options,true);
                else {observer_scope observation(c);response=runtime::run_case(engine,item,options,true);}
                if(c.remote_timing)batch->finish(c,trace_start);
                const double wall=elapsed(request_start);
                if(c.pass=="timing" && c.error.empty() && c.execution_cursor!=c.bindings.at(c.case_id).size())throw std::runtime_error("Execution has fewer graphs than frozen structure");
                if(!c.error.empty())throw std::runtime_error(c.error);
                if(response.value("status","")!="ok")throw std::runtime_error("Profile request failed: "+response.dump());
                if(!c.stack.empty())throw std::runtime_error("Unclosed graph observation");
                if((c.pass=="timing"||c.pass=="structure") && out.trace.size()==trace_start)throw std::runtime_error("No graph execution was observed");
                if(c.pass=="baseline")baseline_ms[c.case_id].push_back(wall);
                if(c.pass=="timing")timed_ms[c.case_id].push_back(wall);
                const auto digest=hash(response.at("output").dump());
                if(replay==0)expected[c.case_id]=digest;else stable&=expected.at(c.case_id)==digest;
                out.requests.push_back({{"device_id",d.id},{"case_id",c.case_id},{"replay",replay},{"pass",c.pass},{"warmup",c.warmup},{"wall_ms",wall},
                    {"output_sha256",digest},{"matches_warmup",digest==expected.at(c.case_id)},{"response",response}});
            }
            if(c.pass=="structure")catalog.select();
        }
        backend_owner target;target.ptr=ggml_backend_init_by_name(backend.c_str(),nullptr);
        if(!target.ptr)throw std::runtime_error("Cannot initialize measurement backend");
        std::cerr<<"profiler: measuring selected candidates on "<<d.id<<std::endl;
        catalog.measure(target.ptr,primary.ptr,primary_id);
        for(const auto & item:baseline_ms){json pairs=json::array();for(size_t i=0;i<item.second.size();++i)pairs.push_back({{"baseline_ms",item.second[i]},{"timing_ms",timed_ms.at(item.first).at(i)},{"observed_delta_ms",timed_ms.at(item.first).at(i)-item.second[i]}});
            overhead.push_back({{"device_id",d.id},{"case_id",item.first},{"paired_replays",pairs},{"interpretation","includes_runtime_variation_and_observation_overhead_not_a_correction_factor"}});}
        for(auto & entry:c.samples) {
            const auto count=entry.second.size();auto duration=detail::sample_statistics(entry.second);
            duration.erase("sample_count");duration.erase("samples");
            out.costs.push_back({{"schema_version","umm-profiler-cost/v3"},{"data_origin","measured"},{"graph_id",entry.first},
                {"device_id",d.id},{"sample_count",count},{"state","post_warmup_replay"},
                {"duration_ms",duration},
                {"timing_scope",d.transport=="rpc"?"rpc_worker_synchronized_host_wall":"synchronized_host_wall"},{"measurement_valid",true},{"decision_usable",false},
                {"memory_peak_bytes",nullptr},{"transfer",nullptr},{"overlap",nullptr},{"kernel_ms",nullptr}});
        }
    }
    out.candidates=catalog.records();out.candidate_costs=catalog.measurements();out.snapshots=catalog.snapshots();
    out.manifest={{"schema_version","umm-profiler/v3"},{"data_origin","measured"},{"status","completed"},{"executed",true},
        {"decision_usable",false},{"request",request_json},{"identity",{{"model_manifest_sha256",hash(model_manifest)},{"model_components_sha256",components},{"profiler_binary_sha256",std::filesystem::exists("/proc/self/exe")?json(file_hash("/proc/self/exe")):json(nullptr)},{"devices",identities}}},
        {"workload",{{"case_count",cases.size()},{"cases_sha256",hash(raw_cases)},{"manifest",workload_manifest},{"cases",original_cases}}},
        {"measurement",{{"passes",{"structure","capture","baseline","timing","baseline","timing"}},{"measured_replays",2},{"observation_overhead",overhead},{"candidate_measurement","two_warmups_five_repetitions_native_weights_real_snapshot"},{"serial_devices",true},{"graph_synchronization",true},{"rpc_completion_fence","bounded_tensor_read_up_to_4096_bytes"},{"rpc_scheduler_completion","reuse_successful_child_fences_without_eval_callback"},{"rpc_timing","worker_batch_per_request"},{"rpc_profile_protocol","6.1"},{"rpc_submission_scope","client_submission_host_wall"},
            {"timing_scope","synchronized_host_wall"},{"nested_events_are_not_additive",true},{"capture_overhead_in_parent_events",true},{"full_json_export_in_timing_pass",false},{"structural_hash_in_timing_pass",false},{"identity_resolution","frozen_case_execution_contract"},{"contract_validation","all_tensor_metadata_topology_and_devices_without_hash"}}},
        {"coverage",{{"workload_loaded",true},{"device_probed",true},{"cut_legality",false},{"isolated_ffn_replay",true},{"selected_boundary_roundtrip",true},{"operator_timing",false},
            {"remote_worker_metrics",std::any_of(devices.begin(),devices.end(),[](const auto & d){return d.transport=="rpc";})},{"phase_coverage","understanding prefill/decode, U1 vision, coarse generation"},
            {"correctness",stable?"within_device_output_repeatable":"within_device_output_changed"}}},
        {"artifacts",{"graphs.jsonl","trace.jsonl","costs.jsonl","requests.jsonl","candidates.jsonl","candidate-costs.jsonl","snapshots.jsonl","report.md"}}};
    out.report="# Structure-led candidate profiler\n\n"
        "Real model execution with six passes per device: structure, snapshot capture, baseline, lightweight timing, baseline, lightweight timing. Devices run serially; local devices run first to supply canonical real activations. The request interface remains model, devices and representative workload.\n\n"
        "Candidate extraction currently recognizes closed, contiguous F32-boundary SwiGLU FFNs with three resident model weights. It rejects views and external consumers of intermediate values. It ranks by estimated matrix FLOPs per boundary byte, selecting one layer per input shape, at most four shapes in first-observed order. This is a structural heuristic, not a profitability proof.\n\n"
        "Selected inputs and reference outputs are captured at scheduler execution boundaries in a separate untimed pass. All devices replay the same canonical inputs with their native resident model weights. Snapshots reference the model hashes and weight identities; weights are not re-uploaded for each measurement. Candidate execution uses two warmups and five timed repetitions.\n\n"
        "Candidate costs include compute, copy-only boundary roundtrip, and boundary plus compute where a local primary is available. Boundary copies use the actual GGML tensor-copy path. The local copy roundtrip is an artificial control; native execution uses the compute-only baseline. Full F32 output error is checked against the captured reference (atol=.05, rtol=.005); this is not end-to-end model quality acceptance.\n\n"
        "Local and structure/capture timings use synchronized host wall intervals with RPC completion-read fences where applicable. RPC timing passes instead collect client submission events without added waits, then fetch worker execution records once per request using protocol 6.1. Only completed worker records enter remote graph costs; worker timestamps are in separate per-window clock domains. These are synchronized worker host intervals, not kernel time. Full JSON export is excluded from timing passes; frozen-contract checks and batch collection still perturb execution. Structure hashing runs only in the structure pass; timed requests use registered graph IDs and verify immutable metadata/topology/device contracts without hashing or serialization. Paired observer-off baselines quantify the observed change, including runtime variation. Nested scheduler/backend intervals overlap and must not be summed.\n\n"
        "Decision usability remains false: isolated replay eligibility does not establish production cut legality, plan correctness, unseen-shape coverage, memory capacity, or end-to-end scheduling benefit. No production scheduling policy changes are made.\n";
    return out;
}
}
