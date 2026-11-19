#include "profiler.h"
#include "inference/profile-context.h"
#include "profiler-annotations.h"
#include "profiler-hierarchy.h"
#include "profiler-graph-contract.h"
#include <memory>
#include "runtime/runtime-support.h"
#include "scheduling/execution-policy.h"
#include "inference/model-package.h"
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
double elapsed(clock::time_point start) {
    return std::chrono::duration<double, std::milli>(clock::now() - start).count();
}
using detail::hash;
std::string read(const std::filesystem::path & path) {
    std::ifstream file(path, std::ios::binary);
    if (!file)
        throw std::invalid_argument("Cannot read " + path.string());
    std::ostringstream out;
    out << file.rdbuf();
    if (file.bad())
        throw std::runtime_error("Read failed: " + path.string());
    return out.str();
}
std::string file_hash(const std::filesystem::path & path) {
    std::ifstream file(path, std::ios::binary);
    if (!file)
        throw std::invalid_argument("Cannot hash " + path.string());
    sha256_t state;
    sha256_init(&state);
    std::array<unsigned char, 1024 * 1024> buffer;
    while (file) {
        file.read(reinterpret_cast<char *>(buffer.data()), buffer.size());
        sha256_update(&state, buffer.data(), size_t(file.gcount()));
    }
    if (!file.eof())
        throw std::runtime_error("Hash read failed");
    unsigned char digest[32];
    sha256_final(&state, digest);
    std::ostringstream out;
    for (auto byte : digest)
        out << std::hex << std::setw(2) << std::setfill('0') << int(byte);
    return out.str();
}
struct environment {
    std::map<std::string, std::pair<bool, std::string>> old;
    void set(const std::string & key, const std::string & value) {
        if (!old.count(key)) {
            const char * p = std::getenv(key.c_str());
            old[key] = {p != nullptr, p ? p : ""};
        }
        if (setenv(key.c_str(), value.c_str(), 1))
            throw std::runtime_error("Cannot configure environment");
    }
    ~environment() {
        for (const auto & item : old) {
            if (item.second.first)
                setenv(item.first.c_str(), item.second.second.c_str(), 1);
            else
                unsetenv(item.first.c_str());
        }
    }
};
struct capture {
    result & output;
    std::string device_id, case_id, requested_backend, error;
    std::string pass = "structure";
    int replay = 0;
    bool warmup = false;
    bool remote_timing = false;
    bool remote = false;
    detail::hierarchy_capture hierarchy;
    clock::time_point origin = clock::now();
    struct frame {
        std::string graph_id, event_id, parent;
        clock::time_point start;
        double export_ms;
    };
    std::vector<frame> stack;
    std::set<std::string> seen;
    struct binding {
        std::string id, phase;
        std::unique_ptr<graph_contract> contract;
        bool scheduler = false;
    };
    std::map<std::string, std::vector<binding>> bindings;
    size_t execution_cursor = 0;
    size_t event_number = 0;
    std::map<std::string, std::vector<double>> samples;
    bool fine() const { return pass == "layer" || pass == "block" || pass == "operator"; }
    bool observe_fine(ggml_backend_t backend, ggml_backend_sched_t sched, ggml_cgraph * graph, bool completed) {
        if (!completed) {
            const auto & binding = bindings.at(case_id).at(execution_cursor++);
            if (binding.phase != current_phase.phase) {
                throw std::runtime_error("Execution phase differs from frozen structure");
            }
            binding.contract->check(backend, sched, graph);
            if (backend && (binding.phase == "understanding_prefill" || binding.phase == "understanding_decode")) {
                hierarchy.begin(backend, graph, binding.id, pass);
                stack.push_back({binding.id, {}, {}, clock::now(), 0});
            }
        } else if (backend && !stack.empty()) {
            auto sample = hierarchy.finish(backend);
            if (sample.is_null()) {
                throw std::runtime_error("Unmatched fine timing graph");
            }
            const auto tokens = current_phase.tokens;
            const auto cached = current_phase.cached_tokens;
            sample.update({
                {"schema_version", "umm-profiler-hierarchy/v1"},
                {"graph_id", stack.back().graph_id},
                {"device_id", device_id},
                {"case_id", case_id},
                {"replay", replay},
                {"level", pass},
                {"submission",
                 {{"tokens", tokens < 0 ? json(nullptr) : json(tokens)},
                  {"cached_tokens", cached < 0 ? json(nullptr) : json(cached)}}},
            });
            output.hierarchy.push_back(std::move(sample));
            stack.pop_back();
        }
        return true;
    }
    bool observe(ggml_backend_t backend, ggml_backend_sched_t sched, ggml_cgraph * graph, bool completed,
                 ggml_status status) {
        if (fine()) {
            if (status != GGML_STATUS_SUCCESS)
                throw std::runtime_error("Fine timing graph execution failed");
            return observe_fine(backend, sched, graph, completed);
        }
        if (!completed) {
            auto start = clock::now();
            std::string fingerprint;
            if (pass == "structure") {
                auto contract = std::make_unique<graph_contract>(backend, sched, graph);
                const auto & descriptor = contract->description();
                fingerprint = descriptor.signature(current_phase.phase, requested_backend, true);
                if (seen.insert(fingerprint).second) {
                    auto description = descriptor.describe(current_phase.phase, current_phase.tokens,
                                                           current_phase.cached_tokens, requested_backend);
                    description["graph_id"] = fingerprint;
                    description["logical_graph_id"] =
                        descriptor.signature(current_phase.phase, requested_backend, false);
                    description["device_id"] = device_id;
                    output.graphs.push_back(std::move(description));
                }
                bindings[case_id].push_back({fingerprint, current_phase.phase, std::move(contract), sched != nullptr});
            } else {
                auto & sequence = bindings.at(case_id);
                if (execution_cursor >= sequence.size())
                    throw std::runtime_error("Execution has more graphs than frozen structure");
                auto & registered = sequence[execution_cursor];
                if (registered.phase != current_phase.phase)
                    throw std::runtime_error("Execution phase differs from frozen structure");
                registered.contract->check(backend, sched, graph);
                fingerprint = registered.id;
            }
            ++execution_cursor;
            const auto parent = stack.empty() ? std::string{} : stack.back().event_id;
            stack.push_back(
                {fingerprint, device_id + "-e" + std::to_string(++event_number), parent, clock::now(), elapsed(start)});
        } else {
            if (stack.empty())
                throw std::runtime_error("Unmatched graph event");
            auto f = stack.back();
            stack.pop_back();
            const double duration = elapsed(f.start);
            output.trace.push_back(
                {{"schema_version", "umm-profiler-event/v3"},
                 {"data_origin", "measured"},
                 {"graph_id", f.graph_id},
                 {"event_id", f.event_id},
                 {"parent_event_id", f.parent.empty() ? json(nullptr) : json(f.parent)},
                 {"device_id", device_id},
                 {"case_id", case_id},
                 {"replay", replay},
                 {"pass", pass},
                 {"warmup", warmup},
                 {"kind", sched ? "scheduler" : "backend"},
                 {"phase", current_phase.phase},
                 {"submission", {{"tokens", current_phase.tokens}, {"cached_tokens", current_phase.cached_tokens}}},
                 {"start_ms", std::chrono::duration<double, std::milli>(f.start - origin).count()},
                 {"duration_ms", duration},
                 {"export_before_ms", f.export_ms},
                 {"status", int(status)},
                 {"timing_scope", remote_timing ? "client_submission_host_wall" : "synchronized_host_wall"},
                 {"event_type", remote_timing ? "submitted" : "completed"},
                 {"clock_domain", "client:" + device_id},
                 {"executed", true},
                 {"kernel_ms", nullptr}});
            if (status != GGML_STATUS_SUCCESS)
                throw std::runtime_error("Graph execution failed");
            if (!warmup && !remote_timing)
                samples[f.graph_id].push_back(duration);
        }
        return true;
    }
    static bool callback(ggml_backend_t backend, ggml_backend_sched_t sched, ggml_cgraph * graph, bool completed,
                         ggml_status status, void * user) noexcept {
        auto & self = *static_cast<capture *>(user);
        try {
            return self.observe(backend, sched, graph, completed, status);
        } catch (const std::exception & e) {
            if (self.error.empty())
                self.error = e.what();
            return false;
        } catch (...) {
            self.error = "Graph capture failed";
            return false;
        }
    }
};
struct observer_scope {
    capture & collector;
    bool submission;
    explicit observer_scope(capture & c) : collector(c), submission(c.remote_timing || (c.remote && c.fine())) {
        if (submission)
            ggml_backend_set_graph_submission_observer(capture::callback, &c);
        else
            ggml_backend_set_graph_observer(capture::callback, &c);
    }
    ~observer_scope() {
        if (submission)
            ggml_backend_set_graph_submission_observer(nullptr, nullptr);
        else
            ggml_backend_set_graph_observer(nullptr, nullptr);
        collector.hierarchy.cancel();
    }
};

// One window per request; its final response fences the ordered worker stream.
class rpc_batch {
#ifdef GGML_USE_RPC
    ggml_backend_t backend = nullptr;
    uint64_t sequence = 0, active = 0;
    std::vector<ggml_rpc_profile_record> records;
    uint32_t drain() {
        uint32_t count = 0;
        auto window = active;
        active = 0;
        if (!ggml_backend_rpc_profile_end(backend, window, records.data(), uint32_t(records.size()), &count))
            throw std::runtime_error("RPC profile records missing, overflowed or out of order");
        return count;
    }
#endif
public:
    explicit rpc_batch(const std::string & name) {
#ifdef GGML_USE_RPC
        backend = ggml_backend_init_by_name(name.c_str(), nullptr);
        if (!backend)
            throw std::runtime_error("Cannot initialize RPC profiling backend");
        try {
            start(1);
            if (drain() != 0)
                throw std::runtime_error("RPC profiling probe returned unexpected graphs");
        } catch (...) {
            ggml_backend_free(backend);
            backend = nullptr;
            throw;
        }
#else
        (void)name;
        throw std::runtime_error("RPC profiling requires GGML_RPC=ON");
#endif
    }
    ~rpc_batch() {
#ifdef GGML_USE_RPC
        if (active) {
            uint32_t count = 0;
            ggml_backend_rpc_profile_end(backend, active, records.data(), uint32_t(records.size()), &count);
        }
        if (backend)
            ggml_backend_free(backend);
#endif
    }
    void start(size_t expected) {
#ifdef GGML_USE_RPC
        if (expected > GGML_RPC_PROFILE_MAX_RECORDS)
            throw std::runtime_error("RPC request exceeds profiling record capacity");
        records.resize(std::max(size_t(64), expected));
        auto window = ++sequence;
        if (!ggml_backend_rpc_profile_begin(backend, window, uint32_t(records.size())))
            throw std::runtime_error(
                "RPC worker does not support batch timing or a window is already active; requires protocol 6.1");
        active = window;
#else
        (void)expected;
#endif
    }
    void finish(capture & c, size_t trace_start) {
#ifdef GGML_USE_RPC
        const auto window = active;
        const auto count = drain();
        std::vector<size_t> submissions;
        const auto end = c.output.trace.size();
        for (size_t i = trace_start; i < end; ++i)
            if (c.output.trace[i].at("kind") == "backend")
                submissions.push_back(i);
        if (count != submissions.size())
            throw std::runtime_error("RPC records do not match frozen graph submissions");
        for (size_t i = 0; i < count; ++i) {
            const auto & record = records[i];
            auto event = c.output.trace[submissions[i]];
            const auto submitted_id = event.at("event_id").get<std::string>();
            event["event_id"] = submitted_id + "-worker";
            event["submission_event_id"] = submitted_id;
            event["parent_event_id"] = nullptr;
            event["clock_domain"] = "worker:" + c.device_id + ":" + std::to_string(window);
            event["window_id"] = window;
            event["event_type"] = "completed";
            event["timing_scope"] = "rpc_worker_synchronized_host_wall";
            event["start_ms"] = (double(record.start_ns) + double(record.prepare_ns)) / 1e6;
            event["duration_ms"] = double(record.compute_ns) / 1e6;
            event["export_before_ms"] = 0;
            event["worker_prepare_ms"] = double(record.prepare_ns) / 1e6;
            event["worker_sequence"] = record.sequence;
            event["worker_device"] = record.device;
            event["worker_nodes"] = record.n_nodes;
            event["worker_reused"] = bool(record.reused);
            event["status"] = record.status;
            c.samples[event.at("graph_id").get<std::string>()].push_back(double(record.compute_ns) / 1e6);
            c.output.trace.push_back(std::move(event));
        }
#else
        (void)c;
        (void)trace_start;
#endif
    }
};
}
result profile(const request & input, bool operator_timing) {
    // Backend configuration and SD callbacks are process-global. Profiling owns this process.
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    json descriptions = json::array();
    for (const auto & d : input.devices)
        descriptions.push_back(
            {{"id", d.id}, {"transport", d.transport}, {"backend", d.backend}, {"endpoint", d.endpoint}});
    const json request_json = {
        {"model", input.model}, {"devices", descriptions}, {"representative_workload", input.representative_workload}};
    parse_request(request_json);
    const auto model = std::filesystem::canonical(input.model);
    const auto workload = std::filesystem::canonical(input.representative_workload);
    auto package = resolve_model(model.string());
    if (package.family != model_family::sensenova_u1)
        throw std::invalid_argument("Real profiler currently supports SenseNova U1 model packages");
    const auto raw_cases = read(workload / "cases.jsonl");
    const auto workload_manifest = json::parse(read(workload / "manifest.json"));
    if (workload_manifest.value("schema_version", "") != "workload/v1")
        throw std::invalid_argument("Unsupported workload schema");
    std::istringstream lines(raw_cases);
    std::string line;
    std::vector<json> cases;
    json original_cases = json::array();
    std::set<std::string> ids;
    size_t max_images = 0;
    bool generation_only = true;
    while (std::getline(lines, line)) {
        if (line.empty())
            continue;
        auto c = json::parse(line);
        original_cases.push_back(c);
        const auto id = c.at("case_id").get<std::string>();
        if (id.empty() || !ids.insert(id).second)
            throw std::invalid_argument("Duplicate or empty workload case ID");
        const auto task = c.at("task_class").get<std::string>();
        if (task != "text_generation" && task != "mixed_understanding" && task != "multimodal_prefill" &&
            task != "image_generation")
            throw std::invalid_argument("Unsupported task class: " + task);
        generation_only &= task == "image_generation";
        size_t images = 0;
        for (auto & item : c.at("content"))
            if (item.at("type") == "image") {
                ++images;
                auto path = std::filesystem::canonical(workload / item.at("path").get<std::string>());
                const auto bytes = read(path); // PNG/JPEG content is checked by the regular UMM request path.
                const auto digest = hash(bytes);
                if (item.contains("sha256") && item.at("sha256") != digest)
                    throw std::invalid_argument("Workload image hash mismatch");
                const auto mime = bytes.size() >= 8 && bytes.compare(0, 8, std::string("\x89PNG\r\n\x1a\n", 8)) == 0
                                      ? "image/png"
                                      : "image/jpeg";
                item["data_url"] = std::string("data:") + mime + ";base64," + ::base64::encode(bytes);
                item.erase("path");
                item.erase("sha256");
            }
        max_images = std::max(max_images, images);
        cases.push_back(std::move(c));
    }
    if (cases.empty() || workload_manifest.at("case_count").get<size_t>() != cases.size())
        throw std::invalid_argument("Workload case count mismatch");
    if (workload_manifest.contains("cases_sha256") && workload_manifest["cases_sha256"] != hash(raw_cases))
        throw std::invalid_argument("Workload cases hash mismatch");
    result out;
    out.graphs = json::array();
    out.trace = json::array();
    out.costs = json::array();
    out.requests = json::array();
    json identities = json::array();
    bool stable = true;
    const auto model_manifest = read(model / "model.json");
    const auto model_info = json::parse(model_manifest);
    const auto lora = model_info.value("merged_lora", json::object());
    const bool fast_lora = lora.is_object() && lora.value("file", std::string{}).find("8step-V2") != std::string::npos;
    json components = json::object();
    for (const auto & comp : package.components) {
        std::cerr << "profiler: hashing model component " << comp.first << std::endl;
        components[comp.first] = file_hash(comp.second);
    }
    // The current 310P NZ implementation latches its environment on first use.
    // Keep one setting for all local CANN workers in this process; RPC worker settings remain remote.
    environment env;
    env.set("GGML_CANN_Q8_NZ", "1");
    env.set("GGML_CANN_SATURATION", "1");
    env.set("GGML_SCHED_STRICT_ACCEL", "1");
    auto devices = input.devices;
    std::stable_partition(devices.begin(), devices.end(), [](const auto & d) { return d.transport == "local"; });
    json overhead = json::array();
    std::vector<std::string> passes = {"structure", "baseline", "timing", "baseline", "timing", "layer", "block"};
    if (operator_timing)
        passes.push_back("operator");
    for (const auto & d : devices) {
        capture c{out};
        c.device_id = d.id;
        c.remote = d.transport == "rpc";
        env.set("GGML_CANN_Q8_NZ", d.transport == "local" ? "1" : "0");
        std::string backend = d.backend;
        if (d.transport == "rpc") {
            if (d.backend != "CANN0")
                throw std::invalid_argument("RPC profiling requires a single-card CANN0 worker");
            env.set("GGML_RPC_NO_RDMA", "1");
            backend = register_remote_cann(d.endpoint);
        } else {
            ggml_backend_load_all();
            auto dev = ggml_backend_dev_by_name(backend.c_str());
            if (!dev)
                throw std::invalid_argument("Device is unavailable: " + backend);
            if (std::string(ggml_backend_dev_description(dev)).find("310P") == std::string::npos)
                throw std::invalid_argument("Local profiling currently requires Ascend310P");
        }
        env.set("GGML_REQUIRED_DEVICE", backend);
        c.requested_backend = backend;
        std::unique_ptr<rpc_batch> batch;
        if (d.transport == "rpc")
            batch = std::make_unique<rpc_batch>(backend);
        runtime::options options = {{"--model", model.string()},
                                    {"--understanding-backend", backend},
                                    {"--generation-backend", backend},
                                    {"--vision-backend", backend},
                                    {"--generation-max-vram", backend + "=40"},
                                    {"--steps", fast_lora ? "8" : "50"},
                                    {"--cfg", fast_lora ? "1" : "4"},
                                    {"--shift", "3"},
                                    {"--n-ctx", generation_only  ? "2048"
                                                : max_images > 4 ? "32768"
                                                                 : "8192"},
                                    {"--n-batch", generation_only  ? "512"
                                                  : max_images > 4 ? "1152"
                                                                   : "4096"},
                                    {"--n-ubatch", generation_only  ? "256"
                                                   : max_images > 4 ? "1152"
                                                                    : "4096"},
                                    {"--cann-q8-nz", d.transport == "local" ? "1" : "0"}};
        const auto init = clock::now();
        session engine(model.string(), "", backend, backend, backend + "=40", backend, std::stoi(options["--n-ctx"]),
                       std::stoi(options["--n-batch"]), std::stoi(options["--n-ubatch"]));
        const auto startup = elapsed(init);
        identities.push_back({{"device_id", d.id},
                              {"resolved_options", options},
                              {"runtime", runtime::runtime_identity(model.string(), options)},
                              {"startup_ms", startup},
                              {"remote_configuration_verified", false}});
        std::map<std::string, std::string> expected;
        std::map<std::string, std::vector<double>> baseline_ms, timed_ms;
        for (size_t replay = 0; replay < passes.size(); ++replay) {
            c.replay = int(replay);
            c.pass = passes[replay];
            c.warmup = c.pass != "timing";
            std::cerr << "profiler: device=" << d.id << " pass=" << c.pass << " replay=" << replay << std::endl;
            for (const auto & item : cases) {
                c.case_id = item.at("case_id").get<std::string>();
                c.execution_cursor = 0;
                c.remote_timing = d.transport == "rpc" && c.pass == "timing";
                const auto phase = item.at("task_class") == "image_generation" ? "generation" : "request";
                phase_scope scope(phase);
                const auto trace_start = out.trace.size();
                json response;
                const auto request_start = clock::now();
                if (c.remote_timing) {
                    size_t expected_graphs = 0;
                    for (const auto & binding : c.bindings.at(c.case_id))
                        expected_graphs += !binding.scheduler;
                    batch->start(expected_graphs);
                }
                if (c.pass == "baseline")
                    response = runtime::run_case(engine, item, options, true);
                else {
                    observer_scope observation(c);
                    response = runtime::run_case(engine, item, options, true);
                }
                if (c.remote_timing)
                    batch->finish(c, trace_start);
                const double wall = elapsed(request_start);
                if ((c.pass == "timing" || c.fine()) && c.error.empty() &&
                    c.execution_cursor != c.bindings.at(c.case_id).size())
                    throw std::runtime_error("Execution has fewer graphs than frozen structure");
                if (!c.error.empty())
                    throw std::runtime_error(c.error);
                if (response.value("status", "") != "ok")
                    throw std::runtime_error("Profile request failed: " + response.dump());
                if (!c.stack.empty())
                    throw std::runtime_error("Unclosed graph observation");
                if ((c.pass == "timing" || c.pass == "structure") && out.trace.size() == trace_start)
                    throw std::runtime_error("No graph execution was observed");
                if (c.pass == "baseline")
                    baseline_ms[c.case_id].push_back(wall);
                if (c.pass == "timing")
                    timed_ms[c.case_id].push_back(wall);
                const auto digest = hash(response.at("output").dump());
                if (replay == 0)
                    expected[c.case_id] = digest;
                else
                    stable &= expected.at(c.case_id) == digest;
                out.requests.push_back({{"device_id", d.id},
                                        {"case_id", c.case_id},
                                        {"replay", replay},
                                        {"pass", c.pass},
                                        {"warmup", c.warmup},
                                        {"wall_ms", wall},
                                        {"output_sha256", digest},
                                        {"matches_warmup", digest == expected.at(c.case_id)},
                                        {"response", response}});
            }
        }
        for (const auto & item : baseline_ms) {
            json pairs = json::array();
            for (size_t i = 0; i < item.second.size(); ++i)
                pairs.push_back({{"baseline_ms", item.second[i]},
                                 {"timing_ms", timed_ms.at(item.first).at(i)},
                                 {"observed_delta_ms", timed_ms.at(item.first).at(i) - item.second[i]}});
            overhead.push_back(
                {{"device_id", d.id},
                 {"case_id", item.first},
                 {"paired_replays", pairs},
                 {"interpretation", "includes_runtime_variation_and_observation_overhead_not_a_correction_factor"}});
        }
        for (auto & entry : c.samples) {
            const auto count = entry.second.size();
            auto duration = detail::sample_statistics(entry.second);
            duration.erase("sample_count");
            duration.erase("samples");
            out.costs.push_back({{"schema_version", "umm-profiler-cost/v3"},
                                 {"data_origin", "measured"},
                                 {"graph_id", entry.first},
                                 {"device_id", d.id},
                                 {"sample_count", count},
                                 {"state", "post_warmup_replay"},
                                 {"duration_ms", duration},
                                 {"timing_scope", d.transport == "rpc" ? "rpc_worker_synchronized_host_wall"
                                                                       : "synchronized_host_wall"},
                                 {"measurement_valid", true},
                                 {"decision_usable", false},
                                 {"memory_peak_bytes", nullptr},
                                 {"transfer", nullptr},
                                 {"overlap", nullptr},
                                 {"kernel_ms", nullptr}});
        }
    }
    out.timings = detail::annotate_timings(out.graphs, out.trace);
    // The legacy bundle files remain readable by existing viewers.
    out.candidates = json::array();
    out.candidate_costs = json::array();
    out.snapshots = json::array();
    out.manifest = {
        {"schema_version", "umm-profiler/v3"},
        {"data_origin", "measured"},
        {"status", "completed"},
        {"executed", true},
        {"decision_usable", false},
        {"request", request_json},
        {"identity",
         {{"model_manifest_sha256", hash(model_manifest)},
          {"model_components_sha256", components},
          {"profiler_binary_sha256",
           std::filesystem::exists("/proc/self/exe") ? json(file_hash("/proc/self/exe")) : json(nullptr)},
          {"devices", identities}}},
        {"workload",
         {{"case_count", cases.size()},
          {"cases_sha256", hash(raw_cases)},
          {"manifest", workload_manifest},
          {"cases", original_cases}}},
        {"measurement",
         {{"passes", passes},
          {"measured_replays", 2},
          {"observation_overhead", overhead},
          {"serial_devices", true},
          {"graph_synchronization", true},
          {"rpc_scheduler_completion", "reuse_successful_child_fences_without_eval_callback"},
          {"rpc_timing", "worker_batch_per_request"},
          {"rpc_profile_protocol", "6.1"},
          {"rpc_submission_scope", "client_submission_host_wall"},
          {"timing_scope", "synchronized_host_wall"},
          {"nested_events_are_not_additive", true},
          {"full_json_export_in_timing_pass", false},
          {"structural_hash_in_timing_pass", false},
          {"identity_resolution", "frozen_case_execution_contract"},
          {"contract_validation", "all_tensor_metadata_topology_and_devices_without_hash"}}},
        {"coverage",
         {{"workload_loaded", true},
          {"device_probed", true},
          {"cut_legality", false},
          {"isolated_ffn_replay", false},
          {"selected_boundary_roundtrip", false},
          {"operator_timing", std::any_of(out.hierarchy.begin(), out.hierarchy.end(),
                                          [](const auto & item) {
                                              return item.at("level") == "operator" && item.contains("spans") &&
                                                     std::any_of(item.at("spans").begin(), item.at("spans").end(),
                                                                 [](const auto & span) {
                                                                     return !span.at("elapsed_ms").is_null();
                                                                 });
                                          })},
          {"logical_graph_timing_annotations", true},
          {"remote_worker_metrics",
           std::any_of(devices.begin(), devices.end(), [](const auto & d) { return d.transport == "rpc"; })},
          {"phase_coverage", "understanding prefill/decode, U1 vision, coarse generation"},
          {"correctness", stable ? "within_device_output_repeatable" : "within_device_output_changed"}}},
        {"artifacts",
         {"graphs.jsonl", "trace.jsonl", "costs.jsonl", "requests.jsonl", "candidates.jsonl", "candidate-costs.jsonl",
          "snapshots.jsonl", "timings.jsonl", "hierarchy.jsonl", "report.md"}}};
    out.report =
        "# Graph profiler\n\n"
        "Runs the structure and paired baseline/timing passes before separate layer, block and optional operator passes. "
        "Baseline requests use the normal inference path with observation off.\n\n"
        "timings.jsonl groups completed backend graph timings by exact logical graph, case and submitted token/context sizes. "
        "Physical graph references retain the actual placement. Local synchronized host intervals and completed RPC worker intervals retain their timing scopes. "
        "Client submission and overlapping scheduler times remain in trace.jsonl and are excluded from timing annotations.\n\n"
        "hierarchy.jsonl contains CANN stream timings for understanding graphs. A cut crossing a measured range has unknown cost at that level. "
        "Boundary transfer, persistent state and peak memory costs are not inferred.\n";
    return out;
}
}
