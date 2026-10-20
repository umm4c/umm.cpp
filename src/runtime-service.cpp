#include "runtime-support.h"
#include "model-package.h"
extern "C" {
#include "sha256.h"
}
#include <array>
#include <iomanip>
#include <atomic>
#include <filesystem>
#include <cmath>
#include <unistd.h>
#include <condition_variable>
#include <fstream>
#include <mutex>
#include <sstream>
#include <thread>

namespace umm::runtime {
namespace {
using clock = std::chrono::steady_clock;
const std::vector<double> buckets{0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300, 600, 1200, 3600, 7200};
std::string escape(const std::string & value) {
    std::string output;
    for (char ch : value) {
        if (ch == '\\' || ch == '"') output += '\\';
        if (ch == '\n') output += "\\n"; else output += ch;
    }
    return output;
}
std::string label(const std::string & key, const std::string & value) { return key + "=\"" + escape(value) + "\""; }
std::string file_hash(const std::string & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot hash model component");
    sha256_t state; sha256_init(&state);
    std::array<unsigned char, 1024 * 1024> buffer;
    while (input) {
        input.read(reinterpret_cast<char *>(buffer.data()), buffer.size());
        sha256_update(&state, buffer.data(), size_t(input.gcount()));
    }
    if (!input.eof()) throw std::runtime_error("Model component hash read failed");
    unsigned char bytes[32]; sha256_final(&state, bytes);
    std::ostringstream result;
    for (auto ch : bytes) result << std::hex << std::setw(2) << std::setfill('0') << int(ch);
    return result.str();
}
struct histogram {
    std::vector<uint64_t> counts = std::vector<uint64_t>(buckets.size(), 0);
    uint64_t count = 0;
    double sum = 0;
    void observe(double value) {
        if (!std::isfinite(value) || value < 0) return;
        ++count; sum += value;
        for (size_t i = 0; i < buckets.size(); ++i) if (value <= buckets[i]) ++counts[i];
    }
};
struct registry {
    std::map<std::string, double> counters;
    std::map<std::string, std::map<std::string, histogram>> histograms;
    void count(const std::string & name, const std::string & labels, double value = 1) {
        if (value >= 0) counters[name + (labels.empty() ? "" : "{" + labels + "}")] += value;
    }
    void observe(const std::string & name, const std::string & labels, double value) {
        histograms[name][labels].observe(value);
    }
    std::string render() const {
        std::ostringstream stream;
        stream << std::setprecision(17);
        for (const auto & family : histograms) {
            stream << "# TYPE " << family.first << " histogram\n";
            for (const auto & entry : family.second) {
                const auto & h = entry.second;
                const auto prefix = entry.first.empty() ? "" : entry.first + ",";
                for (size_t i = 0; i < buckets.size(); ++i) stream << family.first << "_bucket{" << prefix << "le=\"" << buckets[i] << "\"} " << h.counts[i] << '\n';
                stream << family.first << "_bucket{" << prefix << "le=\"+Inf\"} " << h.count << '\n';
                const auto labels = entry.first.empty() ? "" : "{" + entry.first + "}";
                stream << family.first << "_sum" << labels << " " << h.sum << '\n';
                stream << family.first << "_count" << labels << " " << h.count << '\n';
            }
        }
        std::string previous;
        for (const auto & item : counters) {
            const auto name = item.first.substr(0, item.first.find('{'));
            if (name != previous) { stream << "# TYPE " << name << " counter\n"; previous = name; }
            stream << item.first << ' ' << item.second << '\n';
        }
        return stream.str();
    }
};
}

struct service::impl {
    options args;
    system_monitor monitor;
    json ready;
    mutable std::mutex mutex;
    mutable std::mutex stats_mutex;
    std::condition_variable wake;
    bool stopping = false, busy = false;
    struct job { json request; bool http; std::promise<json> result; };
    std::optional<job> pending;
    std::thread worker;
    std::promise<void> initialized;
    registry stats;
    std::ofstream log;
    uint64_t sequence = 0;

    explicit impl(options input) : args(std::move(input)), monitor(args) {
        const char * log_env = std::getenv("UMM_REQUEST_LOG");
        const auto path = option(args, "--request-log", log_env ? log_env : "");
        if (!path.empty()) { log.open(path, std::ios::app); if (!log) throw std::invalid_argument("Cannot open request log"); }
        auto future = initialized.get_future();
        worker = std::thread([this] { run(); });
        try { future.get(); }
        catch (...) { if (worker.joinable()) worker.join(); throw; }
    }
    ~impl() {
        { std::lock_guard<std::mutex> lock(mutex); stopping = true; }
        wake.notify_one();
        if (worker.joinable()) worker.join();
    }
    void record(const json & value, const std::string & task, bool http) {
        std::lock_guard<std::mutex> lock(stats_mutex);
        const auto status = value.value("status", "error");
        const auto labels = label("task", task) + "," + label("status", status);
        stats.count("umm_requests_total", labels);
        if (value.contains("timings_ms")) {
            for (auto it = value["timings_ms"].begin(); it != value["timings_ms"].end(); ++it) {
                if (!it.value().is_number()) continue;
                const double seconds = it.value().get<double>() / 1000;
                if (it.key() == "end_to_end" && !http) {
                    stats.observe("umm_e2e_request_latency_seconds", labels + "," + label("transport", http ? "http" : "jsonl"), seconds);
                } else if (it.key() == "end_to_end") continue;
                else if (it.key() == "engine") stats.observe("umm_engine_request_latency_seconds", labels, seconds);
                else if (it.key() == "ttft") stats.observe("umm_time_to_first_token_seconds", label("task", task), seconds);
                else stats.observe("umm_request_stage_seconds", label("task", task) + "," + label("stage", it.key()), seconds);
            }
            const auto steps = value["timings_ms"].value("diffusion_step", json(nullptr));
            if (steps.is_array()) for (const auto & step : steps) stats.observe("umm_diffusion_step_seconds", "", step.get<double>() / 1000);
        }
        if (value.contains("token_counts")) {
            for (const auto & field : {std::pair<const char *, const char *>{"input", "umm_prompt_tokens_total"}, {"output", "umm_generation_tokens_total"}}) {
                const auto count = value["token_counts"].value(field.first, json(nullptr));
                if (count.is_number()) stats.count(field.second, label("task", task), count.get<double>());
            }
        }
        if (status == "ok" && value.contains("output") && value["output"].value("type", "") == "image") stats.count("umm_generated_images_total", "");
        if (log) {
            auto metadata = value;
            if (metadata.contains("output")) metadata["output"].erase("b64_json");
            log << metadata.dump() << '\n'; log.flush();
        }
    }
    void run() {
        try {
            const auto start = clock::now();
            const auto model = option(args, "--model");
            if (!std::filesystem::is_directory(model)) throw std::invalid_argument("--model must name a model package directory");
            session engine(model, "", option(args, "--understanding-backend"), option(args, "--generation-backend"),
                option(args, "--generation-max-vram"), option(args, "--vision-backend"),
                std::stoi(option(args, "--n-ctx", "0")), std::stoi(option(args, "--n-batch", "0")),
                std::stoi(option(args, "--n-ubatch", "0")), option(args, "--u1-prefill-mode", "optimized"),
                option(args, "--u1-v-cache-layout", "default"));
            ready = {{"type", "ready"}, {"protocol", "umm-session/v1"},
                     {"cold_start_ms", std::chrono::duration<double, std::milli>(clock::now()-start).count()},
                     {"model", model}, {"options", args}, {"runtime_identity", runtime_identity(model, args)},
                     {"monitoring", monitor.snapshot()}, {"startup_transfer", transfer_snapshot()}};
            ready["model_manifest_sha256"] = file_hash((std::filesystem::path(model) / "model.json").string());
            if (const char * hash = std::getenv("UMM_MODEL_MANIFEST_SHA256")) {
                if (ready["model_manifest_sha256"] != hash) throw std::runtime_error("Model manifest hash mismatch");
            }
            if (option(args, "--hash-model", "0") == "1") {
                const auto hash_start = clock::now();
                ready["model_component_sha256"] = json::object();
                for (const auto & entry : resolve_model(model).components) ready["model_component_sha256"][entry.first] = file_hash(entry.second);
                ready["startup_hash_ms"] = std::chrono::duration<double, std::milli>(clock::now()-hash_start).count();
            }
            initialized.set_value();
            std::unique_lock<std::mutex> lock(mutex);
            while (true) {
                wake.wait(lock, [&] { return stopping || pending.has_value(); });
                if (!pending) break;
                auto current = std::move(*pending); pending.reset();
                lock.unlock();
                try {
                    monitor.begin();
                    auto value = run_case(engine, current.request, args, current.http);
                    value["system_metrics"] = monitor.end();
                    value["request_id"] = current.http ? current.request.at("case_id") : json("umm-" + std::to_string(getpid()) + "-" + std::to_string(++sequence));
                    if (current.request.contains("client_case_id")) value["client_case_id"] = current.request["client_case_id"];
                    const auto task = current.request.value("task_class", "unknown");
                    const auto known = task == "text_generation" || task == "mixed_understanding" || task == "multimodal_prefill" || task == "image_generation";
                    record(value, known ? task : "unknown", current.http);
                    lock.lock(); busy = false; lock.unlock();
                    current.result.set_value(std::move(value));
                } catch (...) {
                    lock.lock(); busy = false; lock.unlock();
                    current.result.set_exception(std::current_exception());
                }
                lock.lock();
                if (stopping && !pending) break;
            }
        } catch (...) { initialized.set_exception(std::current_exception()); }
    }
};

service::service(options args) {
    const auto enabled = option(args, "--monitoring", "1");
    if (enabled != "0" && enabled != "1") throw std::invalid_argument("monitoring must be 0 or 1");
    setenv("GGML_CANN_TRANSFER_METRICS", enabled.c_str(), 1);
    state.reset(new impl(std::move(args)));
}
service::~service() = default;
const json & service::info() const { return state->ready; }
std::optional<std::future<json>> service::submit(json request, bool http) {
    std::lock_guard<std::mutex> lock(state->mutex);
    if (state->busy || state->stopping) return std::nullopt;
    state->pending.emplace(impl::job{std::move(request), http, {}});
    auto future = state->pending->result.get_future();
    state->busy = true; state->wake.notify_one();
    return future;
}
void service::reject(const std::string & reason) {
    std::lock_guard<std::mutex> lock(state->stats_mutex);
    state->stats.count("umm_request_rejections_total", label("reason", reason));
}
void service::http_completed(const std::string & id, const std::string & task, const std::string & status, double ms) {
    std::lock_guard<std::mutex> lock(state->stats_mutex);
    state->stats.observe("umm_e2e_request_latency_seconds", label("task", task) + "," + label("status", status) + ",transport=\"http\"", ms / 1000);
    if (state->log) { state->log << json({{"type", "http_response"}, {"request_id", id}, {"http_application_ms", ms}}).dump() << '\n'; state->log.flush(); }
}
std::string service::metrics() const {
    std::ostringstream output;
    output << std::setprecision(17);
    { std::lock_guard<std::mutex> lock(state->stats_mutex); output << state->stats.render(); }
    { std::lock_guard<std::mutex> lock(state->mutex); output << "# TYPE umm_num_requests_running gauge\numm_num_requests_running " << int(state->busy) << '\n'; }
    auto sample = state->monitor.snapshot();
    output << "# TYPE umm_system_sample_errors_total counter\numm_system_sample_errors_total{source=\"system\"} " << sample["errors"] << '\n';
    const auto latest = sample["latest"];
    const double time = std::chrono::duration<double>(clock::now().time_since_epoch()).count();
    bool fresh = !latest.is_null() && time - latest["monotonic_seconds"].get<double>() <= std::max(5.0, 3 * sample["interval_seconds"].get<double>());
    output << "# TYPE umm_system_sample_available gauge\numm_system_sample_available " << int(fresh) << '\n';
    output << "# TYPE umm_system_field_available gauge\n";
    for (const auto * field : {"host_rss_bytes", "device_process_memory_bytes", "compute_utilization_ratio"}) {
        output << "umm_system_field_available{" << label("field", field) << "} "
               << int(fresh && latest.contains(field) && latest[field].is_number()) << '\n';
    }
    if (!latest.is_null()) output << "# TYPE umm_system_sample_age_seconds gauge\numm_system_sample_age_seconds " << time-latest["monotonic_seconds"].get<double>() << '\n';
    if (fresh) {
        for (const auto & field : {std::pair<const char *, const char *>{"host_rss_bytes", "process_resident_memory_bytes"},
            {"device_process_memory_bytes", "umm_device_process_memory_bytes"}}) {
            if (latest.contains(field.first) && latest[field.first].is_number()) output << "# TYPE " << field.second << " gauge\n" << field.second << " " << latest[field.first] << '\n';
        }
        output << "# TYPE umm_device_utilization_ratio gauge\n";
        for (const auto & field : {std::pair<const char *, const char *>{"compute_utilization_ratio", "aicore"}, {"vectorcore_utilization_ratio", "vectorcore"}}) {
            if (latest[field.first].is_number()) output << "umm_device_utilization_ratio{engine=\"" << field.second << "\"} " << latest[field.first] << '\n';
        }
    }
    auto transfers = transfer_snapshot();
    if (!transfers.is_null()) {
        output << "# TYPE umm_transfer_bytes_total counter\n# TYPE umm_transfer_calls_total counter\n";
        for (const auto & row : transfers) for (const auto * direction : {"h2d", "d2h", "d2d"}) {
            const auto labels = "{" + label("device", row["device"]) + "," + label("direction", direction) + "}";
            output << "umm_transfer_bytes_total" << labels << ' ' << row[std::string(direction)+"_bytes"] << '\n';
            output << "umm_transfer_calls_total" << labels << ' ' << row[std::string(direction)+"_calls"] << '\n';
        }
    }
    return output.str();
}
}
