#include "runtime-support.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <regex>
#include <sstream>
#include <thread>
#include <unistd.h>
#include <sys/resource.h>

namespace umm::runtime {
namespace {
using clock = std::chrono::steady_clock;
double now() { return std::chrono::duration<double>(clock::now().time_since_epoch()).count(); }
uint64_t rss_bytes() {
    std::ifstream input("/proc/self/statm");
    uint64_t size = 0, resident = 0;
    if (!(input >> size >> resident)) throw std::runtime_error("Cannot read process RSS");
    return resident * uint64_t(sysconf(_SC_PAGESIZE));
}
std::string command_output(const std::string & arguments) {
    const std::string command = "timeout 3s npu-smi info " + arguments + " 2>&1";
    FILE * file = popen(command.c_str(), "r");
    if (!file) throw std::runtime_error("Cannot start npu-smi");
    std::string result;
    char buffer[1024];
    while (fgets(buffer, sizeof(buffer), file)) result += buffer;
    const int status = pclose(file);
    if (status) throw std::runtime_error("npu-smi query failed: " + result.substr(0, 256));
    return result;
}
json difference(const json & first, const json & last) {
    if (first.is_null() || last.is_null()) return nullptr;
    json result = json::array();
    for (const auto & item : last) {
        auto row = item;
        auto before = std::find_if(first.begin(), first.end(), [&](const json & value) {
            return value.at("device") == item.at("device");
        });
        if (before == first.end()) continue;
        for (const auto * field : {"h2d_bytes", "d2h_bytes", "d2d_bytes", "h2d_calls", "d2h_calls", "d2d_calls"}) {
            row[field] = item.at(field).get<uint64_t>() - before->at(field).get<uint64_t>();
        }
        result.push_back(row);
    }
    return result;
}
}

json transfer_snapshot() {
    auto * reg = ggml_backend_reg_by_name("CANN");
    if (!reg) return nullptr;
    using query = bool (*)(int, uint64_t *, uint64_t *);
    auto get = reinterpret_cast<query>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_cann_get_transfer_stats_v1"));
    if (!get) return nullptr;
    json output = json::array();
    for (size_t i = 0; i < ggml_backend_reg_dev_count(reg); ++i) {
        uint64_t bytes[3]{}, calls[3]{};
        if (get(int(i), bytes, calls)) output.push_back({{"device", "CANN" + std::to_string(i)},
            {"h2d_bytes", bytes[0]}, {"d2h_bytes", bytes[1]}, {"d2d_bytes", bytes[2]},
            {"h2d_calls", calls[0]}, {"d2h_calls", calls[1]}, {"d2d_calls", calls[2]},
            {"scope", "explicit_runtime_copy_submissions"},
            {"d2d_peer_scope", ggml_backend_reg_dev_count(reg) == 1 ? "single_visible_device" : "unspecified_peers"}});
    }
    return output.empty() ? json(nullptr) : output;
}

struct system_monitor::impl {
    bool enabled, stopping = false, active = false;
    double interval, begin_time = 0;
    int card = -1, chip = -1;
    std::string mapping_error;
    mutable std::mutex mutex;
    std::condition_variable wake;
    std::thread thread;
    std::vector<json> window;
    json latest = nullptr, before = nullptr;
    uint64_t peak_rss = 0, peak_device = 0, sample_count = 0, errors = 0, device_samples = 0;
    std::ofstream stream;

    explicit impl(const options & args) : enabled(option(args, "--monitoring", "1") == "1"),
        interval(std::stod(option(args, "--monitor-interval", "2"))) {
        if (!std::isfinite(interval) || interval < 0.1 || interval > 60) throw std::invalid_argument("Invalid monitor interval");
        const char * configured = std::getenv("UMM_SYSTEM_SAMPLES");
        const auto path = option(args, "--system-samples", configured ? configured : "");
        if (!path.empty()) {
            stream.open(path, std::ios::app);
            if (!stream) throw std::invalid_argument("Cannot open system samples file");
        }
        if (!enabled) return;
        try {
            std::istringstream lines(command_output("-m"));
            std::vector<std::pair<int, int>> visible;
            std::string line;
            while (std::getline(lines, line)) {
                std::istringstream fields(line);
                int c, ch, logic;
                if (fields >> c >> ch >> logic && c >= 0 && ch >= 0 && logic >= 0) visible.emplace_back(c, ch);
            }
            if (visible.size() != 1) throw std::runtime_error("Device sampling requires one visible NPU");
            card = visible.front().first; chip = visible.front().second;
        } catch (const std::exception & error) { mapping_error = error.what(); }
        thread = std::thread([this] { loop(); });
    }
    ~impl() {
        { std::lock_guard<std::mutex> lock(mutex); stopping = true; }
        wake.notify_all();
        if (thread.joinable()) thread.join();
    }
    json sample() {
        json value = {{"monotonic_seconds", now()}, {"pid", getpid()}, {"device_process_memory_bytes", nullptr},
                      {"compute_utilization_ratio", nullptr}, {"vectorcore_utilization_ratio", nullptr},
                      {"other_device_processes", nullptr}, {"source", "proc+npu-smi"}, {"utilization_engine", "aicore"},
                      {"card", card}, {"chip", chip}};
        const auto start = clock::now();
        value["collection_begin_monotonic_seconds"] = now();
        try {
            value["host_rss_bytes"] = rss_bytes();
            struct rusage usage{};
            if (getrusage(RUSAGE_SELF, &usage) == 0) value["host_vm_hwm_bytes"] = uint64_t(usage.ru_maxrss) * 1024;
        }
        catch (const std::exception & error) { value["host_error"] = error.what(); }
        if (card >= 0) {
            try {
                const auto suffix = " -i " + std::to_string(card) + " -c " + std::to_string(chip);
                const auto raw = command_output("-t proc-mem" + suffix);
                int others = 0;
                bool found = false;
                std::regex verbose("Process id:\\s*(\\d+)\\s+Process name:\\s*(.*?)\\s+Process memory\\(MB\\):\\s*(\\d+)");
                auto record = [&](int pid, uint64_t mb) {
                    if (pid == getpid()) { value["device_process_memory_bytes"] = mb * 1024 * 1024; found = true; }
                    else ++others;
                };
                for (auto it = std::sregex_iterator(raw.begin(), raw.end(), verbose); it != std::sregex_iterator(); ++it) {
                    record(std::stoi((*it)[1]), std::stoull((*it)[3]));
                }
                if (!found && !others) {
                    std::istringstream lines(raw);
                    std::string line;
                    while (std::getline(lines, line)) {
                        std::replace(line.begin(), line.end(), '|', ' ');
                        std::istringstream fields(line);
                        int c, ch, pid; uint64_t mb; std::string name;
                        if (fields >> c >> ch >> pid >> name >> mb) record(pid, mb);
                    }
                }
                if (raw.find("No process") != std::string::npos) value["device_process_memory_bytes"] = 0;
                else if (!found) throw std::runtime_error("Cannot match NPU process PID");
                value["other_device_processes"] = others;
                const auto usages = command_output("-t usages" + suffix);
                std::smatch match;
                for (const auto & entry : {std::pair<const char *, const char *>{"Aicore", "compute_utilization_ratio"},
                                           {"Vectorcore", "vectorcore_utilization_ratio"}}) {
                    if (std::regex_search(usages, match, std::regex(std::string(entry.first) + " Usage Rate\\(%\\)\\s*:\\s*(\\d+)"))) {
                        value[entry.second] = std::stod(match[1]) / 100.0;
                    }
                }
            } catch (const std::exception & error) { value["device_error"] = error.what(); }
        } else value["device_error"] = mapping_error;
        value["collection_seconds"] = std::chrono::duration<double>(clock::now() - start).count();
        value["monotonic_seconds"] = now();
        return value;
    }
    void loop() {
        std::unique_lock<std::mutex> lock(mutex);
        while (!stopping) {
            const auto next = clock::now() + std::chrono::duration<double>(interval);
            lock.unlock();
            auto value = sample();
            lock.lock();
            latest = value;
            ++sample_count;
            if (value.contains("host_error") || value.contains("device_error")) ++errors;
            if (value.contains("host_rss_bytes")) peak_rss = std::max(peak_rss, value["host_rss_bytes"].get<uint64_t>());
            if (value["device_process_memory_bytes"].is_number()) { peak_device = std::max(peak_device, value["device_process_memory_bytes"].get<uint64_t>()); ++device_samples; }
            if (active && value["collection_begin_monotonic_seconds"].get<double>() >= begin_time) window.push_back(value);
            if (stream) { stream << value.dump() << '\n'; stream.flush(); }
            wake.wait_until(lock, next, [&] { return stopping; });
        }
    }
};

system_monitor::system_monitor(const options & args) : state(new impl(args)) {}
system_monitor::~system_monitor() = default;
void system_monitor::begin() {
    auto before = transfer_snapshot();
    std::lock_guard<std::mutex> lock(state->mutex);
    state->window.clear(); state->before = std::move(before); state->active = true; state->begin_time = now();
    if (state->enabled) {
        try { state->window.push_back({{"monotonic_seconds", now()}, {"host_rss_bytes", rss_bytes()}}); }
        catch (...) {}
    }
}
json system_monitor::end() {
    const auto after = transfer_snapshot();
    std::lock_guard<std::mutex> lock(state->mutex);
    const double end_time = now(), duration = end_time - state->begin_time;
    state->active = false;
    uint64_t host = 0, device = 0;
    bool have_host = false, have_device = false, shared = false, have_shared = false;
    double area = 0, coverage = 0, maximum = 0;
    double previous_t = 0, previous_v = 0;
    size_t count = 0, valid = 0;
    try { host = rss_bytes(); have_host = state->enabled; } catch (...) {}
    for (const auto & sample : state->window) {
        if (sample.contains("host_rss_bytes")) { host = std::max(host, sample["host_rss_bytes"].get<uint64_t>()); have_host = true; }
        if (sample.contains("other_device_processes") && sample["other_device_processes"].is_number()) { shared |= sample["other_device_processes"].get<int>() > 0; have_shared = true; }
        if (!sample.contains("source")) continue;
        ++count;
        if (sample["device_process_memory_bytes"].is_number()) {
            device = std::max(device, sample["device_process_memory_bytes"].get<uint64_t>()); have_device = true;
        }
        const double t = sample["monotonic_seconds"];
        if (sample["compute_utilization_ratio"].is_number()) {
            const double v = sample["compute_utilization_ratio"];
            maximum = std::max(maximum, v); ++valid;
            if (previous_t && t - previous_t <= std::max(5.0, 3 * state->interval)) {
                const double dt = t - previous_t;
                coverage += dt; area += previous_v * dt;
            }
            previous_t = t; previous_v = v;
        } else previous_t = 0;
    }
    json memory = {{"host_rss_sampled_peak_bytes", have_host ? json(host) : json(nullptr)},
                   {"device_process_sampled_peak_bytes", have_device ? json(device) : json(nullptr)}};
    return {{"status", !state->enabled ? "disabled" : count == 0 ? "unavailable" : valid == count && coverage >= duration * 0.8 ? "ok" : "partial"},
            {"memory", memory}, {"sample_count", count}, {"valid_utilization_samples", valid},
            {"utilization_scope", "device"}, {"utilization_engine", "aicore"}, {"shared_device", have_shared ? json(shared) : json(nullptr)},
            {"utilization_mean_ratio", coverage > 0 ? json(area / coverage) : json(nullptr)},
            {"utilization_max_ratio", valid ? json(maximum) : json(nullptr)},
            {"utilization_coverage_ratio", duration > 0 ? coverage / duration : 0},
            {"utilization_covered_seconds", coverage}, {"window_seconds", duration},
            {"transfer", difference(state->before, after)}};
}
json system_monitor::snapshot() const {
    std::lock_guard<std::mutex> lock(state->mutex);
    return {{"enabled", state->enabled}, {"source", "proc+npu-smi"}, {"utilization_engine", "aicore"}, {"interval_seconds", state->interval},
            {"card", state->card}, {"chip", state->chip}, {"mapping", "single_visible_npu"}, {"latest", state->latest},
            {"samples", state->sample_count}, {"errors", state->errors},
            {"host_rss_sampled_peak_bytes", state->sample_count ? json(state->peak_rss) : json(nullptr)},
            {"device_process_sampled_peak_bytes", state->device_samples ? json(state->peak_device) : json(nullptr)}};
}
}
