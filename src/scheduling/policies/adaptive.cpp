#include "adaptive.h"
#include "profiler-index.h"
#include "scheduling/backend-devices.h"
#include "scheduling/backend-environment.h"
#include "scheduling/weights/weight-replicas.h"
#include "inference/profile-context.h"
#include "build-identity.h"

extern "C" {
#include "sha256.h"
}

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace umm {
namespace {

using namespace runtime_placement;
using json = nlohmann::json;

struct calibration {
    double switch_ms = 0;
    double margin_ms = 0;
    std::array<double, 2> submission_ms{};
    std::array<double, 2> transfer_base_ms{};
    std::array<double, 2> transfer_bytes_per_ms{};
    std::array<size_t, 2> workspace_bytes{};
    std::array<size_t, 2> extra_capacity_bytes{};
    std::string endpoint, local_device, profile_capture;

    double transfer_cost(device destination, size_t bytes) const {
        return transfer_base_ms[destination] + bytes / transfer_bytes_per_ms[destination];
    }
};

std::string file_sha256(const std::filesystem::path & path) {
    std::ifstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error("Cannot open planner identity file " + path.string());
    sha256_t hash;
    sha256_init(&hash);
    std::vector<unsigned char> buffer(1024 * 1024);
    while (file.read(reinterpret_cast<char *>(buffer.data()), buffer.size()) || file.gcount()) {
        sha256_update(&hash, buffer.data(), size_t(file.gcount()));
    }
    if (!file.eof())
        throw std::runtime_error("Cannot read planner identity file " + path.string());
    unsigned char digest[32];
    sha256_final(&hash, digest);
    std::ostringstream result;
    for (auto byte : digest)
        result << std::hex << std::setw(2) << std::setfill('0') << int(byte);
    return result.str();
}

calibration load_calibration(const std::string & path) {
    json file;
    std::ifstream input(path);
    input >> file;
    if (file.at("schema_version") != "umm-runtime-placement-calibration/v1" || file.at("data_origin") != "measured") {
        throw std::invalid_argument("Adaptive placement requires measured v1 path calibration");
    }
    calibration result;
    auto cost = [](double v) {
        if (!std::isfinite(v) || v < 0)
            throw std::invalid_argument("Invalid calibration time");
        return v;
    };
    result.switch_ms = cost(file.at("switch_ms").get<double>());
    result.margin_ms = cost(file.at("margin_ms").get<double>());
    result.endpoint = file.at("endpoint");
    result.local_device = file.at("local_device");
    result.profile_capture = file.at("profile_manifest_sha256");
    for (device d : {local, remote}) {
        const auto & device = file.at(d == local ? "local" : "remote");
        result.submission_ms[d] = cost(device.at("submission_ms").get<double>());
        result.transfer_base_ms[d] = cost(device.at("transfer_base_ms").get<double>());
        result.transfer_bytes_per_ms[d] = cost(device.at("transfer_bytes_per_ms").get<double>());
        result.workspace_bytes[d] = device.at("workspace_bytes").get<size_t>();
        result.extra_capacity_bytes[d] = device.at("extra_capacity_bytes").get<size_t>();
        if (!result.transfer_bytes_per_ms[d] || !result.workspace_bytes[d] || !result.extra_capacity_bytes[d]) {
            throw std::invalid_argument("Calibration requires positive measured throughput, workspace and capacity");
        }
    }
    return result;
}

ggml_tensor * storage(ggml_tensor * tensor) {
    while (tensor->view_src)
        tensor = tensor->view_src;
    return tensor;
}

struct graph_problem {
    problem model;
    std::vector<ggml_tensor *> tensors;
    std::vector<ggml_tensor *> value_tensors;

    graph_problem(ggml_cgraph * graph, std::array<ggml_backend_t, 2> backend, const measured_costs & samples,
                  const std::optional<calibration> & path, bool active, size_t remote_free_bytes,
                  clock::time_point deadline) {
        const int count = ggml_graph_n_nodes(graph);
        model.units.resize(count);
        model.current.assign(count, local);
        model.timings = samples.ranges;
        model.together = samples.fused_groups;
        model.switch_ms = path ? path->switch_ms : 0;
        model.margin_ms = path ? path->margin_ms : 0;
        model.capacity.fill(std::numeric_limits<size_t>::max());
        if (path)
            model.submission_ms = path->submission_ms;
        if (active) {
            size_t local_free_bytes = 0, local_total_bytes = 0;
            ggml_backend_dev_memory(ggml_backend_get_device(backend[local]), &local_free_bytes, &local_total_bytes);
            const std::array<size_t, 2> free_bytes = {local_free_bytes, remote_free_bytes};
            for (device d : {local, remote}) {
                const auto budget = std::min(free_bytes[d], path->extra_capacity_bytes[d]);
                model.capacity[d] = budget > path->workspace_bytes[d] ? budget - path->workspace_bytes[d] : 0;
            }
        }
        std::unordered_map<ggml_tensor *, int> nodes;
        std::unordered_map<ggml_tensor *, size_t> values;
        std::unordered_set<ggml_tensor *> mutable_roots;
        tensors.reserve(count);
        for (int i = 0; i < count; ++i) {
            auto * tensor = ggml_graph_node(graph, i);
            tensors.push_back(tensor);
            nodes.emplace(tensor, i);
            if (tensor->view_src && !is_metadata_node(tensor))
                mutable_roots.insert(storage(tensor));
        }
        auto value_for = [&](ggml_tensor * tensor) -> value & {
            auto found = values.find(tensor);
            if (found != values.end())
                return model.values[found->second];
            auto * root = storage(tensor);
            const auto producer = nodes.find(root);
            const auto owner = producer == nodes.end() ? -1 : producer->second;
            value entry;
            entry.producer = owner;
            entry.initial = owner < 0 ? bit_for_local(root) : 0;
            entry.persistent = owner < 0 && root->buffer;
            entry.bytes = ggml_nbytes(tensor);
            for (device d : {local, remote}) {
                entry.transfer_ms[d] = path ? path->transfer_cost(d, entry.bytes) : 0;
            }
            values.emplace(tensor, model.values.size());
            model.values.push_back(std::move(entry));
            value_tensors.push_back(tensor);
            return model.values.back();
        };
        for (int i = 0; i < count; ++i) {
            if (clock::now() >= deadline)
                throw std::runtime_error("deadline during GGML graph analysis");
            auto * tensor = tensors[i];
            auto * root = storage(tensor);
            auto & unit = model.units[i];
            const auto owner = nodes.find(root);
            const bool metadata = is_metadata_node(tensor);
            unit.executes = !metadata;
            if (tensor->buffer || (tensor->view_src && !metadata) || mutable_roots.count(root) ||
                (tensor->flags & (GGML_TENSOR_FLAG_INPUT | GGML_TENSOR_FLAG_OUTPUT)))
                unit.allowed = 1;
            if (metadata && tensor->view_src) {
                if (owner != nodes.end() && owner->second < i)
                    model.equal.push_back({size_t(owner->second), size_t(i)});
                else
                    unit.allowed = 1;
            }
            // Pin consumers of allocations owned by context/KV. GGML's current
            // copy path can also duplicate strided aliases and their backing data.
            for (auto * src : tensor->src) {
                if (!src || metadata)
                    continue;
                auto * base = storage(src);
                if (mutable_roots.count(base) || (base->buffer && bit_for_local(base) == 1) ||
                    (base->flags & GGML_TENSOR_FLAG_INPUT))
                    unit.allowed = 1;
                auto & v = value_for(src);
                if (v.producer < i && (v.readers.empty() || v.readers.back() != size_t(i))) {
                    v.readers.push_back(i);
                }
            }
            if (!metadata && !tensor->view_src) {
                auto & v = value_for(tensor);
                if (tensor->flags & GGML_TENSOR_FLAG_OUTPUT)
                    v.terminal = 1;
            }
        }
    }

    static unsigned bit_for_local(const ggml_tensor * t) {
        return t->buffer && ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS ? 3 : 1;
    }
};

struct adaptive_schedule {
    algorithm method;
    std::string method_name;
    std::shared_ptr<const profiler_index> profile;
    std::optional<calibration> path;
    bool activate;
    double budget_ms;
    std::ofstream log;
    std::array<ggml_backend_t, 2> devices{};
    std::unique_ptr<weight_replicas> replicas;
    size_t remote_free_bytes = 0;
    std::vector<node_placement> applied;
    std::unordered_map<std::string, std::string> physical_graphs;
    std::string source_graph;
    std::vector<ggml_tensor *> source_nodes;
    struct source_reference {
        ggml_tensor * node;
        int slot;
        ggml_tensor * source;
        device target;
    };
    std::vector<source_reference> cross_sources;
    std::array<size_t, 2> expected_bytes{}, reserved_buffers{};
    size_t expected_messages = 0;
    json record;
    clock::time_point allocation_start;
    bool disabled = false;
    std::string phase_name;
    int64_t graph_tokens = -1, graph_cached = -1;

    bool replan(const char * phase, int64_t tokens, int64_t cached) {
        if (!activate || disabled || graph_tokens < 0)
            return false;
        if (graph_tokens == tokens && graph_cached == cached && phase_name == phase)
            return false;
        return !applied.empty() ||
               (phase_name == phase && graph_tokens == tokens && profile->has_submission(phase, tokens, cached));
    }

    void remember_physical_graph(ggml_cgraph * graph) {
        if (source_graph.empty() || source_nodes.size() != size_t(ggml_graph_n_nodes(graph)))
            return;
        for (size_t i = 0; i < source_nodes.size(); ++i) {
            if (source_nodes[i] != ggml_graph_node(graph, int(i)))
                return;
        }
        // A successfully allocated local graph retains the same node positions,
        // including view producers. The scheduler may rewrite operand edges
        // to materialized copies, as it did in the captured device graph.
        auto physical = profiler_index::signature(graph);
        if (const auto * timings = profile->find(physical, phase_name, graph_tokens, graph_cached)) {
            physical_graphs.emplace(source_graph, std::move(physical));
            record["logical_graph_id"] = timings->logical_id;
            record["status"] = "local allocation matched the profiler";
        }
    }

    void verify(ggml_backend_sched_t sched, ggml_cgraph * graph) {
        if (applied.empty()) {
            for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
                auto * node = ggml_graph_node(graph, i);
                if (!is_metadata_node(node) && ggml_backend_sched_get_tensor_backend(sched, node) != devices[local]) {
                    throw std::runtime_error(
                        "Adaptive fallback did not place all computation on the local accelerator");
                }
            }
            if (!record.is_null()) {
                record["actual_placement"] = "local";
                remember_physical_graph(graph);
            }
        } else {
            verify_placements(sched, applied);
            size_t weight_bytes = 0;
            json splits = json::array();
            for (int i = 0; i < ggml_backend_sched_get_n_splits(sched); ++i) {
                ggml_backend_sched_split_info info{};
                if (!ggml_backend_sched_get_split_info(sched, i, &info)) {
                    throw std::runtime_error("Cannot inspect adaptive scheduler split");
                }
                weight_bytes += info.weight_input_bytes;
                splits.push_back({{"begin", info.node_start},
                                  {"end", info.node_end},
                                  {"device", ggml_backend_dev_name(ggml_backend_get_device(info.backend))}});
            }
            record["splits"] = std::move(splits);
            record["weight_input_bytes"] = weight_bytes;
            if (weight_bytes)
                throw std::runtime_error("Adaptive placement has scheduled weight copies");
            std::unordered_set<ggml_tensor *> copied;
            std::array<size_t, 2> actual_bytes{}, buffers{};
            for (const auto & read : cross_sources) {
                auto * copy = read.node->src[read.slot];
                if (!copy || copy == read.source || !copy->buffer ||
                    !ggml_backend_supports_buft(devices[read.target], ggml_backend_buffer_get_type(copy->buffer)) ||
                    copy->type != read.source->type || !ggml_are_same_shape(read.source, copy) ||
                    !ggml_are_same_stride(read.source, copy)) {
                    throw std::runtime_error("A cross-device tensor handoff was not allocated");
                }
                if (copied.insert(copy).second)
                    actual_bytes[read.target] += ggml_nbytes(copy);
            }
            for (device d : {local, remote})
                buffers[d] = ggml_backend_sched_get_buffer_size(sched, devices[d]);
            record["scheduled_transfer_bytes"] = actual_bytes;
            record["transfer_scope"] = "matched_scheduler_boundary_inputs";
            record["scheduled_transfer_messages"] = copied.size();
            record["allocation_bytes"] = buffers;
            std::array<size_t, 2> additional_buffers{};
            for (device d : {local, remote}) {
                additional_buffers[d] = buffers[d] > reserved_buffers[d] ? buffers[d] - reserved_buffers[d] : 0;
                if (additional_buffers[d] > path->extra_capacity_bytes[d]) {
                    record["additional_allocation_bytes"] = additional_buffers;
                    throw std::runtime_error("Scheduler allocation grew beyond the calibrated budget");
                }
            }
            record["additional_allocation_bytes"] = additional_buffers;
            if (actual_bytes != expected_bytes || copied.size() != expected_messages) {
                throw std::runtime_error("Scheduler boundary copies differ from the evaluated plan");
            }
            record["status"] = "verified";
            record["actual_placement"] = "mixed";
        }
        if (log.is_open() && !record.is_null()) {
            record["allocation_and_planning_ms"] =
                std::chrono::duration<double, std::milli>(clock::now() - allocation_start).count();
            log << record.dump() << '\n';
            log.flush();
        }
    }

    void place(ggml_backend_sched_t sched, ggml_cgraph * graph, bool allocated) {
        if (allocated) {
            try {
                verify(sched, graph);
            } catch (const std::exception & error) {
                record["verification_error"] = error.what();
                throw;
            }
            return;
        }
        applied.clear();
        cross_sources.clear();
        source_graph.clear();
        source_nodes.clear();
        expected_bytes.fill(0);
        expected_messages = 0;
        record = nullptr;
        const auto start = allocation_start = clock::now();
        const auto deadline =
            start + std::chrono::duration_cast<clock::duration>(std::chrono::duration<double, std::milli>(budget_ms));
        const auto & phase = profiling::current_phase;
        if (disabled || phase.tokens < 0 || phase.cached_tokens < 0)
            return;
        phase_name = phase.phase;
        graph_tokens = phase.tokens;
        graph_cached = phase.cached_tokens;
        record = {{"phase", phase.phase},
                  {"tokens", phase.tokens},
                  {"cached_tokens", phase.cached_tokens},
                  {"rebuild_ms", phase.rebuild_ms},
                  {"algorithm", method_name}};
        struct time_record {
            json & record;
            clock::time_point start;
            ~time_record() {
                record["planning_ms"] = std::chrono::duration<double, std::milli>(clock::now() - start).count();
            }
        } timing{record, start};
        try {
            if (!profile->has_submission(phase.phase, phase.tokens, phase.cached_tokens)) {
                record["status"] = "missing submission measurements";
                return;
            }
            const std::string structural = profiler_index::signature(graph);
            const auto physical = physical_graphs.find(structural);
            const auto & lookup = physical == physical_graphs.end() ? structural : physical->second;
            const auto * model = profile->find(lookup, phase.phase, phase.tokens, phase.cached_tokens);
            if (!model) {
                record["status"] = "missing graph or submission match";
                if (physical == physical_graphs.end()) {
                    source_graph = structural;
                    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i)
                        source_nodes.push_back(ggml_graph_node(graph, i));
                }
                return;
            }
            record["logical_graph_id"] = model->logical_id;
            record["mode"] = activate ? "apply" : "observe";
            record["cost_basis"] = path ? "calibrated_serial_estimate" : "compute_only_upper_bound";
            record["submission_scope"] = model->submission_scope;
            record["measured_submission_ms"] = model->submission_ms;
            graph_problem input(graph, devices, *model, path, activate, remote_free_bytes, deadline);
            budget limits;
            limits.deadline = start + (deadline - start) * 3 / 4;
            limits.actions = method == algorithm::beam_search ? 17 * input.model.units.size() : 4096;
            const auto plan = choose(input.model, method, limits);
            record.update({{"status", plan.reason},
                           {"evaluated", plan.evaluated},
                           {"candidate_ms", plan.cost.total_ms},
                           {"savings_ms", plan.savings_ms},
                           {"peak_bytes", plan.cost.peak},
                           {"profile_ranges", model->ranges.size()},
                           {"coverage_ranges", plan.cost.coverage.size()},
                           {"search_ms", plan.planning_ms}});
            std::vector<size_t> remote_units;
            for (size_t i = 0; i < plan.placement.size(); ++i) {
                if (plan.placement[i] == remote)
                    remote_units.push_back(i);
            }
            record["remote_units"] = remote_units;
            json handoffs = json::array();
            for (const auto & handoff : plan.cost.handoffs) {
                handoffs.push_back({{"before_unit", handoff.before_unit},
                                    {"source", handoff.source},
                                    {"destination", handoff.destination},
                                    {"value", handoff.value_index},
                                    {"bytes", input.model.values[handoff.value_index].bytes}});
                expected_bytes[handoff.destination] += input.model.values[handoff.value_index].bytes;
            }
            expected_messages = plan.cost.handoffs.size();
            record["expected_transfer_bytes"] = expected_bytes;
            record["expected_transfer_messages"] = expected_messages;
            record["handoffs"] = std::move(handoffs);
            if (activate && phase.single_ubatch && !remote_units.empty() && clock::now() < deadline) {
                for (size_t i = 0; i < input.tensors.size(); ++i) {
                    auto * node = input.tensors[i];
                    if (is_metadata_node(node))
                        continue;
                    applied.push_back({node, devices[plan.placement[i]]});
                }
                for (auto & placement : applied) {
                    if (clock::now() >= deadline || !ggml_backend_supports_op(placement.backend, placement.node)) {
                        record["status"] = "device support verification incomplete";
                        applied.clear();
                        return;
                    }
                }
                for (const auto & handoff : plan.cost.handoffs) {
                    auto * source = input.value_tensors[handoff.value_index];
                    for (const auto reader : input.model.values[handoff.value_index].readers) {
                        if (plan.placement[reader] != handoff.destination)
                            continue;
                        auto * node = input.tensors[reader];
                        for (int slot = 0; slot < GGML_MAX_SRC; ++slot) {
                            if (node->src[slot] == source) {
                                cross_sources.push_back({node, slot, source, handoff.destination});
                            }
                        }
                    }
                }
                if (clock::now() >= deadline) {
                    record["status"] = "deadline before placement binding";
                    applied.clear();
                    return;
                }
                for (device d : {local, remote})
                    reserved_buffers[d] = ggml_backend_sched_get_buffer_size(sched, devices[d]);
                const auto bind_started = clock::now();
                // Any exception after binding begins requires a fresh llama
                // graph. Its scheduler views may already refer to replicas.
                record["status"] = "preparing placement";
                record["weight_bindings"] = replicas->bind(sched, applied);
                apply_placements(sched, applied);
                record["bind_ms"] = std::chrono::duration<double, std::milli>(clock::now() - bind_started).count();
            }
        } catch (const std::exception & error) {
            record["status"] = error.what();
            if (!applied.empty())
                throw;
        }
    }
};

}

execution_configuration adaptive_policy::apply(execution_options & options) const {
    using namespace policy_options;
    const auto endpoint = rpc_endpoint(options, "adaptive");
    const auto local_name = local_device(options);
    const auto profile_dir = get(options, "--planner-profile");
    const auto log_file = get(options, "--planner-log");
    const auto method_name = get(options, "--planner-algorithm", "local-search");
    const auto activation = get(options, "--planner-mode", "observe");
    if (profile_dir.empty() || log_file.empty()) {
        throw std::invalid_argument("Adaptive placement requires --planner-profile and --planner-log");
    }
    if (activation != "observe" && activation != "apply")
        throw std::invalid_argument("Planner mode must be observe or apply");
    const std::map<std::string, algorithm> methods = {{"local-search", algorithm::local_search},
                                                      {"min-cut", algorithm::min_cut},
                                                      {"beam-search", algorithm::beam_search}};
    const auto method = methods.find(method_name);
    if (method == methods.end())
        throw std::invalid_argument("Unknown planner algorithm");
    const auto budget_string = get(options, "--planner-budget-ms", "30");
    size_t consumed = 0;
    const double planning_budget = std::stod(budget_string, &consumed);
    if (!(planning_budget > 0 && planning_budget <= 60'000) || consumed != budget_string.size()) {
        throw std::invalid_argument("Planner budget must be between 0 and 60000 milliseconds");
    }
    if (std::getenv("GGML_REQUIRED_DEVICE"))
        throw std::invalid_argument("adaptive requires separate scheduler devices");
    const auto local_layout = get(options, "--cann-q8-nz", "1");
    require(options, "--cann-q8-nz", local_layout);
    resolve_device_options(options, local_name);
    set_backend_environment("GGML_RPC_NO_RDMA", "1");
    set_backend_environment("GGML_CANN_SATURATION", "1");
    const auto remote_name = register_remote_cann(endpoint);
    auto index = std::make_shared<const profiler_index>(profile_dir);
    const bool active = activation == "apply";
    std::optional<calibration> path;
    const auto calibration_path = get(options, "--planner-calibration");
    if (!calibration_path.empty()) {
        path = load_calibration(calibration_path);
        if (path->endpoint != endpoint || path->local_device != local_name ||
            path->profile_capture != file_sha256(std::filesystem::path(profile_dir) / "manifest.json")) {
            throw std::invalid_argument("Calibration belongs to a different device path or Profiler capture");
        }
    }
    if (active && !path)
        throw std::invalid_argument("Apply mode requires --planner-calibration");
    std::map<std::string, std::string> execution_shape;
    for (const char * key : {"--n-ctx", "--n-batch", "--n-ubatch"})
        execution_shape[key] = get(options, key);
    auto initialize = [index, path, active, log_file, method_name, selected = method->second, planning_budget,
                       local_name, remote_name, local_layout, endpoint,
                       execution_shape](ggml_backend_sched_t sched, const std::vector<ggml_tensor *> & weights,
                                        const std::string & model_path) {
        auto prepared = std::make_shared<adaptive_schedule>();
        prepared->method = selected;
        prepared->method_name = method_name;
        prepared->profile = index;
        prepared->path = path;
        prepared->activate = active;
        prepared->budget_ms = planning_budget;
        prepared->log.open(log_file, std::ios::app);
        if (!prepared->log)
            throw std::invalid_argument("Cannot open planner log");
        prepared->devices = {find_scheduler_backend(sched, ggml_backend_dev_by_name(local_name.c_str())),
                             find_scheduler_backend(sched, ggml_backend_dev_by_name(remote_name.c_str()))};
        if (!prepared->devices[local] || !prepared->devices[remote]) {
            throw std::runtime_error("Adaptive placement requires two configured accelerator backends");
        }
        const auto & identity = index->identity();
        if (active) {
            if (file_sha256(model_path) != identity.at("model_components_sha256").at("understanding")) {
                throw std::invalid_argument("Profiler understanding model hash differs");
            }
            for (const auto & requested : index->devices()) {
                const auto transport = requested.at("transport").get<std::string>();
                const auto at = transport == "local" ? local : remote;
                const auto & records = identity.at("devices");
                const auto found = std::find_if(records.begin(), records.end(), [&](const auto & entry) {
                    return entry.at("device_id") == requested.at("id");
                });
                if (found == records.end())
                    throw std::invalid_argument("Profiler device identity is missing");
                const auto & runtime = found->at("runtime");
                const auto & options = found->at("resolved_options");
                const auto & expected_layout = at == local ? local_layout : "0";
                const auto & expected_device = at == local ? local_name : remote_name;
                const auto hardware = ggml_backend_get_device(prepared->devices[at]);
                const auto & described = runtime.at("hardware").at("understanding");
                if (requested.at("backend") != (at == local ? local_name : std::string("CANN0")) ||
                    requested.at("endpoint") != (at == local ? std::string{} : endpoint) || described.size() != 1 ||
                    described.at(0).at("name") != expected_device ||
                    described.at(0).at("description") != ggml_backend_dev_description(hardware) ||
                    options.at("--understanding-backend") != expected_device) {
                    throw std::invalid_argument("Profiler device or endpoint differs");
                }
                const auto & build = runtime.at("build");
                if (build.at("llama_cpp") != UMM_LLAMA_COMMIT || build.at("dirty").at("llama_cpp") != false ||
                    UMM_LLAMA_DIRTY || options.at("--cann-q8-nz") != expected_layout) {
                    throw std::invalid_argument("Profiler GGML build or CANN weight layout differs");
                }
                for (const auto & [key, value] : execution_shape) {
                    if (options.at(key) != value)
                        throw std::invalid_argument("Profiler " + key + " differs");
                }
            }
        }
        if (prepared->activate) {
            prepared->replicas = std::make_unique<weight_replicas>(
                weights, std::vector<ggml_backend_t>{prepared->devices[local], prepared->devices[remote]});
            // Query after loading replicas. Remote memory queries are synchronous RPCs.
            size_t remote_total_bytes = 0;
            ggml_backend_dev_memory(ggml_backend_get_device(prepared->devices[remote]),
                                    &prepared->remote_free_bytes, &remote_total_bytes);
        }
        return prepared_placement{[prepared](ggml_backend_sched_t sched, ggml_cgraph * graph, bool allocated) {
                                      prepared->place(sched, graph, allocated);
                                  },
                                  [prepared] {
                                      if (prepared->applied.empty())
                                          return false;
                                      prepared->disabled = true;
                                      prepared->applied.clear();
                                      if (!prepared->record.contains("verification_error") &&
                                          prepared->record.value("status", "") == "preparing placement") {
                                          prepared->record["allocation_error"] = "GGML did not allocate the candidate";
                                      }
                                      prepared->record["recovery"] = "rebuilding local placement";
                                      prepared->log << prepared->record.dump() << '\n';
                                      prepared->log.flush();
                                      return true;
                                  },
                                  [prepared](const char * phase, int64_t tokens, int64_t cached) {
                                      return prepared->replan(phase, tokens, cached);
                                  }};
    };
    execution_configuration result;
    result.strict_acceleration = true;
    result.schedule.secondary_backend = remote_name;
    result.schedule.prepare = std::move(initialize);
    return result;
}

}
