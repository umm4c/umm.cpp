#include "execution-policy.h"
#include "ggml-backend.h"

#include <cstdlib>
#include <mutex>
#include <set>
#include <stdexcept>

namespace umm {
namespace {
std::mutex registry_mutex;
std::set<std::string> registered_endpoints;
std::set<std::string> remote_cann_devices;

std::string get(const execution_options & options, const std::string & key,
                const std::string & fallback = "") {
    const auto it = options.find(key);
    return it == options.end() ? fallback : it->second;
}

void environment(const char * name, const std::string & value) {
    if (setenv(name, value.c_str(), 1)) throw std::runtime_error(std::string("Cannot set ") + name);
}

void require_option(execution_options & options, const std::string & key, const std::string & value) {
    const auto it = options.find(key);
    if (it != options.end() && it->second != value) {
        throw std::invalid_argument(key + " conflicts with the static execution policy");
    }
    options[key] = value;
}
}

bool backend_uses_cann(const std::string & name) {
    if (name.rfind("CANN", 0) == 0 || name.rfind("cann", 0) == 0) return true;
    std::lock_guard<std::mutex> lock(registry_mutex);
    return remote_cann_devices.count(name) != 0;
}

std::string register_remote_cann(const std::string & endpoint) {
    ggml_backend_load_all();
    std::lock_guard<std::mutex> lock(registry_mutex);
    auto rpc = ggml_backend_reg_by_name("RPC");
    if (!rpc) throw std::runtime_error("RPC backend is unavailable; rebuild with GGML_RPC=ON");
    using add_server = ggml_backend_reg_t (*)(const char *);
    auto add = reinterpret_cast<add_server>(ggml_backend_reg_get_proc_address(rpc, "ggml_backend_rpc_add_server"));
    if (!add) throw std::runtime_error("RPC registration API is unavailable");
    auto reg = add(endpoint.c_str());
    if (!reg || ggml_backend_reg_dev_count(reg) != 1) {
        throw std::runtime_error("RPC requires a worker exposing exactly one CANN device");
    }
    if (registered_endpoints.insert(endpoint).second) ggml_backend_register(reg);
    std::string selected = ggml_backend_dev_name(ggml_backend_reg_dev_get(reg, 0));
    // The worker is pinned to CANN/910B by the deployment manifest, not RPC device properties.
    remote_cann_devices.insert(selected);
    return selected;
}

execution_options configure_execution_policy(execution_options options) {
    const auto policy = get(options, "--execution-policy");
    if (policy.empty()) {
        if (options.count("--rpc-endpoint") || options.count("--local-device")) {
            throw std::invalid_argument("RPC/local device options require --execution-policy");
        }
        return options;
    }
    if (policy != "local_310p" && policy != "remote_910b") {
        throw std::invalid_argument("execution-policy must be local_310p or remote_910b");
    }
    std::string selected;
    if (policy == "remote_910b") {
        const auto endpoint = get(options, "--rpc-endpoint");
        if (endpoint.empty() || endpoint.find(':') == std::string::npos || options.count("--local-device")) {
            throw std::invalid_argument("remote_910b requires --rpc-endpoint HOST:PORT and no local device");
        }
        require_option(options, "--cann-q8-nz", "0");
        require_option(options, "--monitoring", "0");
        environment("GGML_RPC_NO_RDMA", "1");
        selected = register_remote_cann(endpoint);
        environment("GGML_RPC_CANN_DEVICE", selected);
    } else {
        if (options.count("--rpc-endpoint")) throw std::invalid_argument("local_310p cannot use an RPC endpoint");
        ggml_backend_load_all();
        selected = get(options, "--local-device", "CANN0");
        auto device = ggml_backend_dev_by_name(selected.c_str());
        if (!device || std::string(ggml_backend_reg_name(ggml_backend_dev_backend_reg(device))) != "CANN" ||
            std::string(ggml_backend_dev_description(device)).find("310P") == std::string::npos) {
            throw std::runtime_error("local_310p requires an available Ascend310P CANN device");
        }
        options.emplace("--cann-q8-nz", "1");
    }
    if (const char * previous = std::getenv("GGML_REQUIRED_DEVICE")) {
        if (selected != previous) throw std::invalid_argument("A process cannot change its static execution device");
    }
    for (const char * key : {"--understanding-backend", "--generation-backend", "--vision-backend"}) {
        require_option(options, key, selected);
    }
    options.emplace("--generation-max-vram", selected + "=40");
    if (get(options, "--generation-max-vram").rfind(selected + "=", 0) != 0 ||
        get(options, "--generation-max-vram").find(',') != std::string::npos) {
        throw std::invalid_argument("generation-max-vram must refer only to the static execution device");
    }
    options.emplace("--n-ctx", "2048");
    options.emplace("--n-batch", "512");
    options.emplace("--n-ubatch", "256");
    options.emplace("--u1-prefill-mode", "mixed4");
    options.emplace("--u1-v-cache-layout", "contiguous");
    environment("GGML_CANN_Q8_NZ", get(options, "--cann-q8-nz"));
    environment("GGML_SCHED_STRICT_ACCEL", "1");
    environment("GGML_REQUIRED_DEVICE", selected);
    return options;
}
}
