#include "backend-devices.h"
#include "ggml-backend.h"

#include <mutex>
#include <set>
#include <stdexcept>

namespace umm {
namespace {
std::mutex registry_mutex;
std::set<std::string> registered_endpoints;
std::set<std::string> remote_cann_devices;
}

bool backend_uses_cann(const std::string & name) {
    if (name.rfind("CANN", 0) == 0 || name.rfind("cann", 0) == 0)
        return true;
    std::lock_guard<std::mutex> lock(registry_mutex);
    return remote_cann_devices.count(name) != 0;
}

std::string register_remote_cann(const std::string & endpoint) {
    ggml_backend_load_all();
    std::lock_guard<std::mutex> lock(registry_mutex);
    auto rpc = ggml_backend_reg_by_name("RPC");
    if (!rpc)
        throw std::runtime_error("RPC backend is unavailable; rebuild with GGML_RPC=ON");
    using add_server = ggml_backend_reg_t (*)(const char *);
    auto add = reinterpret_cast<add_server>(ggml_backend_reg_get_proc_address(rpc, "ggml_backend_rpc_add_server"));
    if (!add)
        throw std::runtime_error("RPC registration API is unavailable");
    auto reg = add(endpoint.c_str());
    if (!reg || ggml_backend_reg_dev_count(reg) != 1) {
        throw std::runtime_error("RPC requires a worker exposing exactly one CANN device");
    }
    if (registered_endpoints.insert(endpoint).second)
        ggml_backend_register(reg);
    std::string selected = ggml_backend_dev_name(ggml_backend_reg_dev_get(reg, 0));
    // The worker is pinned to CANN/910B by the deployment manifest, not RPC device properties.
    remote_cann_devices.insert(selected);
    return selected;
}

}
