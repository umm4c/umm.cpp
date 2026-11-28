#include "policies/policy.h"
#include "policies/adaptive.h"
#include "policies/random-plan.h"
#include "ggml-backend.h"

#include <stdexcept>
#include <algorithm>

namespace umm {

namespace policy_options {

std::string get(const execution_options & options, const std::string & key, const std::string & fallback) {
    const auto it = options.find(key);
    return it == options.end() ? fallback : it->second;
}

void require(execution_options & options, const std::string & key, const std::string & value) {
    const auto it = options.find(key);
    if (it != options.end() && it->second != value) {
        throw std::invalid_argument(key + " conflicts with the execution policy");
    }
    options[key] = value;
}

std::string local_device(const execution_options & options) {
    ggml_backend_load_all();
    const auto selected = get(options, "--local-device", "CANN0");
    auto device = ggml_backend_dev_by_name(selected.c_str());
    if (!device || std::string(ggml_backend_reg_name(ggml_backend_dev_backend_reg(device))) != "CANN" ||
        std::string(ggml_backend_dev_description(device)).find("310P") == std::string::npos) {
        throw std::runtime_error("The selected local device must be an available Ascend310P CANN device");
    }
    return selected;
}

std::string rpc_endpoint(const execution_options & options, const char * policy) {
    const auto endpoint = get(options, "--rpc-endpoint");
    if (endpoint.empty() || endpoint.find(':') == std::string::npos) {
        throw std::invalid_argument(std::string(policy) + " requires --rpc-endpoint HOST:PORT");
    }
    return endpoint;
}

void resolve_device_options(execution_options & options, const std::string & selected) {
    for (const char * key : {"--understanding-backend", "--generation-backend", "--vision-backend"}) {
        require(options, key, selected);
    }
    options.emplace("--generation-max-vram", selected + "=40");
    if (get(options, "--generation-max-vram").rfind(selected + "=", 0) != 0 ||
        get(options, "--generation-max-vram").find(',') != std::string::npos) {
        throw std::invalid_argument("generation-max-vram must refer only to the selected generation device");
    }
    options.emplace("--n-ctx", "2048");
    options.emplace("--n-batch", "512");
    options.emplace("--n-ubatch", "256");
}

}

execution_configuration configure_execution_policy(execution_options & options) {
    static const local_310p_policy local;
    static const remote_910b_policy remote;
    static const random_policy random;
    static const adaptive_policy adaptive;
    static const std::map<std::string, const execution_policy *> policies = {
        {"local_310p", &local},
        {"remote_910b", &remote},
        {"random", &random},
        {"adaptive", &adaptive},
    };
    const auto name = policy_options::get(options, "--execution-policy");
    if (name != "random" && (options.count("--policy-seed") || options.count("--policy-cuts"))) {
        throw std::invalid_argument("--policy-seed and --policy-cuts require --execution-policy random");
    }
    if (name != "adaptive" && std::any_of(options.begin(), options.end(), [](const auto & x) {
        return x.first.rfind("--planner-", 0) == 0;
    })) {
        throw std::invalid_argument("Planner options require --execution-policy adaptive");
    }
    if (name.empty()) {
        if (options.count("--rpc-endpoint") || options.count("--local-device")) {
            throw std::invalid_argument("RPC/local device options require --execution-policy");
        }
        return {};
    }
    const auto policy = policies.find(name);
    if (policy == policies.end()) {
        throw std::invalid_argument("execution-policy must be local_310p, remote_910b, random or adaptive");
    }
    return policy->second->apply(options);
}

}
