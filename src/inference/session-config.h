#pragma once

#include "model-package.h"
#include "scheduling/backend-devices.h"
#include "scheduling/execution-policy.h"

#include <string>

namespace umm {

inline execution_configuration configure_model_execution_policy(execution_options & options) {
    const auto policy = options.find("--execution-policy");
    const auto model = options.find("--model");
    if (policy != options.end() && model != options.end() &&
        (policy->second == "local_310p" || policy->second == "remote_910b") &&
        (!options.count("--n-batch") || !options.count("--n-ubatch")) &&
        resolve_model(model->second).family == model_family::sensenova_u1) {
        // A 1024 x 1024 U1 input needs one complete 1024-token image group.
        options.emplace("--n-batch", "1024");
        options.emplace("--n-ubatch", "1024");
    }
    return configure_execution_policy(options);
}

inline int resolve_context_size(model_family family, const std::string & understanding_backend,
                                const std::string & generation_backend, int requested_size) {
    if (requested_size != 0) {
        return requested_size;
    }
    if (family == model_family::sensenova_u1 && understanding_backend == generation_backend &&
        backend_uses_cann(understanding_backend)) {
        // Limit the scheduler reservation when both U1 models share a CANN device.
        return 2048;
    }
    return 0; // Use the llama adapter's model default.
}

}
