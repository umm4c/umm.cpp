#pragma once

#include "model-package.h"
#include "scheduling/backend-devices.h"
#include "scheduling/execution-policy.h"

#include <algorithm>
#include <string>

namespace umm {

inline execution_configuration configure_model_execution_policy(execution_options & options) {
    const auto policy = options.find("--execution-policy");
    const auto model = options.find("--model");
    if (policy != options.end() && policy->second == "local_310p" && model != options.end() &&
        (!options.count("--n-batch") || !options.count("--n-ubatch")) &&
        resolve_model(model->second).family == model_family::sensenova_u1) {
        // A 1024 x 1024 U1 input needs one complete 1024-token image group.
        int default_batch = 1024;
        for (const char * flag : {"--n-ctx", "--n-batch"}) {
            const auto configured = options.find(flag);
            if (configured != options.end()) {
                const int limit = std::stoi(configured->second);
                if (limit > 0)
                    default_batch = std::min(default_batch, limit);
            }
        }
        const auto batch = std::to_string(default_batch);
        options.emplace("--n-batch", batch);
        options.emplace("--n-ubatch", batch);
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
