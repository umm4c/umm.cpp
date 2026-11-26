#pragma once

#include "model-package.h"
#include "scheduling/backend-devices.h"
#include "scheduling/execution-policy.h"

#include <string>

namespace umm {

inline execution_configuration configure_model_execution_policy(execution_options & options) {
    const auto policy = options.find("--execution-policy");
    const auto model = options.find("--model");
    if (policy != options.end() && model != options.end()) {
        const bool local = policy->second == "local_310p";
        const bool needs_batch_defaults = !options.count("--n-batch") || !options.count("--n-ubatch");
        if ((local || policy->second == "remote_910b") &&
            (needs_batch_defaults || (local && !options.count("--n-ctx")))) {
            const auto family = resolve_model(model->second).family;
            int image_group_tokens = 0;
            if (family == model_family::sensenova_u1) {
                // A 1024 x 1024 U1 input needs one complete 1024-token image group.
                image_group_tokens = 1024;
            } else if (local && family == model_family::bagel) {
                // BAGEL fits the patches of a 980 x 980 vision grid and both boundaries in one batch.
                image_group_tokens = (980 / 14) * (980 / 14) + 2;
            }
            if (image_group_tokens > 0) {
                constexpr int text_context_tokens = 1024;
                const auto batch_size = std::to_string(image_group_tokens);
                options.emplace("--n-ctx", std::to_string(image_group_tokens + text_context_tokens));
                options.emplace("--n-batch", batch_size);
                options.emplace("--n-ubatch", batch_size);
            }
        }
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
