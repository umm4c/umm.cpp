#pragma once

#include "model-registry.h"
#include "scheduling/backend-devices.h"

#include <string>

namespace umm {

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
