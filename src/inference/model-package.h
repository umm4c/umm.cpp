#pragma once

#include "model-registry.h"

#include <map>
#include <string>
#include <string_view>

namespace umm {

struct model_package {
    // A package may be assembled from one understanding GGUF plus an
    // optional generation checkpoint, or loaded from model.json.
    model_family family = model_family::unknown;
    std::string architecture;
    std::map<std::string, std::string> components;

    bool has_component(std::string_view name) const;
    const std::string & component(std::string_view name) const;
    // Combines static family support with the components present in this
    // particular package.
    bool supports(model_family resolved_family, model_capability capability) const;
};

// Resolve either a standalone understanding GGUF or a validated package
// directory. The optional generation argument is only valid for a standalone
// model path.
model_package resolve_model(const std::string & model, const std::string & generation = "");

}
