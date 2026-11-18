#include "model-package.h"

#include "json.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace umm {

namespace {

constexpr const char * understanding_component = "understanding";
constexpr const char * generation_component = "generation";
constexpr const char * vision_component = "vision";
constexpr const char * vae_component = "vae";

const std::string & empty_component() {
    static const std::string empty;
    return empty;
}

void validate_manifest(const nlohmann::json & manifest,
                       const model_descriptor * descriptor) {
    const auto version = manifest.at("version").get<int>();
    const auto format = manifest.at("format").get<std::string>();
    if (format != "umm" || (version != 1 && version != 2) || descriptor == nullptr) {
        throw std::runtime_error("Unsupported model package format, version, or architecture");
    }
}

std::string resolve_component_path(const std::filesystem::path & root,
                                   const std::string & name,
                                   const std::string & value) {
    namespace fs = std::filesystem;
    const fs::path relative = value;
    if (relative.empty() || relative.is_absolute()) {
        throw std::runtime_error("Invalid package path: " + name);
    }

    const auto path = fs::weakly_canonical(root / relative);
    const auto within = path.lexically_relative(root);
    if (within.empty() || *within.begin() == ".." || !fs::is_regular_file(path)) {
        throw std::runtime_error("Missing or external package file: " + name);
    }
    return path.string();
}

void load_components(const nlohmann::json & manifest,
                     const std::filesystem::path & root,
                     model_package & package) {
    const auto add_component = [&](const std::string & name, const std::string & value) {
        package.components.emplace(name, resolve_component_path(root, name, value));
    };

    if (manifest.contains("components")) {
        const auto & components = manifest.at("components");
        if (!components.is_object()) {
            throw std::runtime_error("Model package components must be an object");
        }
        for (auto it = components.begin(); it != components.end(); ++it) {
            add_component(it.key(), it.value().get<std::string>());
        }
        return;
    }

    static const std::vector<std::string> metadata_keys = {
        "format", "version", "architecture",
    };
    for (auto it = manifest.begin(); it != manifest.end(); ++it) {
        if (!it.value().is_string() ||
            std::find(metadata_keys.begin(), metadata_keys.end(), it.key()) != metadata_keys.end()) {
            continue;
        }
        add_component(it.key(), it.value().get<std::string>());
    }
}

model_package package_from_files(const std::string & model,
                                 const std::string & generation) {
    model_package package;
    package.components.emplace(understanding_component, model);
    if (!generation.empty()) {
        package.components.emplace(generation_component, generation);
    }
    return package;
}

void require_components(const model_package & package,
                        const model_descriptor & descriptor) {
    for (size_t i = 0; i < descriptor.required_component_count; ++i) {
        if (!package.has_component(descriptor.required_components[i])) {
            throw std::runtime_error(std::string("Model package is missing required component: ") +
                                     descriptor.required_components[i]);
        }
    }
}

} // namespace

bool model_package::has_component(std::string_view name) const {
    return components.find(std::string(name)) != components.end();
}

const std::string & model_package::component(std::string_view name) const {
    const auto found = components.find(std::string(name));
    return found == components.end() ? empty_component() : found->second;
}

bool model_package::supports(model_family resolved_family, model_capability capability) const {
    if (!model_supports(resolved_family, capability)) {
        return false;
    }

    switch (capability) {
        case model_capability::text:
            return has_component(understanding_component);
        case model_capability::image:
            return has_component(generation_component) &&
                   (resolved_family != model_family::bagel || has_component(vae_component));
        case model_capability::understand:
            // U1 keeps its vision encoder in the generation GGUF. BAGEL uses
            // a separate vision projector component.
            return resolved_family == model_family::sensenova_u1
                       ? has_component(generation_component)
                       : has_component(vision_component);
        case model_capability::edit:
            return supports(resolved_family, model_capability::image) &&
                   supports(resolved_family, model_capability::understand);
        case model_capability::image_guidance:
            return supports(resolved_family, model_capability::edit);
    }
    return false;
}

model_package resolve_model(const std::string & model, const std::string & generation) {
    namespace fs = std::filesystem;

    if (!fs::is_directory(model)) {
        return package_from_files(model, generation);
    }
    if (!generation.empty()) {
        throw std::invalid_argument("A model package cannot be combined with --image-model");
    }

    const auto root = fs::canonical(model);
    std::ifstream input(root / "model.json");
    if (!input) {
        throw std::runtime_error("Model package is missing model.json: " + model);
    }

    const auto manifest = nlohmann::json::parse(input);
    const auto architecture = manifest.at("architecture").get<std::string>();
    const auto * descriptor = find_model_descriptor(architecture);
    validate_manifest(manifest, descriptor);

    model_package package;
    package.family = descriptor->family;
    package.architecture = architecture;
    load_components(manifest, root, package);
    require_components(package, *descriptor);
    return package;
}

}
