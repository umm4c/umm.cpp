#include "model-registry.h"

#include <stdexcept>

namespace umm {

namespace {

constexpr uint32_t all_capabilities =
    static_cast<uint32_t>(model_capability::text) |
    static_cast<uint32_t>(model_capability::image) |
    static_cast<uint32_t>(model_capability::understand) |
    static_cast<uint32_t>(model_capability::edit) |
    static_cast<uint32_t>(model_capability::image_guidance);

constexpr const char * u1_components[] = {
    "understanding",
    "generation",
};

constexpr const char * bagel_components[] = {
    "understanding",
    "generation",
    "vision",
    "vae",
};

// Single source of truth for model-specific limits and package components.
constexpr model_descriptor descriptors[] = {
    // SenseNova U1 stores its native vision encoder in the generation GGUF.
    {model_family::sensenova_u1, "sensenova_u1", all_capabilities,
     16384, 8192, 32, 2048, 2048, 0, 0, u1_components, 2},
    // BAGEL keeps its vision projector and VAE as separate package components.
    {model_family::bagel, "bagel", all_capabilities,
     8192, 512, 16, 1024, 1024, 1024, 1024, bagel_components, 4},
};

const model_descriptor * find_descriptor(std::string_view architecture) {
    for (const auto & descriptor : descriptors) {
        if (architecture == descriptor.architecture) {
            return &descriptor;
        }
    }
    return nullptr;
}

const model_descriptor * find_descriptor(model_family family) {
    for (const auto & descriptor : descriptors) {
        if (family == descriptor.family) {
            return &descriptor;
        }
    }
    return nullptr;
}

} // namespace

const model_descriptor * find_model_descriptor(std::string_view architecture) {
    return find_descriptor(architecture);
}

const model_descriptor & model_descriptor_for(model_family family) {
    if (const auto * descriptor = find_descriptor(family)) {
        return *descriptor;
    }
    throw std::invalid_argument("Unsupported model family");
}

model_family model_family_for(std::string_view architecture) {
    const auto * descriptor = find_descriptor(architecture);
    return descriptor ? descriptor->family : model_family::unknown;
}

bool model_supports(model_family family, model_capability capability) {
    const auto mask = static_cast<uint32_t>(capability);
    return (model_descriptor_for(family).capabilities & mask) != 0;
}

}
