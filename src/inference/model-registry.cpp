#include "model-registry-internal.h"
#include "models/u1-rules.h"

#include <stdexcept>

namespace umm {

namespace {

constexpr uint32_t all_capabilities =
    static_cast<uint32_t>(model_capability::text) | static_cast<uint32_t>(model_capability::image) |
    static_cast<uint32_t>(model_capability::understand) | static_cast<uint32_t>(model_capability::edit) |
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

struct registered_model {
    model_descriptor descriptor;
    const char * vision_component;
    const char * generation_auxiliary;
};

constexpr registered_model models[] = {
    // SenseNova U1 stores its native vision encoder in the generation GGUF.
    {{model_family::sensenova_u1, "sensenova_u1", all_capabilities, 16384, 8192, u1_rules::image_stride, 2048, 2048, 0,
      0, u1_components, 2},
     "generation",
     nullptr},
    // BAGEL keeps its vision projector and VAE as separate package components.
    {{model_family::bagel, "bagel", all_capabilities, 8192, 8192, 16, 1024, 1024, 1024, 1024, bagel_components, 4},
     "vision",
     "vae"},
};

const registered_model & model_entry(model_family family) {
    for (const auto & model : models) {
        if (family == model.descriptor.family) {
            return model;
        }
    }
    throw std::invalid_argument("Unsupported model family");
}

} // namespace

const model_descriptor * find_model_descriptor(std::string_view architecture) {
    for (const auto & model : models) {
        if (architecture == model.descriptor.architecture) {
            return &model.descriptor;
        }
    }
    return nullptr;
}

const model_descriptor & model_descriptor_for(model_family family) {
    return model_entry(family).descriptor;
}

model_family model_family_for(std::string_view architecture) {
    const auto * descriptor = find_model_descriptor(architecture);
    return descriptor ? descriptor->family : model_family::unknown;
}

bool model_supports(model_family family, model_capability capability) {
    const auto mask = static_cast<uint32_t>(capability);
    return (model_descriptor_for(family).capabilities & mask) != 0;
}

std::array<const char *, 3> model_capability_components(model_family family, model_capability capability) {
    if (!model_supports(family, capability)) {
        return {};
    }

    const auto & model = model_entry(family);
    switch (capability) {
    case model_capability::text:
        return {"understanding"};
    case model_capability::image:
        return {"generation", model.generation_auxiliary};
    case model_capability::understand:
        return {model.vision_component};
    case model_capability::edit:
    case model_capability::image_guidance:
        if (model_supports(family, model_capability::image) && model_supports(family, model_capability::understand) &&
            (capability != model_capability::image_guidance || model_supports(family, model_capability::edit))) {
            return {"generation", model.generation_auxiliary, model.vision_component};
        }
        return {};
    }
    return {};
}

}
