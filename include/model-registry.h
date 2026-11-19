#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace umm {

enum class model_family {
    unknown,
    sensenova_u1,
    bagel,
};

// Capabilities are bit flags because a model can support several request
// types at once. Package-specific files are checked separately.
enum class model_capability : uint32_t {
    text = 1u << 0,
    image = 1u << 1,
    understand = 1u << 2,
    edit = 1u << 3,
    image_guidance = 1u << 4,
};

// Static facts about one supported model family. Package resolution and
// workflow validation both use this descriptor.
struct model_descriptor {
    model_family family;
    const char * architecture;
    uint32_t capabilities;
    int context_size;   // Language-model context length.
    int batch_size;     // Maximum text batch used by the runtime.
    int image_stride;   // Required image width/height alignment.
    int default_width;  // Default generated image width.
    int default_height; // Default generated image height.
    int max_width;      // Zero means no explicit maximum.
    int max_height;     // Zero means no explicit maximum.
    const char * const * required_components;
    size_t required_component_count;
};

// Look up model metadata by its manifest architecture or runtime family.
const model_descriptor * find_model_descriptor(std::string_view architecture);
const model_descriptor & model_descriptor_for(model_family family);
model_family model_family_for(std::string_view architecture);

// Check static model support. Package-specific component checks live in
// model_package::supports.
bool model_supports(model_family family, model_capability capability);

}
