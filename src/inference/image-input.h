#pragma once

#include "session.h"

#include <stdexcept>

namespace umm {

inline void validate_rgb_image(const image_input & image) {
    if (image.width <= 0 || image.height <= 0 || image.rgb.size() != size_t(image.width) * image.height * 3) {
        throw std::invalid_argument("Expected a nonempty RGB image");
    }
}

}
