// stb is a single-header library. Emit the implementation once in the CLI
// executable; the rest of the application uses this small wrapper.
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "image-io.h"

#include <stdexcept>

namespace umm::cli {

umm::image_input read_image(const std::filesystem::path & path) {
    int width = 0;
    int height = 0;
    int source_channels = 0;
    unsigned char * pixels = stbi_load(path.string().c_str(), &width, &height, &source_channels, 3);
    if (!pixels || width <= 0 || height <= 0) {
        const std::string reason = stbi_failure_reason() ? stbi_failure_reason() : "unknown error";
        stbi_image_free(pixels);
        throw std::runtime_error("Could not read image: " + path.string() + ": " + reason);
    }
    umm::image_input result{width, height,
        std::vector<uint8_t>(pixels, pixels + size_t(width) * height * 3)};
    stbi_image_free(pixels);
    return result;
}

void write_png(const std::filesystem::path & path, const umm::image_result & image) {
    const auto expected_size = size_t(image.width) * image.height * 3;
    if (image.width <= 0 || image.height <= 0 || image.rgb.size() != expected_size) {
        throw std::runtime_error("Image result has invalid dimensions or pixel data");
    }
    if (!stbi_write_png(path.string().c_str(), image.width, image.height, 3,
                        image.rgb.data(), image.width * 3)) {
        throw std::runtime_error("Could not write image: " + path.string());
    }
}

}
