// stb is a single-header library. Emit the implementation once in the CLI
// executable; the rest of the application uses this small wrapper.
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "image-io.h"

#include <stdexcept>
#include <limits>

namespace umm::cli {

static const std::string alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

umm::image_input read_image_data_url(const std::string & value, size_t max_pixels) {
    const auto comma = value.find(',');
    if (comma == std::string::npos || (value.substr(0, comma) != "data:image/png;base64" &&
        value.substr(0, comma) != "data:image/jpeg;base64")) {
        throw std::invalid_argument("Expected a PNG or JPEG base64 data URL");
    }
    const auto length = value.size() - comma - 1;
    if (!length || length % 4 || length > 128 * 1024 * 1024) throw std::invalid_argument("Invalid image base64 length");
    std::vector<unsigned char> bytes;
    bytes.reserve(length / 4 * 3);
    for (size_t i = comma + 1; i < value.size(); i += 4) {
        uint32_t word = 0;
        int padding = 0;
        for (int k = 0; k < 4; ++k) {
            const char ch = value[i + k];
            if (ch == '=') {
                if (k < 2 || i + 4 != value.size()) throw std::invalid_argument("Invalid image base64 padding");
                ++padding;
                word <<= 6;
            } else {
                const auto digit = alphabet.find(ch);
                if (digit == std::string::npos || padding) throw std::invalid_argument("Invalid image base64 character");
                word = (word << 6) | static_cast<uint32_t>(digit);
            }
        }
        if (padding > 2 || (padding && (word & ((1u << (padding * 8)) - 1)))) {
            throw std::invalid_argument("Invalid image base64 tail");
        }
        bytes.push_back(word >> 16);
        if (padding < 2) bytes.push_back(word >> 8);
        if (!padding) bytes.push_back(word);
    }
    int width = 0, height = 0, channels = 0;
    if (bytes.size() > size_t(std::numeric_limits<int>::max()) ||
        !stbi_info_from_memory(bytes.data(), int(bytes.size()), &width, &height, &channels) ||
        width <= 0 || height <= 0 || size_t(width) > max_pixels / size_t(height)) {
        throw std::invalid_argument("Invalid image or decoded pixel limit exceeded");
    }
    unsigned char * pixels = stbi_load_from_memory(bytes.data(), int(bytes.size()), &width, &height, &channels, 3);
    if (!pixels) throw std::invalid_argument("Could not decode image data");
    umm::image_input result{width, height, std::vector<uint8_t>(pixels, pixels + size_t(width) * height * 3)};
    stbi_image_free(pixels);
    return result;
}

std::string png_base64(const umm::image_result & image) {
    if (image.width <= 0 || image.height <= 0 || image.rgb.size() != size_t(image.width) * image.height * 3) {
        throw std::invalid_argument("Invalid image output");
    }
    std::vector<unsigned char> bytes;
    auto append = [](void * context, void * data, int size) {
        auto & output = *static_cast<std::vector<unsigned char> *>(context);
        auto * begin = static_cast<unsigned char *>(data);
        output.insert(output.end(), begin, begin + size);
    };
    if (!stbi_write_png_to_func(append, &bytes, image.width, image.height, 3, image.rgb.data(), image.width * 3)) {
        throw std::runtime_error("Could not encode PNG");
    }
    std::string result;
    result.reserve((bytes.size() + 2) / 3 * 4);
    for (size_t i = 0; i < bytes.size(); i += 3) {
        uint32_t word = uint32_t(bytes[i]) << 16;
        if (i + 1 < bytes.size()) word |= uint32_t(bytes[i + 1]) << 8;
        if (i + 2 < bytes.size()) word |= bytes[i + 2];
        result += alphabet[(word >> 18) & 63];
        result += alphabet[(word >> 12) & 63];
        result += i + 1 < bytes.size() ? alphabet[(word >> 6) & 63] : '=';
        result += i + 2 < bytes.size() ? alphabet[word & 63] : '=';
    }
    return result;
}

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
