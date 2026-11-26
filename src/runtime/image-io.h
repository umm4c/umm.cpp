#pragma once

#include "session.h"

#include <filesystem>
#include <string_view>
#include <utility>

namespace umm::runtime {

inline constexpr size_t default_max_image_pixels = 32000000;

umm::image_input read_image(const std::filesystem::path & path);
void write_png(const std::filesystem::path & path, const umm::image_result & image);
std::pair<int, int> image_dimensions(std::string_view bytes, size_t max_pixels = default_max_image_pixels);
umm::image_input read_image_data_url(const std::string & value, size_t max_pixels = default_max_image_pixels);
std::string png_base64(const umm::image_result & image);

}
