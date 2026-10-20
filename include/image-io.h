#pragma once

#include "session.h"

#include <filesystem>

namespace umm::cli {

umm::image_input read_image(const std::filesystem::path & path);
void write_png(const std::filesystem::path & path, const umm::image_result & image);
umm::image_input read_image_data_url(const std::string & value, size_t max_pixels = 32000000);
std::string png_base64(const umm::image_result & image);

}
