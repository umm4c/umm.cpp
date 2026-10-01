#pragma once

#include "session.h"

#include <filesystem>

namespace umm::cli {

umm::image_input read_image(const std::filesystem::path & path);
void write_png(const std::filesystem::path & path, const umm::image_result & image);

}
