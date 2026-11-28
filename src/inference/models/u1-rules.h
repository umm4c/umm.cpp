#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace umm::u1_rules {

inline constexpr char image_start[] = "<img>";
inline constexpr char image_end[] = "</img>";
inline constexpr int image_stride = 32;
inline constexpr int image_position_count = 1;

// Callers validate the input dimensions before applying the U1 resize rules.
inline std::pair<int, int> prepared_image_size(int width, int height) {
    constexpr int64_t min_pixels = 256LL * 256LL;
    constexpr int64_t max_pixels = 2048LL * 2048LL;
    if (std::max(width, height) / static_cast<double>(std::min(width, height)) > 200.0) {
        throw std::invalid_argument("U1 image aspect ratio must be below 200");
    }

    const auto round_to_stride = [](int size) {
        return std::max<int64_t>(image_stride, ((int64_t(size) + image_stride / 2) / image_stride) * image_stride);
    };
    const auto rounded_width = round_to_stride(width);
    const auto rounded_height = round_to_stride(height);
    const auto rounded_area = rounded_width * rounded_height;
    const auto original_area = int64_t(width) * height;
    if (rounded_area > max_pixels) {
        const double scale = std::sqrt(static_cast<double>(original_area) / max_pixels);
        return {std::max(image_stride, static_cast<int>(std::floor(width / scale / image_stride)) * image_stride),
                std::max(image_stride, static_cast<int>(std::floor(height / scale / image_stride)) * image_stride)};
    }
    if (rounded_area < min_pixels) {
        const double scale = std::sqrt(static_cast<double>(min_pixels) / original_area);
        return {std::max(image_stride, static_cast<int>(std::ceil(width * scale / image_stride)) * image_stride),
                std::max(image_stride, static_cast<int>(std::ceil(height * scale / image_stride)) * image_stride)};
    }
    return {static_cast<int>(rounded_width), static_cast<int>(rounded_height)};
}

// An image group shares one time position. Spatial axes follow the image grid.
inline std::array<int32_t, 4> image_position(int32_t time, std::size_t token, int grid_width) {
    return {time, static_cast<int32_t>(token / grid_width), static_cast<int32_t>(token % grid_width), 0};
}

inline std::array<int32_t, 4> text_position(int32_t time) {
    return {time, 0, 0, 0};
}

}
