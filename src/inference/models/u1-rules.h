#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace umm::u1_rules {

inline constexpr char image_start[] = "<img>";
inline constexpr char image_end[] = "</img>";
inline constexpr int image_stride = 32;
inline constexpr int image_position_count = 1;

// An image group shares one time position. Spatial axes follow the image grid.
inline std::array<int32_t, 4> image_position(int32_t time, std::size_t token, int grid_width) {
    return {time, static_cast<int32_t>(token / grid_width), static_cast<int32_t>(token % grid_width), 0};
}

inline std::array<int32_t, 4> text_position(int32_t time) {
    return {time, 0, 0, 0};
}

}
