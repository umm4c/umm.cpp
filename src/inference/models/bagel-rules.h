#pragma once

namespace umm::bagel_rules {

inline constexpr char vision_boundaries[] = "<|vision_start|><|vision_end|>";
inline constexpr int vision_boundary_tokens = 2;
// Both boundary tokens and all embeddings in a vision group share one position.
inline constexpr int image_position_count = 1;

}
