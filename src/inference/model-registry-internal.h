#pragma once

#include "model-registry.h"

#include <array>

namespace umm {

// Null entries represent unused slots. A valid capability has a first component.
std::array<const char *, 3> model_capability_components(model_family family, model_capability capability);

}
