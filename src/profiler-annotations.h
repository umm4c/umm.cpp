#pragma once

#include "json.hpp"

namespace umm::profiling::detail {
// Joins completed, steady-state graph measurements after collection.
nlohmann::json annotate_timings(const nlohmann::json & graphs, const nlohmann::json & trace);
}
