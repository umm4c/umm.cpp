#pragma once
#include "hash/hash.h"
#include "json.hpp"
#include <algorithm>
#include <vector>
#include <string>
#include <stdexcept>
namespace umm::profiling::detail {
inline std::string hash(const std::string & value) {
    return hash_sha256_hex(value.data(), value.size());
}
inline nlohmann::json sample_statistics(std::vector<double> values) {
    if (values.empty())
        throw std::invalid_argument("Timing statistics require samples");
    std::sort(values.begin(), values.end());
    double sum = 0;
    for (auto value : values)
        sum += value;
    auto q = [&](double f) {
        double n = f * (values.size() - 1);
        size_t a = size_t(n), b = std::min(a + 1, values.size() - 1);
        return values[a] + (values[b] - values[a]) * (n - a);
    };
    return {{"sample_count", values.size()}, {"mean", sum / values.size()}, {"p50", q(.5)},     {"p95", q(.95)},
            {"min", values.front()},         {"max", values.back()},        {"samples", values}};
}
}
