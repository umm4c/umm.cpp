#pragma once
#include "hash/hash.h"
#include "../../third_party/llama.cpp/common/base64.hpp"
#include "json.hpp"
#include <algorithm>
#include <chrono>
#include <vector>
#include <cstdint>
#include <string>
#include <stdexcept>
namespace umm::profiling::detail {
using clock = std::chrono::steady_clock;
inline double elapsed(clock::time_point start) {
    return std::chrono::duration<double, std::milli>(clock::now() - start).count();
}
inline std::string hash(const std::string & value) {
    return hash_sha256_hex(value.data(), value.size());
}
inline std::string base64(const std::vector<uint8_t> & bytes) {
    if (bytes.empty())
        return {};
    return ::base64::encode(reinterpret_cast<const char *>(bytes.data()), bytes.size());
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
