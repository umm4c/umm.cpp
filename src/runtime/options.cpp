#include "options.h"
#include "session.h"

#include <cmath>
#include <stdexcept>
#include <type_traits>

namespace umm::runtime {
namespace {

template <typename T> T numeric_option(const options & args, const char * key, T fallback) {
    const auto entry = args.find(key);
    if (entry == args.end())
        return fallback;

    const auto & text = entry->second;
    std::string::size_type parsed = 0;
    T value;
    try {
        if constexpr (std::is_same_v<T, int>)
            value = std::stoi(text, &parsed);
        else
            value = std::stof(text, &parsed);
    } catch (const std::logic_error &) {
        throw std::invalid_argument(std::string("Invalid value for ") + key + ": " + text);
    }
    if (parsed != text.size() || !std::isfinite(value))
        throw std::invalid_argument(std::string("Invalid value for ") + key + ": " + text);
    return value;
}

}

options parse_options(int argc, char ** argv) {
    std::map<std::string, std::string> options;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key.rfind("--", 0) != 0 || i + 1 == argc || !options.emplace(key, argv[++i]).second) {
            throw std::invalid_argument("Expected unique --option value pairs");
        }
    }
    static const std::map<std::string, bool> allowed = {
        {"--execution-policy", true},
        {"--rpc-endpoint", true},
        {"--local-device", true},
        {"--policy-seed", true},
        {"--policy-cuts", true},
        {"--model", true},
        {"--understanding-backend", true},
        {"--vision-backend", true},
        {"--generation-backend", true},
        {"--generation-max-vram", true},
        {"--steps", true},
        {"--cfg", true},
        {"--image-cfg", true},
        {"--shift", true},
        {"--vae-tiling", true},
        {"--vae-tile-size", true},
        {"--vae-tile-overlap", true},
        {"--n-ctx", true},
        {"--n-batch", true},
        {"--n-ubatch", true},
        {"--cann-q8-nz", true},
        {"--monitoring", true},
        {"--monitor-interval", true},
        {"--system-samples", true},
        {"--request-log", true},
        {"--metrics-port", true},
        {"--host", true},
        {"--port", true},
        {"--served-model-name", true},
        {"--api-key-env", true},
        {"--hash-model", true},
    };
    for (const auto & item : options) {
        if (!allowed.count(item.first)) {
            throw std::invalid_argument("Unknown option: " + item.first);
        }
    }
    if (!options.count("--model")) {
        throw std::invalid_argument("--model is required");
    }
    return options;
}

std::string option(const std::map<std::string, std::string> & options, const std::string & key,
                   const std::string & fallback) {
    const auto found = options.find(key);
    return found == options.end() ? fallback : found->second;
}

image_options parse_generation_options(const options & args, image_options generation) {
    generation.steps = numeric_option(args, "--steps", generation.steps);
    generation.guidance = numeric_option(args, "--cfg", generation.guidance);
    generation.image_guidance = numeric_option(args, "--image-cfg", generation.image_guidance);
    generation.flow_shift = numeric_option(args, "--shift", generation.flow_shift);
    generation.vae_tiling = numeric_option(args, "--vae-tiling", static_cast<int>(generation.vae_tiling)) != 0;
    generation.vae_tile_size = numeric_option(args, "--vae-tile-size", generation.vae_tile_size);
    generation.vae_tile_overlap = numeric_option(args, "--vae-tile-overlap", generation.vae_tile_overlap);

    if (generation.steps < 1)
        throw std::invalid_argument("--steps must be positive");
    if (generation.guidance < 1)
        throw std::invalid_argument("--cfg must be at least 1");
    if (generation.image_guidance < 1)
        throw std::invalid_argument("--image-cfg must be at least 1");
    if (generation.flow_shift <= 0)
        throw std::invalid_argument("--shift must be positive");
    if (generation.vae_tile_size < 0)
        throw std::invalid_argument("--vae-tile-size must be nonnegative");
    if (generation.vae_tile_overlap < 0 || generation.vae_tile_overlap >= 1)
        throw std::invalid_argument("--vae-tile-overlap must be in [0, 1)");
    return generation;
}

}
