#include "options.h"

#include <stdexcept>

namespace umm::runtime {

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

}
