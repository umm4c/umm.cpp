#pragma once

#include "json.hpp"
#ifdef GGML_USE_RPC
#include "ggml-rpc.h"
#endif

#include <map>
#include <string>
#include <vector>

namespace umm::profiling::detail {

// One window per request; its final response fences the ordered worker stream.
class rpc_batch {
#ifdef GGML_USE_RPC
    ggml_backend_t backend = nullptr;
    uint64_t sequence = 0, active = 0;
    std::vector<ggml_rpc_profile_record> records;
    uint32_t drain();
#endif

public:
    explicit rpc_batch(const std::string & name);
    ~rpc_batch();
    rpc_batch(const rpc_batch &) = delete;
    rpc_batch & operator=(const rpc_batch &) = delete;

    void start(size_t expected);
    void finish(nlohmann::json & trace, size_t trace_start, const std::string & device_id,
                std::map<std::string, std::vector<double>> & samples);
};

}
