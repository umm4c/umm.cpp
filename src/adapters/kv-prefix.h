#pragma once

#include "ggml-backend.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace umm {

// Owns contiguous host K/V tensors and their sequence metadata.
class kv_prefix {
public:
    std::vector<int32_t> tokens;
    std::vector<int32_t> positions;
    std::vector<ggml_tensor *> keys;
    std::vector<ggml_tensor *> values;

    static kv_prefix copy_to_host(std::vector<int32_t> tokens, std::vector<int32_t> positions,
                                  const std::vector<ggml_tensor *> & keys,
                                  const std::vector<ggml_tensor *> & values);

private:
    std::unique_ptr<ggml_context, decltype(&ggml_free)> descriptors{nullptr, ggml_free};
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> storage{nullptr, ggml_backend_buffer_free};
};

}
