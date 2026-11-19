#pragma once

#include "session.h"

#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace umm::runtime {

enum class inference_task { text, single_image, mixed, generate_image };

struct image_reference {
    // Resolve the protocol's image reference only when this segment is decoded.
    std::function<image_input()> decode;
};

struct inference_request {
    inference_task task;
    std::vector<std::variant<std::string, image_reference>> content;
    std::string prompt;
    int max_tokens = 0;
    image_options generation;
};

struct inference_error {
    const char * code;
    std::string message;
};

struct inference_result {
    std::variant<std::string, image_result> output;
    request_metrics metrics;
    std::optional<inference_error> error;
};

class unsupported_input_format : public std::invalid_argument {
public:
    using std::invalid_argument::invalid_argument;
};

void validate_request(inference_request & request, const std::string & task);
void execute_request(session & engine, const inference_request & request, inference_result & result);

}
