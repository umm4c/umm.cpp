#pragma once

#include "json.hpp"
#include "options.h"
#include "session.h"

#include <stdexcept>
#include <string>

namespace umm::runtime {

using json = nlohmann::json;

inline const json & required_field(const json & object, const char * key) {
    if (!object.is_object()) {
        throw std::invalid_argument("Expected a JSON object");
    }
    const auto field = object.find(key);
    if (field == object.end()) {
        throw std::invalid_argument(std::string("Missing field: ") + key);
    }
    return *field;
}

template <typename T> T required(const json & object, const char * key) {
    const auto & field = required_field(object, key);
    try {
        return field.get<T>();
    } catch (const json::exception & error) {
        throw std::invalid_argument(std::string("Invalid type for field ") + key + ": " + error.what());
    }
}

struct case_format {
    image_input (*decode)(const json & image);
    void (*configure_generation)(const json & request, const json & params, image_options & generation);
    json (*deliver_image)(const json & request, const image_result & image, request_metrics & metrics,
                          double & output_encode_ms);
};

using case_runner = json (*)(session &, const json &, const options &);

json run_protocol_case(session &, const json & request, const options &, const case_format & format);

// HTTP and Profiler both return inline PNGs and allow per-request generation overrides.
json run_inline_case(session &, const json & request, const options &);

}
