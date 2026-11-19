#pragma once

#include "json.hpp"
#include "options.h"
#include "session.h"

#include <stdexcept>
#include <string>

namespace umm::runtime {

using json = nlohmann::json;

template <typename T> T required(const json & object, const char * key) {
    if (!object.is_object() || !object.contains(key)) {
        throw std::invalid_argument(std::string("Missing field: ") + key);
    }
    return object.at(key).get<T>();
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
