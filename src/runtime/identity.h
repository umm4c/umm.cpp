#pragma once

#include "json.hpp"
#include "options.h"

namespace umm::runtime {

nlohmann::json runtime_identity(const std::string & model, const options & args,
                                const std::string & secondary_backend = "");
void add_model_hashes(nlohmann::json & identity, const std::string & model, const options & args);

}
