#pragma once

#include <map>
#include <string>

namespace umm {

using execution_options = std::map<std::string, std::string>;
execution_options configure_execution_policy(execution_options options);
bool backend_uses_cann(const std::string & name);

}
