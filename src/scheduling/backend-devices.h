#pragma once

#include <string>

namespace umm {

std::string register_remote_cann(const std::string & endpoint);
bool backend_uses_cann(const std::string & name);

}
