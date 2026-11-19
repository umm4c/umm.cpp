#pragma once

#include <string>

namespace umm {

void set_backend_environment(const char * name, const std::string & value);
void set_cann_q8_nz(const std::string & value);
void require_execution_device(const std::string & selected);

}
