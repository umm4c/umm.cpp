#include "backend-environment.h"
#include "execution-policy.h"

#include <cstdlib>
#include <stdexcept>

namespace umm {

void set_backend_environment(const char * name, const std::string & value) {
    if (setenv(name, value.c_str(), 1))
        throw std::runtime_error(std::string("Cannot set ") + name);
}

void set_cann_q8_nz(const std::string & value) {
    if (value != "0" && value != "1")
        throw std::invalid_argument("cann-q8-nz must be 0 or 1");
    if (setenv("GGML_CANN_Q8_NZ", value.c_str(), 1))
        throw std::runtime_error("Could not configure CANN Q8 layout");
}

void apply_execution_environment(const execution_options & options, const execution_configuration & configuration) {
    if (const auto layout = options.find("--cann-q8-nz"); layout != options.end())
        set_cann_q8_nz(layout->second);
    if (configuration.strict_acceleration)
        set_backend_environment("GGML_SCHED_STRICT_ACCEL", "1");
    if (!configuration.rpc_cann_device.empty())
        set_backend_environment("GGML_RPC_CANN_DEVICE", configuration.rpc_cann_device);
    if (!configuration.required_device.empty())
        set_backend_environment("GGML_REQUIRED_DEVICE", configuration.required_device);
}

void require_execution_device(const std::string & selected) {
    if (const char * previous = std::getenv("GGML_REQUIRED_DEVICE")) {
        if (selected != previous)
            throw std::invalid_argument("A process cannot change its static execution device");
    }
}

}
