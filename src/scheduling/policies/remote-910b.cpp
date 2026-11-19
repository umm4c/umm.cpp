#include "policy.h"
#include "scheduling/backend-devices.h"
#include "scheduling/backend-environment.h"

#include <stdexcept>

namespace umm {

execution_configuration remote_910b_policy::apply(execution_options & options) const {
    using namespace policy_options;
    const auto endpoint = rpc_endpoint(options, "remote_910b");
    if (options.count("--local-device")) {
        throw std::invalid_argument("remote_910b cannot use a local device");
    }
    require(options, "--cann-q8-nz", "0");
    require(options, "--monitoring", "0");
    set_backend_environment("GGML_RPC_NO_RDMA", "1");
    const auto selected = register_remote_cann(endpoint);
    require_execution_device(selected);
    resolve_device_options(options, selected);
    return {{}, true, selected, selected};
}

}
