#include "policy.h"

#include <stdexcept>

namespace umm {

understanding_schedule remote_910b_policy::apply(execution_options & options) const {
    using namespace policy_options;
    const auto endpoint = rpc_endpoint(options, "remote_910b");
    if (options.count("--local-device")) {
        throw std::invalid_argument("remote_910b cannot use a local device");
    }
    require(options, "--cann-q8-nz", "0");
    require(options, "--monitoring", "0");
    environment("GGML_RPC_NO_RDMA", "1");
    const auto selected = register_remote_cann(endpoint);
    environment("GGML_RPC_CANN_DEVICE", selected);
    require_execution_device(selected);
    configure_devices(options, selected);
    environment("GGML_REQUIRED_DEVICE", selected);
    return {};
}

}
