#include "policy.h"

#include <stdexcept>

namespace umm {

understanding_schedule local_310p_policy::apply(execution_options & options) const {
    using namespace policy_options;
    if (options.count("--rpc-endpoint")) {
        throw std::invalid_argument("local_310p cannot use an RPC endpoint");
    }
    const auto selected = local_device(options);
    options.emplace("--cann-q8-nz", "1");
    require_execution_device(selected);
    configure_devices(options, selected);
    environment("GGML_REQUIRED_DEVICE", selected);
    return {};
}

}
