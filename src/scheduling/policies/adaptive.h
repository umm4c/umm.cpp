#pragma once
#include "policy.h"

namespace umm {
class adaptive_policy final : public execution_policy {
public:
    execution_configuration apply(execution_options & options) const override;
};
}
