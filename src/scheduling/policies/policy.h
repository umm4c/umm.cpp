#pragma once

#include "scheduling/execution-policy.h"

namespace umm {

class execution_policy {
public:
    virtual ~execution_policy() = default;
    virtual understanding_schedule apply(execution_options & options) const = 0;
};

class local_310p_policy final : public execution_policy {
public:
    understanding_schedule apply(execution_options & options) const override;
};

class remote_910b_policy final : public execution_policy {
public:
    understanding_schedule apply(execution_options & options) const override;
};

namespace policy_options {
std::string get(const execution_options & options, const std::string & key, const std::string & fallback = "");
void environment(const char * name, const std::string & value);
void require(execution_options & options, const std::string & key, const std::string & value);
std::string local_device(const execution_options & options);
std::string rpc_endpoint(const execution_options & options, const char * policy);
void configure_devices(execution_options & options, const std::string & primary);
void require_execution_device(const std::string & selected);
}

}
