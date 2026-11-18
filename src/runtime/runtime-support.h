#pragma once
#include "json.hpp"
#include "session.h"
#include <future>
#include <map>
#include <memory>
#include <optional>
#include <string>

namespace umm::runtime {
using json = nlohmann::json;
using options = std::map<std::string, std::string>;
options parse_options(int argc, char ** argv);
std::string option(const options &, const std::string &, const std::string & fallback = "");
json runtime_identity(const std::string &, const options &);
json run_case(session &, const json &, const options &, bool http = false);
json transfer_snapshot();

class system_monitor {
public:
    explicit system_monitor(const options &);
    ~system_monitor();
    void begin();
    json end();
    json snapshot() const;
private:
    struct impl;
    std::unique_ptr<impl> state;
};

class service {
public:
    explicit service(options);
    ~service();
    const json & info() const;
    std::optional<std::future<json>> submit(json request, bool http = false);
    std::string metrics() const;
    void reject(const std::string & reason);
    void http_completed(const std::string & id, const std::string & task, const std::string & status, double ms);
private:
    struct impl;
    std::unique_ptr<impl> state;
};
std::shared_ptr<void> serve_metrics(service &, const options &);
int serve_http(service &, const options &);
}
