#pragma once
#include "case-protocol.h"
#include <future>
#include <map>
#include <memory>
#include <optional>
#include <string>

namespace umm::runtime {
enum class request_source { jsonl, http };
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
    std::optional<std::future<json>> submit(json request, request_source source, case_runner run);
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
