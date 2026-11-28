#include "runtime/service.h"
#include "runtime/jsonl.h"
#include <cstdio>
#include <iostream>
#include <unistd.h>
#include <utility>

int main(int argc, char ** argv) {
    using namespace umm::runtime;
    const int fd = dup(STDOUT_FILENO);
    if (fd < 0 || dup2(STDERR_FILENO, STDOUT_FILENO) < 0)
        return 1;
    FILE * output = fdopen(fd, "w");
    if (!output)
        return 1;
    auto send = [&](const json & value) {
        const auto line = value.dump() + "\n";
        if (fwrite(line.data(), 1, line.size(), output) != line.size() || fflush(output)) {
            throw std::runtime_error("Could not write protocol response");
        }
    };
    auto failure = [](const char * code, const char * message) {
        return json{{"type", "result"},
                    {"case_id", ""},
                    {"status", "error"},
                    {"error", {{"code", code}, {"message", message}}}};
    };
    try {
        const auto args = parse_options(argc, argv);
        service runtime(args);
        const auto metrics_listener = serve_metrics(runtime, args);
        send(runtime.info());
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty())
                continue;
            json request;
            try {
                request = json::parse(line);
            } catch (const json::parse_error & error) {
                send(failure("invalid_json", error.what()));
                continue;
            } catch (const json::out_of_range & error) {
                send(failure("invalid_request", error.what()));
                continue;
            }

            json response;
            try {
                auto pending = runtime.submit(std::move(request), request_source::jsonl, run_jsonl_case);
                if (!pending)
                    throw std::runtime_error("Session is busy");
                response = pending->get();
            } catch (const std::exception & error) {
                response = failure("request_failed", error.what());
            }
            send(response);
        }
        fclose(output);
        return 0;
    } catch (const std::exception & error) {
        send({{"type", "fatal"}, {"error", error.what()}});
        fclose(output);
        return 1;
    }
}
