#include "runtime-support.h"
#include <cstdio>
#include <iostream>
#include <unistd.h>

int main(int argc, char ** argv) {
    using namespace umm::runtime;
    const int fd = dup(STDOUT_FILENO);
    if (fd < 0 || dup2(STDERR_FILENO, STDOUT_FILENO) < 0) return 1;
    FILE * output = fdopen(fd, "w");
    if (!output) return 1;
    auto send = [&](const json & value) {
        const auto line = value.dump() + "\n";
        if (fwrite(line.data(), 1, line.size(), output) != line.size() || fflush(output)) {
            throw std::runtime_error("Could not write protocol response");
        }
    };
    try {
        const auto args = parse_options(argc, argv);
        service runtime(args);
        const auto metrics_listener = serve_metrics(runtime, args);
        send(runtime.info());
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty()) continue;
            try {
                auto pending = runtime.submit(json::parse(line));
                if (!pending) throw std::runtime_error("Session is busy");
                send(pending->get());
            } catch (const std::exception & error) {
                send({{"type", "result"}, {"case_id", ""}, {"status", "error"},
                      {"error", {{"code", "invalid_json"}, {"message", error.what()}}}});
            }
        }
        fclose(output);
        return 0;
    } catch (const std::exception & error) {
        send({{"type", "fatal"}, {"error", error.what()}});
        fclose(output);
        return 1;
    }
}
