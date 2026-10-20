#include "runtime-support.h"
#include <cstdio>
#include <iostream>
#include <unistd.h>
int main(int argc, char ** argv) {
    if (dup2(STDERR_FILENO, STDOUT_FILENO) < 0) return 1;
    try {
        auto args = umm::runtime::parse_options(argc, argv);
        if (!args.count("--hash-model")) args["--hash-model"] = "1";
        umm::runtime::service runtime(args);
        return umm::runtime::serve_http(runtime, args);
    } catch (const std::exception & error) {
        std::cerr << "umm-server: " << error.what() << '\n';
        return 1;
    }
}
