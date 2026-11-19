#include "profiling/profiler.h"
#include <fstream>
#include <filesystem>
#include <iostream>
#include <stdexcept>

int main(int argc, char ** argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        std::cout << "Usage: umm-profiler --request REQUEST.json --out NEW_DIRECTORY [--operators]\n"
                     "Real graph profiling: loads the model and replays the workload on each device.\n";
        return 0;
    }
    try {
        const bool operators = argc == 6 && std::string(argv[5]) == "--operators";
        if ((argc != 5 && !operators) || std::string(argv[1]) != "--request" || std::string(argv[3]) != "--out") {
            throw std::invalid_argument("Usage: umm-profiler --request REQUEST.json --out NEW_DIRECTORY [--operators]");
        }
        if (std::filesystem::exists(argv[4]))
            throw std::invalid_argument("Output directory already exists");
        std::ifstream stream(argv[2]);
        if (!stream)
            throw std::invalid_argument("Cannot open request file");
        nlohmann::json value;
        stream >> value;
        const auto result = umm::profiling::profile(umm::profiling::parse_request(value), operators);
        umm::profiling::write_result(result, argv[4]);
        std::cout << "Profile written to " << argv[4] << "; not usable for scheduling decisions.\n";
    } catch (const std::exception & error) {
        std::cerr << "umm-profiler: " << error.what() << '\n';
        return 1;
    }
}
