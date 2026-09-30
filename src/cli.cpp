#include "cli.h"

#include "image-io.h"
#include "json.hpp"
#include "session.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace umm::cli {

namespace {

using option_map = std::map<std::string, std::string>;

void print_help() {
    std::cout
        << "umm-cli --model PACKAGE --mode text|image|think-image --prompt TEXT\n"
           "        [--output image.png] [--max-tokens 256]\n"
           "        [--understanding-backend CANN0] [--generation-backend CANN0]\n"
           "        [--generation-max-vram CANN0=40]\n"
           "        [--width 2048] [--height 2048] [--steps 50] [--cfg 4] [--shift 3] [--seed 42]\n";
}

option_map parse_arguments(int argc, char ** argv) {
    option_map args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help") {
            print_help();
            return {};
        }
        if (key.rfind("--", 0) != 0 || i + 1 == argc ||
            !args.emplace(key, argv[++i]).second) {
            throw std::invalid_argument("Expected unique --option value pairs; use --help");
        }
    }

    const std::vector<std::string> known = {
        "--model", "--mode", "--prompt", "--output", "--max-tokens",
        "--width", "--height", "--steps", "--cfg", "--shift", "--seed",
        "--understanding-backend", "--generation-backend", "--generation-max-vram",
    };
    for (const auto & entry : args) {
        if (std::find(known.begin(), known.end(), entry.first) == known.end()) {
            throw std::invalid_argument("Unknown option: " + entry.first);
        }
    }
    return args;
}

std::string value(const option_map & args, const std::string & key,
                  const std::string & fallback = {}) {
    const auto found = args.find(key);
    return found == args.end() ? fallback : found->second;
}

const std::string & required(const option_map & args, const std::string & key) {
    const auto found = args.find(key);
    if (found == args.end()) {
        throw std::invalid_argument(key + " is required; use --help");
    }
    return found->second;
}

void write_metadata(const std::filesystem::path & output,
                    const std::string & mode,
                    const std::string & prompt,
                    const umm::image_options & options,
                    const umm::image_result & image) {
    const nlohmann::json metadata = {
        {"mode", mode},
        {"prompt", prompt},
        {"width", options.width},
        {"height", options.height},
        {"steps", options.steps},
        {"guidance", options.guidance},
        {"flow_shift", options.flow_shift},
        {"seed", options.seed},
        {"reasoning", image.reasoning},
        {"reasoning_tokens", image.reasoning_tokens},
        {"prefix_tokens", image.prefix_tokens},
    };

    std::ofstream file(output.string() + ".json");
    if (!file) {
        throw std::runtime_error("Could not write metadata: " + output.string() + ".json");
    }
    file << metadata.dump(2) << '\n';
}

}

int run(int argc, char ** argv) {
    const auto args = parse_arguments(argc, argv);
    if (args.empty() && argc > 1 && std::string(argv[1]) == "--help") {
        return 0;
    }

    const auto & model = required(args, "--model");
    const auto & prompt = required(args, "--prompt");
    if (!std::filesystem::is_directory(model)) {
        throw std::invalid_argument("--model must name a model package directory");
    }

    const auto mode = value(args, "--mode", "text");
    if (mode != "text" && mode != "image" && mode != "think-image") {
        throw std::invalid_argument("Mode must be text, image, or think-image");
    }

    umm::session session(model, "", value(args, "--understanding-backend"),
                         value(args, "--generation-backend"), value(args, "--generation-max-vram"));
    if (mode == "text") {
        std::cout << session.text(prompt, std::stoi(value(args, "--max-tokens", "256"))) << '\n';
        return 0;
    }

    umm::image_options options;
    options.width = std::stoi(value(args, "--width", "2048"));
    options.height = std::stoi(value(args, "--height", "2048"));
    options.steps = std::stoi(value(args, "--steps", "50"));
    options.guidance = std::stof(value(args, "--cfg", "4"));
    options.flow_shift = std::stof(value(args, "--shift", "3"));
    options.seed = std::stoll(value(args, "--seed", "42"));
    options.think = mode == "think-image";
    options.max_think_tokens = std::stoi(value(args, "--max-tokens", "1024"));

    const auto image = session.image(prompt, options);
    const auto output = std::filesystem::path(value(args, "--output", "image.png"));
    write_png(output, image);
    write_metadata(output, mode, prompt, options, image);
    std::cout << image.reasoning << '\n' << output << '\n';
    return 0;
}

}
