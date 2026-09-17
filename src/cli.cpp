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
        << "umm-cli --model PACKAGE --mode MODE --prompt TEXT\n"
           "        MODE: text|image|think-image|interleave|think-interleave|understand|think-understand|edit|think-edit\n"
           "        [--input image.png] [--output image.png] [--max-tokens 256] [--max-images 2]\n"
           "        [--understanding-backend CANN0] [--vision-backend CANN0]\n"
           "        [--generation-backend CANN0] [--generation-max-vram CANN0=40]\n"
           "        [--width MODEL_DEFAULT] [--height MODEL_DEFAULT] [--steps 50]\n"
           "        [--cfg 4] [--image-cfg 1.5] [--shift 3] [--seed 42]\n"
           "        [--vae-tiling 0|1] [--vae-tile-size 64] [--vae-tile-overlap 0.5]\n"
           "        (interleave defaults: 512x512, cfg 1)\n";
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
        "--model", "--mode", "--prompt", "--input", "--output", "--max-tokens", "--max-images",
        "--width", "--height", "--steps", "--cfg", "--image-cfg", "--shift", "--seed",
        "--vae-tiling", "--vae-tile-size", "--vae-tile-overlap",
        "--understanding-backend", "--vision-backend", "--generation-backend", "--generation-max-vram",
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
        {"image_guidance", options.image_guidance},
        {"flow_shift", options.flow_shift},
        {"seed", options.seed},
        {"vae_tiling", options.vae_tiling},
        {"vae_tile_size", options.vae_tile_size},
        {"vae_tile_overlap", options.vae_tile_overlap},
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
    const std::vector<std::string> modes = {
        "text", "image", "think-image", "interleave", "think-interleave",
        "understand", "think-understand", "edit", "think-edit",
    };
    if (std::find(modes.begin(), modes.end(), mode) == modes.end())
        throw std::invalid_argument("Unsupported mode: " + mode);

    umm::session session(model, "", value(args, "--understanding-backend"),
                         value(args, "--generation-backend"), value(args, "--generation-max-vram"),
                         value(args, "--vision-backend"));
    if (mode == "text") {
        std::cout << session.text(prompt, std::stoi(value(args, "--max-tokens", "256"))) << '\n';
        return 0;
    }

    const bool interleave_mode = mode == "interleave" || mode == "think-interleave";
    const bool understanding = mode == "understand" || mode == "think-understand";
    const bool editing = mode == "edit" || mode == "think-edit";
    if (understanding || editing) {
        const auto input = read_image(required(args, "--input"));
        if (understanding) {
            std::cout << session.understand(input, prompt,
                std::stoi(value(args, "--max-tokens", "256")), mode == "think-understand") << '\n';
            return 0;
        }

        umm::image_options options;
        options.width = std::stoi(value(args, "--width", "0"));
        options.height = std::stoi(value(args, "--height", "0"));
        options.steps = std::stoi(value(args, "--steps", "50"));
        options.guidance = std::stof(value(args, "--cfg", "4"));
        options.image_guidance = std::stof(value(args, "--image-cfg", "1.5"));
        options.flow_shift = std::stof(value(args, "--shift", "3"));
        options.seed = std::stoll(value(args, "--seed", "42"));
        options.vae_tiling = std::stoi(value(args, "--vae-tiling", "0")) != 0;
        options.vae_tile_size = std::stoi(value(args, "--vae-tile-size", "0"));
        options.vae_tile_overlap = std::stof(value(args, "--vae-tile-overlap", "0.5"));
        options.think = mode == "think-edit";
        options.max_think_tokens = std::stoi(value(args, "--max-tokens", "1024"));
        const auto image = session.edit(input, prompt, options);
        const auto output = std::filesystem::path(value(args, "--output", "image.png"));
        write_png(output, image);
        write_metadata(output, mode, prompt, options, image);
        std::cout << image.reasoning << '\n' << output << '\n';
        return 0;
    }

    umm::image_options options;
    options.width = std::stoi(value(args, "--width", interleave_mode ? "512" : "0"));
    options.height = std::stoi(value(args, "--height", interleave_mode ? "512" : "0"));
    options.steps = std::stoi(value(args, "--steps", "50"));
    options.guidance = std::stof(value(args, "--cfg", interleave_mode ? "1" : "4"));
    options.image_guidance = std::stof(value(args, "--image-cfg", "1.5"));
    options.flow_shift = std::stof(value(args, "--shift", "3"));
    options.seed = std::stoll(value(args, "--seed", "42"));
    options.vae_tiling = std::stoi(value(args, "--vae-tiling", "0")) != 0;
    options.vae_tile_size = std::stoi(value(args, "--vae-tile-size", "0"));
    options.vae_tile_overlap = std::stof(value(args, "--vae-tile-overlap", "0.5"));
    options.think = mode == "think-image" || mode == "think-interleave";
    options.max_think_tokens = std::stoi(value(args, "--max-tokens", "1024"));

    if (interleave_mode) {
        const auto result = session.interleave(prompt, options,
            std::stoi(value(args, "--max-tokens", "256")),
            std::stoi(value(args, "--max-images", "2")));
        const auto output = std::filesystem::path(value(args, "--output", "interleave"));
        for (size_t i = 0; i < result.images.size(); ++i) {
            write_png(output.string() + "_image_" + std::to_string(i) + ".png", result.images[i]);
        }
        std::ofstream text_file(output.string() + ".txt");
        if (!text_file) {
            throw std::runtime_error("Could not write interleaved text output");
        }
        text_file << result.text;
        std::cout << result.text << '\n';
        return 0;
    }

    const auto image = session.image(prompt, options);
    const auto output = std::filesystem::path(value(args, "--output", "image.png"));
    write_png(output, image);
    write_metadata(output, mode, prompt, options, image);
    std::cout << image.reasoning << '\n' << output << '\n';
    return 0;
}

}
