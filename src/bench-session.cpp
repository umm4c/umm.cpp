#include "image-io.h"
#include "json.hpp"
#include "session.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

#include <unistd.h>

namespace {

using json = nlohmann::json;
using clock_type = std::chrono::steady_clock;

std::map<std::string, std::string> parse_options(int argc, char ** argv) {
    std::map<std::string, std::string> options;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key.rfind("--", 0) != 0 || i + 1 == argc || !options.emplace(key, argv[++i]).second) {
            throw std::invalid_argument("Expected unique --option value pairs");
        }
    }
    static const std::map<std::string, bool> allowed = {
        {"--model", true}, {"--understanding-backend", true}, {"--vision-backend", true},
        {"--generation-backend", true}, {"--generation-max-vram", true},
        {"--steps", true}, {"--cfg", true}, {"--image-cfg", true}, {"--shift", true},
        {"--vae-tiling", true}, {"--vae-tile-size", true}, {"--vae-tile-overlap", true},
        {"--n-ctx", true}, {"--n-batch", true}, {"--n-ubatch", true},
    };
    for (const auto & item : options) {
        if (!allowed.count(item.first)) {
            throw std::invalid_argument("Unknown option: " + item.first);
        }
    }
    if (!options.count("--model")) {
        throw std::invalid_argument("--model is required");
    }
    return options;
}

std::string option(const std::map<std::string, std::string> & options,
                   const std::string & key, const std::string & fallback = "") {
    const auto found = options.find(key);
    return found == options.end() ? fallback : found->second;
}

template <typename T>
T required(const json & object, const char * key) {
    if (!object.is_object() || !object.contains(key)) {
        throw std::invalid_argument(std::string("Missing field: ") + key);
    }
    return object.at(key).get<T>();
}

double elapsed_ms(clock_type::time_point start) {
    return std::chrono::duration<double, std::milli>(clock_type::now() - start).count();
}

void send(FILE * protocol, const json & message) {
    const std::string line = message.dump() + "\n";
    if (std::fwrite(line.data(), 1, line.size(), protocol) != line.size() || std::fflush(protocol) != 0) {
        throw std::runtime_error("Could not write protocol response");
    }
}

json available(double value) { return value < 0 ? json(nullptr) : json(value); }
json available(int64_t value) { return value < 0 ? json(nullptr) : json(value); }

json timings(const umm::request_metrics & metrics, clock_type::time_point start) {
    return {{"end_to_end", elapsed_ms(start)},
            {"image_decode", available(metrics.image_decode_ms)},
            {"input_preprocess", available(metrics.input_preprocess_ms)},
            {"generation_model_load", available(metrics.generation_model_load_ms)},
            {"vision_model_load", available(metrics.vision_model_load_ms)},
            {"vision_encode", available(metrics.vision_encode_ms)},
            {"text_prefill", available(metrics.text_prefill_ms)},
            {"text_decode", available(metrics.text_decode_ms)},
            {"ttft", available(metrics.ttft_ms)},
            {"conditioning", available(metrics.conditioning_ms)},
            {"diffusion_steps", available(metrics.diffusion_steps_ms)},
            {"diffusion_step", metrics.diffusion_step_ms.empty() ? json(nullptr) : json(metrics.diffusion_step_ms)},
            {"vae_decode", available(metrics.vae_decode_ms)},
            {"image_write", available(metrics.image_write_ms)}};
}

json token_counts(const umm::request_metrics & metrics) {
    return {{"input", available(metrics.input_tokens)},
            {"vision", metrics.vision_encode_ms < 0 ? json(nullptr) : json(metrics.vision_tokens)},
            {"output", available(metrics.output_tokens)}};
}

json run_case(umm::session & session, const json & request,
              const std::map<std::string, std::string> & options) {
    const auto start = clock_type::now();
    umm::request_metrics metrics;
    std::string case_id;
    try {
        case_id = required<std::string>(request, "case_id");
        const auto task = required<std::string>(request, "task_class");
        const auto & content = request.at("content");
        const auto & params = request.at("params");
        if (!content.is_array() || !params.is_object()) {
            throw std::invalid_argument("content must be an array and params an object");
        }
        std::string prompt;
        std::string image_path;
        int image_count = 0;
        int image_position = -1;
        int content_position = 0;
        for (const auto & item : content) {
            const auto type = required<std::string>(item, "type");
            if (type == "text") {
                prompt += required<std::string>(item, "value");
            } else if (type == "image") {
                image_path = required<std::string>(item, "path");
                ++image_count;
                image_position = content_position;
            } else {
                throw std::invalid_argument("Unknown content type: " + type);
            }
            ++content_position;
        }
        if (prompt.empty()) {
            throw std::invalid_argument("Text prompt is empty");
        }
        json output;
        if (task == "mixed_understanding") {
            if (image_count != 1 || image_position != 0) {
                return {{"type", "result"}, {"case_id", case_id}, {"status", "error"},
                        {"error", {{"code", "unsupported_input_format"}, {"message", "Exactly one leading image is supported"}}},
                        {"timings_ms", timings(metrics, start)}, {"token_counts", token_counts(metrics)}};
            }
            const int max_tokens = required<int>(params, "max_new_tokens");
            if (max_tokens <= 0) throw std::invalid_argument("max_new_tokens must be positive");
            const auto decode_start = clock_type::now();
            const auto image = umm::cli::read_image(image_path);
            metrics.image_decode_ms = elapsed_ms(decode_start);
            output = {{"type", "text"}, {"text", session.understand(image, prompt, max_tokens, false, &metrics)}};
        } else if (task == "image_generation") {
            if (image_count != 0) {
                return {{"type", "result"}, {"case_id", case_id}, {"status", "error"},
                        {"error", {{"code", "unsupported_input_format"}, {"message", "Image generation expects text only"}}},
                        {"timings_ms", timings(metrics, start)}, {"token_counts", token_counts(metrics)}};
            }
            umm::image_options image_options;
            image_options.width = required<int>(params, "width");
            image_options.height = required<int>(params, "height");
            image_options.seed = required<int64_t>(params, "seed");
            if (image_options.width <= 0 || image_options.height <= 0) {
                throw std::invalid_argument("width and height must be positive");
            }
            image_options.steps = std::stoi(option(options, "--steps", "50"));
            image_options.guidance = std::stof(option(options, "--cfg", "4"));
            image_options.image_guidance = std::stof(option(options, "--image-cfg", "1.5"));
            image_options.flow_shift = std::stof(option(options, "--shift", "3"));
            image_options.vae_tiling = option(options, "--vae-tiling", "0") == "1";
            image_options.vae_tile_size = std::stoi(option(options, "--vae-tile-size", "0"));
            image_options.vae_tile_overlap = std::stof(option(options, "--vae-tile-overlap", "0.5"));
            const auto output_path = std::filesystem::path(required<std::string>(request, "output_path"));
            if (!output_path.is_absolute() || output_path.extension() != ".png") {
                throw std::invalid_argument("output_path must be an absolute PNG path");
            }
            const auto image = session.image(prompt, image_options, &metrics);
            std::filesystem::create_directories(output_path.parent_path());
            const auto write_start = clock_type::now();
            umm::cli::write_png(output_path, image);
            std::filesystem::permissions(output_path,
                std::filesystem::perms::group_read | std::filesystem::perms::others_read,
                std::filesystem::perm_options::add);
            metrics.image_write_ms = elapsed_ms(write_start);
            output = {{"type", "image"}, {"path", output_path.string()},
                      {"width", image.width}, {"height", image.height}};
        } else if (task == "multimodal_prefill") {
            return {{"type", "result"}, {"case_id", case_id}, {"status", "error"},
                    {"error", {{"code", "unsupported_input_format"}, {"message", "Multimodal prefill is not supported by this session interface"}}},
                    {"timings_ms", timings(metrics, start)}, {"token_counts", token_counts(metrics)}};
        } else {
            throw std::invalid_argument("Unknown task_class: " + task);
        }
        return {{"type", "result"}, {"case_id", case_id}, {"status", "ok"},
                {"output", output}, {"timings_ms", timings(metrics, start)},
                {"token_counts", token_counts(metrics)}};
    } catch (const std::exception & error) {
        return {{"type", "result"}, {"case_id", case_id}, {"status", "error"},
                {"error", {{"code", "request_failed"}, {"message", error.what()}}},
                {"timings_ms", timings(metrics, start)}, {"token_counts", token_counts(metrics)}};
    }
}

} // namespace

int main(int argc, char ** argv) {
    // Keep protocol responses separate from engine output on stdout.
    const int protocol_fd = dup(STDOUT_FILENO);
    if (protocol_fd < 0 || dup2(STDERR_FILENO, STDOUT_FILENO) < 0) return 1;
    FILE * protocol = fdopen(protocol_fd, "w");
    if (!protocol) return 1;
    try {
        const auto options = parse_options(argc, argv);
        const auto model = option(options, "--model");
        if (!std::filesystem::is_directory(model)) {
            throw std::invalid_argument("--model must name a model package directory");
        }
        const auto startup_start = clock_type::now();
        umm::session session(model, "", option(options, "--understanding-backend"),
                             option(options, "--generation-backend"),
                             option(options, "--generation-max-vram"), option(options, "--vision-backend"),
                             std::stoi(option(options, "--n-ctx", "0")),
                             std::stoi(option(options, "--n-batch", "0")),
                             std::stoi(option(options, "--n-ubatch", "0")));
        json ready = {{"type", "ready"}, {"protocol", "umm-session/v1"},
                      {"cold_start_ms", elapsed_ms(startup_start)},
                      {"model", model}, {"options", options}};
        if (const char * model_hash = std::getenv("UMM_MODEL_MANIFEST_SHA256")) {
            ready["model_manifest_sha256"] = model_hash;
        }
        send(protocol, ready);
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty()) continue;
            try {
                send(protocol, run_case(session, json::parse(line), options));
            } catch (const std::exception & error) {
                send(protocol, {{"type", "result"}, {"case_id", ""}, {"status", "error"},
                                {"error", {{"code", "invalid_json"}, {"message", error.what()}}}});
            }
        }
        std::fclose(protocol);
        return 0;
    } catch (const std::exception & error) {
        send(protocol, {{"type", "fatal"}, {"error", error.what()}});
        std::fclose(protocol);
        return 1;
    }
}
