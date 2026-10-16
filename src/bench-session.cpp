#include "image-io.h"
#include "build-identity.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "json.hpp"
#include "model-package.h"
#include "session.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

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
        {"--u1-prefill-mode", true},
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
            {"image_encoder", available(metrics.image_encoder_ms)},
            {"image_prefill", available(metrics.image_prefill_ms)},
            {"text_prefill_compute", available(metrics.text_prefill_compute_ms)},
            {"text_prefill", available(metrics.text_prefill_ms)},
            {"text_decode", available(metrics.text_decode_ms)},
            {"ttft", available(metrics.ttft_ms)},
            {"conditioning", available(metrics.conditioning_ms)},
            {"diffusion_steps", available(metrics.diffusion_steps_ms)},
            {"diffusion_step", metrics.diffusion_step_ms.empty() ? json(nullptr) : json(metrics.diffusion_step_ms)},
            {"vae_decode", available(metrics.vae_decode_ms)},
            {"image_write", available(metrics.image_write_ms)}};
}

json prefill_metrics(const umm::request_metrics & metrics) {
    if (metrics.prefill_batches.empty()) return nullptr;
    json batches = json::array();
    int64_t outputs = 0;
    int64_t images = 0;
    for (const auto & batch : metrics.prefill_batches) {
        batches.push_back({{"kind", batch.kind}, {"tokens", batch.tokens},
                           {"cached_tokens", batch.cached_tokens}, {"outputs", batch.outputs},
                           {"elapsed_ms", batch.elapsed_ms}});
        outputs += batch.outputs;
        images += batch.kind == "image";
    }
    return {{"call_count", batches.size()}, {"logits_rows", outputs},
            {"attention_mode_switches", 2 * images},
            {"input_hash_algorithm", "fnv1a64-token-position"},
            {"input_hash", std::to_string(metrics.prefill_input_hash)}, {"batches", batches}};
}

json token_counts(const umm::request_metrics & metrics) {
    return {{"input", available(metrics.input_tokens)},
            {"vision", metrics.vision_encode_ms < 0 ? json(nullptr) : json(metrics.vision_tokens)},
            {"output", available(metrics.output_tokens)}};
}

json backend_devices(const std::string & requested) {
    if (requested.empty()) return nullptr;
    json devices = json::array();
    size_t offset = 0;
    while (offset < requested.size()) {
        const auto end = requested.find('&', offset);
        const auto name = requested.substr(offset, end == std::string::npos ? end : end - offset);
        auto * device = ggml_backend_dev_by_name(name.c_str());
        if (!device) throw std::runtime_error("Unknown backend device: " + name);
        devices.push_back({{"name", ggml_backend_dev_name(device)},
                           {"description", ggml_backend_dev_description(device)}});
        if (end == std::string::npos) break;
        offset = end + 1;
    }
    return devices;
}

json weight_precision(const std::string & path) {
    if (std::filesystem::path(path).extension() != ".gguf") {
        return {{"tensor_count", nullptr}, {"tensor_types", nullptr},
                {"file_bytes", std::filesystem::file_size(path)},
                {"precision_status", "unavailable"}};
    }
    std::unique_ptr<gguf_context, decltype(&gguf_free)> context(
        gguf_init_from_file(path.c_str(), {true, nullptr}), gguf_free);
    if (!context) throw std::runtime_error("Could not inspect GGUF precision: " + path);
    json counts = json::object();
    for (int64_t i = 0; i < gguf_get_n_tensors(context.get()); ++i) {
        const char * type = ggml_type_name(gguf_get_tensor_type(context.get(), i));
        if (!counts.contains(type)) counts[type] = 0;
        counts[type] = counts[type].get<int64_t>() + 1;
    }
    return {{"tensor_count", gguf_get_n_tensors(context.get())},
            {"tensor_types", counts},
            {"file_bytes", std::filesystem::file_size(path)}};
}

json runtime_identity(const std::string & model,
                      const std::map<std::string, std::string> & options) {
    const auto package = umm::resolve_model(model);
    json weights = json::object();
    for (const auto & component : package.components) {
        weights[component.first] = weight_precision(component.second);
    }
    json merged_lora = nullptr;
    if (std::filesystem::is_directory(model)) {
        std::ifstream manifest(std::filesystem::path(model) / "model.json");
        const auto data = json::parse(manifest);
        if (data.contains("merged_lora")) merged_lora = data.at("merged_lora");
    }
    const auto strict = std::getenv("GGML_SCHED_STRICT_ACCEL");
    const auto physical_device = std::getenv("UMM_PHYSICAL_DEVICE");
    return {{"architecture", package.architecture},
            {"weights", weights},
            {"merged_lora", merged_lora},
            {"hardware", {{"physical_device", physical_device ? json(physical_device) : json(nullptr)},
                           {"understanding", backend_devices(option(options, "--understanding-backend"))},
                           {"generation", backend_devices(option(options, "--generation-backend"))},
                           {"vision", backend_devices(option(options, "--vision-backend"))}}},
            {"policy", {{"mode", option(options, "--understanding-backend").empty() &&
                                  option(options, "--generation-backend").empty() ?
                                      "automatic_backend" : "explicit_backend"},
                        {"strict_accelerator", strict != nullptr},
                        {"generation_max_vram", option(options, "--generation-max-vram")},
                        {"u1_prefill_mode", option(options, "--u1-prefill-mode", "optimized")}}},
            {"build", {{"umm", UMM_BUILD_COMMIT},
                       {"llama_cpp", UMM_LLAMA_COMMIT},
                       {"stable_diffusion_cpp", UMM_SD_COMMIT},
                       {"dirty", {{"umm", UMM_BUILD_DIRTY},
                                  {"llama_cpp", UMM_LLAMA_DIRTY},
                                  {"stable_diffusion_cpp", UMM_SD_DIRTY}}}}}};
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
            if (image_count == 0) {
                throw std::invalid_argument("Multimodal prefill needs at least one image");
            }
            const int max_tokens = required<int>(params, "max_new_tokens");
            if (max_tokens <= 0) throw std::invalid_argument("max_new_tokens must be positive");
            std::vector<umm::understanding_segment> ordered;
            ordered.reserve(content.size());
            metrics.image_decode_ms = 0;
            for (const auto & item : content) {
                const auto type = required<std::string>(item, "type");
                if (type == "text") {
                    ordered.emplace_back(required<std::string>(item, "value"));
                } else {
                    const auto decode_start = clock_type::now();
                    ordered.emplace_back(umm::cli::read_image(required<std::string>(item, "path")));
                    metrics.image_decode_ms += elapsed_ms(decode_start);
                }
            }
            output = {{"type", "text"}, {"text", session.understand_sequence(ordered, max_tokens, &metrics)}};
        } else {
            throw std::invalid_argument("Unknown task_class: " + task);
        }
        return {{"type", "result"}, {"case_id", case_id}, {"status", "ok"},
                {"output", output}, {"timings_ms", timings(metrics, start)},
                {"token_counts", token_counts(metrics)}, {"prefill", prefill_metrics(metrics)}};
    } catch (const std::exception & error) {
        return {{"type", "result"}, {"case_id", case_id}, {"status", "error"},
                {"error", {{"code", "request_failed"}, {"message", error.what()}}},
                {"timings_ms", timings(metrics, start)}, {"token_counts", token_counts(metrics)},
                {"prefill", prefill_metrics(metrics)}};
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
                             std::stoi(option(options, "--n-ubatch", "0")),
                             option(options, "--u1-prefill-mode", "optimized"));
        json ready = {{"type", "ready"}, {"protocol", "umm-session/v1"},
                      {"cold_start_ms", elapsed_ms(startup_start)},
                      {"model", model}, {"options", options},
                      {"runtime_identity", runtime_identity(model, options)}};
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
