#include "identity.h"
#include "build-identity.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "inference/model-package.h"

extern "C" {
#include "sha256.h"
}

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace umm::runtime {
namespace {

using json = nlohmann::json;
using clock = std::chrono::steady_clock;

json backend_devices(const std::string & requested) {
    if (requested.empty())
        return nullptr;
    json devices = json::array();
    size_t offset = 0;
    while (offset < requested.size()) {
        const auto end = requested.find('&', offset);
        const auto name = requested.substr(offset, end == std::string::npos ? end : end - offset);
        auto * device = ggml_backend_dev_by_name(name.c_str());
        if (!device)
            throw std::runtime_error("Unknown backend device: " + name);
        devices.push_back(
            {{"name", ggml_backend_dev_name(device)}, {"description", ggml_backend_dev_description(device)}});
        if (end == std::string::npos)
            break;
        offset = end + 1;
    }
    return devices;
}

json weight_precision(const std::string & path) {
    if (std::filesystem::path(path).extension() != ".gguf") {
        return {{"tensor_count", nullptr},
                {"tensor_types", nullptr},
                {"file_bytes", std::filesystem::file_size(path)},
                {"precision_status", "unavailable"}};
    }
    std::unique_ptr<gguf_context, decltype(&gguf_free)> context(gguf_init_from_file(path.c_str(), {true, nullptr}),
                                                                gguf_free);
    if (!context)
        throw std::runtime_error("Could not inspect GGUF precision: " + path);
    json counts = json::object();
    for (int64_t i = 0; i < gguf_get_n_tensors(context.get()); ++i) {
        const char * type = ggml_type_name(gguf_get_tensor_type(context.get(), i));
        if (!counts.contains(type))
            counts[type] = 0;
        counts[type] = counts[type].get<int64_t>() + 1;
    }
    return {{"tensor_count", gguf_get_n_tensors(context.get())},
            {"tensor_types", counts},
            {"file_bytes", std::filesystem::file_size(path)}};
}

std::string file_hash(const std::string & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("Cannot hash model component");
    sha256_t state;
    sha256_init(&state);
    std::array<unsigned char, 1024 * 1024> buffer;
    while (input) {
        input.read(reinterpret_cast<char *>(buffer.data()), buffer.size());
        sha256_update(&state, buffer.data(), size_t(input.gcount()));
    }
    if (!input.eof())
        throw std::runtime_error("Model component hash read failed");
    unsigned char bytes[32];
    sha256_final(&state, bytes);
    std::ostringstream result;
    for (auto ch : bytes)
        result << std::hex << std::setw(2) << std::setfill('0') << int(ch);
    return result.str();
}

} // namespace

json runtime_identity(const std::string & model, const std::map<std::string, std::string> & options,
                      const std::string & secondary_backend) {
    const auto package = umm::resolve_model(model);
    json weights = json::object();
    for (const auto & component : package.components) {
        weights[component.first] = weight_precision(component.second);
    }
    json merged_lora = nullptr;
    if (std::filesystem::is_directory(model)) {
        std::ifstream manifest(std::filesystem::path(model) / "model.json");
        const auto data = json::parse(manifest);
        if (data.contains("merged_lora"))
            merged_lora = data.at("merged_lora");
    }
    const auto strict = std::getenv("GGML_SCHED_STRICT_ACCEL");
    const auto physical_device = std::getenv("UMM_PHYSICAL_DEVICE");
    const auto q8_env = std::getenv("GGML_CANN_Q8_NZ");
    std::string q8_layout = q8_env ? q8_env : "off";
    std::transform(q8_layout.begin(), q8_layout.end(), q8_layout.begin(),
                   [](unsigned char ch) { return std::tolower(ch); });
    const bool q8_nz = q8_layout == "on" || q8_layout == "1" || q8_layout == "yes" || q8_layout == "y" ||
                       q8_layout == "enable" || q8_layout == "true";
    const auto policy = option(options, "--execution-policy");
    const bool mixed = !secondary_backend.empty();
    const bool rpc = policy == "remote_910b";
    const bool explicit_devices =
        !option(options, "--understanding-backend").empty() || !option(options, "--generation-backend").empty();
    const char * policy_mode =
        mixed ? "graph_placement" : (explicit_devices ? "explicit_backend" : "automatic_backend");
    auto understanding = backend_devices(option(options, "--understanding-backend"));
    if (mixed) {
        for (const auto & device : backend_devices(secondary_backend)) {
            understanding.push_back(device);
        }
    }
    return {{"architecture", package.architecture},
            {"weights", weights},
            {"merged_lora", merged_lora},
            {"hardware",
             {{"physical_device", physical_device ? json(physical_device) : json(nullptr)},
              {"understanding", understanding},
              {"generation", backend_devices(option(options, "--generation-backend"))},
              {"vision", backend_devices(option(options, "--vision-backend"))}}},
            {"transport",
             {{"kind", mixed ? "mixed" : (rpc ? "rpc" : "local")},
              {"endpoint", option(options, "--rpc-endpoint")},
              {"remote_metrics", (mixed || rpc) ? "collected_on_worker" : "not_applicable"}}},
            {"policy",
             {{"name", policy},
              {"static_policy", mixed ? "" : policy},
              {"mode", policy_mode},
              {"strict_accelerator", strict != nullptr},
              {"generation_max_vram", option(options, "--generation-max-vram")},
              {"u1_prefill_mode", package.family == model_family::sensenova_u1 ? json("mixed4") : json(nullptr)},
              {"u1_v_cache_layout", package.family == model_family::sensenova_u1 ? json("contiguous") : json(nullptr)},
              {"cann_q8_nz", q8_nz}}},
            {"build",
             {{"umm", UMM_BUILD_COMMIT},
              {"llama_cpp", UMM_LLAMA_COMMIT},
              {"stable_diffusion_cpp", UMM_SD_COMMIT},
              {"dirty",
               {{"umm", UMM_BUILD_DIRTY}, {"llama_cpp", UMM_LLAMA_DIRTY}, {"stable_diffusion_cpp", UMM_SD_DIRTY}}}}}};
}

void add_model_hashes(nlohmann::json & identity, const std::string & model, const options & options) {
    identity["model_manifest_sha256"] = file_hash((std::filesystem::path(model) / "model.json").string());
    if (const char * hash = std::getenv("UMM_MODEL_MANIFEST_SHA256")) {
        if (identity["model_manifest_sha256"] != hash)
            throw std::runtime_error("Model manifest hash mismatch");
    }
    if (option(options, "--hash-model", "0") == "1") {
        const auto hash_start = clock::now();
        identity["model_component_sha256"] = json::object();
        for (const auto & entry : resolve_model(model).components)
            identity["model_component_sha256"][entry.first] = file_hash(entry.second);
        identity["startup_hash_ms"] = std::chrono::duration<double, std::milli>(clock::now() - hash_start).count();
    }
}

} // namespace umm::runtime
