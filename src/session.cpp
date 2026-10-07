#include "session.h"

#include "diffusion-bridge.h"
#include "ggml-backend.h"
#include "llama-cpp-adapter.h"
#include "model-package.h"
#include "model-workflow.h"
#include "stable-diffusion.h"
#include "sd-cpp-adapter.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <utility>

namespace umm {

namespace {

void log_stable_diffusion(enum sd_log_level_t, const char * message, void *) {
    std::fputs(message, stderr);
}

void validate_edit_dimensions(const image_options & options) {
    const bool width_set = options.width != 0;
    const bool height_set = options.height != 0;
    if (width_set != height_set) {
        throw std::invalid_argument("Editing output width and height must be provided together");
    }
}

bool has_image_prefix_data(const sd_kv_prefix_t & prefix) {
    return prefix.token_ids && prefix.token_count &&
           prefix.keys && prefix.values && prefix.layer_count;
}

int context_size_for_single_cann_u1(model_family family, const std::string & understanding_backend,
                                    const std::string & generation_backend) {
    if (family != model_family::sensenova_u1 || understanding_backend != generation_backend ||
        (understanding_backend.rfind("CANN", 0) != 0 && understanding_backend.rfind("cann", 0) != 0)) {
        return 0;
    }
    // Limit the scheduler reservation while both U1.5 GGUFs share one 310P.
    return 2048;
}

prefix_view stage_prefix_to_host(const std::vector<ggml_tensor *> & keys,
                                 const std::vector<ggml_tensor *> & values) {
    if (keys.empty() || keys.size() != values.size()) {
        throw std::invalid_argument("Invalid prefix tensors");
    }
    prefix_view result;
    result.descriptors.reset(ggml_init({2*keys.size()*ggml_tensor_overhead(), nullptr, true}));
    if (!result.descriptors) {
        throw std::runtime_error("Could not allocate prefix staging descriptors");
    }
    result.keys.reserve(keys.size());
    result.values.reserve(values.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        if (!keys[i] || !values[i] || !keys[i]->buffer || !values[i]->buffer ||
            !ggml_is_contiguous(keys[i]) || !ggml_is_contiguous(values[i])) {
            throw std::runtime_error("External prefix tensors must be contiguous backend tensors");
        }
        result.keys.push_back(ggml_dup_tensor(result.descriptors.get(), keys[i]));
        result.values.push_back(ggml_dup_tensor(result.descriptors.get(), values[i]));
    }
    result.storage.reset(ggml_backend_alloc_ctx_tensors_from_buft(
        result.descriptors.get(), ggml_backend_cpu_buffer_type()));
    if (!result.storage) {
        throw std::runtime_error("Could not allocate prefix staging buffer");
    }
    for (size_t i = 0; i < keys.size(); ++i) {
        for (const auto pair : {std::pair{keys[i], result.keys[i]},
                                std::pair{values[i], result.values[i]}}) {
            std::vector<uint8_t> bytes(ggml_nbytes(pair.first));
            ggml_backend_tensor_get(pair.first, bytes.data(), 0, bytes.size());
            ggml_backend_tensor_set(pair.second, bytes.data(), 0, bytes.size());
        }
    }
    return result;
}

std::vector<llama_pos> image_prefix_positions(const sd_kv_prefix_t & prefix) {
    if (prefix.positions) {
        return {prefix.positions, prefix.positions + prefix.token_count};
    }

    std::vector<llama_pos> positions(prefix.token_count);
    for (size_t i = 0; i < positions.size(); ++i) {
        positions[i] = llama_pos(i);
    }
    return positions;
}

} // namespace

// The implementation owns model state and supplies the callbacks used by each
// model workflow. Keeping this plumbing here leaves the workflow classes
// focused on model-specific prompt and generation logic.
struct session::impl {
    model_package package;
    llama_cpp_adapter language_model;
    std::unique_ptr<model_workflow> workflow;
    std::string generation_model;
    std::string generation_backend;
    std::string generation_max_vram;
    std::string vision_backend;
    std::string vae_model;
    std::string sd_vision_model;
    std::unique_ptr<sd_cpp_adapter> sd_vision_adapter;
    std::unique_ptr<sd_ctx_t, decltype(&free_sd_ctx)> image_engine{nullptr, free_sd_ctx};

    impl(model_package package_, const std::string & understanding_backend,
         const std::string & generation_backend_, const std::string & generation_max_vram_,
         const std::string & vision_backend_, int context_size, int batch_size, int microbatch_size);

    void load_image_engine();
    void append_image(const image_input & image);
    void append_native_image(const image_input & image);
    void append_vision_image(const image_input & image);
    void transfer_prefix(conditioning_slot slot);
    void append_latent_image(const image_input & image, int64_t seed);
    void import_image_prefix(const sd_kv_prefix_t & prefix);
    workflow_context workflow_context_for_request();
};

// Model setup ---------------------------------------------------------------

session::impl::impl(model_package package_, const std::string & understanding_backend,
                    const std::string & generation_backend_, const std::string & generation_max_vram_,
                    const std::string & vision_backend_, int context_size, int batch_size, int microbatch_size)
    : package(std::move(package_)),
      language_model(package.component("understanding"),
                     context_size ? context_size : context_size_for_single_cann_u1(
                         package.family, understanding_backend, generation_backend_),
                     99, false,
                     understanding_backend, batch_size, microbatch_size),
      workflow(create_model_workflow(language_model.family())),
      generation_model(package.component("generation")),
      generation_backend(generation_backend_),
      generation_max_vram(generation_max_vram_),
      vision_backend(vision_backend_),
      vae_model(package.component("vae")),
      sd_vision_model(package.component("vision")) {
    if (package.family != model_family::unknown && package.family != language_model.family()) {
        throw std::runtime_error("Model package architecture does not match its understanding GGUF");
    }
    if (!package.supports(language_model.family(), model_capability::text)) {
        throw std::runtime_error("Model package is missing its understanding component");
    }
}

// Image and diffusion plumbing ---------------------------------------------

void session::impl::load_image_engine() {
    if (image_engine) {
        return;
    }
    if (generation_model.empty()) {
        throw std::runtime_error("Provide the generation checkpoint for image output");
    }

    sd_ctx_params_t params;
    sd_ctx_params_init(&params);
    params.model_path = generation_model.c_str();
    params.vae_path = vae_model.empty() ? nullptr : vae_model.c_str();
    params.n_threads = 8;
    params.enable_mmap = true;
    const bool multi_device = generation_backend.find('&') != std::string::npos;
    const bool cann = generation_backend.find("CANN") != std::string::npos ||
                      generation_backend.find("cann") != std::string::npos;
    const bool disable_flash = cann || std::getenv("UMM_DISABLE_FLASH_ATTN") != nullptr;
    params.flash_attn = params.diffusion_flash_attn = !disable_flash;
    params.backend = generation_backend.empty() ? nullptr : generation_backend.c_str();
    params.params_backend = generation_backend.empty() || multi_device ? nullptr : generation_backend.c_str();
    params.split_mode = multi_device ? "layer" : nullptr;
    params.max_vram = generation_max_vram.empty() ? nullptr : generation_max_vram.c_str();
    params.external_kv_prefix = true;

    if (language_model.family() == model_family::bagel) {
        // sd.cpp loads the understanding GGUF under its private LLM prefix.
        params.llm_path = package.component("understanding").c_str();
    }

    image_engine.reset(new_sd_ctx(&params));
    if (!image_engine) {
        throw std::runtime_error("Could not load generation checkpoint");
    }
}

void session::impl::append_image(const image_input & image) {
    if (language_model.family() == model_family::sensenova_u1) {
        // U1 carries its image encoder in the generation/diffusion side.
        append_native_image(image);
        return;
    }
    // BAGEL supplies vision weights as a separate model-package component.
    append_vision_image(image);
}

void session::impl::append_native_image(const image_input & image) {
    load_image_engine();
    const auto stride = model_descriptor_for(model_family::sensenova_u1).image_stride;
    language_model.append_u1_image_embeddings(
        encode_u1_image_with_diffusion(image_engine.get(), image),
        image.width / stride,
        image.height / stride);
}

void session::impl::append_vision_image(const image_input & image) {
    if (!model_supports(language_model.family(), model_capability::understand) ||
        sd_vision_model.empty()) {
        throw std::runtime_error("Image understanding requires a model package with vision weights");
    }
    if (!sd_vision_adapter) {
        sd_vision_adapter = create_sd_cpp_adapter(language_model.family(), sd_vision_model, vision_backend);
    }
    language_model.append_bagel_image_embeddings(sd_vision_adapter->encode(image));
}

void session::impl::transfer_prefix(conditioning_slot slot) {
    const auto prefix = language_model.prefix();
    const auto & tokens = language_model.tokens();
    const sd_kv_prefix_t data{tokens.data(), tokens.size(), prefix.keys.data(), prefix.values.data(),
                              prefix.keys.size(), language_model.positions().data()};
    if (!set_kv_prefix(image_engine.get(), slot, data)) {
        throw std::runtime_error("Could not transfer understanding prefix to generation");
    }
}

void session::impl::append_latent_image(const image_input & image, int64_t seed) {
    transfer_prefix(conditioning_slot::conditional);

    const sd_image_t input{uint32_t(image.width), uint32_t(image.height), 3,
                           const_cast<uint8_t *>(image.rgb.data())};
    sd_kv_prefix_t prefix{};
    if (!encode_image_prefix(image_engine.get(), conditioning_slot::conditional, input, seed, prefix)) {
        throw std::runtime_error("Could not encode image into generation prefix");
    }
    if (!has_image_prefix_data(prefix)) {
        throw std::runtime_error("Image prefix is incomplete");
    }
    import_image_prefix(prefix);
}

void session::impl::import_image_prefix(const sd_kv_prefix_t & prefix) {
    const auto positions = image_prefix_positions(prefix);
    const std::vector<ggml_tensor *> keys(prefix.keys, prefix.keys + prefix.layer_count);
    const std::vector<ggml_tensor *> values(prefix.values, prefix.values + prefix.layer_count);
    auto staged = stage_prefix_to_host(keys, values);
    language_model.import_prefix({prefix.token_ids, prefix.token_ids + prefix.token_count},
                              positions,
                              staged.keys,
                              staged.values);
}

workflow_context session::impl::workflow_context_for_request() {
    workflow_context context{language_model};
    context.load_image_engine = [this] { load_image_engine(); };
    context.image_engine = [this] { return image_engine.get(); };
    context.transfer_prefix = [this](conditioning_slot slot) { transfer_prefix(slot); };
    context.append_latent_image = [this](const image_input & image, int64_t seed) {
        append_latent_image(image, seed);
    };
    context.append_vision_image = [this](const image_input & image) {
        append_image(image);
    };
    return context;
}

// Public session API --------------------------------------------------------

session::session(const std::string & model,
                 const std::string & generation_model,
                 const std::string & understanding_backend,
                 const std::string & generation_backend,
                 const std::string & generation_max_vram,
                 const std::string & vision_backend,
                 int context_size, int batch_size, int microbatch_size) {
    auto package = resolve_model(model, generation_model);
    ggml_backend_load_all();
    llama_backend_init();
    sd_set_log_callback(log_stable_diffusion, nullptr);
    impl_ = std::make_unique<impl>(std::move(package), understanding_backend, generation_backend,
                                  generation_max_vram, vision_backend,
                                  context_size, batch_size, microbatch_size);
}

session::~session() = default;

bool session::supports(model_capability capability) const {
    return impl_->package.supports(impl_->language_model.family(), capability);
}

std::string session::text(const std::string & prompt, int max_tokens) {
    if (max_tokens < 1) {
        throw std::invalid_argument("max_tokens must be positive");
    }

    auto & engine = impl_->language_model;
    engine.reset();
    engine.append(engine.tokenize(impl_->workflow->text_prompt(prompt)));

    std::string result;
    for (int i = 0; i < max_tokens; ++i) {
        const auto token = engine.greedy();
        if (engine.is_end(token)) {
            break;
        }
        const auto piece = engine.piece(token);
        if (piece == "<img>") {
            throw std::runtime_error("The model requested image output during a text-only request");
        }
        result += piece;
        engine.append({token});
    }
    return result;
}

std::string session::understand(const image_input & image, const std::string & prompt,
                                int max_tokens, bool think) {
    if (!supports(model_capability::understand)) {
        throw std::invalid_argument("Image understanding is not supported by the selected model");
    }

    const auto prepared = prepare_image_input(impl_->language_model.family(), image);
    auto context = impl_->workflow_context_for_request();
    return impl_->workflow->understand(context, prepared, prompt, max_tokens, think);
}

image_result session::image(const std::string & prompt, const image_options & options) {
    if (!supports(model_capability::image)) {
        throw std::invalid_argument("Image generation is not supported by the selected model");
    }

    auto context = impl_->workflow_context_for_request();
    return impl_->workflow->generate(context, prompt, options, nullptr);
}

image_result session::edit(const image_input & image, const std::string & prompt,
                           const image_options & requested) {
    if (!supports(model_capability::edit)) {
        throw std::invalid_argument("Image editing is not supported by the selected model");
    }

    const auto prepared = prepare_image_input(impl_->language_model.family(), image);
    auto options = requested;
    validate_edit_dimensions(requested);
    if (options.width == 0) {
        options.width = prepared.width;
        options.height = prepared.height;
    }

    auto context = impl_->workflow_context_for_request();
    return impl_->workflow->generate(context, prompt, options, &prepared);
}

interleave_result session::interleave(const std::string & prompt, const image_options & options,
                                     int max_text_tokens, int max_images) {
    if (impl_->language_model.family() != model_family::sensenova_u1) {
        throw std::invalid_argument("Interleaved generation is supported only by SenseNova U1");
    }
    auto context = impl_->workflow_context_for_request();
    return impl_->workflow->interleave(context, prompt, options, max_text_tokens, max_images);
}

}
