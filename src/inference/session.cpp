#include "profile-context.h"
#include "session.h"
#include "scheduling/execution-policy.h"

#include "adapters/diffusion-adapter.h"
#include "adapters/vision-encoder.h"
#include "session-config.h"
#include "ggml-backend.h"
#include "adapters/llama-cpp-adapter.h"
#include "model-package.h"
#include "model-workflow.h"

#include <stdexcept>
#include <utility>

namespace umm {

namespace {

struct active_metric_scope {
    request_metrics *& slot;
    request_metrics * previous;
    active_metric_scope(request_metrics *& slot_, request_metrics * current) : slot(slot_), previous(slot_) {
        slot = current;
    }
    ~active_metric_scope() { slot = previous; }
};

void validate_edit_dimensions(const image_options & options) {
    const bool width_set = options.width != 0;
    const bool height_set = options.height != 0;
    if (width_set != height_set) {
        throw std::invalid_argument("Editing output width and height must be provided together");
    }
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
    std::string vision_model;
    std::unique_ptr<bagel_vision_encoder> vision_encoder;
    std::unique_ptr<diffusion_adapter> image_engine;
    request_metrics * active_metrics = nullptr;

    impl(model_package package_, const std::string & understanding_backend, const std::string & generation_backend_,
         const std::string & generation_max_vram_, const std::string & vision_backend_, int context_size,
         int batch_size, int microbatch_size, const understanding_schedule & schedule);

    diffusion_adapter & diffusion();
    bagel_vision_encoder & vision();
    void transfer_prefix(conditioning_slot slot);
    workflow_context workflow_context_for_request();
};

// Model setup ---------------------------------------------------------------

session::impl::impl(model_package package_, const std::string & understanding_backend,
                    const std::string & generation_backend_, const std::string & generation_max_vram_,
                    const std::string & vision_backend_, int context_size, int batch_size, int microbatch_size,
                    const understanding_schedule & schedule)
    : package(std::move(package_)),
      language_model(package.component("understanding"),
                     resolve_context_size(package.family, understanding_backend, generation_backend_, context_size), 99,
                     false, understanding_backend, batch_size, microbatch_size, schedule.secondary_backend),
      workflow(create_model_workflow(language_model.family())), generation_model(package.component("generation")),
      generation_backend(generation_backend_), generation_max_vram(generation_max_vram_),
      vision_backend(vision_backend_), vae_model(package.component("vae")), vision_model(package.component("vision")) {
    if (package.family != model_family::unknown && package.family != language_model.family()) {
        throw std::runtime_error("Model package architecture does not match its understanding GGUF");
    }
    if (!package.supports(language_model.family(), model_capability::text)) {
        throw std::runtime_error("Model package is missing its understanding component");
    }
    if (schedule.prepare) {
        auto prepared = schedule.prepare(language_model.scheduler(), language_model.weights(), package.component("understanding"));
        language_model.set_placement(std::move(prepared.place), std::move(prepared.recover), std::move(prepared.replan));
    } else {
        language_model.set_placement(schedule.place);
    }
}

// Image and diffusion plumbing ---------------------------------------------

diffusion_adapter & session::impl::diffusion() {
    if (!image_engine) {
        image_engine = std::make_unique<diffusion_adapter>(
            generation_model, vae_model,
            workflow->diffusion_uses_understanding_weights() ? package.component("understanding") : std::string{},
            generation_backend, generation_max_vram, active_metrics);
    }
    return *image_engine;
}

bagel_vision_encoder & session::impl::vision() {
    if (!model_supports(language_model.family(), model_capability::understand) || vision_model.empty()) {
        throw std::runtime_error("Image understanding requires a model package with vision weights");
    }
    if (!vision_encoder) {
        const auto load_start = metrics_clock::now();
        vision_encoder = std::make_unique<bagel_vision_encoder>(vision_model, vision_backend);
        if (active_metrics)
            active_metrics->vision_model_load_ms = metrics_elapsed_ms(load_start);
    }
    return *vision_encoder;
}

void session::impl::transfer_prefix(conditioning_slot slot) {
    auto & image_adapter = diffusion();
    image_adapter.set_prefix(slot, language_model.export_prefix());
}

workflow_context session::impl::workflow_context_for_request() {
    workflow_context context{language_model};
    context.metrics = active_metrics;
    context.diffusion = [this]() -> diffusion_adapter & { return diffusion(); };
    context.transfer_prefix = [this](conditioning_slot slot) { transfer_prefix(slot); };
    context.vision = [this]() -> bagel_vision_encoder & { return vision(); };
    return context;
}

// Public session API --------------------------------------------------------

session::session(const std::string & model, const std::string & generation_model,
                 const std::string & understanding_backend, const std::string & generation_backend,
                 const std::string & generation_max_vram, const std::string & vision_backend, int context_size,
                 int batch_size, int microbatch_size)
    : session(model, generation_model, understanding_backend, generation_backend, generation_max_vram, vision_backend,
              context_size, batch_size, microbatch_size, {}) {}

session::session(const std::string & model, const std::string & generation_model,
                 const std::string & understanding_backend, const std::string & generation_backend,
                 const std::string & generation_max_vram, const std::string & vision_backend, int context_size,
                 int batch_size, int microbatch_size, const understanding_schedule & schedule) {
    auto package = resolve_model(model, generation_model);
    ggml_backend_load_all();
    llama_backend_init();
    diffusion_adapter::install_log_callback();
    impl_ = std::make_unique<impl>(std::move(package), understanding_backend, generation_backend, generation_max_vram,
                                   vision_backend, context_size, batch_size, microbatch_size, schedule);
}

session::~session() = default;

bool session::supports(model_capability capability) const {
    return impl_->package.supports(impl_->language_model.family(), capability);
}

std::string session::text(const std::string & prompt, int max_tokens, request_metrics * metrics) {
    if (max_tokens < 1)
        throw std::invalid_argument("max_tokens must be positive");
    active_metric_scope binding(impl_->active_metrics, metrics);
    auto & engine = impl_->language_model;
    engine.reset();
    const auto tokens = engine.tokenize(impl_->workflow->text_prompt(prompt));
    if (tokens.size() + size_t(max_tokens) > engine.context_capacity())
        throw std::invalid_argument("Text and output reserve exceed context capacity");
    const auto start = metrics_clock::now();
    engine.append(tokens, true, metrics);
    if (metrics) {
        metrics->text_prefill_ms = metrics_elapsed_ms(start);
        metrics->input_tokens = engine.tokens().size();
        metrics->output_tokens = 0;
        metrics->decode_start = metrics_clock::now();
        metrics->finish_reason = "length";
    }
    std::string result;
    for (int i = 0; i < max_tokens; ++i) {
        const auto token = engine.greedy();
        if (engine.is_end(token)) {
            if (metrics)
                metrics->finish_reason = "stop";
            break;
        }
        const auto piece = engine.piece(token);
        if (piece == "<img>")
            throw std::runtime_error("The model requested image output during a text-only request");
        if (metrics) {
            if (metrics->ttft_ms < 0)
                metrics->ttft_ms = metrics_elapsed_ms(metrics->request_start);
            ++metrics->output_tokens;
        }
        result += piece;
        profiling::phase_scope decode_phase("understanding_decode");
        engine.append({token});
    }
    if (metrics) {
        engine.synchronize();
        metrics->text_decode_ms = metrics_elapsed_ms(metrics->decode_start);
    }
    return result;
}

std::string session::understand(const image_input & image, const std::string & prompt, int max_tokens, bool think,
                                request_metrics * metrics) {
    if (!supports(model_capability::understand)) {
        throw std::invalid_argument("Image understanding is not supported by the selected model");
    }

    active_metric_scope binding(impl_->active_metrics, metrics);
    const auto preprocess_start = metrics_clock::now();
    const auto prepared = impl_->workflow->prepare_image(image);
    if (metrics)
        metrics->input_preprocess_ms = metrics_elapsed_ms(preprocess_start);
    if (metrics)
        metrics->inference_start = metrics_clock::now();
    auto context = impl_->workflow_context_for_request();
    return impl_->workflow->understand(context, prepared, prompt, max_tokens, think);
}

std::string session::understand_sequence(const std::vector<understanding_segment> & content, int max_tokens,
                                         request_metrics * metrics) {
    if (impl_->language_model.family() != model_family::sensenova_u1 || !supports(model_capability::understand)) {
        throw std::invalid_argument("Ordered multimodal understanding requires SenseNova U1");
    }
    active_metric_scope binding(impl_->active_metrics, metrics);
    const auto preprocess_start = metrics_clock::now();
    std::vector<understanding_segment> prepared;
    prepared.reserve(content.size());
    for (const auto & segment : content) {
        if (const auto * image = std::get_if<image_input>(&segment)) {
            prepared.emplace_back(impl_->workflow->prepare_image(*image));
        } else {
            prepared.emplace_back(std::get<std::string>(segment));
        }
    }
    if (metrics) {
        metrics->input_preprocess_ms = metrics_elapsed_ms(preprocess_start);
        metrics->inference_start = metrics_clock::now();
    }
    auto context = impl_->workflow_context_for_request();
    return impl_->workflow->understand_sequence(context, prepared, max_tokens);
}

image_result session::image(const std::string & prompt, const image_options & options, request_metrics * metrics) {
    if (!supports(model_capability::image)) {
        throw std::invalid_argument("Image generation is not supported by the selected model");
    }

    active_metric_scope binding(impl_->active_metrics, metrics);
    if (metrics)
        metrics->inference_start = metrics_clock::now();
    auto context = impl_->workflow_context_for_request();
    return impl_->workflow->generate(context, prompt, options, nullptr);
}

image_result session::edit(const image_input & image, const std::string & prompt, const image_options & requested) {
    if (!supports(model_capability::edit)) {
        throw std::invalid_argument("Image editing is not supported by the selected model");
    }

    const auto prepared = impl_->workflow->prepare_image(image);
    auto options = requested;
    validate_edit_dimensions(requested);
    if (options.width == 0) {
        options.width = prepared.width;
        options.height = prepared.height;
    }

    auto context = impl_->workflow_context_for_request();
    return impl_->workflow->generate(context, prompt, options, &prepared);
}

interleave_result session::interleave(const std::string & prompt, const image_options & options, int max_text_tokens,
                                      int max_images) {
    if (impl_->language_model.family() != model_family::sensenova_u1) {
        throw std::invalid_argument("Interleaved generation is supported only by SenseNova U1");
    }
    auto context = impl_->workflow_context_for_request();
    return impl_->workflow->interleave(context, prompt, options, max_text_tokens, max_images);
}

}
