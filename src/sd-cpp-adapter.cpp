#include "sd-cpp-adapter.h"

#include "stable-diffusion.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "clip.h"
#include "mtmd-image.h"
#include "clip-impl.h"

namespace umm {

namespace {

bool is_cann_backend(const std::string & backend) {
    return backend.size() >= 4 &&
           (backend.substr(0, 4) == "CANN" || backend.substr(0, 4) == "cann");
}

// Common image validation and resizing --------------------------------------

void validate_rgb_image(const image_input & image) {
    if (image.width <= 0 || image.height <= 0 || image.rgb.size() != size_t(image.width) * image.height * 3) {
        throw std::invalid_argument("Expected a nonempty RGB image");
    }
}

int round_to_factor(int value, int factor) {
    return std::max(factor, ((value + factor / 2) / factor) * factor);
}

image_input resize_rgb_image(const image_input & image, int width, int height) {
    if (image.width == width && image.height == height) {
        return image;
    }
    image_input result{width, height, std::vector<uint8_t>(size_t(width) * height * 3)};
    for (int y = 0; y < height; ++y) {
        const float source_y = height == 1 ? 0.f :
            static_cast<float>(y) * static_cast<float>(image.height - 1) / static_cast<float>(height - 1);
        const int y0 = static_cast<int>(std::floor(source_y));
        const int y1 = std::min(y0 + 1, image.height - 1);
        const float fy = source_y - static_cast<float>(y0);
        for (int x = 0; x < width; ++x) {
            const float source_x = width == 1 ? 0.f :
                static_cast<float>(x) * static_cast<float>(image.width - 1) / static_cast<float>(width - 1);
            const int x0 = static_cast<int>(std::floor(source_x));
            const int x1 = std::min(x0 + 1, image.width - 1);
            const float fx = source_x - static_cast<float>(x0);
            for (int c = 0; c < 3; ++c) {
                const auto sample = [&](int sx, int sy) {
                    return static_cast<float>(image.rgb[(size_t(sy) * image.width + sx) * 3 + c]);
                };
                const float top = sample(x0, y0) * (1.f - fx) + sample(x1, y0) * fx;
                const float bottom = sample(x0, y1) * (1.f - fx) + sample(x1, y1) * fx;
                const float value = top * (1.f - fy) + bottom * fy;
                result.rgb[(size_t(y) * width + x) * 3 + c] =
                    static_cast<uint8_t>(std::clamp(std::lround(value), 0L, 255L));
            }
        }
    }
    return result;
}

image_input prepare_u1_input(const image_input & image) {
    validate_rgb_image(image);
    constexpr int factor = 32;
    constexpr int64_t min_pixels = 256LL * 256LL;
    constexpr int64_t max_pixels = 2048LL * 2048LL;
    if (std::max(image.width, image.height) / static_cast<double>(std::min(image.width, image.height)) > 200.0) {
        throw std::invalid_argument("U1 image aspect ratio must be below 200");
    }

    int height = round_to_factor(image.height, factor);
    int width = round_to_factor(image.width, factor);
    const int64_t rounded_area = int64_t(height) * width;
    const int64_t original_area = int64_t(image.height) * image.width;
    if (rounded_area > max_pixels) {
        const double scale = std::sqrt(static_cast<double>(original_area) / max_pixels);
        height = std::max(factor, static_cast<int>(std::floor(image.height / scale / factor)) * factor);
        width = std::max(factor, static_cast<int>(std::floor(image.width / scale / factor)) * factor);
    } else if (rounded_area < min_pixels) {
        const double scale = std::sqrt(static_cast<double>(min_pixels) / original_area);
        height = std::max(factor, static_cast<int>(std::ceil(image.height * scale / factor)) * factor);
        width = std::max(factor, static_cast<int>(std::ceil(image.width * scale / factor)) * factor);
    }
    return resize_rgb_image(image, width, height);
}

// BAGEL input preparation and CLIP/MTMD projection -------------------------

image_input prepare_bagel_input(const image_input & image) {
    validate_rgb_image(image);
    clip_image_u8 input;
    input.set_size({image.width, image.height}, false);
    input.cpy_buf(image.rgb);
    const auto prepared = mtmd_image_preprocessor_bagel::prepare_image(input);
    return {prepared.get_size().width, prepared.get_size().height, prepared.get_ro_buf()};
}

class bagel_sd_cpp_adapter final : public sd_cpp_adapter {
public:
    bagel_sd_cpp_adapter(const std::string & path, const std::string & backend) : context_(nullptr, clip_free) {
        clip_context_params params{};
        params.use_gpu = true;
        if (!backend.empty()) {
            params.device = ggml_backend_dev_by_name(backend.c_str());
            if (!params.device) {
                throw std::runtime_error("Vision backend was not found: " + backend);
            }
        }
        params.flash_attn_type = is_cann_backend(backend)
            ? CLIP_FLASH_ATTN_TYPE_DISABLED
            : CLIP_FLASH_ATTN_TYPE_ENABLED;
        const auto loaded = clip_init(path.c_str(), params);
        context_.reset(loaded.ctx_v);
        clip_free(loaded.ctx_a);
        clip_free(loaded.ctx_gen_a);
        if (!context_ || clip_n_mmproj_embd(context_.get()) != 3584) {
            throw std::runtime_error("Could not load BAGEL vision projector");
        }
    }

    std::vector<float> encode(const image_input & image) override {
        validate_rgb_image(image);
        clip_image_u8 input;
        input.set_size({image.width, image.height}, false);
        input.cpy_buf(image.rgb);
        const auto images = mtmd_image_preprocessor_bagel(context_.get()).preprocess(input);
        if (images.entries.size() != 1) throw std::runtime_error("Expected one image embedding group");
        std::vector<float> embeddings(size_t(clip_n_output_tokens(context_.get(), &images.entries.front())) *
                                      clip_n_mmproj_embd(context_.get()));
        if (!clip_image_encode(context_.get(), 8, &images.entries.front(), embeddings)) {
            throw std::runtime_error("Image encoding failed");
        }
        return embeddings;
    }

private:
    std::unique_ptr<clip_ctx, void (*)(clip_ctx *)> context_;
};

}

// Public adapter/factory and model-specific preprocessing -------------------

sd_cpp_adapter::~sd_cpp_adapter() = default;

std::unique_ptr<sd_cpp_adapter> create_sd_cpp_adapter(model_family family, const std::string & path,
                                                       const std::string & backend) {
    if (family == model_family::bagel) {
        return std::make_unique<bagel_sd_cpp_adapter>(path, backend);
    }
    throw std::runtime_error("The selected model has no sd.cpp vision adapter");
}

image_input prepare_image_input(model_family family, const image_input & image) {
    if (family == model_family::sensenova_u1) {
        return prepare_u1_input(image);
    }
    if (family == model_family::bagel) {
        return prepare_bagel_input(image);
    }
    throw std::invalid_argument("Editing is not supported by the selected model");
}

std::vector<float> encode_u1_image_with_diffusion(sd_ctx_t * context, const image_input & image) {
    if (context == nullptr) {
        throw std::invalid_argument("A loaded U1 image engine is required");
    }
    validate_rgb_image(image);
    if (image.width % 32 || image.height % 32) {
        throw std::invalid_argument("U1 image dimensions must be divisible by 32 after preprocessing");
    }
    sd_image_t input{static_cast<uint32_t>(image.width), static_cast<uint32_t>(image.height), 3,
                     const_cast<uint8_t *>(image.rgb.data())};
    float * values = nullptr;
    size_t token_count = 0;
    size_t embedding_dim = 0;
    if (!sd_encode_sensenova_u1_image(context, &input, &values, &token_count, &embedding_dim) ||
        values == nullptr || token_count == 0 || embedding_dim == 0) {
        throw std::runtime_error("SenseNova U1 image encoding failed");
    }
    std::vector<float> result(values, values + token_count * embedding_dim);
    sd_free_buffer(values);
    return result;
}

}
