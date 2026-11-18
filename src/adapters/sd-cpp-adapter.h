#pragma once

#include "model-registry.h"
#include "session.h"

#include <memory>
#include <string>
#include <vector>

struct sd_ctx_t;

namespace umm {

// Adapter around the stable-diffusion.cpp backend and its bundled CLIP/MTMD
// image path. Session and workflow code should not depend on those APIs.
class sd_cpp_adapter {
public:
    virtual ~sd_cpp_adapter();
    virtual std::vector<float> encode(const image_input & image) = 0;
};

std::unique_ptr<sd_cpp_adapter> create_sd_cpp_adapter(model_family family, const std::string & path,
                                                       const std::string & backend = "");

// Convert an input image to the dimensions/layout expected by a model's
// editing or understanding path.
image_input prepare_image_input(model_family family, const image_input & image);

// SenseNova U1's image encoder is provided by stable-diffusion.cpp rather than
// this adapter's virtual encoder interface.
std::vector<float> encode_u1_image_with_diffusion(sd_ctx_t * context, const image_input & image);

}
