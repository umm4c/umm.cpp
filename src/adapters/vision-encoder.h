#pragma once

#include "session.h"

#include <memory>
#include <string>
#include <vector>

struct clip_ctx;

namespace umm {

// BAGEL uses the MTMD/CLIP vision projector.
class bagel_vision_encoder {
public:
    bagel_vision_encoder(const std::string & path, const std::string & backend);
    std::vector<float> encode(const image_input & image);

private:
    std::unique_ptr<clip_ctx, void (*)(clip_ctx *)> context_;
};

image_input prepare_bagel_image(const image_input & image);

}
