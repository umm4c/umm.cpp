#include "jsonl.h"
#include "image-io.h"

#include <filesystem>

namespace umm::runtime {
namespace {

image_input decode_file(const json & image) {
    return read_image(required<std::string>(image, "path"));
}

std::filesystem::path png_path(const json & request) {
    const auto path = std::filesystem::path(required<std::string>(request, "output_path"));
    if (!path.is_absolute() || path.extension() != ".png")
        throw std::invalid_argument("output_path must be an absolute PNG path");
    return path;
}

void configure_generation(const json & request, const json & params, image_options &) {
    (void)png_path(request);
    for (const char * field : {"steps", "cfg", "shift"}) {
        if (params.contains(field))
            throw std::invalid_argument("Invalid generation override");
    }
}

json deliver_file(const json & request, const image_result & image, request_metrics & metrics, double &) {
    const auto path = png_path(request);
    std::filesystem::create_directories(path.parent_path());
    const auto start = metrics_clock::now();
    write_png(path, image);
    std::filesystem::permissions(path, std::filesystem::perms::group_read | std::filesystem::perms::others_read,
                                 std::filesystem::perm_options::add);
    metrics.image_write_ms = metrics_elapsed_ms(start);
    return {{"type", "image"}, {"path", path.string()}, {"width", image.width}, {"height", image.height}};
}

}

json run_jsonl_case(session & engine, const json & request, const options & args) {
    return run_protocol_case(engine, request, args, {decode_file, configure_generation, deliver_file});
}

}
