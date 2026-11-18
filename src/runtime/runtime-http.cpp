#include "runtime-support.h"
#include "httplib.h"
#include <atomic>
#include <thread>

namespace umm::runtime {
namespace {
using clock = std::chrono::steady_clock;
std::atomic<uint64_t> sequence{0};
std::string request_id() {
    const auto time = std::chrono::system_clock::now().time_since_epoch().count();
    return "umm-" + std::to_string(time) + "-" + std::to_string(++sequence);
}
void error(httplib::Response & response, int status, const std::string & code, const std::string & message) {
    response.status = status;
    response.set_content(json({{"error", {{"message", message}, {"type", status >= 500 ? "server_error" : "invalid_request_error"},
                                         {"param", nullptr}, {"code", code}}}}).dump(), "application/json");
}
std::string token(const options & args) {
    const auto name = option(args, "--api-key-env");
    if (name.empty()) return "";
    const char * value = std::getenv(name.c_str());
    if (!value || !*value) throw std::invalid_argument("API key environment variable is empty");
    return value;
}
int port_number(const std::string & text) {
    size_t used = 0; int port = std::stoi(text, &used);
    if (used != text.size() || port < 1 || port > 65535) throw std::invalid_argument("Invalid listen port");
    return port;
}
void configure(httplib::Server & server, service & runtime, const std::string & secret) {
    server.new_task_queue = [] { return new httplib::ThreadPool(4, 16); };
    server.set_payload_max_length(128 * 1024 * 1024);
    server.set_read_timeout(30, 0);
    server.set_write_timeout(30, 0);
    server.set_pre_routing_handler([&runtime, secret](const httplib::Request & request, httplib::Response & response) {
        if (!secret.empty() && request.path != "/health" && request.get_header_value("Authorization") != "Bearer " + secret) {
            runtime.reject("authentication"); error(response, 401, "invalid_api_key", "Invalid bearer token");
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });
    server.Get("/metrics", [&runtime](const httplib::Request &, httplib::Response & response) {
        response.set_content(runtime.metrics(), "text/plain; version=0.0.4; charset=utf-8");
    });
    server.Get("/health", [](const httplib::Request &, httplib::Response & response) { response.set_content("{\"status\":\"ok\"}", "application/json"); });
    server.Get("/ready", [](const httplib::Request &, httplib::Response & response) { response.set_content("{\"status\":\"ready\"}", "application/json"); });
    server.set_error_handler([&runtime](const httplib::Request &, httplib::Response & response) {
        if (response.get_header_value("Content-Type").rfind("application/json", 0) == 0) return;
        runtime.reject("http_protocol");
        error(response, response.status, "http_" + std::to_string(response.status), "HTTP request rejected");
    });
    server.set_exception_handler([](const httplib::Request &, httplib::Response & response, std::exception_ptr ptr) {
        try { if (ptr) std::rethrow_exception(ptr); }
        catch (const std::exception & exc) { error(response, 500, "internal_error", exc.what()); }
    });
}
void only(const json & value, const std::vector<std::string> & allowed) {
    if (!value.is_object()) throw std::invalid_argument("Expected a JSON object");
    for (auto it = value.begin(); it != value.end(); ++it) {
        if (std::find(allowed.begin(), allowed.end(), it.key()) == allowed.end()) throw std::invalid_argument("Unsupported parameter: " + it.key());
    }
}
int positive(const json & value, int limit) {
    if (!value.is_number_integer()) throw std::invalid_argument("Expected a positive integer");
    const auto number = value.get<int64_t>();
    if (number <= 0 || number > limit) throw std::invalid_argument("Integer exceeds supported range");
    return int(number);
}
json chat_request(const json & input, const std::string & id) {
    only(input, {"model", "messages", "max_tokens", "max_completion_tokens", "temperature", "stream", "n"});
    if (input.contains("stream") && input.at("stream") != false) throw std::invalid_argument("Streaming is not supported");
    if (input.contains("temperature") && (!input["temperature"].is_number() || input["temperature"].get<double>() != 0)) throw std::invalid_argument("Only greedy decoding (temperature=0) is supported");
    if (input.contains("n") && positive(input["n"], 1) != 1) throw std::invalid_argument("Only n=1 is supported");
    if (input.contains("max_tokens") && input.contains("max_completion_tokens")) throw std::invalid_argument("Conflicting token limits");
    const int maximum = positive(input.value("max_completion_tokens", input.value("max_tokens", json(256))), 262144);
    const auto & messages = input.at("messages");
    if (!messages.is_array() || messages.size() != 1 || messages[0].at("role") != "user") throw std::invalid_argument("One user message is supported");
    only(messages[0], {"role", "content"});
    const auto & content = messages[0].at("content");
    json parts = json::array();
    size_t images = 0;
    if (content.is_string()) parts.push_back({{"type", "text"}, {"value", content}});
    else if (content.is_array()) {
        for (const auto & item : content) {
            const auto type = item.at("type").get<std::string>();
            if (type == "text") {
                only(item, {"type", "text"});
                parts.push_back({{"type", "text"}, {"value", item.at("text").get<std::string>()}});
            } else if (type == "image_url") {
                only(item, {"type", "image_url"}); only(item.at("image_url"), {"url"});
                const auto url = item.at("image_url").at("url").get<std::string>();
                if (url.rfind("data:image/", 0) != 0) throw std::invalid_argument("Images require base64 PNG/JPEG data URLs");
                parts.push_back({{"type", "image"}, {"data_url", url}}); ++images;
            } else throw std::invalid_argument("Unsupported content part");
        }
    } else throw std::invalid_argument("Invalid message content");
    const std::string task = images == 0 ? "text_generation" : images == 1 && parts[0]["type"] == "image" ? "mixed_understanding" : "multimodal_prefill";
    return {{"case_id", id}, {"task_class", task}, {"content", parts}, {"params", {{"max_new_tokens", maximum}}}};
}
json image_request(const json & input, const std::string & id) {
    only(input, {"model", "prompt", "n", "size", "response_format", "umm_options"});
    if (input.contains("n")) positive(input["n"], 1);
    if (input.value("response_format", "b64_json") != "b64_json") throw std::invalid_argument("Only b64_json output is supported");
    const auto prompt = input.at("prompt").get<std::string>();
    const auto size = input.value("size", "1024x1024");
    const auto cross = size.find('x');
    if (cross == std::string::npos || cross != size.rfind('x')) throw std::invalid_argument("Invalid image size");
    auto dimension = [](const std::string & text) {
        if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos) throw std::invalid_argument("Invalid image size");
        return positive(json(std::stoll(text)), 2048);
    };
    const int width = dimension(size.substr(0, cross)), height = dimension(size.substr(cross + 1));
    if (width % 32 || height % 32) throw std::invalid_argument("Image size must be divisible by 32");
    auto custom = input.value("umm_options", json::object());
    only(custom, {"seed", "steps", "cfg", "shift"});
    if (custom.contains("seed") && !custom["seed"].is_number_integer()) throw std::invalid_argument("seed must be an integer");
    custom["seed"] = custom.value("seed", int64_t(42)); custom["width"] = width; custom["height"] = height;
    return {{"case_id", id}, {"task_class", "image_generation"},
            {"content", json::array({{{"type", "text"}, {"value", prompt}}})}, {"params", custom}};
}
struct listener {
    httplib::Server server;
    std::thread thread;
    ~listener() { server.stop(); if (thread.joinable()) thread.join(); }
};
}
std::shared_ptr<void> serve_metrics(service & runtime, const options & args) {
    const auto port = option(args, "--metrics-port", "0");
    if (port == "0") return {};
    auto handle = std::make_shared<listener>();
    configure(handle->server, runtime, token(args));
    if (!handle->server.bind_to_port(option(args, "--host", "127.0.0.1"), port_number(port))) throw std::runtime_error("Cannot bind metrics listener");
    auto * raw = handle.get();
    handle->thread = std::thread([raw] { raw->server.listen_after_bind(); });
    handle->server.wait_until_ready();
    return handle;
}
int serve_http(service & runtime, const options & args) {
    httplib::Server server;
    configure(server, runtime, token(args));
    const auto model = option(args, "--served-model-name", "u15");
    server.Get("/v1/models", [model](const httplib::Request &, httplib::Response & response) {
        response.set_content(json({{"object", "list"}, {"data", json::array({{{"id", model}, {"object", "model"}, {"created", 0}, {"owned_by", "umm"}}})}}).dump(), "application/json");
    });
    server.Get("/umm/info", [&runtime, model](const httplib::Request &, httplib::Response & response) {
        auto value = runtime.info(); value["served_model_name"] = model;
        value["capabilities"] = {{"chat", "single_user_nonstreaming"}, {"image_input", "png_jpeg_base64"}, {"max_image_dimension", 2048}};
        response.set_content(value.dump(), "application/json");
    });
    auto handler = [&](bool image) {
        return [&, image](const httplib::Request & request, httplib::Response & response) {
            const auto start = clock::now();
            const auto id = request_id(); response.set_header("X-Request-ID", id);
            json normalized;
            try {
                const auto input = json::parse(request.body);
                if (input.at("model").get<std::string>() != model) {
                    runtime.reject("model"); error(response, 404, "model_not_found", "Unknown model alias"); return;
                }
                normalized = image ? image_request(input, id) : chat_request(input, id);
                if (request.has_header("X-Client-Case-ID")) normalized["client_case_id"] = request.get_header_value("X-Client-Case-ID").substr(0, 128);
            } catch (const std::exception & exc) {
                runtime.reject("invalid_request"); error(response, 400, "invalid_request", exc.what()); return;
            }
            auto future = runtime.submit(normalized, true);
            if (!future) { runtime.reject("busy"); error(response, 429, "server_busy", "An inference request is already running"); return; }
            const auto result = future->get();
            if (result.at("status") != "ok") {
                const auto & failure = result.at("error");
                const auto code = failure.at("code").get<std::string>();
                error(response, code == "invalid_request" || code == "unsupported_input_format" ? 400 : 500,
                      code, failure.at("message"));
            } else {
                json body;
                const auto created = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
                if (image) body = {{"created", created}, {"data", json::array({{{"b64_json", result.at("output").at("b64_json")}}})}};
                else {
                    const auto input_tokens = result.at("token_counts").at("input").get<int64_t>();
                    const auto output_tokens = result.at("token_counts").at("output").get<int64_t>();
                    body = {{"id", "chatcmpl-" + id}, {"object", "chat.completion"}, {"created", created}, {"model", model},
                            {"choices", json::array({{{"index", 0}, {"message", {{"role", "assistant"}, {"content", result.at("output").at("text")}}},
                                                      {"finish_reason", result.value("finish_reason", "stop")}}})},
                            {"usage", {{"prompt_tokens", input_tokens}, {"completion_tokens", output_tokens}, {"total_tokens", input_tokens+output_tokens}}}};
                }
                response.set_content(body.dump(), "application/json");
            }
            runtime.http_completed(id, normalized.at("task_class"), result.at("status"),
                std::chrono::duration<double, std::milli>(clock::now()-start).count());
        };
    };
    server.Post("/v1/chat/completions", handler(false));
    server.Post("/v1/images/generations", handler(true));
    return server.listen(option(args, "--host", "127.0.0.1"), port_number(option(args, "--port", "8080"))) ? 0 : 1;
}
}
