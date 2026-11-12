#pragma once
#include "json.hpp"
#include "ggml-backend.h"
#include <memory>
#include <string>

namespace umm::profiling {
class candidate_catalog {
public:
    explicit candidate_catalog(std::string model_key);
    ~candidate_catalog();
    void start_device(const std::string & id);
    void discover(ggml_cgraph * graph, const std::string & graph_id);
    void select();
    void begin_capture(ggml_backend_sched_t sched,ggml_cgraph * graph,const std::string & case_id);
    void end_capture(ggml_backend_sched_t sched);
    void measure(ggml_backend_t target,ggml_backend_t primary,const std::string & primary_id);
    nlohmann::json records() const;
    nlohmann::json measurements() const;
    nlohmann::json snapshots() const;
private:
    struct impl;
    std::unique_ptr<impl> state;
};
}
