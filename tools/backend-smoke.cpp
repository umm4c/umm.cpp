#include "execution-policy.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

int main(int argc, char ** argv) {
    try {
        if (argc < 2 || argc > 4) throw std::invalid_argument("Usage: umm-backend-smoke local_310p|remote_910b|guard_negative [ENDPOINT|DEVICE] [saturation]");
        umm::execution_options options{{"--execution-policy", argv[1]}};
        if (std::string(argv[1]) == "guard_negative") {
            ggml_backend_load_all();
            options["--understanding-backend"] = "CPU";
            setenv("GGML_REQUIRED_DEVICE", "FORBIDDEN_TEST_DEVICE", 1);
        } else {
            if (argc >= 3) options[std::string(argv[1]) == "local_310p" ? "--local-device" : "--rpc-endpoint"] = argv[2];
            options = umm::configure_execution_policy(std::move(options));
        }
        if (argc == 4 && std::string(argv[3]) != "saturation") throw std::invalid_argument("Unknown backend smoke mode");
        auto backend = ggml_backend_init_by_name(options.at("--understanding-backend").c_str(), nullptr);
        if (!backend) throw std::runtime_error("Cannot initialize selected backend");
        if (argc == 4 && std::string(argv[3]) == "saturation") {
            auto ctx = ggml_init({1024 * 1024, nullptr, true});
            auto a = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 64);
            auto b = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 64);
            auto sum = ggml_add(ctx, a, b);
            auto graph = ggml_new_graph_custom(ctx, 8, false);
            ggml_build_forward_expand(graph, sum);
            auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
            if (!buffer) throw std::runtime_error("Saturation allocation failed");
            std::vector<ggml_fp16_t> input(64), result(64);
            for (int i = 0; i < 64; ++i) input[i] = ggml_fp32_to_fp16(i % 2 ? -45056.f : 45056.f);
            ggml_backend_tensor_set(a, input.data(), 0, ggml_nbytes(a));
            ggml_backend_tensor_set(b, input.data(), 0, ggml_nbytes(b));
            if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) throw std::runtime_error("Saturation graph failed");
            ggml_backend_tensor_get(sum, result.data(), 0, ggml_nbytes(sum));
            for (int i = 0; i < 64; ++i) {
                const float actual = ggml_fp16_to_fp32(result[i]);
                if (actual != (i % 2 ? -65504.f : 65504.f)) throw std::runtime_error("FP16 overflow did not saturate");
            }
            std::puts("PASS FP16 positive/negative ADD overflow saturates to +/-65504");
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
            ggml_backend_free(backend);
            return 0;
        }
        for (auto type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0}) {
            for (int columns : {1, 16}) {
                const int k = 512, n = 64;
                auto ctx = ggml_init({2 * 1024 * 1024, nullptr, true});
                auto a = ggml_new_tensor_2d(ctx, type, k, n);
                auto b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, columns);
                // CANN quantized storage is repacked; arbitrary quantized subviews need separate validation.
                const bool sliced = type != GGML_TYPE_Q8_0;
                const int output_rows = sliced ? 32 : n;
                const int row_offset = sliced ? 16 : 0;
                auto weight = sliced ? ggml_view_2d(ctx, a, k, output_rows, a->nb[1], row_offset * a->nb[1]) : a;
                auto c = ggml_mul_mat(ctx, weight, b);
                auto indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);
                auto gathered = ggml_get_rows(ctx, a, indices);
                ggml_set_name(a, "rpc_smoke_weights");
                auto graph = ggml_new_graph_custom(ctx, 32, false);
                ggml_build_forward_expand(graph, c);
                ggml_build_forward_expand(graph, gathered);
                auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
                if (!buffer) throw std::runtime_error("Backend tensor allocation failed");
                std::vector<float> weights(k*n), reference(k*n), input(k*columns), result(output_rows*columns);
                for (int i = 0; i < k*n; ++i) weights[i] = float(i % 29 - 14) / 128;
                std::vector<unsigned char> packed(ggml_nbytes(a)), readback(packed.size());
                if (type == GGML_TYPE_F32) std::memcpy(packed.data(), weights.data(), packed.size());
                else ggml_quantize_chunk(type, weights.data(), packed.data(), 0, n, k, nullptr);
                if (type == GGML_TYPE_F32) reference = weights;
                else ggml_get_type_traits(type)->to_float(packed.data(), reference.data(), k*n);
                // Exercise chunked uploads before the backend applies its weight layout.
                const size_t half = packed.size()/2;
                ggml_backend_tensor_set(a, packed.data(), 0, half);
                ggml_backend_tensor_set(a, packed.data()+half, half, packed.size()-half);
                const int32_t row_indices[] = {0, 5, 5, 63};
                ggml_backend_tensor_set(indices, row_indices, 0, sizeof(row_indices));
                ggml_backend_tensor_get(a, readback.data(), 0, readback.size());
                if (packed != readback) throw std::runtime_error("Tensor readback differs from uploaded bytes");
                for (int round = 0; round < 2; ++round) {
                    for (int i = 0; i < k*columns; ++i) input[i] = float((i+round*3) % 17 - 8) / 64;
                    ggml_backend_tensor_set(b, input.data(), 0, ggml_nbytes(b));
                    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
                        throw std::runtime_error("Backend graph execution failed");
                    }
                    ggml_backend_tensor_get(c, result.data(), 0, ggml_nbytes(c));
                    float error = 0;
                    for (int col = 0; col < columns; ++col) {
                        for (int row = 0; row < output_rows; ++row) {
                            float expected = 0;
                            for (int q = 0; q < k; ++q) expected += reference[(row+row_offset)*k+q] * input[col*k+q];
                            const float actual = result[col*output_rows+row];
                            if (!std::isfinite(actual)) throw std::runtime_error("Non-finite matrix result");
                            error = std::max(error, std::abs(actual-expected));
                        }
                    }
                    if (error > 0.015f) throw std::runtime_error("Matrix result exceeds absolute tolerance 0.015");
                    std::vector<float> rows(k*4);
                    ggml_backend_tensor_get(gathered, rows.data(), 0, ggml_nbytes(gathered));
                    float gather_error = 0;
                    for (int row = 0; row < 4; ++row) {
                        for (int q = 0; q < k; ++q) {
                            const float actual = rows[row*k+q];
                            if (!std::isfinite(actual)) throw std::runtime_error("Non-finite GET_ROWS result");
                            gather_error = std::max(gather_error, std::abs(actual-reference[row_indices[row]*k+q]));
                        }
                    }
                    if (gather_error > 0.001f) throw std::runtime_error("GET_ROWS result exceeds tolerance 0.001");
                    std::printf("PASS type=%s columns=%d round=%d max_abs_error=%g gather_error=%g\n",ggml_type_name(type),columns,round,error,gather_error);
                    std::fflush(stdout);
                }
                ggml_backend_buffer_free(buffer);
                ggml_free(ctx);
            }
        }
        ggml_backend_free(backend);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "Backend smoke failed: %s\n", error.what());
        return 1;
    }
}
