# 随机放置文本对照

`umm-random-model-run` 用同一组 U1.5 文本请求对照本地、远端和随机切分的输出与耗时。日常推理可直接使用 [CLI 的 `random` 策略](../README.md#命令行调用)。

## 构建和运行

在 CANN 容器中构建，运行时映射一张本地 310P，并按 [RPC 部署指南](../scripts/rpc/README.md) 准备远端 worker 和隧道。

```bash
cmake -S . -B build-random-dispatch -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CANN=ON -DSOC_TYPE=Ascend310P3 -DUSE_ACL_GRAPH=OFF \
  -DGGML_RPC=ON -DGGML_RPC_RDMA=OFF -DUMM_BUILD_BACKEND_SMOKE=ON
cmake --build build-random-dispatch --target umm-random-model-run -j 8
```

先把以下配置保存为 `local.json`。`model` 指向理解侧 GGUF。

```json
{
  "mode": "local",
  "model": "/path/to/u1.5-v2/understanding.gguf",
  "local_device": "CANN0",
  "n_ctx": 512,
  "n_batch": 128,
  "max_tokens": 8,
  "requests": [{"id": "count", "prompt": "请从 1 数到 5。"}]
}
```

```bash
build-random-dispatch/bin/umm-random-model-run local.json results/random-local
```

复制为 `remote.json` 和 `random.json`，将 `mode` 分别改为 `remote` 和 `random`，两份配置都添加 `"endpoint": "127.0.0.1:50052"` 和 `"reference_dir": "results/random-local"`，保持模型、请求及容量相同。

```bash
build-random-dispatch/bin/umm-random-model-run remote.json results/random-remote
build-random-dispatch/bin/umm-random-model-run random.json results/random-mixed
```

输出目录必须尚未存在。工具固定关闭 Q8 NZ。随机模式可指定 `seed`（默认 0）、`cuts`（默认 1）、`flip_devices`（默认 `false`）及 `replicate_weights`（默认 `false`）。启用权重副本时，两端都需有足够显存。用 `logit_atol`、`logit_rtol` 设置比较容差，默认分别为 0.05、0.005。

## 读结果

- `summary.json` 中的 `load_ms` 记录模型加载，`replica_init_ms` 单独记录权重副本初始化。
- `requests.jsonl` 记录请求和对照结果。`model_ms` 包含放置回调观测，`request_wall_ms` 还包含对照分析和中间文件写出。两者都来自带观测的实验工具。
- `graphs.jsonl` 记录切点、实际执行分段和设备，`forwards.jsonl` 将每次 prefill 或 decode 关联到执行图和 logits 文件。`weight_input_bytes` 是调度器列出的待复制权重逻辑字节量。
- 加载 `reference_dir` 后，比较过程沿用参考输出 token，保持每步前缀相同。Top-1 一致率用于对照两条计算路径，回答质量需用任务评测衡量。

## 可选的后端冒烟测试

使用上面的 CANN + RPC 构建配置。下面的命令用于核对设备读写、数值和切分，再运行模型级对照。测试结果不能替代真实模型验收。

```bash
cmake --build build-random-dispatch --target umm-backend-smoke umm-random-plan-smoke -j 8
build-random-dispatch/bin/umm-backend-smoke local_310p CANN0
build-random-dispatch/bin/umm-backend-smoke remote_910b 127.0.0.1:50052 saturation
GGML_RPC_NO_RDMA=1 GGML_CANN_Q8_NZ=0 GGML_CANN_SATURATION=1 \
  build-random-dispatch/bin/umm-random-plan-smoke 127.0.0.1:50052 CANN0 4
```
