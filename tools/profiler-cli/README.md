# 计算图分析工具

`umm-profiler` 用代表性请求运行 SenseNova U1.5，记录实际执行的计算图、设备放置和耗时。可采集本地 310P 和远端 CANN RPC worker 的结果。

## 构建

先按 [本地 310P 部署指南](../../scripts/ascend/README.md) 准备 CANN 容器和源码。在仓库根目录执行：

```bash
cmake -S . -B build-profiler -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CANN=ON -DSOC_TYPE=Ascend310P3 -DUSE_ACL_GRAPH=OFF \
  -DGGML_RPC=ON -DGGML_RPC_RDMA=OFF
cmake --build build-profiler --target umm-profiler -j 8
```

运行时映射准备采集的本地卡。采集远端设备前，按 [RPC 部署指南](../../scripts/rpc/README.md) 启动同源 worker 和 SSH 隧道，再让容器通过 host 网络访问隧道。远端整图计时需要支持批量计时的 worker（RPC 6.1 起），层、块和算子计时需要匹配的 RPC 6.3 worker。

## 配置并运行

复制 [request.example.json](request.example.json)，将下面三个字段改成实际值。路径以执行 profiler 的环境为准。

| 字段 | 填写方式 |
| --- | --- |
| `model` | SenseNova U1.5 模型包目录 |
| `devices` | 待测设备列表，每项有唯一 `id`。`transport` 为 `local` 或 `rpc`，`backend` 填设备名，例如 `CANN0`。仅 RPC 项需要 `endpoint` |
| `representative_workload` | 含 `manifest.json` 和 `cases.jsonl` 的 `workload/v1` 目录。相对图片路径以该目录为基准 |

负载支持纯文本、单图理解、图文混合 prefill 和文生图。建议先选少量典型请求，采集时各设备依次运行同一份负载。每台设备依次运行结构轮、两组基线与计时轮，以及理解图的层和块计时轮。

准备最小负载时，可以把 [JSONL 调用示例](../../scripts/rpc/README.md#同一进程发送多条-jsonl-请求)的两条文本请求保存为 `cases.jsonl`，同目录的 `manifest.json` 写入 `{"schema_version":"workload/v1","case_count":2}`，再将该目录填入请求配置。

| 负载 | context / batch / ubatch 默认值 |
| --- | --- |
| 只有文生图 | `2048 / 512 / 256` |
| 有案例使用超过四张输入图像 | context 为 `32768`，batch 和 ubatch 取 `1152` 与预处理后最大单图视觉 token 数中的较大值 |
| 其他 | `8192 / 4096 / 4096` |

实际参数以结果的 manifest 为准。

```bash
cp tools/profiler-cli/request.example.json profile-request.json
# 编辑 profile-request.json
mkdir -p profiles
build-profiler/bin/umm-profiler --request profile-request.json --out profiles/u15-test
```

输出目录必须尚未存在。需要算子计时时，在命令末尾加 `--operators`。Profiler 应独立运行，便于把该轮耗时与普通推理区分开。CANN 使用 ACL Graph 或后端不支持细粒度计时时，对应记录标记为不可用。

## 查看与校验

| 文件 | 内容 |
| --- | --- |
| `manifest.json` | 模型与负载身份、设备、实际生效参数、采集范围和产物清单 |
| `graphs.jsonl` | 节点、依赖、张量、视图和设备放置 |
| `trace.jsonl` | 各次执行的阶段、图和请求关联、提交规模及计时 |
| `costs.jsonl` | 按设备和图统计的热态整图耗时 |
| `timings.jsonl` | 按逻辑图、请求与实际提交规模对齐的后端整图计时 |
| `hierarchy.jsonl` | 理解图的层、块及可选算子计时，以及对应的节点范围 |
| `requests.jsonl` | 请求响应、耗时、token 计数和输出一致性 |
| `report.md` | 该次采集的范围与限制 |

比较结果时，先对齐模型、请求、设备和计时口径。整图计时与其内部的层、块、算子计时覆盖同一段工作，不能相加。基线请求耗时可以用来估计采集带来的扰动。远端 worker 的同步完成时间、客户端提交时间和请求墙钟分别记录。

```bash
python3 tools/profiler-cli/validate_capture.py profiles/u15-test
```

校验器也支持 manifest 中声明了候选文件的历史采集包。声明的文件缺失时会报错。可用 [Graph Atlas](https://github.com/umm4c/utils/tree/master/profiler-viewer) 打开采集包，浏览图层级、依赖和设备放置。

```bash
# 如已将 utils 与 UMM.cpp 并列检出，可生成离线页面
python3 ../utils/profiler-viewer/export_viewer.py \
  --profile profiles/u15-test --out profiles/u15-test.html
```

## 回归检查

```bash
cmake --build build-profiler --target umm-profiler-observer-test umm-profiler-contract-test -j 4
build-profiler/bin/umm-profiler-observer-test
build-profiler/bin/umm-profiler-contract-test
python3 tests/profiling/test_interface.py build-profiler/bin/umm-profiler

# 如已有可用的 RPC worker，可补跑 RPC 完成边界检查
cmake --build build-profiler --target umm-profiler-rpc-fence-test -j 4
GGML_RPC_NO_RDMA=1 build-profiler/bin/umm-profiler-rpc-fence-test 127.0.0.1:50052
```

Profiler 如何为后续调度提供数据，见 [设计文档](DESIGN.md)。
