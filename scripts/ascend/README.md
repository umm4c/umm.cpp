# 本地昇腾 310P 部署

在一张 310P 上运行 U1.5 的文本生成、单图理解和文生图。下面的命令从仓库根目录执行。

## 构建

准备支持 Ascend310P3 的宿主机驱动和 `umm-cann:8.5.0-mvp` 镜像。镜像获取和恢复见 [310P 运行镜像](../../docker/README.runtime.md)。先在宿主机初始化子模块，再在容器里编译。编译阶段无需映射 NPU。

```bash
git submodule update --init --checkout -- third_party/llama.cpp third_party/stable-diffusion.cpp
docker run --rm --security-opt seccomp=unconfined \
  -v "$PWD:/workspace/umm" -w /workspace/umm umm-cann:8.5.0-mvp bash -lc '
    set -e
    cmake -S . -B build-static-310p -DCMAKE_BUILD_TYPE=Release \
      -DGGML_CANN=ON -DSOC_TYPE=Ascend310P3 -DGGML_RPC=OFF
    cmake --build build-static-310p --target umm-cli umm-session umm-server -j 8
  '
```

## 选卡和运行

先用 `npu-smi info` 查看空闲卡。脚本通过 `LOCAL_NPU` 选择宿主机的物理卡号，默认是 `0`。容器只映射选中的 `/dev/davinciN`，同时映射设备管理节点和驱动目录。直接在容器中调用 CLI、JSONL 或 HTTP 时，用 `--execution-policy local_310p --local-device CANN0` 选择单卡映射后的逻辑设备。该策略默认启用 Q8 NZ。

```bash
export MODEL_DIR=/path/to/u1.5-v2
export LOCAL_NPU=2

scripts/run-static.sh local_310p text '用一句话介绍月球。'
scripts/run-static.sh local_310p understand '图中有什么？' /path/to/photo.png
OUTPUT_DIR="$PWD/outputs/u15-local" \
  scripts/run-static.sh local_310p image '白色背景上的红色立方体'
```

| 入口 | context / batch / ubatch 默认值 | 覆盖参数 |
| --- | --- | --- |
| CLI、JSONL 和 HTTP 的 `local_310p` 策略（U1.5） | `2048 / 1024 / 1024` | `--n-ctx`、`--n-batch`、`--n-ubatch` |
| CLI、JSONL 和 HTTP 的 `local_310p` 策略（BAGEL） | `2048 / 512 / 256` | `--n-ctx`、`--n-batch`、`--n-ubatch` |
| `scripts/run-static.sh local_310p` | `8192 / 4096 / 4096` | `N_CTX`、`N_BATCH`、`N_UBATCH` |
| `scripts/ascend/run-u15.sh` | `2048 / 512 / 256` | `N_CTX`、`N_BATCH`、`N_UBATCH` |

`run-static.sh` 生图默认 1024 × 1024、8 步、CFG 1、shift 3、seed 7，图片写到 `OUTPUT_DIR/image.png`。可通过 `WIDTH`、`HEIGHT`、`STEPS`、`CFG`、`SHIFT`、`SEED` 覆盖。

直接使用 `local_310p` 做 U1.5 单图理解时，默认微批次可容纳 1024 × 1024 输入的 1024 个视觉 token。上下文还需容纳提示词和输出。只指定上下文或批次时，未指定的默认批次和微批次会按已有的容量限制缩小。

### 直接使用容器

需要运行 `umm-server`、`umm-session` 或调试 CLI 参数时，可以从仓库根目录进入容器。下例沿用上面的 `MODEL_DIR` 和 `LOCAL_NPU`，模型目录在容器中挂载为 `/model`。host 网络让本机终端可以访问容器内监听回环地址的 HTTP 服务。

```bash
docker run --rm -it --network host --security-opt seccomp=unconfined \
  --device "/dev/davinci${LOCAL_NPU:-0}" --device /dev/davinci_manager \
  --device /dev/hisi_hdc --device /dev/devmm_svm \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v "$PWD:/workspace/umm:ro" -v "$MODEL_DIR:/model:ro" \
  -w /workspace/umm umm-cann:8.5.0-mvp bash -lc '
    export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH:-}
    exec bash
  '
```

容器内可按 [HTTP 快速开始](../../README.md#http-服务调用) 启动服务。若需写入图像，在启动容器时额外挂载可写的输出目录。

## 模型专用脚本

需要图文交错生成或 BAGEL 多卡理解时，可用下面的入口。手动映射多卡时，`DEVICES` 的顺序决定容器内 CANN 编号。下例中物理卡 2、3、4 依次对应 `CANN0`、`CANN1`、`CANN2`。

```bash
BUILD_DIR=build-static-310p DEVICES=2 \
  scripts/ascend/run-u15.sh interleave '用图文介绍月球。'

BUILD_DIR=build-static-310p MODEL_DIR=/path/to/BAGEL-Q8U-F16G-UMM DEVICES=2 \
  WIDTH=1024 HEIGHT=1024 scripts/ascend/run-bagel.sh image '木桌上的红苹果'

BUILD_DIR=build-static-310p MODEL_DIR=/path/to/BAGEL-F16-UMM DEVICES=2,3,4 \
  UNDERSTANDING_BACKEND=CANN0 VISION_BACKEND=CANN1 \
  GENERATION_BACKEND='diffusion=CANN1&CANN2,vae=CANN0' \
  GENERATION_MAX_VRAM=CANN1=14,CANN2=14 \
  scripts/ascend/run-bagel.sh understand '这是什么？' /path/to/photo.png
```

`run-u15.sh` 默认构建目录为 `build-cann-rebase`，生图默认 1024 × 1024、交错生图默认 512 × 512，两者默认 8 步、CFG 1、seed 42。BAGEL 单卡示例用于文生图，理解使用 F16 多卡配置。BAGEL 编辑仍处于实验阶段，图像质量尚未通过验证。通过各脚本的 `--help` 查看完整参数。

## 简单测速

```bash
BUILD_DIR=build-static-310p DEVICES=2 WARMUP=1 REPEATS=3 \
  scripts/ascend/bench-u15.sh image '白色背景上的红色立方体'
```

BAGEL 可换用 `bench-bagel.sh` 并指定对应的 `MODEL_DIR`。默认输出到 `benchmark-results/`，可用 `BENCH_DIR` 覆盖。`results.csv` 的 `elapsed_seconds` 包含容器启动、模型加载、推理和写出，每轮启动新进程。`summary.txt` 汇总成功的正式轮次，`logs/`、`npu/` 和 `outputs/` 保存日志、设备采样和输出。

如需单独记录一个容器内命令的进程显存，可运行 `python3 scripts/ascend/monitor-command.py --card <npu-smi卡号> --output <目录> -- <命令>`。容器需能访问所选卡、驱动和 `npu-smi`。峰值见 `summary.json`，采样见 `memory.jsonl`。
