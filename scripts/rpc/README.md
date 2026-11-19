# 远端昇腾 910B RPC 部署

`remote_910b` 将理解、视觉编码和生图固定在远端单张 910B 上。客户端保留模型包，通过 RPC 向 worker 传输权重。下面分别在客户端和远端执行命令。

## 准备 worker 源码和镜像

worker 与客户端使用 **同一份 llama.cpp/GGML 源码**，以 UMM 当前提交锁定的子模块为准，CANN 改动已包含在对应版本中。两端都使用 `GGML_MAX_NAME=160`。仅检查 RPC 协议版本不足以确认兼容。

在客户端仓库根目录执行。下面以 SSH 别名 `hw19` 为例，将远端路径统一替换为实际的专用目录。

```bash
git submodule update --init --checkout -- third_party/llama.cpp third_party/stable-diffusion.cpp
git -C third_party/llama.cpp rev-parse HEAD
git -C third_party/llama.cpp status --porcelain
ssh hw19 'mkdir -p /path/to/umm-rpc-910b/llama.cpp /path/to/umm-rpc-910b/deploy'
rsync -a --exclude=.git --exclude='/build*/' \
  third_party/llama.cpp/ hw19:/path/to/umm-rpc-910b/llama.cpp/
rsync -a scripts/rpc/ hw19:/path/to/umm-rpc-910b/deploy/
```

使用新目录同步源码，避免沿用旧文件和构建缓存。若子模块有未提交修改，部署时同时记录这些差异。

在远端准备与 [worker.env.example](worker.env.example) 所列 Image ID 匹配的镜像。已验证的组合为 aarch64、Ascend910B1、驱动 24.1.0.3 和 CANN 9.1.0，镜像标签为 `quay.io/ascend/vllm-ascend:v0.23.0.post1`。示例锁定的是本地 Image ID，可通过 `docker image inspect` 核对。更换镜像或驱动后重新完成三任务验收。

## 构建和启动 worker

在远端用 `npu-smi info` 选一张空闲卡，再复制并编辑配置，至少设置 `RPC_WORKDIR` 和 `RPC_NPU`。配置文件是 shell 文件，保存在仓库外。

```bash
deploy=/path/to/umm-rpc-910b/deploy
cp "$deploy/worker.env.example" "$HOME/umm-rpc-worker.env"
# 编辑 $HOME/umm-rpc-worker.env，填写实际目录、空闲物理卡号和镜像身份
bash "$deploy/worker.sh" build "$HOME/umm-rpc-worker.env"
bash "$deploy/worker.sh" start "$HOME/umm-rpc-worker.env"
bash "$deploy/worker.sh" logs "$HOME/umm-rpc-worker.env"
```

脚本构建 `ggml-rpc-server`，构建日志写入 `RPC_WORKDIR/logs/build-worker.log`。默认芯片目标为 `Ascend910B1`，关闭 ACL Graph 和 RDMA。worker 映射单张物理卡，在容器内使用 `CANN0`，默认监听远端 `127.0.0.1:50052`。启动后核对设备枚举和饱和模式日志。

| worker 设置 | 值及用途 |
| --- | --- |
| `GGML_RPC_NO_RDMA` | `1`，通过 TCP 通信 |
| `GGML_CANN_Q8_NZ` | `0`，关闭 310P 专用 Q8 布局 |
| `GGML_CANN_SATURATION` | `1`，保持已验证的 FP16 溢出饱和行为，U1.5 生图依赖此设置 |
| `LLAMA_CACHE` | `/work/cache`，存放 RPC 权重缓存 |
| `OMP_NUM_THREADS` | `1`，限制 worker 的 OpenMP 线程数 |

这些设置由 worker 脚本传入。客户端的环境变量无法修改远端进程的设置。停止时调用 `bash "$deploy/worker.sh" stop "$HOME/umm-rpc-worker.env"`。重新启动之前，只移除本次配置对应且已停止的旧容器。

## SSH 隧道与客户端构建

客户端准备好 SSH 别名和已验证的主机公钥后，在独立终端启动隧道。

```bash
scripts/rpc/tunnel.sh hw19
```

默认把本地 `127.0.0.1:50052` 转发到远端 `127.0.0.1:50052`。通过 `RPC_LOCAL_PORT`、`RPC_REMOTE_PORT` 修改两端端口。按 Ctrl-C 关闭隧道，远端 worker 继续运行。

在客户端仓库根目录构建 RPC-only 版本。可在宿主机编译，或在能运行这些二进制的客户端容器中编译。使用容器访问宿主机的回环隧道时，应使用 host 网络。

```bash
cmake -S . -B build-rpc-client -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CANN=OFF -DGGML_RPC=ON -DGGML_RPC_RDMA=OFF
cmake --build build-rpc-client --target umm-cli umm-session umm-server -j 8
```

客户端无需映射 NPU。使用下文的容器脚本时，在同一镜像中构建，默认镜像为 `umm-cann:8.5.0-mvp`。镜像管理见 [310P 运行镜像](../../docker/README.runtime.md)。

## 三任务验收

```bash
export MODEL_DIR=/path/to/u1.5-v2
export RPC_ENDPOINT=127.0.0.1:50052
scripts/run-static.sh remote_910b text '用一句话介绍月球。'
scripts/run-static.sh remote_910b understand '图中有什么？' /path/to/photo.png
OUTPUT_DIR="$PWD/outputs/u15-remote" \
  scripts/run-static.sh remote_910b image '白色背景上的红色立方体'
```

脚本各次启动独立进程，默认 context / batch / ubatch 为 `8192 / 4096 / 4096`，可用 `N_CTX`、`N_BATCH`、`N_UBATCH` 覆盖。直接调用 CLI、JSONL 或 HTTP 并选择 `remote_910b` 时，策略默认值是 `2048 / 512 / 256`。脚本默认生图配置为 1024 × 1024、8 步、CFG 1、shift 3、seed 7。

检查文本、图像理解答案和 PNG 是否符合预期。对照 worker 日志和远端 `npu-smi info` 确认实际用卡。记录两端版本、数值配置、首轮权重加载开销、后续请求耗时和输出。

### 同一进程发送多条 JSONL 请求

在能运行客户端二进制的环境中执行。一次启动 `umm-session`，从标准输入逐行提交 JSON。下面用两条短文本请求验证同一进程能复用已加载的模型。

```bash
build-rpc-client/bin/umm-session --model "$MODEL_DIR" \
  --execution-policy remote_910b --rpc-endpoint "$RPC_ENDPOINT" \
  --steps 8 --cfg 1 --shift 3 <<'JSONL'
{"case_id":"text-1","task_class":"text_generation","content":[{"type":"text","value":"用一句话介绍月球。"}],"params":{"max_new_tokens":64}}
{"case_id":"text-2","task_class":"text_generation","content":[{"type":"text","value":"用一句话介绍火星。"}],"params":{"max_new_tokens":64}}
JSONL
```

标准输出先返回一条 `ready`，随后每条请求对应一条 `result`。检查 `case_id` 和 `status`，用 `timings_ms` 比较耗时。日志走标准错误。单图理解使用 `mixed_understanding`，`content` 的第一项为带 `path` 的图片。文生图使用 `image_generation`，在 `params` 提供 `width`、`height` 和 `seed`，并在请求顶层提供绝对 `.png` 路径 `output_path`。容器调用时路径以容器内为准。

## 可选的独立私网隧道

已有对应私网路由和 SSH 配置时，可单独运行 [private-tunnel.py](private-tunnel.py)。默认监听本地 `50353`，转发到远端 `50053`。修改路由和 SSH 参数前先查看脚本的 `--help`。

```bash
python3 scripts/rpc/private-tunnel.py --state-dir "$HOME/.local/state/umm-rpc-private"
```

连接状态、worker 协议版本和连接编号见状态目录的 `state.json`。确认就绪后用 `127.0.0.1:50353` 重新验收推理。如需交给 systemd 管理，可参考 [用户服务示例](private-tunnel.service.example)。隧道的版本检查不能替代实际推理验收。
