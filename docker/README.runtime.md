# Ascend 310P 固定运行环境

2026-10-20，Linux arm64 / Ascend310P3 / CANN 8.5.0 / Python 3.11.14。

| 项目 | 固定值 |
|---|---|
| 日常运行标签 | `umm-cann:8.5.0-mvp` |
| 版本标签 | `umm-cann:8.5.0-310p-runtime-20261020` |
| 基础镜像标签 | `umm-cann-base:8.5.0-310p-20261020` |
| 基础来源 | `quay.io/ascend/cann:8.5.0-310p-openeuler24.03-py3.11` |
| 基础 Image ID | `sha256:7c5790b80ec63ac20e91d7f587a3cfb42cc0d282bc6c8f5dcc4d654d25fa0068` |
| 环境身份 | `runtime-cann310p.lock.json` |
| Python 依赖 | `requirements-cann-runtime.lock`，19 个精确版本及 arm64 wheel SHA256 |
| 保留容器 | `umm-cann-runtime-keep`，保持 Created，标签 `umm.environment.keep=true` |

此镜像用于 C++ 编译、推理和评测。包含 CANN 算子编译依赖及图片检查依赖；源码、二进制、模型和输出由宿主机挂载。PyTorch 权重转换/官方参考环境使用原有 `Dockerfile.cann-mvp`，本镜像不包含 PyTorch/Transformers，也不恢复历史 reference 镜像。

## 重建、备份和恢复

在 umm.cpp 根目录运行：

```bash
# 首次联网准备固定 wheel；代理从当前 shell 传入容器
./docker/rebuild-runtime.sh --download

# wheel 齐全后的离线重建；校验基础 Image ID 和依赖 SHA256
./docker/rebuild-runtime.sh

# 保存运行镜像、基础镜像及全部镜像层，并生成归档 SHA256
./docker/export-runtime.sh

# 校验归档、离线加载、核对运行镜像 ID、恢复标签和保留容器
./docker/restore-runtime.sh

# 验证权限、19 个依赖版本、模块导入与依赖声明
./docker/verify-runtime.sh
```

离线包位于 `artifacts/runtime/cann310p-20261020/`：

- `umm-cann-runtime-20261020.tar.gz` / `.sha256`：完整镜像备份及校验值。
- `wheels/`、依赖锁、Dockerfile：离线重建输入。
- `build-final.log`、`dependency-verification-final.json`、`smoke-results.json`、`container-signal.json`：最终构建、依赖、实机验证及停止信号记录。

大文件保存在本地并由 `.gitignore` 排除；Dockerfile、脚本、依赖锁和环境身份文件由 Git 跟踪。离线包需随环境一起保留。

保留容器不启动、不占 NPU，保证当前运行镜像有容器引用；清理临时容器时排除它。`docker image prune -a` 会删除失去容器引用的命名镜像；仅有版本标签不足以保留镜像。

## 使用与验证

现有 `scripts/ascend/` 以及 bench-runner 的正式构建/JSONL/HTTP 脚本继续使用日常标签。运行保持单卡，沿用驱动挂载和 `seccomp=unconfined`，HTTP 父进程使用镜像 Python，宿主机无需 docker-init 或 Docker 升级。

基础镜像的 `op-compile-tool` 元数据将 Python 标准库 `getopt`、`inspect`、`multiprocessing` 误写为第三方依赖，`pip check` 保留这三项提示；标准库模块已实际导入核对。其余声明的缺失依赖已补齐。验证结果与镜像身份见环境锁及本地离线包。

固定入口设置 `umask 022`，确保挂载目录中的请求日志和系统采样可被宿主机用户读取。运行 Python/NumPy 或 CANN 时同样需要上述 seccomp 参数。

2026-10-20：单物理 NPU2 的 TextVQA 单例和 1024²/8 步 WISE 单例通过，答案/PNG 及 token 数与原镜像记录一致，系统采样为宿主机可读的 0644。CMake/C++ 编译运行、全部依赖版本/导入、无需 docker-init 的父进程信号测试通过。这些是环境回归记录，正式性能数据沿用已有基线。

完整离线归档约 4.3 GiB，已通过 gzip 校验、归档 SHA256 校验和实际 Docker 加载恢复；恢复后的 Image ID 与环境锁一致。校验值记录在环境锁和归档旁的 `.sha256` 文件中。

从离线归档恢复可保留原 Image ID；从 Dockerfile 全新构建的镜像因构建时间等元数据可能得到新 ID，需要验证后更新环境身份与备份。
