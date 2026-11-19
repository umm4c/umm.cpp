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

此镜像用于 C++ 编译、推理和评测。源码、模型和输出由宿主机挂载。使用方法见 [本地 310P 部署](../scripts/ascend/README.md) 或 [RPC 客户端部署](../scripts/rpc/README.md)。需要 PyTorch 的权重转换可使用 [Dockerfile.cann-mvp](Dockerfile.cann-mvp)。

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

运行时使用日常标签 `umm-cann:8.5.0-mvp`。容器的单卡映射和 seccomp 参数见 310P 部署指南。

从备份恢复会保留原 Image ID。全新构建后，使用 `verify-runtime.sh` 校验环境，再更新环境锁和备份。基础镜像的依赖元数据会让 `pip check` 报告 `getopt`、`inspect`、`multiprocessing`，这三个标准库模块已验证可以导入。
