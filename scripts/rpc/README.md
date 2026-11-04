# Reproduce the 910B RPC worker

These scripts deploy the worker for `remote_910b`. Run `worker.sh` on the remote
host and `tunnel.sh` on the client. They do not change host drivers, SSH credentials
or other containers. The worker listens on loopback and maps one physical NPU.

## Validated environment and source

The 2026-11-04 validation used aarch64, Ascend910B1 physical card 4, host driver
24.1.0.3, and CANN 9.1.0/GCC 11 from
`quay.io/ascend/vllm-ascend:v0.23.0.post1`. The example pins its observed image config
ID. This ID verifies local image content; it is **not** a pullable registry manifest
digest. Obtain the matching image with `docker pull` or `docker load`, and check
`docker image inspect`. For an image archive, also record its SHA256. A different
image/CANN version needs new validation; do not silently replace the expected ID.
A registry digest can be used as RPC_IMAGE once verified to resolve to the same ID.

This deployment branch inherits the tested UMM implementation
`cfe6e50386f76a6b53aaab0fdd2e4f986239952c`. Its submodules lock llama.cpp to
`9897082a26924e96d3016907be1fa360fc44b0ac` and SD to
`7ab496fba8e45ac8b1ccfa81ab9c15373682b59a`. Use the llama.cpp submodule of this
checkout, including its CANN changes, for the server. RPC protocol 6.0.0 alone is
insufficient to establish compatible operators. Both sides require GGML_MAX_NAME=160.

From the **client checkout**, initialize submodules and verify no local edits before
synchronizing. Replace the example remote directory consistently in all commands:

```sh
git submodule update --init --recursive
git rev-parse HEAD
git -C third_party/llama.cpp rev-parse HEAD
git -C third_party/llama.cpp status --porcelain
ssh hw19 'mkdir -p /path/to/umm-rpc-910b/llama.cpp /path/to/umm-rpc-910b/deploy'
rsync -a --delete --exclude=.git --exclude='/build*/' \
  third_party/llama.cpp/ hw19:/path/to/umm-rpc-910b/llama.cpp/
rsync -a scripts/rpc/ hw19:/path/to/umm-rpc-910b/deploy/
```

Use a dedicated destination: `--delete` makes its source mirror exact. Record the
client `git rev-parse HEAD`, submodule commit, dirty status and source hashes with
run artifacts. The remote copy has no Git directory. No model package is needed
there; weights are sent over RPC and may be cached.

## Build and start on the remote host

Check `npu-smi info` and select an available card. Copy `worker.env.example` to an
external file such as `$HOME/umm-rpc-worker.env`, set RPC_WORKDIR to the dedicated
absolute remote path and choose RPC_NPU. Config files are trusted shell scripts.
Do not edit the expected image ID merely to bypass a mismatch.

```sh
bash /path/to/umm-rpc-910b/deploy/worker.sh build "$HOME/umm-rpc-worker.env"
bash /path/to/umm-rpc-910b/deploy/worker.sh start "$HOME/umm-rpc-worker.env"
bash /path/to/umm-rpc-910b/deploy/worker.sh logs "$HOME/umm-rpc-worker.env"
```

Build uses Release, static libraries, `GGML_RPC=ON`, `GGML_CANN=ON`,
`GGML_RPC_RDMA=OFF`, `SOC_TYPE=Ascend910B1`, `USE_ACL_GRAPH=OFF`, and C/C++
`GGML_MAX_NAME=160`. It builds only ggml-rpc-server. Logs go to logs/build-worker.log.
Use a fresh workdir when changing source or toolchain to avoid stale build outputs.

Runtime maps davinciN, davinci_manager, hisi_hdc and devmm_svm; mounts the driver
read-only and the workdir read-write. Host networking permits the SSH server to
reach the loopback listener. CANN0 is the expected single-card logical name;
confirm device enumeration and the worker's saturation startup message in logs.
The protocol does not attest the remote hardware identity.

Required environment:

| Variable | Value | Reason |
| --- | --- | --- |
| GGML_RPC_NO_RDMA | 1 | TCP transport |
| GGML_CANN_Q8_NZ | 0 | Disable 310P-specific layout |
| GGML_CANN_SATURATION | 1 | Match finite F16 overflow behavior; required for validated image generation |
| LLAMA_CACHE | /work/cache | RPC weight cache under cache/rpc |
| OMP_NUM_THREADS | 1 | Worker host threading |

The container has no restart policy. Stop with `worker.sh stop CONFIG`; to replace
it, explicitly remove that stopped container with `docker rm NAME` before starting
again. `status` and `logs` inspect the named container. Build/start refuse an image
ID mismatch; start refuses an existing container name. No running service is
replaced automatically. Never use these commands with another service's name.

## Tunnel and three-task validation on the client

Configure your existing SSH alias (for example hw19) with its private key and
verified known_hosts entry. These files stay outside Git. Then run in a dedicated
terminal:

```sh
scripts/rpc/tunnel.sh hw19
```

The tunnel binds 127.0.0.1:50052 and forwards to the remote loopback port. Set
RPC_LOCAL_PORT/RPC_REMOTE_PORT to override. This host-SSH implementation provides
the same forwarding as the validation's SSH container, without depending on a
client container image. Closing it with Ctrl-C ends the tunnel. Worker stays up.

Build the client using the RPC-only instructions in `docs/static-execution.md`
inside the validated client CANN 8.5.0 environment (`umm-cann:8.5.0-mvp`). The
historical client image ID is
`sha256:8d9a763fb39a70b80540dd88e0129a59aa05c6dbf7bbf5a9d24452860f3ffd88`.
The client build requires no local NPU. With the model package available locally:

```sh
export MODEL_DIR=/path/to/SenseNova-U1.5-8B-MoT-LoRA-8step-V2-Q8U-F16G-UMM
export RPC_ENDPOINT=127.0.0.1:50052
scripts/run-static.sh remote_910b text 'Describe the moon.'
scripts/run-static.sh remote_910b understand 'Describe this image.' /path/to/input.png
scripts/run-static.sh remote_910b image 'A red apple on a wooden table'
```

Runner defaults are 8192/4096/4096 context/batch/ubatch; generation is 1024x1024,
8 steps, CFG 1, shift 3, seed 7. Repeat requests and check outputs. These runner
commands create separate processes: additionally send two requests within one
umm-session process to check persistent model/KV reuse (see the session interface
in the main README). Record cold weight upload, warm request timing, output
text/images, worker logs, `npu-smi info`, effective config, image IDs and source
identities. Confirm no client NPU mapping and that the remote worker uses the
chosen card. A connected socket alone is not a successful inference validation.

Keep weights, RPC caches, builds, private configuration and raw run artifacts
outside Git. Save artifacts in the benchmark results directory. This deployment
commit provides reproducible commands; re-running against another environment
still requires the three-task correctness checks.
