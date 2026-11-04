# Static execution policies

The CLI, JSONL session and HTTP server accept `--execution-policy local_310p|remote_910b`.
The policy pins understanding, vision, generation and generation parameters to one device.
Conflicting component devices are rejected. `GGML_REQUIRED_DEVICE` rejects graph computation
on any other device.
A process cannot change its static device after initialization.

`local_310p` requires a local CANN device whose description identifies Ascend310P.
`--local-device CANN0` selects the container-visible device. Map only the chosen physical
NPU into the container. Q8 NZ is enabled by default for this policy.

`remote_910b --rpc-endpoint HOST:PORT` requires a worker exposing exactly one device.
The worker must be deployed with exactly one CANN 910B and its identity recorded separately.
The RPC protocol does not attest the remote hardware: the client device type and description
cannot establish that the worker is CANN. This policy uses that deployment contract to select
CANN-compatible attention behavior, disables the 310P-only Q8 NZ path and local NPU sampling,
and forces TCP transport. Remote metrics must be sampled on the worker host.

## Builds

Use separate build directories. A CPU+RPC client avoids initializing local CANN:

```sh
cmake -S . -B build-rpc-client -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CANN=OFF -DGGML_RPC=ON -DGGML_RPC_RDMA=OFF
cmake --build build-rpc-client --target umm-cli umm-session umm-server -j4
```

For local execution:

```sh
cmake -S . -B build-static-310p -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CANN=ON -DSOC_TYPE=Ascend310P3 -DGGML_RPC=OFF
cmake --build build-static-310p --target umm-cli umm-session umm-server -j4
```

For both policies the initial defaults are context/batch/ubatch 2048/512/256,
`mixed4` prefill and contiguous V cache. Explicit task-specific sizes and prefill modes
can override these defaults. Generation memory budget defaults to 40 GiB on the selected device.

Example argument sets (run using the corresponding container/device setup):

```sh
build-static-310p/bin/umm-cli --model MODEL --execution-policy local_310p \
  --local-device CANN0 --mode text --prompt 'Describe the moon.'
build-rpc-client/bin/umm-cli --model MODEL --execution-policy remote_910b \
  --rpc-endpoint 127.0.0.1:50052 --mode text --prompt 'Describe the moon.'
```

Enable `UMM_BUILD_BACKEND_SMOKE=ON` to build `umm-backend-smoke`. It checks tensor
allocation, chunked upload/readback, F32/F16 tensor views, full-weight Q8_0 MatMul and repeated
execution against a numerical reference. The reference arithmetic is test-only.
These checks do not replace model-level text, vision and generation validation.

KV/embedding staging through the client remains supported and is not zero-copy.
Host control, tokenization, sampling and image processing are permitted; model graphs
must execute on the selected accelerator. Dynamic scheduling is outside this change.

## Diagnostic limitation

Arbitrary offset Q8_0 weight subviews are not supported on this CANN fork.
The regression probe tests complete Q8_0 weights. This is distinct from F16 KV/cache views and
must not be bypassed with CPU fallback. Model validation must still succeed.

## Container runner

Set MODEL_DIR to the frozen package and use scripts/run-static.sh. The remote policy
maps no local NPU. The local policy maps only LOCAL_NPU (default 0). OUTPUT_DIR controls
where images and metadata are saved. The runner defaults to 8192/4096/4096 so the
validated single-image understanding request fits; override N_CTX/N_BATCH/N_UBATCH
for task-specific configurations. The CLI policy defaults remain 2048/512/256.

```sh
MODEL_DIR=/path/to/package scripts/run-static.sh local_310p image 'A red apple on a wooden table'
MODEL_DIR=/path/to/package RPC_ENDPOINT=127.0.0.1:50052 scripts/run-static.sh remote_910b text 'Describe the moon.'
```

For this controlled worker deployment, UMM publishes GGML_RPC_CANN_DEVICE to the SD
backend helper. This preserves the native CANN F16 generation graph over RPC instead
of accidentally using the generic F32 graph. It is a deployment capability hint,
not a protocol-level hardware attestation.

## Worker numerical configuration

Build the worker from the client's llama.cpp checkout, including the CANN fixes,
with `GGML_MAX_NAME=160` for C and C++, `GGML_RPC=ON`, `GGML_CANN=ON`,
`SOC_TYPE=Ascend910B1`, `GGML_RPC_RDMA=OFF` and `USE_ACL_GRAPH=OFF`.
Start it with only the chosen physical NPU mapped into its container:

```sh
GGML_CANN_Q8_NZ=0 GGML_CANN_SATURATION=1 GGML_RPC_NO_RDMA=1 \
  build-rpc-910b/bin/ggml-rpc-server --host 127.0.0.1 --port 50052 --device CANN0 --cache
```

`GGML_CANN_SATURATION=1` explicitly calls `aclrtSetDeviceSatMode` in the CANN
backend. This preserves 310P's finite FP16 overflow behavior on the 910B worker.
Without it, the second U1.5 denoising step can produce Inf at a residual ADD and
NaN at the following RMS_NORM, resulting in a black image. The weights remain
Q8/F16 and the generation graph remains F16; no CPU fallback is introduced.
Record this mode alongside CANN and driver versions when comparing outputs.
The client cannot set a worker process's environment over RPC, so the deployment
must set and verify it explicitly. Worker startup logs confirm the selected mode.

The optional backend probe's fourth argument `saturation` tests positive and
negative FP16 ADD overflow against +/-65504. The `guard_negative` test deliberately
requests a forbidden device and must fail before computation.
