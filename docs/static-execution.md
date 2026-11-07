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

## Reproducible worker deployment

See [scripts/rpc/README.md](../scripts/rpc/README.md) for the pinned worker image,
parameterized build/start scripts, SSH tunnel and three-task validation procedure.
Machine-specific settings and credentials remain outside the repository.

## Random partition integration probe

`UMM_BUILD_BACKEND_SMOKE=ON` also builds `umm-random-plan-smoke` when CANN and RPC
are enabled together. It exercises the shared `random-plan.h` interface without
changing the two production static policies. It is not yet a model inference policy.

```sh
build-random-dispatch/bin/umm-random-plan-smoke 127.0.0.1:50052 CANN0 4
```

Use the validated CANN container with one local NPU mapped and the existing SSH
forward. Set `GGML_RPC_NO_RDMA=1`, `GGML_CANN_Q8_NZ=0`, and
`GGML_CANN_SATURATION=1`. The worker must retain its validated saturation setting.
Build with `GGML_CANN=ON`, `GGML_RPC=ON`, `GGML_RPC_RDMA=OFF`,
`SOC_TYPE=Ascend310P3`, and `USE_ACL_GRAPH=OFF` in a separate build directory.

The probe tests one cut, two cuts, then a cut between each pair of compute nodes.
For each seed it picks cuts and a starting device, then tests the opposite mapping.
Views stay with their storage owner; allocated/in-place outputs cannot migrate.
Unsupported placements are rejected. Multi-cut search is bounded to 64 attempts;
failure means no valid sampled candidate was found, not proof that none exists.

Apply a plan after scheduler reset, then call `ggml_backend_sched_alloc_graph`
explicitly, check its resulting assignments, and only then execute. Plans hold graph
pointers and are valid only for that graph's lifetime. Repeating a seed reproduces
selection for the same graph/build; portable serialized plan replay is not implemented.

The probe checks F32/F16 weights with F32 activations in a branched MatMul/SiLU/scale/residual graph, a view, changed inputs
on repeated execution, and rejection of pinned in-place alias partitions. It checks
intermediates and outputs against a host arithmetic reference and compares final
outputs to both pure-device executions with absolute tolerance 0.001. No CPU graph
compute fallback is permitted. This does not validate model KV state, general dtype
or operator coverage, model weight residency, or mixed model inference. Those require
the next text/vision/generation integration steps.

## Real-model random placement probe

`umm-random-model-run CONFIG.json NEW_OUTPUT_DIR` runs the U1.5 understanding GGUF
through the normal llama adapter. The adapter's optional secondary backend keeps
weights and KV on the primary device and exposes both backends to the scheduler.
The GGML placement callback runs before allocation and validates assignments after
allocation. Its default is null, preserving normal scheduler behavior.

Example configuration for the reference run:

```json
{
  "mode": "local",
  "model": "/model/understanding.gguf",
  "local_device": "CANN0",
  "n_ctx": 512,
  "n_batch": 128,
  "max_tokens": 12,
  "requests": [{"id": "count-r0", "prompt": "Count from 1 to 5, separated by commas, without any other text."}]
}
```

For comparisons use the same requests and sizes. Set mode to `remote` or `random`,
add endpoint `127.0.0.1:50052`, and set `reference_dir` to the local run's output.
Random mode accepts seed, cuts (tested on the real model with 1), and flip_devices.
Use the same CANN+RPC container/device setup and numerical environment as the small
probe. Q8 NZ is disabled in all modes for this comparison. The output directory must
not exist. The generic CLI/session execution-policy options remain the two static
policies; this experimental model runner is a separate entry point.

Each request records load/request/forward timings, token predictions and output.
`graphs.jsonl` stores planned cuts and actual split ranges, backend, node names,
operators, tensor dtype and shape. `forwards.jsonl` links each prefill/decode to the
active graph ID, including graph reuse. Each step saves full float32 logits.
Compared runs use the reference continuation (teacher forcing) to keep prefixes
identical; top-1 agreement is not an answer-correctness benchmark. Logit tolerances
are configured before execution (defaults: atol 0.05, rtol 0.005).

Recorded model_ms includes the allocation callback's instrumentation. Request wall
time also includes snapshot IO and comparison; model load time is separate.
Callback metadata timing includes record preparation. These measurements describe
the experimental runner, not uninstrumented production latency. RPC weight-input
copies can dominate random placement. Initial reserve is on the primary device,
so later dynamic placement can increase remote buffers beyond that initial estimate.

Real-model single-cut validation does not establish support for arbitrary KV
migration, multi-cut model plans, visual input or diffusion graphs. Cut locations
remain constrained by persistent storage ownership. Results and complete configs
are under results/random-model-20261105 in the shared workspace.

## Session-local replicated weights

For the real-model random probe, `replicate_weights: true` eagerly prepares one
copy of each understanding weight on each selected backend. Existing weights on
the primary device are reused. The other copy is allocated outside the scheduler's
temporary graph allocator and uploaded once per model session using backend tensor
copy (canonical get/set when needed). The replica upload is reported separately
as `replica_init_ms`; it is not part of request `model_ms`.

The placement callback binds each compute node's weight inputs to its selected
backend, including views with offsets and strides. Replica buffers are marked
WEIGHTS and survive graph rebuilds, decode reuse and request KV resets. KV remains
on the primary device. Activations and mutable state still use scheduler transfers.
The callback rejects any resulting split that still lists weight inputs to copy.

`weight_residency` reports tensor count, logical bytes, allocated buffer bytes and
uploaded logical bytes for each device. Graph and forward records include
`weight_input_bytes`: the sum of logical weight input sizes listed for copying by
the scheduler on each execution. This is not a measured network byte count; expert
slicing and backend transport can change physical bytes. The current U1.5 text
probe has no expert slicing. The count excludes initialization uploads.

This option currently covers the experimental text runner with Q8 NZ disabled.
It does not add remote weight persistence across client process exits or change
the RPC wire protocol. The two static production policies remain unchanged.
