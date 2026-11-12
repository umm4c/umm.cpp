# Real graph profiler

The profiler accepts **model, devices and representative workload**. It now
executes real SenseNova U1 requests and exports graphs and measured execution
intervals. The synthetic mock implementation has been removed.

## Build and run

Configure the UMM root using the same CANN environment as normal inference:

```bash
cmake -S . -B build-profiler -DGGML_CANN=ON -DGGML_RPC=ON -DSOC_TYPE=Ascend310P3 -DCMAKE_BUILD_TYPE=Release
cmake --build build-profiler --target umm-profiler -j 4
build-profiler/bin/umm-profiler --request tools/profiler/request.example.json --out /path/to/new-profile
```

Run inside the deployment container, with the requested local device and input
paths visible and output parent writable. The RPC endpoint must already be
reachable (for example through the existing SSH tunnel). The profiler does not
start workers or tunnels. Building from `tools/profiler` alone is no longer
supported: the real collector links the production runtime and shared GGML.
No new product feature flag is required; regular inference does not enable the
observer or add graph synchronization.

## Input and execution defaults

Only three top-level fields are accepted:

- `model`: a real SenseNova U1 model-package directory.
- `devices`: a nonempty list of uniquely named device descriptions.
- `representative_workload`: a real workload/v1 directory with manifest.json and cases.jsonl.

A device has `id`, `transport` (`local` or `rpc`) and `backend` (e.g. CANN0).
RPC requires `endpoint`; local must not have a nonempty endpoint. In this first
collector, local devices must be Ascend310P; RPC must expose one CANN0 worker.
Worker chip, physical card, numeric configuration and software are deployment
facts not independently verified by client-side profiling.

The existing text_generation, mixed_understanding, multimodal_prefill and
image_generation request formats are replayed via the normal UMM request path.
Image paths resolve against the workload directory; image and cases hashes are
checked when provided. Input order and image groups are preserved. Malformed
or failed requests abort the profile rather than producing mock success.

Devices run sequentially, with local devices first for canonical input capture.
Each receives six full-workload passes: structure, capture, observer-off baseline,
lightweight timing, observer-off baseline, lightweight timing. Each replay uses the same cases, order and seeds;
requests reset model context through the regular workflow. Reuse one session
per device; record initialization separately. This is not a cold file-cache test
(model hashes are read before loading), nor proof that every allocation is warm.
Use a small representative workload rather than a full quality benchmark.

Runtime defaults are resolved internally and saved in the manifest:

| Workload | Context / batch / microbatch |
| --- | --- |
| Generation only | 2048 / 512 / 256 |
| Any case has more than four images | 32768 / 1152 / 1152 |
| Other understanding or mixed workloads | 8192 / 4096 / 4096 |

Local 310P uses Q8 NZ and saturation. RPC uses its existing worker configuration;
client options do not reconfigure the worker. Known merged 8step-V2 packages
use 8 steps and CFG 1; others use 50 steps and CFG 4, with shift 3. Generation
parameters provided by workload cases use the normal runtime override rules.
No input resizing, truncation or automatic cut search is performed.

Run in a **dedicated process**: backend environment and SD callbacks are
process-global. Concurrent ordinary inference in the same process is unsupported.
The C++ API serializes profiler calls and restores modified environment values;
some backend configuration (including NZ) is latched on first use. Do not initialize
or use backends before calling this API.

## C++ API

`include/profiler.h`: `umm::profiling::request`, `device`, `result`,
`parse_request(json)`, `profile(request)` and `write_result(result, directory)`.
Link `umm-profiler-api`. The public three-field request remains unchanged.
The writer never overwrites an existing directory; its parent must exist.
The CLI rejects existing output before loading. Collection is in memory and
written after success; file-system failures can leave a partial new directory.

## Output v3

| Artifact | Content |
| --- | --- |
| `manifest.json` | Model/workload hashes, original cases, resolved per-device settings, runtime identity, initialization time, coverage and limitations |
| `graphs.jsonl` | Actual graph structure, tensor shapes/strides/types, view/root relationships, op parameters, dependencies and scheduler splits |
| `trace.jsonl` | Each executed scheduler/backend graph, case/device/replay association, phase, submitted token/KV metadata and host interval |
| `costs.jsonl` | Post-warmup statistics per exact graph fingerprint and device; sample count, mean/p50/p95/min/max |
| `requests.jsonl` | Actual responses, stage/token/prefill metrics, output hashes and within-device repeatability |
| `report.md` | Measurement scope and unsupported capabilities |

The manifest and graph/event/cost schemas are v3. The structure pass alone builds
structural fingerprints and exports descriptors. It freezes a per-case execution
sequence with each graph ID and an address-independent contract. The timing pass
references those IDs after checking phase, node order, all tensor names/types,
shapes/strides, operator parameters, topology, aliases, buffer roles/types and
actual node devices. Checks use preallocated arrays and a flat uniqueness table;
no SHA, structural serialization or pointer-map rebuild runs in the timing pass.
This remains a linear metadata check, not zero-cost or constant-time lookup.

New graph objects and allocation addresses are allowed when the complete contract
matches. Changed inputs are allowed; input values are not structure. Changed
structure, extra/fewer executions or changed phase are rejected rather than
silently reusing stale metadata or collecting new structure inside a timing run.
Submission token/KV metadata is recorded on events; tensor shapes describe the
actual micrograph. `graph_id` includes placement; `logical_graph_id` excludes
explicit device/buffer names and scheduler copy prefixes. IDs are build-scoped,
not a portable graph ABI. Node/tensor IDs are local to their graph.

Logical GGML layout is exported; backend-internal packed layout is unknown.
Only explicit GGML views are identified as aliases; general cut legality remains
unproven. Structural fingerprints are build-scoped, not a portable graph ABI.

### Automatic candidate selection and isolated measurement

| Artifact | Content |
| --- | --- |
| `candidates.jsonl` | Stable candidate identity, source graph, closed FFN pattern, eligibility/rejection reason, rank, estimated FLOPs, boundaries, weight identities and selection |
| `snapshots.jsonl` | Canonical real F32 input/reference bytes (base64), hashes, shapes, source case/device and model/weight references |
| `candidate-costs.jsonl` | Per-device isolated compute, copy-only roundtrip, boundary plus compute distributions, preparation cost, numerical error and unavailable reasons |

The first pattern is four nodes: gate and up MatMul, split SwiGLU, down MatMul.
All boundary/intermediate tensors must be contiguous F32 matrices; weights must
be resident, named model weights. Views, nonstandard operands and consumers of
intermediate results outside the candidate are rejected. This deliberately narrow
pattern excludes KV/state mutation; isolated eligibility is not general production
cut legality. Original node parameters are preserved when reconstructing it.

Candidates rank by estimated matrix FLOPs per boundary byte (not measured benefit).
Choose one layer per input shape, up to four first-observed shapes. The selection
freezes after the first device structure pass. This bounded heuristic does not
claim every layer with the same shape has equal performance. Stable candidate IDs
contain model component hashes, weight names/types/shapes, node math and shapes;
they exclude placement and changing KV context.

A separate untimed capture pass uses scheduler execution callbacks to read the
candidate input after its producer and reference after the candidate. Reading at
the end of the whole graph would be unsafe because graph allocation reuses memory.
Canonical snapshots are shared across devices. Each device binds its own live
model weight tensors while its session exists; no candidate weight readback,
reconversion or repeated upload is required. Missing snapshots/bindings or backend
operator rejection produce explicit unavailable records.

Isolated measurements use two warmups and five repetitions. Native compute is
measured separately from preparation and full boundary roundtrip through the
actual GGML tensor-copy path, with the first local device as primary. Copy-only and
combined measurements are separate experiments; do not assume additive costs.
The local-to-local roundtrip is an artificial copy control; compare a remote
roundtrip against native local compute for a first indication. Adjacent remote
candidates need a full-plan model to avoid charging redundant roundtrips.
No local primary means boundary costs are unavailable. No CPU compute fallback
is used by the real device collector.

Every candidate output is checked against the canonical reference (F32,
`abs(error) <= .05 + .005 * abs(reference)`); max absolute error, RMSE, finiteness
and exact equality are recorded. These are initial measurement tolerances, not
model-quality acceptance. Weight upload time remains null: weights are already
resident and their original loading is not separately decomposed.

Phase metadata covers understanding prefill/decode and native U1 vision graphs.
Generation is a coarse phase, without individual conditioning/diffusion/VAE graph
labels. Exact tensor shapes remain available for unclassified graphs.

**Timing is synchronized host wall time**, including submission and waits, not
device kernel time. The legacy `export_before_ms` field records structure work in structure events
and contract-check/lookup preparation in timing events. Full JSON
export and candidate discovery occur only in the structure pass.
Scheduler intervals contain backend intervals and child export overhead; they
must not be summed. Event start_ms is relative to its per-device collector origin. Observation adds synchronization and changes overlap, so
these values are not uninstrumented production latency. RPC completion is fenced
with a bounded response-bearing tensor read (up to 4096 bytes); its round-trip/read overhead is
included, since ordinary RPC synchronization only drains client submissions. Only timing-pass events enter
cost statistics; a percentile with few samples is descriptive only.

On successful scheduler execution without an eval callback, child backend events
already fence the device streams. The scheduler retains its final synchronization
but reuses those completion fences instead of performing another RPC read. Direct
backend calls, scheduler entry, callback execution and failure paths retain their
completion reads. A single-split RPC scheduler call therefore uses three reads
instead of four; backend event timing still includes its own completion fence.

General cut legality, per-operator timing, pure network/DMA costs, overlap, memory
peaks and remote worker metrics remain unavailable. Actual tensor-copy roundtrip
cost is supplied for selected FFNs, not a general transfer model. The manifest
records paired observer-off/timing request walls; their difference includes runtime
variation and must not be used as a universal correction factor.
Production placement is unchanged; isolated copies of selected subgraphs are executed. Output repeatability is checked
within each device; it is not cross-device logits or task accuracy validation.
`decision_usable` remains false until profiling overhead, coverage, reference
numerics and prediction have been validated. Real measurements must not yet be
used as a validated scheduling cost model.

## Checks

```bash
cmake --build build-profiler --target umm-profiler-observer-test umm-profiler-candidate-test umm-profiler-contract-test -j 4
build-profiler/bin/umm-profiler-candidate-test
build-profiler/bin/umm-profiler-contract-test
build-profiler/bin/umm-profiler-observer-test
python3 tools/profiler/test_interface.py build-profiler/bin/umm-profiler
python3 tools/profiler/validate_capture.py /path/to/new-profile
```

The observer check uses CPU graphs to cover direct/scheduler/reused execution,
thread isolation, synchronized output, abort and disable. A simulated RPC queue
also checks completion before END, exact read counts, and work queued by an eval
callback after child completion. CLI checks cover request
validation, missing inputs without mock success and rejection of existing output.
The candidate check covers topology closure, capture at execution boundaries, exact
CPU replay, boundary timing, and rejection of an extra intermediate consumer.
Real-model captures and paired observer-off controls are separate integration checks.

The completion-read size is intentional: an integration probe through the deployed
SSH tunnel showed roughly 44 ms for 1/64-byte responses versus about 2.2 ms for
4096-byte responses. This localizes the disturbance to the small-read path; it
is not proof of the exact TCP/SSH mechanism. The bounded read preserves ordered
server completion without changing the worker/protocol. Small graphs with no
4096-byte result may still incur that disturbance. Compute records include the
completion-read overhead and are not pure device compute. Full roundtrip timing
uses its result copy as the completion fence, avoiding an extra read.

Optional RPC integration check against an existing worker:

The probe also executes an explicitly placed CPU/RPC/CPU graph three times with
changed inputs. It verifies the host-resident final output inside scheduler END,
without an additional RPC read that could hide an early completion callback.

```bash
cmake --build build-profiler --target umm-profiler-rpc-fence-test -j 4
GGML_RPC_NO_RDMA=1 build-profiler/bin/umm-profiler-rpc-fence-test 127.0.0.1:50052
```

## Offline graph viewer

[Graph Atlas](../../../utils/profiler-viewer/README.md) packages existing profiler output into an offline
interactive page, with phase/device/length filters, folded layers, operator
inspection, exact candidate boundaries and measured cost comparisons. It does not
run the model or change scheduling. Previewed plans are clearly marked unexecuted.

```bash
python3 ../utils/profiler-viewer/export_viewer.py --profile /path/to/profile --out /path/to/graph-viewer.html
```

The viewer is maintained in the independent `workspace/utils` repository. To serve
an existing profile directly, run from `workspace/utils`:

```bash
python3 profiler-viewer/serve.py --profile /path/to/profile --port 8787
```

Open `http://127.0.0.1:8787/`. Only the page and health endpoint are served; the
profile directory is not exposed as a file server.
