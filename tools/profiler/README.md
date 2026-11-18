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

Devices run sequentially. The structure pass and two baseline/timing pairs run
first. Separate `layer` and `block` passes collect understanding graph timings.
Add `--operators` after the output path to collect an independent operator pass.
Replays use the same cases, order and seeds and reset model context through the regular workflow.
One session per device keeps its weights loaded across all passes. Use a small
representative workload for calibration.

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
| `timings.jsonl` | Steady-state backend timings grouped by logical graph, case, submitted tokens and cached context |
| `hierarchy.jsonl` | CANN stream durations for layers, blocks and optional operators, with physical graph references and half-open node ranges |
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

### Timings associated with the graph

`timings.jsonl` uses `umm-profiler-timing/v1`. Each row identifies an exact
`logical_graph_id`, a workload case and the token/context metadata of the actual
submission. Its `timings` entries refer to physical graphs by device and graph ID.
Each entry covers a half-open `node_range` in that graph and records a median,
sample count and timing scope. Entries are derived after collection from completed
backend timing events. A graph with a different structural fingerprint gets its
own row, even if it has the same stage name or execution order.

`timings.jsonl` covers whole backend submissions. A measured range can
cover a cut candidate when the range fits entirely inside the candidate and its
device, inputs and timing scope match. A partial range has unknown time. Request
baselines in `requests.jsonl` provide the coarser end-to-end reference. Client
submission and inclusive scheduler intervals stay in the trace and must be
kept separate from backend timings.

`hierarchy.jsonl` provides finer ranges for U1.5 understanding graphs. Each
record references the same `graphs.jsonl` and has one coverage level. Norm and
the first residual belong to the Attention block, and the FFN block includes
its Norm and the layer output residual. Preamble and epilogue have their own
ranges. Fused operators share an interval. View-only nodes are grouped with
adjacent compute. Each range is
measured between events on the CANN stream. An interval can include idle time
between launches. The collector reads the events after graph completion.
With ACL graph capture active, or when a backend cannot provide the requested
timing, the record is marked unavailable. Invalid event intervals have null
durations. The tool follows the actual execution mode throughout the measurement.

Remote range timing requires a 6.3 worker built from the matching GGML. The
ordinary backend timing continues to support 6.1 and 6.2 workers. A fine RPC
pass queues its boundaries for the next graph and reads back the ranges once
at graph completion. This adds a response per profiled graph, with no
per-operator RPC or stream synchronization. Prefer the coarsest matching range
when scoring a cut. Parent and child ranges and different timing scopes are
alternative descriptions of the same work.

The regular profiler no longer selects or replays a fixed set of FFNs. The
three previous candidate files are emitted empty for existing bundle readers.
The isolated FFN fixture remains in the standalone candidate test. See
[Profiler design](DESIGN.md) for planned finer measurements and scheduling inputs.

Phase metadata covers understanding prefill/decode and native U1 vision graphs.
Generation is a coarse phase, without individual conditioning/diffusion/VAE graph
labels. Exact tensor shapes remain available for unclassified graphs.

**Completed timing is synchronized host wall time, not device kernel time.**
Local timing and structure observation retain their existing waits.
RPC timing passes require a protocol 6.1 worker and instead open a window per
request, record client submissions without observer-added waits, and collect
worker records in one final response. Normal logits, activation and result reads
are unchanged. Ordinary inference does not send profiling commands. A 6.1 client
can use a 6.0 worker for ordinary inference, but batch profiling rejects it before
sending unknown commands; a 6.0 client cannot connect to a 6.1 worker.

The legacy `export_before_ms` field records structure work in structure events
and contract-check/lookup preparation in timing events. Full JSON
export occurs only in the structure pass.
Client events use their per-device collector clock. Synchronized client scheduler
intervals contain backend intervals and child preparation, so must not be summed.
RPC timing emits `event_type: submitted`, `timing_scope: client_submission_host_wall`;
its after callback means submission returned, not completion. These events never
enter compute cost statistics. After collection, each backend submission has a
separate `completed` worker event linked by `submission_event_id`, with
`timing_scope: rpc_worker_synchronized_host_wall` and an independent per-window
`clock_domain`. Worker start_ms is relative to BEGIN on the worker; prepare time
is separate from synchronized compute. Worker events have no client parent.
Never subtract timestamps across clock domains or equate worker backend time to
full scheduler/network time. Only completed timing events enter cost statistics;
RPC scheduler graphs therefore remain unmeasured. A percentile with few samples
is descriptive only. Request wall time includes the final batch collection.

For synchronized observation (including structure), successful scheduler execution without an eval callback means child backend events
already fence the device streams. The scheduler retains its final synchronization
but reuses those completion fences instead of performing another RPC read. Direct
backend calls, scheduler entry, callback execution and failure paths retain their
completion reads. A single-split RPC scheduler call therefore uses three reads
instead of four; backend event timing still includes its own completion fence.
This does not apply to RPC timing's submission observer, which adds no completion
reads at graph boundaries. BEGIN is ordered but has no response; END returns a
bounded, at least 4096-byte response and validates count, device, node count,
GRAPH/RECOMPUTE and submission sequence. Capacity comes from the frozen case
sequence (minimum 64, maximum 4096 records). Overflow or mismatched records fail
the request. The collector exclusively owns the endpoint while a window is active.

General cut legality, transfer calibration and memory peaks are
still needed for a validated scheduling model. The manifest records paired
observer-off/timing request walls. Their difference includes runtime variation.
Output repeatability checks compare runs within each device. The bundle still
sets `decision_usable: false`.

## Checks

Structure capture now collects one `graph_description`, shared by JSON export,
placement/logical identity and the frozen contract. Timed checks still traverse
preallocated contract storage without JSON, hashing or old tensor addresses.
Graph IDs and the v3 output schema are preserved. SHA256 and Base64 encoding
reuse the vendored implementations; file hashes remain streaming and image-input
validation is unchanged. Graph/candidate timing distributions share one helper.

Keep generic operator microbenchmarks outside this profiler. The deleted upstream
operator harness is not restored or exposed as a build target.

For generic text throughput, use `third_party/llama.cpp/tools/llama-bench` and
`scripts/compare-llama-bench.py` in a standalone llama.cpp tools/common build.
Match model/context/backend/NZ settings first; when using RPC, both sides must
also use `GGML_MAX_NAME=160`. This does not substitute for real U1 multimodal
workloads or native-resident candidate replay. Graph-copy comparison remains a
diagnostic path because it allocates/copies reachable tensors, including weights.

Offline hardware inspection uses [msprof-import](../../../utils/msprof-import/README.md)
with existing CANN exports. It does not collect hardware traces inside ordinary
timing passes or infer cross-host clock offsets/node-to-kernel mappings.

```bash
cmake --build build-profiler --target umm-profiler-observer-test umm-profiler-candidate-test umm-profiler-contract-test -j 4
build-profiler/bin/umm-profiler-candidate-test
build-profiler/bin/umm-profiler-contract-test
build-profiler/bin/umm-profiler-observer-test
python3 tests/profiling/test_interface.py build-profiler/bin/umm-profiler
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

Candidate replay still uses completion-read timing; it is not converted into
worker-only cost by this change. The completion-read size is intentional: an integration probe through the deployed
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
It validates batch ordering, new/cached graph execution, window state, overflow,
empty windows and reuse. Pass `legacy` after an older worker endpoint to verify
that profiling is rejected locally without sending a new command.

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

RPC transport/cache costs depend on the endpoint and worker version. Protocol 6.2
keeps immutable `WEIGHTS` uploads cached and sends large dynamic buffers without
content hashing or disk caching; 6.0/6.1 workers keep the old behavior. Re-measure
candidate boundaries after changing either the route or worker policy. Repeated
snapshot inputs may hit the old activation cache, so their timings do not model
all changing-input workloads. Keep endpoint, SSH route/state, worker binary and
cache policy with the capture; do not reuse old boundary costs across these changes.
