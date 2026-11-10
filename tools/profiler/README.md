# Profiler interface (v1 prototype)

This version returns **mock data only**. It does not load a model or workload,
contact devices, execute graphs, or implement profiling or scheduling.

## Build and run

The tool can be built independently without GGML, CANN or inference libraries:

```bash
cmake -S tools/profiler -B build-profiler -DCMAKE_BUILD_TYPE=Release
cmake --build build-profiler -j 2
build-profiler/umm-profiler --request tools/profiler/request.example.json --out /tmp/new-profile
python3 tools/profiler/test_interface.py build-profiler/umm-profiler
```

The parent build also provides `umm-profiler` and `umm-profiler-api` targets.
No new runtime feature flag is introduced. Normal inference does not call this API.

## Input

Only three top-level fields are accepted:

- `model`: nonempty model-package path string.
- `devices`: nonempty list of device descriptions.
- `representative_workload`: nonempty path string to a representative workload directory.

Each device has a unique `id`, `transport` (`local` or `rpc`), and a `backend`
(device name such as `CANN0`, interpreted on the local host or remote worker).
RPC descriptions additionally require a nonempty `endpoint`; local descriptions
must not specify a nonempty endpoint. The endpoint refers to an existing RPC
connection, such as an SSH-forwarded address. This prototype does not open it.
A strategy is not part of the request. Measurement policies remain internal.

See `tools/profiler/request.example.json`. Paths are opaque in this prototype:
existence, model identity, workload format and device capabilities are not checked.
No buckets are extracted from the real workload yet.

## C++ API

`include/profiler.h` exposes `umm::profiling::request`, `device`, `result`,
`parse_request(json)`, `profile(request)` and `write_result(result, directory)`.
Link the `umm-profiler-api` target; no inference session is required.
The result contains manifest, graph, trace and cost records plus a Markdown report.
The writer requires a new directory whose parent exists, and never overwrites
an existing output directory. Invalid requests throw before writing artifacts;
file-system errors can leave an incomplete new output directory.

## Output

| Artifact | Contract |
| --- | --- |
| `manifest.json` | `umm-profiler/v1`; request echo, identity/coverage placeholders, artifact list |
| `graphs.jsonl` | `umm-profiler-graph/v1`; graph/phase/shape, tensors, node dependencies, aliases, state and boundary placeholders |
| `trace.jsonl` | `umm-profiler-event/v1`; graph/node/split/device associations and synthetic time intervals |
| `costs.jsonl` | `umm-profiler-cost/v1`; device-specific cost records, statistics, residency/memory/transfer placeholders |
| `report.md` | Explicit mock status and limitations |

All records carry `data_origin: mock`. The manifest additionally declares
`executed: false` and `decision_usable: false`. Costs have zero real samples and
`measurement_valid: false`. The one-node ADD graph and 1 ms values are synthetic
fixtures independent of the requested model, content, shape or hardware.
Unavailable real measurements are `null`, not zero. No actual device placement,
support, correctness, legal cuts or performance benefits are asserted.
Consumers must reject mock/invalid records for scheduling decisions.

Schema versions identify the prototype contract; incompatible changes require
a version change. Future collectors should replace mock production behind this
API while keeping the user inputs limited to model, devices and workload.
