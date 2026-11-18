"""Validate a real profiler bundle: python3 validate_capture.py PROFILE_DIRECTORY."""
from collections import Counter
import base64
import hashlib
import math
import re
import json
from statistics import median
from pathlib import Path
import sys


def validate(root):
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["schema_version"] == "umm-profiler/v3"
    assert manifest["data_origin"] == "measured" and manifest["executed"]
    assert not manifest["decision_usable"]
    def rows(name):
        return [json.loads(line) for line in (root / name).read_text().splitlines() if line.strip()]
    graphs, events, costs, requests = (rows(name + ".jsonl") for name in ["graphs", "trace", "costs", "requests"])
    assert graphs and events and costs and requests
    by_graph = {(x["device_id"], x["graph_id"]): x for x in graphs}
    assert len(by_graph) == len(graphs)
    event_map = {x["event_id"]: x for x in events}
    assert len(event_map) == len(events)
    for graph in graphs:
        assert graph["data_origin"] == "measured"
        tensors = {x["id"] for x in graph["tensors"]}
        nodes = {x["id"] for x in graph["nodes"]}
        for tensor in graph["tensors"]:
            assert tensor["storage_id"] in tensors
            assert tensor["alias_of"] is None or tensor["alias_of"] in tensors
        for node in graph["nodes"]:
            assert set(node["inputs"]) <= tensors and set(node["outputs"]) <= tensors
            assert set(node["dependencies"]) <= nodes
    # Equivalent descriptors must share their logical ID despite scheduler copy prefixes.
    backend_names = {d["device_id"]: d["resolved_options"]["--understanding-backend"] for d in manifest["identity"]["devices"]}
    logical_groups = {}
    for graph in graphs:
        tensors = []
        prefix = re.compile(r"^" + re.escape(backend_names[graph["device_id"]]) + r"(?:\[[^]]+\])?#")
        for tensor in graph["tensors"]:
            t = {k: tensor[k] for k in ["id", "name", "shape", "strides_bytes", "op_params_bytes", "flags", "alias_of", "view_offset_bytes", "dtype"]}
            t["name"] = prefix.sub("", t["name"], count=1)
            tensors.append(t)
        nodes = [{k: n[k] for k in ["id", "tensor_id", "op", "inputs", "dependencies", "outputs"]} for n in graph["nodes"]]
        key = json.dumps([graph["kind"], graph["phase"], tensors, nodes], sort_keys=True)
        if key in logical_groups:
            assert logical_groups[key] == graph["logical_graph_id"], "Placement leaked into logical graph identity"
        logical_groups[key] = graph["logical_graph_id"]
    sample_counts = Counter()
    for event in events:
        assert event["data_origin"] == "measured" and event["executed"] and event["status"] == 0
        assert (event["device_id"], event["graph_id"]) in by_graph
        assert event["duration_ms"] >= 0
        if event["timing_scope"] == "rpc_worker_synchronized_host_wall":
            submitted = event_map[event["submission_event_id"]]
            assert submitted["event_type"] == "submitted" and event["event_type"] == "completed"
            assert submitted["graph_id"] == event["graph_id"] and submitted["device_id"] == event["device_id"]
            assert submitted["case_id"] == event["case_id"] and submitted["replay"] == event["replay"]
            assert submitted["clock_domain"] != event["clock_domain"]
            assert event["parent_event_id"] is None and event["kind"] == "backend"
            assert event["worker_nodes"] == len(by_graph[event["device_id"], event["graph_id"]]["nodes"])
        if event.get("event_type") == "submitted":
            assert event["timing_scope"] == "client_submission_host_wall"
        parent = event["parent_event_id"]
        if parent is not None:
            assert parent in event_map
            assert event_map[parent]["device_id"] == event["device_id"]
            assert event_map[parent].get("clock_domain") == event.get("clock_domain")
            assert event_map[parent]["start_ms"] <= event["start_ms"]
            assert event["start_ms"] + event["duration_ms"] <= event_map[parent]["start_ms"] + event_map[parent]["duration_ms"] + .01
        if not event["warmup"] and event.get("event_type") != "submitted":
            sample_counts[event["device_id"], event["graph_id"]] += 1
    assert {(x["device_id"], x["graph_id"]) for x in costs} == set(sample_counts)
    for cost in costs:
        assert cost["sample_count"] == sample_counts[cost["device_id"], cost["graph_id"]]
        assert cost["measurement_valid"] and not cost["decision_usable"]
        assert cost["duration_ms"]["min"] <= cost["duration_ms"]["p50"] <= cost["duration_ms"]["p95"] <= cost["duration_ms"]["max"]
    passes = manifest["measurement"]["passes"]
    expected = {(d["id"], c["case_id"], i) for d in manifest["request"]["devices"] for c in manifest["workload"]["cases"] for i in range(len(passes))}
    assert {(x["device_id"], x["case_id"], x["replay"]) for x in requests} == expected
    assert len(requests) == len(expected)
    assert all(x["response"]["status"] == "ok" for x in requests)
    task_types = {c["case_id"]: c["task_class"] for c in manifest["workload"]["cases"]}
    request_map = {(x["device_id"], x["case_id"], x["replay"]): x for x in requests}
    for event in events:
        if task_types[event["case_id"]] != "text_generation":
            continue
        graph = by_graph[event["device_id"], event["graph_id"]]
        shape = event["submission"]
        prompt_tokens = request_map[event["device_id"], event["case_id"], event["replay"]]["response"]["token_counts"]["input"]
        if shape["tokens"] == 1 and shape["cached_tokens"] is not None and shape["cached_tokens"] >= prompt_tokens:
            assert graph["phase"] == "understanding_decode", "Text decode mislabeled as prefill"

    candidates, snapshots, candidate_costs = (rows(name + ".jsonl") for name in ["candidates", "snapshots", "candidate-costs"])
    by_candidate = {c["candidate_id"]: c for c in candidates}
    assert len(by_candidate) == len(candidates)
    assert sum(c["selected"] for c in candidates) <= 4
    for candidate in candidates:
        for origin in candidate["origins"]:
            graph = by_graph[origin["device_id"], origin["graph_id"]]
            assert all(0 <= i < len(graph["nodes"]) for i in origin["node_indices"])
            if candidate["isolated_replay_eligible"]:
                assert len(origin["node_indices"]) == 4
                assert [graph["nodes"][i]["op"] for i in origin["node_indices"]] == ["MUL_MAT", "MUL_MAT", "GLU", "MUL_MAT"]
    for snap in snapshots:
        candidate = by_candidate[snap["candidate_id"]]
        assert candidate["selected"] and candidate["isolated_replay_eligible"]
        for field, shape in [("input", "input_shape"), ("reference", "output_shape")]:
            raw = base64.b64decode(snap[field], validate=True)
            assert hashlib.sha256(raw).hexdigest() == snap[field + "_sha256"]
            assert len(raw) == 4 * math.prod(snap[shape])
    for cost in candidate_costs:
        assert by_candidate[cost["candidate_id"]]["selected"]
        assert not cost["decision_usable"]
        if cost["status"] == "completed":
            assert cost["compute_ms"]["sample_count"] == 5
            assert cost["measurement_valid"] == (cost["accuracy"]["all_finite"] and cost["accuracy"]["within_tolerance"])
    for req in requests:
        assert req["pass"] == passes[req["replay"]]

    annotations = []
    if "timings.jsonl" in manifest["artifacts"]:
        annotations = rows("timings.jsonl")
        expected_samples = {}
        for event in events:
            if (event["pass"], event["event_type"], event["kind"]) != ("timing", "completed", "backend") or event["warmup"]:
                continue
            graph = by_graph[event["device_id"], event["graph_id"]]
            shape = event["submission"]
            key = (graph["logical_graph_id"], event["case_id"], shape["tokens"], shape["cached_tokens"],
                   event["device_id"], event["graph_id"], event["timing_scope"])
            expected_samples.setdefault(key, []).append(event["duration_ms"])

        seen_inputs = set()
        for row in annotations:
            assert row["schema_version"] == "umm-profiler-timing/v1"
            shape = row["submission"]
            key = (row["logical_graph_id"], row["case_id"], shape["tokens"], shape["cached_tokens"])
            assert key not in seen_inputs and row["timings"]
            seen_inputs.add(key)
            for timing in row["timings"]:
                graph = by_graph[timing["device_id"], timing["graph_id"]]
                assert graph["kind"] == "backend" and graph["logical_graph_id"] == row["logical_graph_id"]
                assert timing["node_range"] == [0, len(graph["nodes"])]
                source = (timing["device_id"], timing["graph_id"], timing["timing_scope"])
                samples = expected_samples.pop(key + source)
                assert timing["sample_count"] == len(samples)
                assert math.isclose(timing["median_ms"], median(samples))
        assert not expected_samples, "Completed backend timings missing from annotations"

    hierarchy_counts = Counter()
    if "hierarchy.jsonl" in manifest["artifacts"]:
        def hierarchy_key(row):
            submission = row["submission"]
            shape = tuple(None if value is None or value < 0 else value
                          for value in (submission["tokens"], submission["cached_tokens"]))
            return row["device_id"], row["case_id"], row["graph_id"], shape

        fine_replays = [i for i, name in enumerate(passes) if name in ("layer", "block", "operator")]
        expected_hierarchy = Counter(
            (replay, hierarchy_key(event))
            for event in events
            if event["pass"] == "structure" and event["kind"] == "backend"
            and event["phase"] in ("understanding_prefill", "understanding_decode")
            for replay in fine_replays
        )
        recorded_hierarchy = Counter()
        for row in rows("hierarchy.jsonl"):
            assert row["schema_version"] == "umm-profiler-hierarchy/v1"
            graph = by_graph[row["device_id"], row["graph_id"]]
            assert graph["kind"] == "backend" and graph["phase"] in ("understanding_prefill", "understanding_decode")
            assert row["level"] in ("layer", "block", "operator") and row["level"] == passes[row["replay"]]
            assert (row["device_id"], row["case_id"], row["replay"]) in request_map
            recorded_hierarchy[row["replay"], hierarchy_key(row)] += 1
            hierarchy_counts["/".join((row["device_id"], row["level"], row["status"]))] += 1
            if row["status"] == "unavailable":
                assert row["reason"] and "spans" not in row
                continue
            assert row["status"] in ("completed", "partial") and row["timing_scope"] == "cann_stream_events"
            end = 0
            missing = 0
            for span in row["spans"]:
                begin, next_end = span["node_range"]
                assert begin == end and begin < next_end and span["labels"]
                if span["elapsed_ms"] is None:
                    assert span["reason"] == "invalid_stream_elapsed_time"
                    missing += 1
                else:
                    assert math.isfinite(span["elapsed_ms"]) and span["elapsed_ms"] >= 0
                end = next_end
            assert end == len(graph["nodes"]), "Fine timings must cover the submitted node sequence once"
            assert bool(missing) == (row["status"] == "partial")
        assert recorded_hierarchy == expected_hierarchy, "Fine timing records differ from the frozen execution sequence"

    return {"status": "passed", "candidates": len(candidates), "selected": sum(c["selected"] for c in candidates), "snapshots": len(snapshots), "candidate_costs": len(candidate_costs), "graphs": len(graphs), "events": len(events), "cost_records": len(costs), "timing_annotations": len(annotations), "hierarchy": dict(hierarchy_counts), "requests": len(requests), "repeatable": all(x["matches_warmup"] for x in requests)}


if __name__ == "__main__":
    print(json.dumps(validate(Path(sys.argv[1])), indent=2))
