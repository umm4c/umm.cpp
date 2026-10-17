#!/usr/bin/env python3
"""Run a command and sample process memory on one exclusive Ascend device."""

import argparse
import datetime
import json
import os
import re
import signal
import subprocess
import threading
import time
from pathlib import Path


def parse_memory(output):
    if "No process" in output:
        return []
    processes = re.findall(r"Process id:\s*(\d+)\s+Process name:\s*(.*?)\s+Process memory\(MB\):\s*(\d+)", output)
    if processes:
        return [{"pid": int(pid), "name": name, "memory_mb": int(memory)}
                for pid, name, memory in processes]
    rows = []
    for line in output.splitlines():
        fields = line.strip().strip("|").split()
        if len(fields) >= 5 and all(item.isdigit() for item in (fields[0], fields[1], fields[2], fields[-1])):
            rows.append({"pid": int(fields[2]), "name": " ".join(fields[3:-1]),
                         "memory_mb": int(fields[-1])})
    if not rows:
        raise ValueError("unrecognized npu-smi proc-mem output: " + output[-1500:])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--card", type=int, required=True)
    parser.add_argument("--chip", type=int, default=0)
    parser.add_argument("--interval", type=float, default=2)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.interval <= 0:
        parser.error("a command and a positive sampling interval are required")
    args.output.mkdir(parents=True, exist_ok=False)
    stop = threading.Event()
    samples = []
    start = time.monotonic()
    child = None
    pending_signal = None

    def sample(stream):
        tick = time.monotonic()
        row = {"elapsed_s": tick - start,
               "time": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        try:
            result = subprocess.run(["npu-smi", "info", "-t", "proc-mem", "-i", str(args.card),
                                     "-c", str(args.chip)], text=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, timeout=15, check=True)
            processes = parse_memory(result.stdout)
            row.update(processes=processes, process_memory_mb=sum(p["memory_mb"] for p in processes))
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            row.update(process_memory_mb=None, error=str(exc))
        row["collection_s"] = time.monotonic() - tick
        samples.append(row)
        stream.write(json.dumps(row) + "\n")
        stream.flush()

    def monitor(stream):
        while not stop.wait(args.interval):
            sample(stream)

    def forward(signum, _frame):
        nonlocal pending_signal
        pending_signal = signum
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, forward)
    returncode = None
    with (args.output / "memory.jsonl").open("x") as stream:
        sample(stream)
        thread = threading.Thread(target=monitor, args=(stream,))
        thread.start()
        command_start = time.monotonic() - start
        try:
            if pending_signal is not None:
                returncode = -pending_signal
            else:
                child = subprocess.Popen(command, start_new_session=True)
                if pending_signal is not None:
                    forward(pending_signal, None)
                returncode = child.wait()
        finally:
            command_end = time.monotonic() - start
            stop.set()
            thread.join()
            if child is not None:
                sample(stream)
            valid = [r for r in samples if r["process_memory_mb"] is not None]
            during = [r for r in valid if command_start <= r["elapsed_s"] <= command_end]
            boundaries = [command_start] + [r["elapsed_s"] for r in during] + [command_end]
            max_gap = max((b - a for a, b in zip(boundaries, boundaries[1:])), default=0)
            summary = {"command": command, "returncode": returncode, "card": args.card,
                       "chip": args.chip, "interval_s": args.interval,
                       "command_duration_s": command_end - command_start,
                       "samples": len(samples), "valid_samples": len(valid),
                       "error_samples": len(samples) - len(valid),
                       "valid_samples_during_command": len(during), "max_valid_gap_s": max_gap,
                       "sampling_complete": bool(during) and len(valid) == len(samples)
                       and max_gap <= max(3 * args.interval, 5),
                       "peak_process_memory_mb": max((r["process_memory_mb"] for r in during), default=None),
                       "scope": "sum of device processes; exclusive device required; driver memory excluded"}
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return returncode if returncode is not None and returncode >= 0 else 128 - (returncode or 0)


if __name__ == "__main__":
    raise SystemExit(main())
