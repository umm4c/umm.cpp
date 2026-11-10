"""Profiler contract checks. Usage: python3 test_interface.py /path/to/umm-profiler"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

BINARY = str(Path(sys.argv.pop(1)).resolve())


class InterfaceTests(unittest.TestCase):
    def request(self):
        return {"model": "/does-not-exist/model", "representative_workload": "/does-not-exist/workload",
                "devices": [{"id": "local", "transport": "local", "backend": "CANN0"},
                            {"id": "remote", "transport": "rpc", "backend": "CANN0", "endpoint": "unreachable.invalid:50052"}]}

    def run_request(self, root, request):
        path = root / "request.json"
        path.write_text(json.dumps(request))
        return subprocess.run([BINARY, "--request", str(path), "--out", str(root / "output")],
                              capture_output=True, text=True, timeout=10)

    def test_mock_contract_without_model_or_devices(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = self.request()
            run = self.run_request(root, request)
            self.assertEqual(run.returncode, 0, run.stderr)
            output = root / "output"
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["request"]["model"], request["model"])
            self.assertEqual(manifest["status"], "mock")
            self.assertFalse(manifest["executed"])
            self.assertFalse(manifest["decision_usable"])
            graphs = [json.loads(x) for x in (output / "graphs.jsonl").read_text().splitlines()]
            graph_ids = {g["graph_id"] for g in graphs}
            for filename in ["trace.jsonl", "costs.jsonl"]:
                records = [json.loads(x) for x in (output / filename).read_text().splitlines()]
                self.assertEqual({x["device_id"] for x in records}, {"local", "remote"})
                for item in records:
                    self.assertIn(item["graph_id"], graph_ids)
                    self.assertEqual(item["data_origin"], "mock")
                    if filename == "costs.jsonl":
                        self.assertFalse(item["measurement_valid"])
                        self.assertEqual(item["sample_count"], 0)
            self.assertIn("MOCK DATA", (output / "report.md").read_text())

    def test_invalid_requests_do_not_create_output(self):
        bad = [dict(self.request(), strategy="random"), dict(self.request(), devices=[])]
        duplicate = self.request()
        duplicate["devices"][1]["id"] = "local"
        bad.append(duplicate)
        missing_endpoint = self.request()
        del missing_endpoint["devices"][1]["endpoint"]
        bad.append(missing_endpoint)
        bad.append(dict(self.request(), model=123))
        for request in bad:
            with self.subTest(request=request), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.assertNotEqual(self.run_request(root, request).returncode, 0)
                self.assertFalse((root / "output").exists())

    def test_existing_results_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(self.run_request(root, self.request()).returncode, 0)
            before = {p.name: p.read_bytes() for p in (root / "output").iterdir()}
            self.assertNotEqual(self.run_request(root, self.request()).returncode, 0)
            self.assertEqual(before, {p.name: p.read_bytes() for p in (root / "output").iterdir()})


if __name__ == "__main__":
    unittest.main()
