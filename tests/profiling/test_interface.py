"""Profiler validation checks. Usage: python3 test_interface.py /path/to/umm-profiler"""
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
                              capture_output=True, text=True, timeout=30)

    def test_invalid_requests_do_not_create_output(self):
        bad = [(dict(self.request(), strategy="random"), "Unknown profiler field"),
               (dict(self.request(), devices=[]), "must be nonempty")]
        duplicate = self.request()
        duplicate["devices"][1]["id"] = "local"
        bad.append((duplicate, "unique"))
        missing_endpoint = self.request()
        del missing_endpoint["devices"][1]["endpoint"]
        bad.append((missing_endpoint, "endpoint"))
        for request, error in bad:
            with self.subTest(error=error), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                result = self.run_request(root, request)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(error, result.stderr)
                self.assertFalse((root / "output").exists())

    def test_missing_model_fails_without_mock_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = self.run_request(root, self.request())
            self.assertNotEqual(run.returncode, 0)
            self.assertFalse((root / "output").exists())

    def test_existing_results_are_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "output").mkdir()
            (root / "output/sentinel").write_bytes(b"keep")
            run = self.run_request(root, self.request())
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("already exists", run.stderr)
            self.assertEqual((root / "output/sentinel").read_bytes(), b"keep")
            self.assertEqual(len(list((root / "output").iterdir())), 1)


if __name__ == "__main__":
    unittest.main()
