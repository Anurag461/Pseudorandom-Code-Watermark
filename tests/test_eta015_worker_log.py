"""Mocked wrapper regression: no Modal import, subprocess or model execution."""
import ast
import json
from pathlib import Path
import subprocess
import tempfile
import types
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "online_8b_eta015_remaining400.py"


class WorkerLogRegression(unittest.TestCase):
    def check_case(self, outcome, attempt_namespace=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results, scratch = root / "results", root / "scratch"
            scratch.mkdir()
            commits = []
            original_marker = results / "test-run/attempts/prepare/cpu/failure.json"
            if attempt_namespace:
                original_marker.parent.mkdir(parents=True)
                original_marker.write_text("preserved first failure")

            def write_json(path, value):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value))

            def launch(command, stdout, stderr):
                self.assertEqual(Path(stdout.name).parent, scratch)
                self.assertIs(stderr, stdout)
                stdout.write("saved diagnostic\n")
                write_json(Path(command[-1]), {"files": []})
                process = types.SimpleNamespace(killed=False)
                process.kill = lambda: setattr(process, "killed", True)

                def wait(timeout=None):
                    if outcome == "timeout" and not process.killed:
                        raise subprocess.TimeoutExpired(command, timeout)
                    return 1 if outcome == "failure" else 0

                process.wait = wait
                return process

            selected = [node for node in ast.parse(SOURCE.read_text()).body
                        if isinstance(node, ast.FunctionDef) and node.name == "bounded"]
            namespace = {
                "Path": lambda path: results if path == "/results" else scratch if path == "/tmp" else Path(path),
                "RUN": "test-run", "verify_plan": lambda plan: None, "check_code": lambda plan: None,
                "rt": types.SimpleNamespace(redetect_results=types.SimpleNamespace(
                    reload=lambda: None, commit=lambda: commits.append(True))),
                "write_json": write_json, "json": json, "time": types.SimpleNamespace(monotonic=lambda: 1.0),
                "os": types.SimpleNamespace(getpid=lambda: 123), "sys": types.SimpleNamespace(executable="unused"),
                "subprocess": types.SimpleNamespace(Popen=launch, TimeoutExpired=subprocess.TimeoutExpired),
                "__file__": str(SOURCE), "file_ref": lambda path, volume: {"path": str(path)},
            }
            exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"), namespace)
            payload = {"plan": {"attempt_namespace": "prepare_retry1"} if original_marker.exists() else {}}
            if outcome == "success":
                namespace["bounded"]("prepare", payload, "cpu", 600)
            else:
                with self.assertRaises(TimeoutError if outcome == "timeout" else RuntimeError):
                    namespace["bounded"]("prepare", payload, "cpu", 600)
            folder = results / "test-run/attempts"
            if payload["plan"]:
                self.assertEqual(original_marker.read_text(), "preserved first failure")
                folder = folder / "prepare_retry1"
            folder = folder / "prepare/cpu"
            self.assertEqual((folder / "worker.log").read_text(), "saved diagnostic\n")
            self.assertTrue((folder / "timing.json").exists())
            self.assertEqual((folder / "failure.json").exists(), outcome != "success")
            self.assertEqual(len(commits), 2)
            self.assertEqual(list(scratch.iterdir()), [])

    def test_success_preserves_log(self): self.check_case("success")
    def test_failure_preserves_log(self): self.check_case("failure")
    def test_timeout_preserves_log(self): self.check_case("timeout")
    def test_retry_preserves_initial_failure(self): self.check_case("success", "prepare_retry1")


if __name__ == "__main__": unittest.main()
