"""Metadata/launch-guard checks only: no Modal imports, tensors or scoring."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/online_8b_eta015_remaining400_setup"
PLAN = json.loads((OUT / "setup.json").read_text())
SOURCE = ROOT / "online_8b_eta015_remaining400.py"
STAGES = ("prepare", "generate", "freeze", "detect", "score")


def guards(folder):
    selected = [n for n in ast.parse(SOURCE.read_text()).body if isinstance(n, ast.FunctionDef)
                and n.name in ("verify_plan", "approved_plan")]
    ns = {"OUT": folder, "RUN": "online_8b_eta015_remaining400_v1", "STAGES": STAGES,
          "Path": Path, "json": json, "check_code": lambda p: None,
          "os": types.SimpleNamespace(environ={"MODAL_PROFILE": "new-prc-watermark"}),
          "subprocess": types.SimpleNamespace(check_output=lambda *a, **k: "redetection\n"),
          "rt": types.SimpleNamespace(_redetect_sha=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest())}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"), ns)
    return ns


class Remaining400Metadata(unittest.TestCase):
    def test_exact_coverage_and_parallel_limit(self):
        guards(OUT)["verify_plan"](PLAN)
        self.assertEqual([i for batch in PLAN["generation_batches"] for i in batch], list(range(100, 500)))
        self.assertEqual(sum(len(g) for g in PLAN["detector_groups"].values()), 10)
        self.assertEqual([i for group in PLAN["detector_groups"]["0.6B"] for i in group], list(range(8)))
        self.assertEqual([i for group in PLAN["detector_groups"]["8B"] for i in group], list(range(4)))
        for field, value in (("null_count", 500), ("T", 4096), ("automatic_retries", 1),
                             ("reference_replays", 1), ("prompt_indices", list(range(500)))):
            changed = copy.deepcopy(PLAN); changed[field] = value
            with self.assertRaises(ValueError): guards(OUT)["verify_plan"](changed)

    def test_approval_and_no_retry_guards(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d); setup = folder / "setup.json"
            setup.write_text(json.dumps(PLAN))
            ns = guards(folder)
            with self.assertRaises(ValueError): ns["approved_plan"]("prepare", "")
            with self.assertRaises(FileNotFoundError): ns["approved_plan"]("prepare", "test-only")
            approval = {"stage": "prepare", "plan_sha256": hashlib.sha256(setup.read_bytes()).hexdigest(),
                        "approval_reference": "test-only", "explicit_user_approval": True,
                        "authorized_spend_usd": .10}
            (folder / "approval_prepare.json").write_text(json.dumps(approval))
            self.assertEqual(ns["approved_plan"]("prepare", "test-only")["N_new"], 400)
            approval["authorized_spend_usd"] = .01
            (folder / "approval_prepare.json").write_text(json.dumps(approval))
            with self.assertRaises(ValueError): ns["approved_plan"]("prepare", "test-only")
            approval["authorized_spend_usd"] = .10
            (folder / "approval_prepare.json").write_text(json.dumps(approval))
            (folder / "attempt_prepare.json").write_text("{}")
            with self.assertRaises(ValueError): ns["approved_plan"]("prepare", "test-only")

    def test_pilot_reports_and_source_snapshots(self):
        self.assertEqual(PLAN["pilot"]["8B"]["counts"]["6144"]["map"]["wm"], {"detected": 94, "count": 100})
        self.assertEqual(PLAN["pilot"]["0.6B"]["counts"]["6144"]["map"]["wm"], {"detected": 92, "count": 100})
        left = PLAN["pilot"]["8B"]["prepared"]["run"]["case"]["records"]
        right = PLAN["pilot"]["0.6B"]["prepared"]["run"]["case"]["records"]
        self.assertEqual([(r["prompt_idx"], r["tokens_sha256"]) for r in left],
                         [(r["prompt_idx"], r["tokens_sha256"]) for r in right])
        for name, digest in PLAN["runtime_source_sha256"].items():
            # Historical setup snapshots stay immutable after an approved fix.
            # The runtime itself verifies current source hashes at each launch.
            self.assertEqual(hashlib.sha256((OUT / "execution_sources" / name).read_bytes()).hexdigest(), digest)
        self.assertAlmostEqual(sum(s["allowance_usd"] for s in PLAN["stages"].values()), PLAN["sum_stage_allowances_usd"])

    def test_primary_only_and_import_order(self):
        tree = ast.parse(SOURCE.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(node.func.attr, ("generate_null", "online_build_artifacts", "validate_replay"))
                if node.func.attr == "_recover_redetection_batch":
                    self.assertTrue(any(k.arg == "validate" and isinstance(k.value, ast.Constant)
                                        and k.value.value is False for k in node.keywords))
        child = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "child")
        imports = [n.names[0].name for n in child.body if isinstance(n, ast.Import)]
        self.assertEqual(imports[:7], ["torch", "scipy.special", "galois", "transformers", "safetensors.torch", "qwen", "detectors"])


if __name__ == "__main__": unittest.main()
