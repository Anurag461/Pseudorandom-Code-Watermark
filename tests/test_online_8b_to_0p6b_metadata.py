"""Metadata-only guards; never import model/scoring or contact Modal."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/online_8b_to_0p6b_redetect_setup"
PLAN = json.loads((OUT / "setup.json").read_text())


def guards(folder):
    tree = ast.parse((ROOT / "online_8b_to_0p6b.py").read_text())
    chosen = [node for node in tree.body if isinstance(node, ast.FunctionDef)
              and node.name in ("case_folder", "verify_plan", "run")]
    for node in chosen:
        node.decorator_list = []
    spy = types.SimpleNamespace(calls=0)
    def never_paid(**kwargs):
        spy.calls += 1
        raise AssertionError("paid boundary reached")
    ns = dict(OUT=folder, STAGES=("prepare", "replay", "score"), Path=Path, json=json,
              os=types.SimpleNamespace(environ={"MODAL_PROFILE": "new-prc-watermark"}),
              subprocess=types.SimpleNamespace(check_output=lambda *a, **k: "redetection\n"),
              check_code=lambda plan: None,
              rt=types.SimpleNamespace(REDETECT_PROTOCOL="completion_only_raw_abstain_v1",
                  _redetect_sha=lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()),
              paid_stage=types.SimpleNamespace(with_options=never_paid))
    exec(compile(ast.Module(body=chosen, type_ignores=[]), "guards", "exec"), ns)
    return ns, spy


class ProposalGuards(unittest.TestCase):
    def test_selected_cohorts_and_excluded_4096(self):
        ns, _ = guards(OUT)
        for case in PLAN["cases"]:
            self.assertEqual(ns["verify_plan"](PLAN, case["id"]), case)
            self.assertNotIn(4096, case["lengths"])
        with self.assertRaises(ValueError):
            ns["case_folder"]("eta015_T4096_N500")
        wrong = copy.deepcopy(PLAN)
        wrong["cases"][-1]["N"] = 500
        with self.assertRaises(ValueError):
            ns["verify_plan"](wrong, "eta015_T6144_N100")

    def test_frozen_inputs_and_source_snapshots(self):
        for case in PLAN["cases"]:
            source = OUT / case["source_manifest_local"]
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), case["source_manifest_sha256"])
            records = [r for r in json.loads(source.read_text())["case"]["records"] if r["source"] == "wm"]
            self.assertEqual(records, case["records"])
            self.assertEqual([r["prompt_idx"] for r in records], list(range(case["N"])))
        for name, digest in PLAN["runtime_source_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest)
            self.assertEqual(hashlib.sha256((OUT / "execution_sources" / name).read_bytes()).hexdigest(), digest)

    def test_no_approval_or_repeated_attempt_blocks_paid_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "setup.json").write_text(json.dumps(PLAN))
            ns, spy = guards(folder)
            case = PLAN["cases"][0]["id"]
            with self.assertRaises(ValueError):
                ns["run"](case, "prepare", "")
            attempt = folder / "cases" / case / "attempt_prepare.json"
            attempt.parent.mkdir(parents=True)
            attempt.write_text("{}")
            with self.assertRaises(ValueError):
                ns["run"](case, "prepare", "test only")
            self.assertEqual(spy.calls, 0)

    def test_changed_manifest_blocks_replay_and_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "setup.json").write_text(json.dumps(PLAN))
            ns, spy = guards(folder)
            for case in PLAN["cases"]:
                path = folder / "cases" / case["id"] / "attempt_prepare.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({"plan_sha256": "different"}))
                for stage in ("replay", "score"):
                    with self.assertRaisesRegex(ValueError, "setup changed"):
                        ns["run"](case["id"], stage, "test only")
            self.assertEqual(spy.calls, 0)


if __name__ == "__main__":
    unittest.main()
