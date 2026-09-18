"""Prepare the frozen Stage A analysis and minimal completion-only replay requests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from .config import PREFIX_LENGTHS
from .self_bleu_config import digest, verify_reference
from .self_bleu_validation import ROOT, RATE, save, sha
from .self_bleu_validation_results import collect

VALIDATION = ROOT / "outputs/self_bleu_validation/step3-v4"
RAW = ROOT / "outputs/self_bleu_validation/raw/743658bc40e7f52c910d9538266bbd0ff461bba949e2a842cc8af36fa32507d8"
SETUP = ROOT / "outputs/self_bleu_pilot/stage_a_v1"
METHODS = ("online_prc", "textseal", "synthid_text", "gumbel_max", "null")
CODE = ("baseline_comparison/self_bleu_pilot.py", "baseline_comparison/self_bleu_pilot_modal.py",
        "baseline_comparison/self_bleu_validation.py", "baseline_comparison/self_bleu_validation_modal.py",
        "baseline_comparison/textseal_modal.py", "baseline_comparison/textseal_redetect.py",
        "baseline_comparison/textseal_completion.py", "baseline_comparison/textseal_source_audit.json",
        "baseline_comparison/requirements-textseal.txt", "baseline_comparison/modal_app.py",
        "baseline_comparison/comparison_runner.py", "baseline_comparison/config.py",
        "baseline_comparison/official.py", "baseline_comparison/scoring.py",
        "baseline_comparison/self_bleu_config.py", "baseline_comparison/self_bleu_reference.json",
        "qwen.py", "prc.py", "online_prc.py", "detectors.py")


def load_pairs():
    verification = collect(VALIDATION, RAW)
    report = json.loads((VALIDATION / "generation_report.json").read_text())
    batches = {}
    for row in report["settings"]:
        setting = row["setting"]
        if setting["method"] == "textseal" and setting["alpha"] != .1:
            continue
        for bid in row["batches"]:
            batch = json.loads((RAW / "batches" / f"{bid}.json").read_text())
            batches[(setting["method"], batch["manifest"]["response_index"])] = batch
    if set(batches) != {(m, r) for m in METHODS for r in (0, 1)}:
        raise ValueError("Stage A pair coverage differs")
    return batches, verification


def clean_records(batches):
    rows = []
    for (method, replicate), batch in sorted(batches.items()):
        for row in batch["responses"]:
            rows.append({"response_id": row["response_id"], "method": method,
                         "prompt_index": row["prompt_index"], "response_index": replicate,
                         "completion_sha256": row["completion_sha256"], "token_ids": row["token_ids"]})
    return rows


def prepare(output=SETUP, failed_attempt_allowance_usd=0.):
    from .textseal_results import validate_record
    reference = verify_reference()
    batches, verification = load_pairs()
    rows = clean_records(batches)
    old_setup = ROOT / "outputs/comparison_redetect/textseal_setup/direct_prefix"
    old_manifest = json.loads((old_setup / "native8b_manifest.json").read_text())
    old_report = json.loads((old_setup / "full_report.json").read_text())
    old_by_id = {r["record_id"]: r for r in old_report["rows"]}
    validation_ts = json.loads((VALIDATION / "textseal_report.json").read_text())
    if old_report["runtime"] != validation_ts["execution"]:
        raise ValueError("TextSeal replay runtimes differ")
    by_response = {r["response_id"]: r for r in rows}
    reused = {}
    sources = {str(VALIDATION.relative_to(ROOT) / name): sha(VALIDATION / name)
               for name in ("manifest.json", "verification.json", "generation_report.json", "textseal_report.json")}
    for row in rows:
        if row["method"] != "textseal" or row["response_index"] != 0:
            continue
        rid = f"textseal/{row['prompt_index']:04d}"
        path = old_setup / "cache" / f"{rid}.json"
        if sha(path) != old_report["record_sha256"][rid]:
            raise ValueError("historical TextSeal record changed")
        clean = {k: row[k] for k in ("method", "prompt_index", "token_ids")}
        data = validate_record(json.loads(path.read_text()), clean, old_by_id[rid], old_manifest, old_report["runtime"])
        reused[row["response_id"]] = {"response_id": row["response_id"], "completion_sha256": row["completion_sha256"],
                                      "results": data["results"], "source": str(path.relative_to(ROOT)), "source_sha256": sha(path)}
    for name, expected in validation_ts["files"].items():
        path = RAW / name
        if sha(path) != expected:
            raise ValueError("step-3 TextSeal evidence changed")
        payload = json.loads(path.read_text())
        rid = payload["response_id"]
        if rid not in by_response or payload["alpha"] != .1 or rid in reused:
            continue
        data, row = payload["data"], by_response[rid]
        if not data["actual_model_inputs_verified"] or data["completion_sha256"] != row["completion_sha256"] or not data["validation"]["passed"]:
            raise ValueError("step-3 replay identity or validation differs")
        reused[rid] = {"response_id": rid, "completion_sha256": row["completion_sha256"], "results": data["results"],
                       "source": str(path.relative_to(ROOT)), "source_sha256": expected}
    requests = {"prc": [r for r in rows if r["method"] in ("online_prc", "null")],
                "textseal": [r for r in rows if r["method"] in ("textseal", "null") and r["response_id"] not in reused]}
    if len(rows) != 500 or len(reused) != 53 or [len(requests[m]) for m in ("prc", "textseal")] != [200, 147]:
        raise ValueError("unexpected replay/reuse coverage")
    old_validation = json.loads((VALIDATION / "manifest.json").read_text())
    manifest = {"schema_version": 1, "stage": "A", "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "protocol": reference["protocol"], "model": old_validation["model"], "artifact": reference["prc_generation_artifact"],
                "prefix_lengths": list(PREFIX_LENGTHS), "primary_lengths": [400, 1024], "nominal_fpr": .001,
                "prompts": list(range(50)), "sampling_seeds": [12345, 67890], "input_sha256": digest(rows),
                "requests": {m: {"count": len(r), "sha256": digest(r)} for m, r in requests.items()},
                "reused_textseal_sha256": digest(list(reused.values())), "reused_textseal_records": len(reused),
                "analysis": {"sacrebleu_version": "2.4.3", "tokenize": "13a", "smooth_method": "exp", "effective_order": True,
                             "lowercase": False, "scale": "0-1", "skip_special_tokens": True, "clean_up_tokenization_spaces": False,
                             "bootstrap_resamples": 2000, "bootstrap_seed": 20260918, "bootstrap_unit": "paired prompt cluster",
                             "synthid_primary_mask": "official compute_context_repetition_mask", "synthid_reproduction_mask": "historical unique (context,token), start=4",
                             "operating_point": "nominal only; no empirical tail calibration on pilot"},
                "textseal_runtime": old_report["runtime"], "code_sha256": {p: sha(ROOT / p) for p in CODE},
                "sources": sources, "validation_id": verification["manifest_id"],
                "cost": {"resource_usd_per_second": RATE, "timeout_seconds_per_stage": 600,
                         "maximum_new_resource_reservation_usd": 1204*RATE,
                         "previous_planning_charge_usd": verification["total_planning_charge_usd"] + failed_attempt_allowance_usd,
                         "pilot_failed_attempt_allowance_usd": failed_attempt_allowance_usd,
                         "initial_allocation_usd": 10, "total_study_ceiling_usd": 200,
                         "dispatch": "one H100 per stage, sequential, max_containers=1, retries=0; no generation"}}
    manifest["id"] = digest(manifest)
    for method, records in requests.items():
        validate_request(manifest, records, method, ROOT)
    save(output / "manifest.json", manifest)
    save(output / "inputs.json", rows)
    save(output / "reused_textseal.json", list(reused.values()))
    return {"manifest_id": manifest["id"], "responses": len(rows), "reused_textseal_records": len(reused),
            "requests": manifest["requests"], "reserved_total_with_previous_allowances_usd": manifest["cost"]["previous_planning_charge_usd"]+1204*RATE}


def validate_request(manifest, records, stage, root):
    if digest({k: v for k, v in manifest.items() if k != "id"}) != manifest["id"]:
        raise ValueError("pilot manifest identity differs")
    if manifest["protocol"] != "completion_only_raw_abstain_v1" or manifest["stage"] != "A" or manifest["nominal_fpr"] != .001:
        raise ValueError("pilot protocol differs")
    for name, expected in manifest["code_sha256"].items():
        if sha(Path(root) / name) != expected:
            raise ValueError(f"pilot worker source differs: {name}")
    expected = manifest["requests"][stage]
    if len(records) != expected["count"] or digest(records) != expected["sha256"]:
        raise ValueError("pilot detector request differs")
    seen = set()
    for row in records:
        if (set(row) != {"response_id", "method", "prompt_index", "response_index", "completion_sha256", "token_ids"}
                or row["response_id"] in seen or len(row["token_ids"]) != 1024
                or any(type(t) is not int or not 0 <= t < 151936 for t in row["token_ids"])
                or digest(row["token_ids"]) != row["completion_sha256"]):
            raise ValueError("detector inputs must be unique raw completions with identity metadata only")
        seen.add(row["response_id"])
    cost = manifest["cost"]
    if (cost["timeout_seconds_per_stage"] != 600 or cost["resource_usd_per_second"] != RATE
            or cost["previous_planning_charge_usd"] + 1204*RATE > 10):
        raise ValueError("pilot allocation would be exceeded")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SETUP)
    parser.add_argument("--failed-attempt-allowance-usd", type=float, default=0.)
    args = parser.parse_args()
    if not 0 <= args.failed_attempt_allowance_usd <= 5:
        raise ValueError("invalid failed-attempt allowance")
    print(json.dumps(prepare(args.output, args.failed_attempt_allowance_usd), indent=2))
