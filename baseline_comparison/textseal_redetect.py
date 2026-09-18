"""Local setup and cache contracts for the completion-only TextSeal replay.

Running this module prepares files only. Remote execution is a separate command
in textseal_modal.py and requires an explicitly selected stage and manifest hash.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import time

from .config import PREFIX_LENGTHS, TEXTSEAL_ALPHA, TEXTSEAL_COMMIT, TEXTSEAL_KEY_A, TEXTSEAL_KEY_B
from .textseal_completion import PROTOCOL

REPO = Path(__file__).resolve().parents[1]
SETUP = REPO / "outputs/comparison_redetect/textseal_setup/direct_prefix"
CODE_FILES = [f"baseline_comparison/{name}" for name in (
    "__init__.py", "config.py", "official.py", "scoring.py", "textseal_completion.py",
    "textseal_redetect.py", "textseal_modal.py", "textseal_source_audit.json", "requirements-textseal.txt",
)]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    temporary.write_bytes(canonical(value) + b"\n")
    temporary.replace(path)


def record_identity(record):
    if set(record) != {"method", "prompt_index", "token_ids"}:
        raise ValueError("worker records must contain only method, pairing index, and completion IDs")
    method, index, ids = record["method"], record["prompt_index"], record["token_ids"]
    if (method not in ("textseal", "null") or type(index) is not int or not 0 <= index < 500
            or type(ids) is not list or any(type(t) is not int or not 0 <= t < 151936 for t in ids)):
        raise ValueError("invalid completion record")
    return {"record_id": f"{method}/{index:04d}", "completion_sha256": digest(ids), "length": len(ids)}


def build_setup(output=SETUP):
    output = Path(output)
    preflight_path = REPO / "outputs/comparison_redetect/preflight/preflight.json"
    preflight = json.loads(preflight_path.read_text())
    input_path = preflight_path.parent / preflight["completion_inputs"]["path"]
    if not preflight["status"] == "passed" or file_sha(input_path) != preflight["completion_inputs"]["sha256"]:
        raise ValueError("completion preflight/export changed")
    records = []
    for line in input_path.read_text().splitlines():
        row = json.loads(line)
        if set(row) != {"method", "prompt_index", "token_ids", "historical_token_sha256"}:
            raise ValueError("unexpected preflight input fields")
        if row["method"] not in ("textseal", "null"):
            continue
        ids = row["token_ids"]
        historical = hashlib.sha256(f"int64:({len(ids)},):".encode() + struct.pack(f"<{len(ids)}q", *ids)).hexdigest()
        if historical != row["historical_token_sha256"]:
            raise ValueError("historical completion hash differs")
        records.append({key: row[key] for key in ("method", "prompt_index", "token_ids")})
    records.sort(key=lambda row: (row["method"], row["prompt_index"]))
    identities = [record_identity(record) for record in records]
    expected_ids = {f"{method}/{i:04d}" for method in ("textseal", "null") for i in range(500)}
    if len(identities) != 1000 or {row["record_id"] for row in identities} != expected_ids or any(row["length"] != 1024 for row in identities):
        raise ValueError("expected 500 TextSeal and 500 original shared-null completions of length 1024")
    model = json.loads((REPO / "outputs/redetection/.archive/manifests/same_8b_eta005_n1280.json").read_text())["model"]
    model["config_sha256"] = file_sha(REPO / "outputs/redetection/.archive/setup_8b_n1280/config.json")
    output.mkdir(parents=True, exist_ok=True)
    clean = output / "completion_inputs.jsonl"
    clean.write_bytes(b"".join(canonical(row) + b"\n" for row in records))
    manifest = {
        "schema_version": 1, "protocol": PROTOCOL, "method": "textseal", "model": model,
        "length": 1024, "prefix_lengths": list(PREFIX_LENGTHS), "batch_size": 1,
        "prefix_strategy": "direct",
        "scoring": {"ngram": 3, "alpha": TEXTSEAL_ALPHA, "keys": [TEXTSEAL_KEY_A, TEXTSEAL_KEY_B],
                    "method": "v2", "score_field": "p_value_weighted", "nominal_fpr": .001,
                    "upstream_result": "preserve the complete returned dictionary, including its 0.01 combined decision"},
        "upstream_commit": TEXTSEAL_COMMIT, "code_sha256": {name: file_sha(REPO / name) for name in CODE_FILES},
        "inputs": {"file": clean.name, "sha256": file_sha(clean), "records": identities},
        "pilot_ids": [f"{method}/{i:04d}" for method in ("textseal", "null") for i in range(5)],
        "source_preflight_sha256": file_sha(preflight_path),
        "null_source": "_nulls/qwen3_8b_base/T13088", "generation_run": preflight["run_id"],
        "runtime": {"attention": "eager", "dtype": "bfloat16", "use_cache": False,
                    "tf32": False, "bf16_reduced_precision_reduction": False,
                    "gpu": "H100", "cpu": 4, "memory_mib": 65536, "max_containers": 1,
                    "retries": 0, "scaledown_seconds": 2,
                    "dependencies": [line for line in (REPO / "baseline_comparison/requirements-textseal.txt").read_text().splitlines()
                                     if line and not line.startswith("#")]},
        "execution": {"profile": "new-prc-watermark", "pilot_timeout_seconds": 900,
                      "full_timeout_seconds": 3600, "automatic_full_launch": False,
                      "require_exact_prefix_validation": True, "repeat_handling": False,
                      "result_volume": "prc-completion-only", "model_volume": "prc-hf-cache"},
        "cost": {"pricing_source": "https://modal.com/pricing", "checked_date": "2026-09-17",
                 "resource_usd_per_second": .001097 + 4*.0000131 + 64*.00000222,
                 "pilot_planning_usd": [0.3, 1.5], "full_planning_usd": [2, 5],
                 "note": "Provisional, including CPU scoring on the GPU worker; pilot measures actual loading/replay/scoring time before full approval. No selected-region premium or automatic retries."},
    }
    write_json(output / "native8b_manifest.json", manifest)
    print(json.dumps({"manifest": str(output / "native8b_manifest.json"), "manifest_sha256": digest(manifest),
                      "pilot_records": 10, "full_records": 1000, "remote_calls": 0}, indent=2))
    return manifest


def validate_request(manifest, records, stage, approved_sha):
    if stage not in ("pilot", "full") or not approved_sha or digest(manifest) != approved_sha:
        raise ValueError("explicit stage and matching approved manifest SHA-256 are required")
    if manifest.get("prefix_strategy") != "direct":
        raise ValueError("production replay requires direct detection at each prefix; longest-trace reuse is diagnostic only")
    for name, expected in manifest["code_sha256"].items():
        if name not in CODE_FILES or file_sha(REPO / name) != expected:
            raise ValueError(f"execution code changed since setup: {name}")
    if set(manifest["code_sha256"]) != set(CODE_FILES):
        raise ValueError("incomplete execution code identity")
    expected = {row["record_id"]: row for row in manifest["inputs"]["records"]}
    selected = set(manifest["pilot_ids"]) if stage == "pilot" else set(expected)
    identities = [record_identity(row) for row in records]
    if len(identities) != len(selected) or {row["record_id"] for row in identities} != selected:
        raise ValueError("stage candidate coverage differs")
    if any(row != expected[row["record_id"]] for row in identities):
        raise ValueError("completion input identity differs")


def load_request(path, stage, approved_sha):
    path = Path(path)
    manifest = json.loads(path.read_text())
    inputs = path.parent / manifest["inputs"]["file"]
    if file_sha(inputs) != manifest["inputs"]["sha256"]:
        raise ValueError("clean input export changed")
    records = [json.loads(line) for line in inputs.read_text().splitlines()]
    if stage == "pilot":
        records = [row for row in records if record_identity(row)["record_id"] in manifest["pilot_ids"]]
    validate_request(manifest, records, stage, approved_sha)
    return manifest, records


def run_record(detector, record, lengths, validate_prefixes, prefix_strategy="direct"):
    """Observe actual model forwards: each is exactly the raw causal prefix."""
    ids = record["token_ids"]
    if prefix_strategy not in ("direct", "reuse"):
        raise ValueError("unknown prefix strategy")
    calls = []

    def observe(module, args, kwargs):
        if kwargs or len(args) != 1 or args[0].ndim != 2 or args[0].shape[0] != 1:
            raise ValueError("unexpected model inputs or external cache")
        n = args[0].shape[1]
        if args[0].tolist() != [ids[:n]] or not 1 <= n <= len(ids):
            raise ValueError("model received something other than raw completion tokens")
        calls.append(n)

    handle = detector._detector.model.register_forward_pre_hook(observe, with_kwargs=True)
    started = time.monotonic()
    try:
        function = detector.detect_prefixes if prefix_strategy == "direct" else detector.detect_prefixes_reusing_longest
        result = function(ids, prefix_lengths=lengths, validate_prefixes=validate_prefixes)
    finally:
        handle.remove()
    expected = ([n for n in sorted(lengths) for _ in range(2 if validate_prefixes else 1)]
                if prefix_strategy == "direct" else [len(ids)] + (sorted(lengths)[:-1] if validate_prefixes else []))
    if calls != expected:
        raise ValueError(f"unexpected forward call schedule: {calls}")
    return {"input": record_identity(record), **result, "forward_lengths": calls,
            "seconds": time.monotonic()-started, "actual_model_inputs_verified": True}


def require_pilot(report, manifest_sha, runtime, pilot_ids):
    if (not report or report.get("manifest_sha256") != manifest_sha or report.get("stage") != "pilot"
            or report.get("passed") is not True or report.get("runtime") != runtime
            or set(report.get("record_sha256", {})) != set(pilot_ids)):
        raise ValueError("full replay requires a passing pilot with the same manifest, runtime and candidates")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SETUP)
    args = parser.parse_args()
    build_setup(args.output)


if __name__ == "__main__":
    main()
