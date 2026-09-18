"""Retrieve and verify step-3 artifacts without dispatching any compute."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path, PurePosixPath

from .self_bleu_config import digest
from .self_bleu_validation import save, sha, validate_manifest, ROOT


def collect(setup, raw, *, download=False):
    manifest = json.loads((setup / "manifest.json").read_text())
    validate_manifest(manifest, ROOT)
    generation = json.loads((setup / "generation_report.json").read_text())
    textseal = json.loads((setup / "textseal_report.json").read_text())
    for report in (generation, textseal):
        if not report["passed"] or report["manifest_id"] != manifest["id"]:
            raise ValueError("requires matching passing validation reports")
    prefix = f"self_bleu_validation/{manifest['id']}"
    files = {**generation["files"], **textseal["files"]}
    for name in ("manifest.json", "generation_report.json", "textseal_report.json"):
        files[name] = sha(setup / name)
    for name in files:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("invalid artifact path")
    if download:
        import modal
        volume = modal.Volume.from_name("prc-completion-only", create_if_missing=False)
        def retrieve(name):
            path = raw / name
            if path.exists():
                if sha(path) != files[name]:
                    raise ValueError(f"existing artifact differs: {name}")
                return
            data = b"".join(volume.read_file(f"{prefix}/{name}"))
            import hashlib
            if hashlib.sha256(data).hexdigest() != files[name]:
                raise ValueError(f"download checksum differs: {name}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(retrieve, files))
    for name, expected in files.items():
        if sha(raw / name) != expected:
            raise ValueError(f"saved artifact checksum differs: {name}")
    batches, response_ids = {}, set()
    full = short = synthid_checks = 0
    for name in generation["files"]:
        if not name.startswith("batches/"):
            continue
        batch = json.loads((raw / name).read_text())
        identity = batch["manifest"]
        base = {k: v for k, v in identity.items() if k not in ("batch_id", "namespace", "response_ids")}
        if (digest(base) != identity["batch_id"] or digest(identity["setting"]) != identity["setting_sha256"]
                or identity["prompt_indices"] != list(range(50)) or len(batch["responses"]) != 50):
            raise ValueError("batch identity or coverage differs")
        for i, row in enumerate(batch["responses"]):
            if (row["response_id"] in response_ids or row["response_id"] != identity["response_ids"][i]
                    or row["prompt_index"] != i or row["response_index"] != identity["response_index"]
                    or row["sampling_seed"] != identity["sampling_seed"]
                    or row["setting_sha256"] != identity["setting_sha256"]
                    or len(row["token_ids"]) != identity["generation"]["max_new_tokens"]
                    or digest(row["token_ids"]) != row["completion_sha256"]):
                raise ValueError("response identity or token coverage differs")
            response_ids.add(row["response_id"])
        if identity["generation"]["max_new_tokens"] == 1024:
            full += len(batch["responses"])
        else:
            short += len(batch["responses"])
        if identity["setting"]["method"] == "synthid_text":
            check = batch["telemetry"]["synthid_official_smoke_reference"]
            if not check["indices_equal"] or check["max_abs_score_difference"] != 0:
                raise ValueError("SynthID official update parity failed")
            synthid_checks += 1
        batches[identity["batch_id"]] = batch
    if (full, short, len(batches), len(textseal["records"]), synthid_checks) != (600, 200, 16, 7, 5):
        raise ValueError("validation coverage differs")
    reuse = []
    for row in generation["settings"]:
        first, second = (batches[identifier] for identifier in row["batches"])
        if first["manifest"]["sampling_seed"] != 12345 or second["manifest"]["sampling_seed"] != 67890:
            raise ValueError("replicate seed differs")
        if first["manifest"]["setting"] != second["manifest"]["setting"]:
            raise ValueError("watermark configuration changed across replicates")
        matches = row["historical_first_response_matches"]
        if matches is not None:
            expected = manifest["source_audit"]["expected_completion_sha256"][row["setting"]["method"]]
            actual = [r["completion_sha256"] == old for r, old in zip(first["responses"], expected)]
            if actual != matches:
                raise ValueError("historical reuse audit differs")
        reuse.append({"setting": row["setting"], "historical_exact_matches": None if matches is None else sum(matches),
                      "responses_changed_across_seeds": row["responses_changed_across_seeds"]})
    measured = generation["measured_resource_usd"] + generation.get("repair_measured_resource_usd", 0) + textseal["measured_resource_usd"]
    result = {"passed": True, "manifest_id": manifest["id"], "volume": "prc-completion-only",
              "remote_path": prefix, "verified_files": files, "verified_file_count": len(files),
              "full_length_response_records": full, "short_response_records": short,
              "synthid_official_update_checks": synthid_checks, "reuse": reuse,
              "measured_resource_usd": measured, "failed_startup_allowance_usd": .5,
              "image_startup_storage_allowance_usd": 2, "total_planning_charge_usd": measured + 2.5,
              "remaining_initial_allocation_after_allowances_usd": 10-measured-2.5,
              "billing_note": "Resource time is measured for generation, PRC repair and TextSeal replay. Separate allowances are conservative reservations, not invoice amounts."}
    save(setup / "verification.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    result = collect(args.setup, args.raw, download=args.download)
    print(json.dumps({k: v for k, v in result.items() if k not in ("verified_files", "reuse")}, indent=2))
