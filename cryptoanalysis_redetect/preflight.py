"""Inspect saved artifact metadata only; never load an LM or score a detector.

Exit 2 means required source data is missing. Exit 0 means source field presence
checks passed, NOT that hard-detector reproduction passed or compute is approved.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import zipfile

ARTIFACT_REPO = "https://github.com/1234wangtr/PRC_estimator"
ARTIFACT_COMMIT = "8593e86aeb50b5f82d6c88e390b12a30f581dbaa"
ARCHIVE_PATH = "llm/data/Deepseek_t_3_temp_all.zip"
ARCHIVE_SHA256 = "f8ac4b3a45a533f0ea31d579d4125b3f36d19e791ad2d4192d6db927a6f40f91"
MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
REVISION = "916b56a44061fd5cd7d6a8fb632557ed4f724f60"
TEMPERATURES = (1.0, 1.2, 1.4, 1.6, 1.8)
PARAMETERS = dict(completion_tokens=1024, vocab_size=152064, bits_per_token=18,
                  n=18432, r=17510, t=3, eta=0.1)
REQUIRED_PRIMARY = (
    "secret_key", "one_time_pad",
    "origin_sentence_tokens", "watermark_sentence_tokens",
)
REQUIRED_ORACLE = ("prompt_tokens",)
SAVED_FIELDS = ("origin_sentence", "watermark_sentence", "correct_rate",
                "avg_entropy", "det", "origin_det")
PATH_PATTERN = re.compile(r"^gen_result/temperature_(1\.[02468])/(\d+)\.json$")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def text_group_hash(sentences):
    return hashlib.sha256(json.dumps(sentences, ensure_ascii=False).encode()).hexdigest()


def inspect_record(name, raw):
    """Inspect keys and saved scalar metadata without reconstructing scores."""
    match = PATH_PATTERN.fullmatch(name)
    if not match:
        raise ValueError(f"Unexpected group path: {name}")
    temperature, group_id = float(match[1]), match[2]
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {name}")
    for key in SAVED_FIELDS:
        if not isinstance(data.get(key), list) or len(data[key]) != 16:
            raise ValueError(f"{name}: {key} must be a 16-element list")
    for key in ("origin_sentence", "watermark_sentence"):
        if not all(isinstance(x, str) for x in data[key]):
            raise ValueError(f"{name}: {key} must contain strings")
    for key in ("det", "origin_det"):
        if not all(type(x) is bool for x in data[key]):
            raise ValueError(f"{name}: {key} must contain booleans")
    cfg = data.get("generation_config")
    if cfg is None:
        if temperature != 1.0:
            raise ValueError(f"{name}: missing generation_config outside T=1.0")
        temperature_source = "directory; matches plot_entropy.py default"
    else:
        if cfg.get("temperature") != temperature:
            raise ValueError(f"{name}: directory/config temperature mismatch")
        expected = dict(max_new_tokens=1024, top_k=0, top_p=1.0, do_sample=True)
        for key, value in expected.items():
            if cfg.get(key) != value:
                raise ValueError(f"{name}: unexpected generation_config.{key}")
        temperature_source = "directory and saved generation_config"
    return {
        "file": name, "group_id": group_id, "temperature": temperature,
        "temperature_source": temperature_source,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "fields": sorted(data), "prompts": 16,
        "saved_det_true": sum(data["det"]),
        "saved_det": data["det"],
        "watermark_text_group_sha256": text_group_hash(data["watermark_sentence"]),
        "missing_primary_fields": [k for k in REQUIRED_PRIMARY if k not in data],
        "missing_oracle_fields": [k for k in REQUIRED_ORACLE if k not in data],
        "schema": {k: {"type": type(v).__name__,
                       "length": len(v) if isinstance(v, (list, dict, str)) else None,
                       "element_types": sorted({type(x).__name__ for x in v})
                       if isinstance(v, list) else None}
                   for k, v in data.items()},
    }


def split_groups(records, max_groups_per_temp=None):
    """Use the same numeric-ID sorting/half split rule independently at each T.

    Archive timestamps differ across temperatures; rank does not establish key
    identity. Keys must be checked for reuse/leakage once supplied.
    """
    if max_groups_per_temp is not None and (
        max_groups_per_temp < 2 or max_groups_per_temp % 2
    ):
        raise ValueError("--max-groups-per-temp must be a positive even count >= 2")
    result = []
    for temp in TEMPERATURES:
        groups = sorted((r for r in records if r["temperature"] == temp),
                        key=lambda r: (int(r["group_id"]), r["file"]))
        if len({r["group_id"] for r in groups}) != len(groups):
            raise ValueError(f"Duplicate group ID at T={temp}")
        selected = groups[:max_groups_per_temp]
        if len(selected) < 2 or len(selected) % 2:
            raise ValueError(f"Cannot split {len(selected)} groups 50/50 at T={temp}")
        for rank, group in enumerate(selected):
            result.append({**group, "sorted_rank": rank,
                           "split": "calibration" if rank < len(selected) // 2 else "evaluation"})
    return result


def summarize_flags(records, selected):
    rows = []
    for temp in TEMPERATURES:
        all_groups = [r for r in records if r["temperature"] == temp]
        chosen = [r for r in selected if r["temperature"] == temp]
        for scope, groups in (("all_available", all_groups), ("selected_diagnostic", chosen)):
            n = sum(g["prompts"] for g in groups)
            positives = sum(g["saved_det_true"] for g in groups)
            rows.append(dict(temperature=temp, scope=scope, groups=len(groups), N=n,
                             stored_detected=positives, stored_detection_rate=positives / n,
                             source="saved det flags; NOT independently reproduced"))
    return rows


def inspect_companion(path, records):
    """Check exact saved text-group identity; never assign a merely similar key."""
    lookup = {r["watermark_text_group_sha256"]: r["file"] for r in records}
    groups = []
    with zipfile.ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            if not name.endswith(".json"):
                continue
            data = json.loads(archive.read(name))
            groups.append({
                "file": name, "fields": sorted(data),
                "exact_sweep_watermark_text_group_match": lookup.get(
                    text_group_hash(data["watermark_sentence"])),
                "secret_key_rows": len(data.get("secret_key", [])),
                "otp_coordinates": len(data.get("one_time_pad", [])),
                "watermark_token_lengths": sorted({len(x) for x in data.get("watermark_sentence_tokens", [])}),
                "null_token_lengths": sorted({len(x) for x in data.get("origin_sentence_tokens", [])}),
            })
    return {"archive_sha256": sha256_file(path), "groups": groups,
            "note": "Companion is T=1.8 only; exact text matching is necessary, not sufficient for key provenance."}


def inspect_archive(archive_path, output, *, max_groups_per_temp=None,
                    companion=None, extract_to=None):
    archive_path, output = Path(archive_path), Path(output)
    digest = sha256_file(archive_path)
    if digest != ARCHIVE_SHA256:
        raise ValueError(f"Pinned archive SHA-256 mismatch: {digest}")
    records = []
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            if not PATH_PATTERN.fullmatch(member.filename):
                raise ValueError(f"Unexpected archive member: {member.filename}")
            raw = archive.read(member)
            records.append(inspect_record(member.filename, raw))
            if extract_to is not None:
                root = Path(extract_to).resolve()
                target = (root / member.filename).resolve()
                if not target.is_relative_to(root):
                    raise ValueError(f"Unsafe archive member: {member.filename}")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
    records.sort(key=lambda r: (r["temperature"], int(r["group_id"])))
    for temp in TEMPERATURES:
        if sum(r["temperature"] == temp for r in records) != 64:
            raise ValueError(f"Pinned artifact must contain 64 files at T={temp}")
    selected = split_groups(records, max_groups_per_temp)
    flags = summarize_flags(records, selected)
    missing = [r for r in selected if r["missing_primary_fields"]]
    report = {
        "status": "blocked_missing_source_data" if missing else "source_fields_present_only",
        "artifact_repo": ARTIFACT_REPO, "artifact_commit": ARTIFACT_COMMIT,
        "archive_path": ARCHIVE_PATH, "archive_sha256": digest,
        "model": MODEL, "model_revision": REVISION, "parameters": PARAMETERS,
        "max_groups_per_temp": max_groups_per_temp,
        "available_groups": len(records), "selected_groups": len(selected),
        "selected_watermarked_completions": 16 * len(selected),
        "selected_null_completions": 16 * len(selected),
        "primary_missing_in_selected_groups": len(missing),
        "primary_missing_in_all_groups": sum(bool(r["missing_primary_fields"]) for r in records),
        "hard_reproduction_passed": False,
        "hard_reproduction_status": "not_attempted_required_source_fields_absent" if missing else "not_attempted",
        "keys_verified": False, "thresholds_frozen": False,
        "lm_calls": 0, "detector_scoring_calls": 0, "generated_texts": 0,
        "modal_runs_launched": 0, "gpu_seconds": 0, "compute_cost_usd": 0,
        "stop_reason": "Cannot independently reproduce hard detection or recover exact completion token IDs from decoded strings without saved keys, OTPs, and token IDs.",
        "saved_flag_summary": flags,
    }
    output.mkdir(parents=True, exist_ok=True)
    if companion:
        report["companion"] = inspect_companion(companion, records)
    json_write(output / "preflight.json", report)
    # Compact inventory preserves exact IDs and planned splits, without copying text.
    fields = ("file", "group_id", "temperature", "file_sha256", "prompts", "saved_det_true",
              "missing_primary_fields", "missing_oracle_fields")
    json_write(output / "inventory.json", [{k: r[k] for k in fields} for r in records])
    json_write(output / "selected_groups.json", [
        {k: r[k] for k in (*fields, "split", "sorted_rank")} for r in selected])
    schema_examples = {}
    for record in records:
        variant = ",".join(record["fields"])
        entry = schema_examples.setdefault(variant, {"example": record["file"], "groups": 0,
                                                    "schema": record["schema"]})
        entry["groups"] += 1
    json_write(output / "schema.json", list(schema_examples.values()))
    with (output / "stored_flags_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flags[0]))
        writer.writeheader()
        writer.writerows(flags)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--companion-archive", type=Path)
    parser.add_argument("--extract-to", type=Path)
    parser.add_argument("--max-groups-per-temp", type=int)
    args = parser.parse_args(argv)
    report = inspect_archive(args.archive, args.output,
                             max_groups_per_temp=args.max_groups_per_temp,
                             companion=args.companion_archive, extract_to=args.extract_to)
    print("Temperature | groups | N | stored positives | stored rate (NOT reproduction)")
    for row in report["saved_flag_summary"]:
        if row["scope"] == "selected_diagnostic":
            print(f"{row['temperature']:.1f} | {row['groups']} | {row['N']} | "
                  f"{row['stored_detected']} | {row['stored_detection_rate']:.6f}")
    print(report["status"] + ": " + report["stop_reason"])
    print("LM calls=0; scoring calls=0; GPU seconds=0; compute cost=$0.")
    return 2 if report["status"] == "blocked_missing_source_data" else 0


if __name__ == "__main__":
    raise SystemExit(main())
