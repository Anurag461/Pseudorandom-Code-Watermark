"""Validate and aggregate existing SynthID/Gumbel results; no redetection.

Only local cached files are read. No model, detector, generation or remote job
is called. Raw generation shards establish completion identities, not scores.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess

from .config import PREFIX_LENGTHS, SYNTHID_COMMIT, TEXTSEAL_COMMIT
from .textseal_redetect import REPO, file_sha, write_json

RUN_ID = "qwen3-8b-batch50-validation-20260823-v1"
METHODS = ("synthid_text", "gumbel_max")
OUTPUT = REPO / "outputs/comparison_redetect/token_baseline_reuse"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def token_hash(ids):
    return hashlib.sha256(f"int64:({len(ids)},):".encode() + struct.pack(f"<{len(ids)}q", *ids)).hexdigest()


def prepare(generation_cache, output=OUTPUT):
    import torch
    from .comparison_runner import _numpy_pickle_compat
    _numpy_pickle_compat()
    output = Path(output)
    source = REPO / "outputs/controlled_baseline_full" / RUN_ID
    provenance = json.loads((source / "controlled_baseline_full_provenance_manifest.json").read_text())
    source_hashes = {}
    for name in ("controlled_baseline_full_artifact_manifest.json", "controlled_baseline_full_prompt_level.jsonl",
                 "controlled_baseline_full_prefix_summary.csv", "controlled_baseline_full_audit.json"):
        sha = file_sha(source/name)
        require(sha == provenance["compact_artifacts"][name], f"historical source changed: {name}")
        source_hashes[name] = sha
    audit = json.loads((source / "controlled_baseline_full_audit.json").read_text())
    require(audit["checks"]["all_18000_baseline_exact_prefix_deltas_zero"] is True,
            "historical exact prefix check failed")
    tokens = {}
    for i in range(10):
        name = f"shard_{i:02d}.pt"
        path = Path(generation_cache) / "controlled_baseline_full" / RUN_ID / "generated" / name
        require(file_sha(path) == provenance["raw_generation_shards"][name], "raw generation shard changed")
        raw = torch.load(path, map_location="cpu", weights_only=False)
        require(raw["run_id"] == RUN_ID and raw["prompt_indices"] == list(range(i*50,(i+1)*50)),
                "raw generation identity differs")
        for method in METHODS:
            sequences = raw["sequences"][method]
            require(len(sequences) == 50, "generation shard coverage differs")
            for index, sequence in zip(raw["prompt_indices"], sequences):
                ids = list(map(int, sequence["token_ids"]))
                require(len(ids) == 1024, "completion length differs")
                tokens[(method, "watermarked", index)] = token_hash(ids)
    setup = REPO / "outputs/comparison_redetect/textseal_setup/direct_prefix"
    manifest = json.loads((setup / "native8b_manifest.json").read_text())
    require(file_sha(setup / "completion_inputs.jsonl") == manifest["inputs"]["sha256"], "shared null export changed")
    for line in (setup / "completion_inputs.jsonl").open():
        row = json.loads(line)
        if row["method"] == "null":
            for method in METHODS:
                tokens[(method, "null", row["prompt_index"])] = token_hash(row["token_ids"])
    require(len(tokens) == 2000, "completion cohort coverage differs")
    expected = {(method,kind,i,n) for method in METHODS for kind in ("watermarked","null")
                for i in range(500) for n in PREFIX_LENGTHS}
    seen, configs, commits = set(), {}, {"synthid_text": SYNTHID_COMMIT, "gumbel_max": TEXTSEAL_COMMIT}
    counts = {method:{str(n):{kind:{"detected":0,"count":0,"detected_indices":[]} for kind in ("watermarked","null")}
                      for n in PREFIX_LENGTHS} for method in METHODS}
    for line in (source / "controlled_baseline_full_prompt_level.jsonl").open():
        row = json.loads(line)
        method = row["method"]
        if method not in METHODS:
            continue
        key = (method,row["sample_type"],row["prompt_index"],row["prefix_length"])
        require(key in expected and key not in seen, "duplicate or unexpected cached result")
        seen.add(key)
        require(row["generated_token_hash"] == tokens[key[:3]] and row["generated_token_count"] == 1024,
                "cached result completion identity differs")
        require(row["model_revision"] == manifest["model"]["revision"]
                and row["source_repository_commit"] == commits[method], "model or detector revision differs")
        require(0 <= row["p_value"] <= 1 and row["decision"] == (row["p_value"] < .001),
                "cached cutoff or decision differs")
        configs.setdefault(method,row["method_configuration"])
        require(configs[method] == row["method_configuration"], "method configuration varies across cached results")
        c = counts[method][str(row["prefix_length"])][row["sample_type"]]
        c["count"] += 1
        c["detected"] += row["decision"]
        if row["decision"]:
            c["detected_indices"].append(row["prompt_index"])
    require(seen == expected, "cached prefix coverage differs")
    for row in csv.DictReader((source / "controlled_baseline_full_prefix_summary.csv").open()):
        if row["method"] not in METHODS:
            continue
        c = counts[row["method"]][row["prefix_length"]]
        require(c["watermarked"]["count"] == c["null"]["count"] == 500
                and c["watermarked"]["detected"]/500 == float(row["tpr"])
                and c["null"]["detected"] == int(row["false_positive_count"])
                and c["null"]["detected"]/500 == float(row["observed_fpr"])
                and row["nominal_decision_rule"] == "p < 0.001", "cached aggregate differs")
    for method in METHODS:
        for c in counts[method].values():
            c["watermarked"].pop("detected_indices")
    csv_path = OUTPUT.parent / "baseline_comparisons.csv"
    existing = json.loads(csv_path.with_suffix(".provenance.json").read_text())
    require(file_sha(csv_path) == existing["csv_sha256"], "existing CSV changed without matching provenance")
    require(existing["null_source"] == existing["textseal"]["shared_null_source"] == manifest["null_source"],
            "PRC/TextSeal null source differs")
    output.mkdir(parents=True,exist_ok=True)
    for name,src in (("before.csv",csv_path),("before.provenance.json",csv_path.with_suffix(".provenance.json"))):
        target = output/name
        if target.exists():
            require(target.read_bytes() == src.read_bytes(), "frozen before snapshot differs; do not overwrite")
        else:
            shutil.copyfile(src,target)
    # Retain the historical scorer for the prompt-dependency audit. This is
    # source inspection only: it is never imported or executed here.
    historical_source = subprocess.check_output(["git","show","a21263b:baseline_comparison/smoke_runner.py"],cwd=REPO)
    (output/"historical_scorer.py.txt").write_bytes(historical_source)
    summary = {"methods":list(METHODS), "mode":"reuse_existing_detection_results", "counts":counts,
        "prefix_lengths":list(PREFIX_LENGTHS), "nominal_fpr":.001, "shared_null_source":manifest["null_source"],
        "generation_model":manifest["model"]["id"], "model_revision":manifest["model"]["revision"],
        "method_configurations":configs, "source_commits":commits, "source_run":RUN_ID,
        "source_directory":str(source.relative_to(REPO)), "source_sha256":source_hashes,
        "raw_generation_shard_sha256":provenance["raw_generation_shards"],
        "shared_null_export_sha256":manifest["inputs"]["sha256"],
        "historical_scorer_git_ref":"a21263b:baseline_comparison/smoke_runner.py",
        "historical_scorer_sha256":hashlib.sha256(historical_source).hexdigest(),
        "aggregation_source_sha256":file_sha(__file__),
        "input_audit":"Detection uses raw completion prefixes, key-derived Gumbel scores or SynthID g-values, and fixed calibration. Prompt/log-probability fields are used only for provenance and quality metrics; entropy is not used by either test.",
        "checks":{"cached_records_verified":len(seen),"watermarked_completions_match_original_shards":1000,
                  "shared_nulls_match_textseal_and_prc":500,"cached_aggregate_matches":True,
                  "historical_exact_prefix_evidence_checks_passed":True,"detector_calls":0,
                  "model_forwards":0,"remote_calls":0,"additional_compute_usd":0}}
    write_json(output/"summary.json",summary)
    print(json.dumps({"counts":counts,"checks":summary["checks"]},indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-cache",type=Path,required=True)
    parser.add_argument("--output",type=Path,default=OUTPUT)
    args = parser.parse_args()
    prepare(args.generation_cache,args.output)
