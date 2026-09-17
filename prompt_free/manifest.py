"""Frozen input manifests, deterministic run identities and local planning."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess

from prompt_free.core import PROTOCOL

SOURCE_FILES = (
    "prompt_free/__init__.py", "prompt_free/core.py", "prompt_free/manifest.py",
    "prompt_free/storage.py", "prompt_free/modal_redetect.py", "prompt_free/validation.py",
    "prompt_free/requirements.txt", "qwen.py", "detectors.py", "prc.py", "online_prc.py",
)


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("expected a relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(p in (".", "..") for p in value.split("/")):
        raise ValueError("paths must stay inside their configured volume")
    return value


def sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("expected a lowercase SHA-256 digest")


def _reference(ref):
    if set(ref) != {"volume", "path", "sha256", "bytes"}:
        raise ValueError("source references require exactly volume, path, sha256 and bytes")
    if ref["volume"] not in ("data", "archive"):
        raise ValueError("source volume must be data or archive")
    relative_path(ref["path"])
    sha(ref["sha256"])
    if type(ref["bytes"]) is not int or ref["bytes"] <= 0:
        raise ValueError("invalid source file size")


def validate(manifest):
    if set(manifest) != {"schema_version", "protocol", "model", "cases"}:
        raise ValueError("unexpected manifest fields; prompts and old traces are not inputs")
    if manifest["schema_version"] != 1 or manifest["protocol"] != PROTOCOL:
        raise ValueError("only the raw-completion abstention protocol is accepted")
    model = manifest["model"]
    if set(model) != {"id", "size", "revision", "weights_sha256", "tokenizer_sha256", "cache_directory", "dtype"}:
        raise ValueError("model checkpoint identity is incomplete")
    if (model["id"], model["size"], model["dtype"]) != ("Qwen/Qwen3-0.6B-Base", "0.6B", "bfloat16"):
        raise ValueError("this audited configuration supports only BF16 Qwen3-0.6B-Base detection")
    if not isinstance(model["revision"], str) or len(model["revision"]) != 40 or any(c not in "0123456789abcdef" for c in model["revision"]):
        raise ValueError("model revision must be an immutable commit")
    sha(model["weights_sha256"]); sha(model["tokenizer_sha256"])
    relative_path(model["cache_directory"])
    if not isinstance(manifest["cases"], list) or not manifest["cases"]:
        raise ValueError("at least one experiment case is required")
    names = set()
    for case in manifest["cases"]:
        required = {"id", "generation_model", "construction", "artifact", "lengths", "fpr", "fpr_policy", "weights", "batch_size", "cache", "records"}
        if set(case) != required:
            raise ValueError("unexpected or missing case fields")
        name = case["id"]
        if not isinstance(name, str) or not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in name) or name in names:
            raise ValueError("case ids must be unique lowercase identifiers")
        names.add(name)
        if not isinstance(case["generation_model"], str) or not case["generation_model"]:
            raise ValueError("generation model label is required")
        _reference(case["artifact"])
        lengths = case["lengths"]
        if not isinstance(lengths, list) or not lengths or any(type(n) is not int or not 1 <= n <= 40960 for n in lengths) or len(set(lengths)) != len(lengths):
            raise ValueError("lengths must be unique positive integers within model context")
        if type(case["batch_size"]) is not int or not 1 <= case["batch_size"] <= 100:
            raise ValueError("choose an explicit document batch size in [1,100]")
        if case["cache"] not in ("concat", "static"):
            raise ValueError("unsupported cache implementation")
        if not isinstance(case["fpr"], (int, float)) or isinstance(case["fpr"], bool) or not 0 < case["fpr"] < 1:
            raise ValueError("invalid FPR")
        policies = {"fixed": ("block_or_bonferroni",), "online": ("one_shot", "alpha_spending_v1")}
        if case["construction"] not in policies or case["fpr_policy"] not in policies[case["construction"]]:
            raise ValueError("FPR policy must match the original PRC construction")
        if not isinstance(case["weights"], list) or not case["weights"] or len(set(case["weights"])) != len(case["weights"]) or any(w not in ("map", "entropy") for w in case["weights"]):
            raise ValueError("weights must be map and/or entropy")
        if not isinstance(case["records"], list) or not case["records"]:
            raise ValueError("explicit frozen source records are required")
        ids = set()
        for record in case["records"]:
            if set(record) != {"source", "prompt_idx", "file", "tokens_sha256"}:
                raise ValueError("record schema accepts source references, not prompt/probability inputs")
            if record["source"] not in ("wm", "null") or type(record["prompt_idx"]) is not int or record["prompt_idx"] < 0:
                raise ValueError("invalid source identity")
            ident = (record["source"], record["prompt_idx"])
            if ident in ids:
                raise ValueError("duplicate candidate")
            ids.add(ident)
            _reference(record["file"]); sha(record["tokens_sha256"])
        if {s for s, _ in ids} != {"wm", "null"}:
            raise ValueError("both watermarked and null candidates are required")
    return manifest


def source_identity(root, *, require_commit=False):
    root = Path(root)
    hashes = {name: file_sha(root/name) for name in SOURCE_FILES}
    result = {"files": hashes, "sha256": digest_json(hashes)}
    if require_commit:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        for name, expected in hashes.items():
            raw = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=root)
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError(f"execution source is not committed: {name}")
        result["git_commit"] = commit
    return result


def plan(manifest):
    validate(manifest)
    return {"protocol": PROTOCOL, "manifest_sha256": digest_json(manifest),
            "model": manifest["model"]["id"], "inference_launched": False,
            "cases": [{"id": c["id"], "documents": len(c["records"]),
                       "lengths": c["lengths"], "batch_size": c["batch_size"],
                       "batches": (len(c["records"])+c["batch_size"]-1)//c["batch_size"],
                       "model_forward_token_positions": len(c["records"])*(max(c["lengths"])-1),
                       "fpr_policy": c["fpr_policy"]} for c in manifest["cases"]]}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate and plan a campaign without accessing Modal")
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    print(json.dumps(plan(json.loads(args.manifest.read_text())), indent=2))
