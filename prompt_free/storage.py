"""Hash-checked sources and resumable batches; no model inference at import."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import torch

from detectors import semantic_sha256, tensor_sha256
from prompt_free.core import PROTOCOL, score, validate_partition, validate_tokens
from prompt_free.manifest import digest_json, file_sha, relative_path

TRACE_FIELD = "partition_probability_coordinates_2_to_T"


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def load_pt(path):
    # Original caches were authored using NumPy 2; the validated GPU image
    # uses NumPy 1.26. Only already hash-verified, user-owned caches are loaded.
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
    return torch.load(path, weights_only=False, map_location="cpu")


def read_source(ref, source_roots):
    path = Path(source_roots[ref["volume"]])/relative_path(ref["path"])
    raw = path.read_bytes()
    if len(raw) != ref["bytes"] or hashlib.sha256(raw).hexdigest() != ref["sha256"]:
        raise ValueError(f"source hash/size mismatch: {ref['volume']}:{ref['path']}")
    return load_pt(io.BytesIO(raw))


def token_sha(tokens):
    return hashlib.sha256(validate_tokens(tokens).contiguous().numpy().tobytes()).hexdigest()


def save_immutable_pt(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if semantic_sha256(load_pt(path)) != semantic_sha256(value):
            raise ValueError(f"existing prepared input differs: {path}")
    else:
        temporary = path.with_suffix(".partial")
        torch.save(value, temporary)
        temporary.replace(path)
    return file_sha(path)


def prepare_case(case, model, source, source_roots, result_root):
    """CPU-only preflight. GPU payloads have exactly tokens and partition."""
    from concurrent.futures import ThreadPoolExecutor
    from online_prc import OnlinePRCKey
    raw_artifact = read_source(case["artifact"], source_roots)
    key_name = "decoding_key" if case["construction"] == "fixed" else "online_key"
    artifact = {"partition": raw_artifact["partition"], key_name: raw_artifact[key_name]}
    validate_partition(artifact["partition"])
    if case["construction"] == "online":
        OnlinePRCKey.from_dict(artifact["online_key"])
    else:
        key = artifact["decoding_key"]
        matrix = key[1]
        if (matrix.shape[1] <= 0 or key[0].shape[0] != matrix.shape[1]
                or np.asarray(key[2]).shape != (matrix.shape[1],)
                or not np.all((np.asarray(key[2]) == 0) | (np.asarray(key[2]) == 1))
                or not np.all(np.diff(matrix.indptr) == key[-1]) or not np.all(matrix.data == 1)):
            raise ValueError("invalid fixed PRC key")
    identity = {"protocol": PROTOCOL, "model": model, "case": case, "source": source,
                "artifact_sha256": semantic_sha256(artifact),
                "partition_sha256": tensor_sha256(artifact["partition"]),
                "key_sha256": semantic_sha256(artifact[key_name]),
                "first_coordinate_score": 0., "prepended_token_count": 0,
                "token_step": 1, "threshold_policy": "original_formula_new_V"}
    run_id = digest_json(identity)[:24]
    root = Path(result_root)/PROTOCOL/case["id"]/run_id
    root.mkdir(parents=True, exist_ok=True)
    identity_file = root/"identity.json"
    if identity_file.exists() and json.loads(identity_file.read_text()) != identity:
        raise ValueError("run identity collision")
    json_write(identity_file, identity)
    artifact_sha = save_immutable_pt(root/"scoring_artifact.pt", artifact)
    maximum = max(case["lengths"])

    def extract(ref):
        raw = read_source(ref["file"], source_roots)
        if raw.get("prompt_idx") != ref["prompt_idx"] or raw.get("watermark") != (ref["source"] == "wm"):
            raise ValueError("candidate label/id differs from manifest")
        tokens = validate_tokens(raw["tokens"])
        if len(tokens) < maximum:
            raise ValueError("candidate is shorter than the requested prefix")
        tokens = tokens[:maximum].clone()
        if token_sha(tokens) != ref["tokens_sha256"]:
            raise ValueError("candidate tokens differ from frozen manifest")
        if torch.any(tokens < 0) or torch.any(tokens >= artifact["partition"].shape[1]):
            raise ValueError("candidate token outside partition")
        # No prompt or generation-probability field is extracted.
        return tokens

    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(extract, case["records"]))
    batches = []
    size = case["batch_size"]
    for start in range(0, len(rows), size):
        selected = case["records"][start:start+size]
        tokens = torch.stack(rows[start:start+size])
        batch_root = root/"batches"/f"{start:06d}"
        gpu_input = {"tokens": tokens, "partition": artifact["partition"].to(torch.bfloat16)}
        input_sha = save_immutable_pt(batch_root/"inputs.pt", gpu_input)
        trace_identity = {"protocol": PROTOCOL, "run_id": run_id, "case_id": case["id"],
                          "model": model, "code_sha256": source["sha256"],
                          "input_sha256": input_sha, "partition_sha256": tensor_sha256(gpu_input["partition"]),
                          "maximum_length": maximum, "cache": case["cache"], "token_step": 1,
                          "actual_batch_size": len(selected), "requested_batch_size": size,
                          "records": [{k: r[k] for k in ("source", "prompt_idx", "tokens_sha256")} for r in selected],
                          "prepended_token_count": 0, "first_coordinate_score": 0.}
        descriptor = {"root": str(batch_root.relative_to(result_root)), "identity": trace_identity}
        json_write(batch_root/"batch.json", descriptor)
        if (batch_root/"trace.pt").exists():
            validate_trace(load_pt(batch_root/"trace.pt"), trace_identity)
        batches.append(descriptor)
    prepared = {"root": str(root.relative_to(result_root)), "run_id": run_id,
                "identity": identity, "artifact_file_sha256": artifact_sha, "batches": batches}
    json_write(root/"prepared.json", prepared)
    return prepared


def load_gpu_input(descriptor, result_root):
    path = Path(result_root)/relative_path(descriptor["root"])/"inputs.pt"
    identity = descriptor["identity"]
    if file_sha(path) != identity["input_sha256"]:
        raise ValueError("prepared GPU input hash changed")
    value = load_pt(path)
    if set(value) != {"tokens", "partition"}:
        raise ValueError("GPU input may contain only tokens and partition")
    validate_partition(value["partition"])
    tokens = value["tokens"]
    expected = (identity["actual_batch_size"], identity["maximum_length"])
    if tokens.shape != expected or tokens.dtype != torch.int64:
        raise ValueError("GPU input has different shape or dtype")
    if tensor_sha256(value["partition"]) != identity["partition_sha256"]:
        raise ValueError("GPU partition identity changed")
    for row, ref in zip(tokens, identity["records"]):
        if token_sha(row) != ref["tokens_sha256"]:
            raise ValueError("GPU input batch order or token identity changed")
    return value


def trace_payload(trace, identity):
    payload = {"identity": identity, TRACE_FIELD: trace,
               "probabilities_sha256": tensor_sha256(trace)}
    validate_trace(payload, identity)
    return payload


def validate_trace(payload, expected):
    if set(payload) != {"identity", TRACE_FIELD, "probabilities_sha256"}:
        raise ValueError("legacy or incomplete trace payload")
    if payload["identity"] != expected or expected.get("protocol") != PROTOCOL:
        raise ValueError("trace protocol/execution/source identity differs")
    trace = payload[TRACE_FIELD]
    shape = (expected["actual_batch_size"], expected["maximum_length"]-1)
    if not isinstance(trace, torch.Tensor) or trace.dtype != torch.float32 or trace.shape != shape:
        raise ValueError("trace must have exactly T-1 float32 stored probabilities")
    if not torch.isfinite(trace).all() or torch.any((trace < 0) | (trace > 1)):
        raise ValueError("invalid probability values")
    if tensor_sha256(trace) != payload["probabilities_sha256"]:
        raise ValueError("cached probability contents changed")
    return trace


def aggregate(prepared, result_root):
    root = Path(result_root)/relative_path(prepared["root"])
    if file_sha(root/"scoring_artifact.pt") != prepared["artifact_file_sha256"]:
        raise ValueError("original scoring key/partition changed")
    artifact = load_pt(root/"scoring_artifact.pt")
    if semantic_sha256(artifact) != prepared["identity"]["artifact_sha256"]:
        raise ValueError("artifact identity differs")
    case = prepared["identity"]["case"]
    rows, trace_hashes = [], {}
    for descriptor in prepared["batches"]:
        data = load_gpu_input(descriptor, result_root)
        path = Path(result_root)/descriptor["root"]/"trace.pt"
        trace = validate_trace(load_pt(path), descriptor["identity"]).numpy()
        trace_hashes[descriptor["root"]] = file_sha(path)
        for i, ref in enumerate(descriptor["identity"]["records"]):
            scores = {}
            for length in case["lengths"]:
                scores[str(length)] = {weight: score(
                    artifact, data["tokens"][i, :length], trace[i, :length-1],
                    construction=case["construction"], fpr=case["fpr"],
                    fpr_policy=case["fpr_policy"], weight=weight) for weight in case["weights"]}
            rows.append({**ref, "scores": scores})
    fields = ("source", "prompt_idx", "tokens_sha256")
    expected_order = [tuple(r[k] for k in fields) for r in case["records"]]
    actual_order = [tuple(r[k] for k in fields) for r in rows]
    if actual_order != expected_order or len({(r["source"], r["prompt_idx"]) for r in rows}) != len(rows):
        raise ValueError("missing or duplicated scored candidate")
    counts = {}
    for length in case["lengths"]:
        counts[str(length)] = {}
        for weight in case["weights"]:
            counts[str(length)][weight] = {}
            for source in ("wm", "null"):
                selected = [r for r in rows if r["source"] == source]
                detected = sum(r["scores"][str(length)][weight]["decision"] for r in selected)
                counts[str(length)][weight][source] = {"detected": detected, "count": len(selected), "rate": detected/len(selected)}
    report = {"passed": True, "run_id": prepared["run_id"], "identity": prepared["identity"],
              "counts": counts, "trace_shard_sha256": trace_hashes, "records": rows}
    json_write(root/"full.json", report)
    json_write(root/"summary.json", {k: v for k, v in report.items() if k != "records"})
    return {"root": prepared["root"], "counts": counts, "passed": True}
