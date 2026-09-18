"""Read-only Modal-volume inventory for the completion-only comparison.

This is a local SDK client, not a Modal app: it launches no remote CPU/GPU jobs.
Source files are copied to a local cache and checked against historical tokens.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from .config import MAX_NEW_TOKENS, ONLINE_PRC_SOURCE_TAG, SHARED_NULL_SOURCE_T


RUN_ID = "qwen3-8b-batch50-validation-20260823-v1"
GENERATED_ROOT = f"controlled_baseline_full/{RUN_ID}/generated"
WM_ROOT = f"{ONLINE_PRC_SOURCE_TAG}/wm"
NULL_ROOT = f"_nulls/qwen3_8b_base/T{SHARED_NULL_SOURCE_T}"
REPO = Path(__file__).resolve().parents[1]


def historical_token_hashes() -> dict:
    root = REPO / "outputs/controlled_baseline_full" / RUN_ID
    manifest = json.loads((root / "controlled_baseline_full_artifact_manifest.json").read_text())
    filename = "controlled_baseline_full_prompt_level.jsonl"
    expected = next(item["sha256"] for item in manifest["artifacts"] if item["path"] == filename)
    path = root / filename
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("historical comparison records changed")
    hashes = {}
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["prefix_length"] != MAX_NEW_TOKENS:
                continue
            method = "null" if row["sample_type"] == "null" else row["method"]
            if method not in {"null", "textseal", "online_prc"}:
                continue
            key = (method, row["prompt_index"])
            digest = row["generated_token_hash"]
            if key in hashes and hashes[key] != digest:
                raise ValueError("shared null hash differs across historical methods")
            hashes[key] = digest
    if set(hashes) != {(method, index) for method in ("null", "textseal", "online_prc") for index in range(500)}:
        raise ValueError("historical comparison coverage differs")
    return hashes


def validate_source(path: Path, kind: str, index: int | None, expected_hashes: dict) -> list[dict]:
    import numpy as np
    import torch
    from .comparison_runner import _numpy_pickle_compat, _token_sha256

    _numpy_pickle_compat()
    source = torch.load(path, map_location="cpu", weights_only=False)
    if kind == "textseal":
        indices = list(range(index * 50, (index + 1) * 50))
        if source["prompt_indices"] != indices or source["run_id"] != RUN_ID:
            raise ValueError("TextSeal shard identity differs")
        sequences = source["sequences"]["textseal"]
        if len(sequences) != 50:
            raise ValueError("TextSeal shard coverage differs")
        rows = [(i, sequence["token_ids"]) for i, sequence in zip(indices, sequences)]
    else:
        if source["prompt_idx"] != index or bool(source["watermark"]) != (kind == "online_prc"):
            raise ValueError("PRC/null identity differs")
        rows = [(index, source["tokens"])]
    clean = []
    for prompt_index, values in rows:
        tokens = np.asarray(values).reshape(-1)
        if tokens.dtype.kind not in "iu" or len(tokens) < MAX_NEW_TOKENS or np.any(tokens < 0):
            raise ValueError("invalid completion tokens")
        ids = list(map(int, tokens[:MAX_NEW_TOKENS]))
        digest = _token_sha256(ids)
        if digest != expected_hashes[(kind, prompt_index)]:
            raise ValueError(f"completion changed: {kind}/{prompt_index}")
        # No prompt, cached probability, entropy, or generation cache is exported.
        clean.append({"method": kind, "prompt_index": prompt_index, "token_ids": ids,
                      "historical_token_sha256": digest})
    return clean


async def inventory(cache_dir: Path, output_dir: Path) -> dict:
    import modal

    expected_hashes = historical_token_hashes()
    original = REPO / "outputs/controlled_baseline_full" / RUN_ID / "controlled_baseline_full_provenance_manifest.json"
    provenance = json.loads(original.read_text())
    volume = modal.Volume.from_name("prc-data", create_if_missing=False)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    requests = []
    for root, kind in ((GENERATED_ROOT, "textseal"), (WM_ROOT, "online_prc"), (NULL_ROOT, "null")):
        entries = {entry.path: entry async for entry in volume.iterdir.aio(root)}
        for i in range(10 if kind == "textseal" else 500):
            name = f"shard_{i:02d}.pt" if kind == "textseal" else f"{'wm' if kind == 'online_prc' else 'null'}_{i:04d}.pt"
            path = f"{root}/{name}"
            entry = entries[path]  # Missing required files stop the preflight.
            expected = provenance["raw_generation_shards"][name] if kind == "textseal" else None
            requests.append({"path": path, "bytes": entry.size, "kind": kind, "index": i,
                             "expected_sha256": expected})

    semaphore = asyncio.Semaphore(8)
    completed = 0

    async def fetch(request):
        nonlocal completed
        async with semaphore:
            path = cache_dir / request["path"]
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".partial")
                with temporary.open("wb") as stream:
                    async for chunk in volume.read_file.aio(request["path"]):
                        stream.write(chunk)
                temporary.replace(path)
            if path.stat().st_size != request["bytes"]:
                raise ValueError(f"source size differs: {request['path']}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if request["expected_sha256"] and digest != request["expected_sha256"]:
                raise ValueError(f"source hash differs: {request['path']}")
            rows = validate_source(path, request["kind"], request["index"], expected_hashes)
            completed += 1
            if completed % 100 == 0 or completed == len(requests):
                print(f"Validated {completed}/{len(requests)} cached source files", flush=True)
            return ({**request, "sha256": digest}, rows)

    fetched = await asyncio.gather(*(fetch(request) for request in requests))
    rows = sorted((row for _, batch in fetched for row in batch), key=lambda row: (row["method"], row["prompt_index"]))
    if len(rows) != 1500:
        raise ValueError("incomplete clean input coverage")

    # Freeze the original PRC key/partition artifact; never invent new keys.
    artifact_path = f"{ONLINE_PRC_SOURCE_TAG}/artifacts.pt"
    artifact_bytes = b"".join([chunk async for chunk in volume.read_file.aio(artifact_path)])
    artifact_local = cache_dir / artifact_path
    artifact_local.parent.mkdir(parents=True, exist_ok=True)
    artifact_local.write_bytes(artifact_bytes)
    import torch
    from detectors import semantic_sha256
    from online_prc import OnlinePRCKey
    artifact = torch.load(artifact_local, map_location="cpu", weights_only=False)
    unhashed = {key: value for key, value in artifact.items() if key != "artifact_fingerprint"}
    if (semantic_sha256(unhashed) != artifact["artifact_fingerprint"]
            or artifact["artifact_fingerprint"] != provenance["fingerprints"]["online_prc_artifact_sha256"]):
        raise ValueError("PRC artifact fingerprint differs")
    key = OnlinePRCKey.from_dict(artifact["online_key"])

    hf = modal.Volume.from_name("prc-hf-cache", create_if_missing=False)
    models = {}
    for name, revision in (("Qwen3-8B-Base", "49e3418fbbbca6ecbdf9608b4d22e5a407081db4"),
                           ("Qwen3-0.6B-Base", "da87bfb608c14b7cf20ba1ce41287e8de496c0cd")):
        entries = [entry async for entry in hf.iterdir.aio(f"models/{name}", recursive=True)]
        weights = [{"path": entry.path, "bytes": entry.size} for entry in entries if entry.path.endswith(".safetensors")]
        paths = {entry.path for entry in entries}
        if not weights or f"models/{name}/tokenizer.json" not in paths:
            raise ValueError(f"model weights/tokenizer missing: {name}")
        metadata_files = [Path(weight["path"]).name for weight in weights] + ["tokenizer.json"]
        missing_configs = [filename for filename in ("config.json", "tokenizer_config.json")
                           if f"models/{name}/{filename}" not in paths]
        metadata_files += [filename for filename in ("config.json", "tokenizer_config.json")
                           if filename not in missing_configs]
        metadata_hashes = {}
        for filename in metadata_files:
            metadata_path = f"models/{name}/.cache/huggingface/download/{filename}.metadata"
            metadata = b"".join([chunk async for chunk in hf.read_file.aio(metadata_path)])
            if metadata.decode().splitlines()[0] != revision:
                raise ValueError(f"model cache revision differs: {metadata_path}")
            metadata_hashes[metadata_path] = hashlib.sha256(metadata).hexdigest()
        models[name] = {"revision": revision, "weights": weights,
                        "download_metadata_sha256": metadata_hashes,
                        "missing_hf_config_files": missing_configs,
                        "hf_configuration_ready": not missing_configs,
                        "verification": "weight/tokenizer/config download revisions and weight-file inventory; weights not loaded or rehashed"}

    inputs_path = output_dir / "completion_inputs.jsonl"
    with inputs_path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    report = {
        "status": "passed", "run_id": RUN_ID, "mode": "read_only_modal_volume_client",
        "remote_compute_jobs": 0, "model_loads": 0, "generation_attempts": 0,
        "coverage": {"textseal": 500, "online_prc": 500, "null": 500, "completion_length": MAX_NEW_TOKENS},
        "source_files": [source for source, _ in fetched],
        "source_bytes": sum(source["bytes"] for source, _ in fetched) + len(artifact_bytes),
        "prc_artifact": {"path": artifact_path, "sha256": hashlib.sha256(artifact_bytes).hexdigest(),
                         "artifact_fingerprint": artifact["artifact_fingerprint"], "online_key_fingerprint": key.fingerprint},
        "models": models,
        "replay_prerequisites": [f"Stage {', '.join(info['missing_hf_config_files'])} from pinned {name} revision before HF replay"
                                 for name, info in models.items() if info["missing_hf_config_files"]],
        "completion_inputs": {"path": inputs_path.name, "sha256": hashlib.sha256(inputs_path.read_bytes()).hexdigest()},
    }
    (output_dir / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(inventory(args.cache_dir, args.output_dir))
    print(json.dumps({key: result[key] for key in ("status", "coverage", "source_bytes", "remote_compute_jobs")}))


if __name__ == "__main__":
    main()
