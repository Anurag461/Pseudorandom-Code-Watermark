"""Short-lived, cache-only Modal replay for Phase 0/1 fixed PRC detection."""
from __future__ import annotations

import json
import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path

import modal


MODEL_SIZE = "0.6B"
EXPECTED_N416_ARTIFACT = (
    "620983bbf6d62136656cdd20f7ea3de94ae66b7450659d43f4c6de84c805d7be"
)
EXPECTED_N749_8B_ARTIFACT = (
    "42542eb23a7feb5c2981d734d0cd0d53951e12d2521cb0c9142fca691a8cefd1"
)
RESULT_SCHEMA_VERSION = 2

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "scipy", "galois", "numpy")
    .add_local_python_source(
        "prc",
        "detectors",
        "online_prc",
        "weighted_rademacher",
        "adaptive_parity_basis",
        "low_entropy_replay",
    )
)
data_vol = modal.Volume.from_name("prc-data", create_if_missing=False)
app = modal.App("prc-low-entropy-phase01", image=image)


def _normalized_model_size(model_size: str) -> str:
    value = str(model_size).strip().upper()
    return value if value.endswith("B") else f"{value}B"


def _model_tag(model_size: str) -> str:
    size = _normalized_model_size(model_size).lower().replace(".", "p")
    return f"qwen3_{size}_base"


def _model_display(model_size: str) -> str:
    return f"Qwen3-{_normalized_model_size(model_size)}-Base"


def _tag(n: int, t: int, eta: float, r: int,
         generation_model_size: str = MODEL_SIZE) -> str:
    base = f"n{n}_t{t}_eta{eta:.2f}_T{n}"
    if _normalized_model_size(generation_model_size) != MODEL_SIZE:
        base += f"__gen-{_model_tag(generation_model_size)}"
    return f"{base}_r{r}"


def _known_run(n: int, t: int, eta: float, r: int,
               generation_model_size: str) -> dict:
    identity = (
        int(n), int(t), round(float(eta), 12), int(r),
        _normalized_model_size(generation_model_size),
    )
    known = {
        (416, 3, 0.05, 412, "0.6B"): {
            "artifact_fingerprint": EXPECTED_N416_ARTIFACT,
            "hoeffding_tp": 456,
            "hoeffding_fp": 0,
        },
        (749, 3, 0.05, 742, "8B"): {
            "artifact_fingerprint": EXPECTED_N749_8B_ARTIFACT,
            "hoeffding_tp": 455,
            "hoeffding_fp": 0,
        },
    }
    if identity not in known:
        raise ValueError(f"no authoritative baseline registered for {identity}")
    return known[identity]


def _fpr_slug(fpr: float) -> str:
    return f"{float(fpr):.12g}".replace("-", "m").replace(".", "p")


def _result_cache_path(tag: str, fpr: float,
                       generation_model_size: str) -> str:
    return (
        f"/data/{tag}/low_entropy_phase01/fpr-{_fpr_slug(fpr)}/"
        f"fixed_{_model_tag(generation_model_size)}_phase01_v2.json"
    )


def _null_candidates(required_T: int, num_prompts: int,
                     generation_model_size: str) -> list[dict]:
    candidates = []
    root = "/data/_nulls"
    if _normalized_model_size(generation_model_size) != MODEL_SIZE:
        root = f"{root}/{_model_tag(generation_model_size)}"
    if not os.path.isdir(root):
        return candidates
    for name in os.listdir(root):
        if not name.startswith("T") or not name[1:].isdigit():
            continue
        length = int(name[1:])
        if length < required_T:
            continue
        directory = os.path.join(root, name)
        present = sum(
            os.path.exists(os.path.join(directory, f"null_{idx:04d}.pt"))
            for idx in range(num_prompts)
        )
        candidates.append(
            {"T": length, "directory": directory, "present": present}
        )
    return sorted(candidates, key=lambda item: item["T"])


@app.function(volumes={"/data": data_vol}, timeout=300)
def audit_fixed_run(
    n: int = 416,
    t: int = 3,
    eta: float = 0.05,
    r: int = 412,
    num_prompts: int = 500,
    generation_model_size: str = MODEL_SIZE,
) -> dict:
    import torch

    data_vol.reload()
    generation_model_size = _normalized_model_size(generation_model_size)
    tag = _tag(n, t, eta, r, generation_model_size)
    artifact_path = f"/data/{tag}/artifacts.pt"
    wm_directory = f"/data/{tag}/wm"
    result = {
        "tag": tag,
        "artifact_path": artifact_path,
        "artifact_exists": os.path.exists(artifact_path),
        "wm_directory": wm_directory,
        "watermarked_present": sum(
            os.path.exists(os.path.join(wm_directory, f"wm_{idx:04d}.pt"))
            for idx in range(num_prompts)
        ),
        "null_candidates": _null_candidates(
            n, num_prompts, generation_model_size
        ),
        "requested_generation_model_size": generation_model_size,
    }
    if result["artifact_exists"]:
        artifact = torch.load(
            artifact_path, weights_only=False, map_location="cpu"
        )
        result.update(
            {
                "artifact_fingerprint": artifact.get("artifact_fingerprint"),
                "artifact_n": artifact.get("n"),
                "artifact_T": artifact.get("T"),
                "generation_model_size": artifact.get(
                    "generation_model_size", MODEL_SIZE
                ),
                "parity_check_rank_info": artifact.get(
                    "parity_check_rank_info"
                ),
            }
        )
    return result


@app.function(volumes={"/data": data_vol}, timeout=1800)
def replay_fixed_run(
    n: int = 416,
    t: int = 3,
    eta: float = 0.05,
    r: int = 412,
    fpr: float = 1e-3,
    num_prompts: int = 500,
    generation_model_size: str = MODEL_SIZE,
    expected_artifact_fingerprint: str = "",
    expected_hoeffding_tp: int = -1,
    expected_hoeffding_fp: int = -1,
    persist: bool = True,
) -> dict:
    import numpy as np
    import torch
    import low_entropy_replay as replay_module
    import weighted_rademacher as rademacher_module
    from low_entropy_replay import replay_cached_fixed_map_record

    def _semantic_array_sha256(values, dtype) -> str:
        if torch.is_tensor(values):
            array = values.detach().cpu().numpy()
        else:
            array = np.asarray(values)
        array = np.ascontiguousarray(array, dtype=dtype)
        header = f"{array.dtype}:{array.shape}:".encode()
        return hashlib.sha256(header + array.tobytes()).hexdigest()

    def _source_sha256(path: str) -> str:
        with open(path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()

    data_vol.reload()
    generation_model_size = _normalized_model_size(generation_model_size)
    authoritative = _known_run(n, t, eta, r, generation_model_size)
    if not expected_artifact_fingerprint:
        expected_artifact_fingerprint = authoritative["artifact_fingerprint"]
    if expected_hoeffding_tp < 0:
        expected_hoeffding_tp = authoritative["hoeffding_tp"]
    if expected_hoeffding_fp < 0:
        expected_hoeffding_fp = authoritative["hoeffding_fp"]
    tag = _tag(n, t, eta, r, generation_model_size)
    artifact_path = f"/data/{tag}/artifacts.pt"
    wm_directory = f"/data/{tag}/wm"
    if not os.path.exists(artifact_path):
        raise FileNotFoundError(artifact_path)
    artifact = torch.load(
        artifact_path, weights_only=False, map_location="cpu"
    )
    actual_fingerprint = artifact.get("artifact_fingerprint")
    if actual_fingerprint != expected_artifact_fingerprint:
        raise ValueError(
            "artifact fingerprint mismatch: "
            f"expected {expected_artifact_fingerprint}, got {actual_fingerprint}"
        )
    if int(artifact.get("n", -1)) != n or int(artifact.get("T", -1)) != n:
        raise ValueError("artifact is not the requested single-block fixed run")
    model_size = _normalized_model_size(
        artifact.get("generation_model_size", MODEL_SIZE)
    )
    if model_size != generation_model_size:
        raise ValueError(
            f"artifact generation model is {model_size}, expected "
            f"{generation_model_size}"
        )

    candidates = [
        candidate
        for candidate in _null_candidates(
            n, num_prompts, generation_model_size
        )
        if candidate["present"] == num_prompts
    ]
    if not candidates:
        raise FileNotFoundError(
            f"no complete null cache with T >= {n} for {num_prompts} prompts"
        )
    null_cache = candidates[0]
    records = []
    for watermark, directory, prefix in (
        (True, wm_directory, "wm"),
        (False, null_cache["directory"], "null"),
    ):
        for prompt_idx in range(num_prompts):
            path = os.path.join(directory, f"{prefix}_{prompt_idx:04d}.pt")
            record = torch.load(path, weights_only=False, map_location="cpu")
            record_model = record.get("generation_model_size")
            if record_model is not None and _normalized_model_size(
                record_model
            ) != generation_model_size:
                raise ValueError(
                    f"{path} belongs to model {record_model}, expected "
                    f"{generation_model_size}"
                )
            for field in ("tokens", "p_trace"):
                if field not in record:
                    raise KeyError(f"{path} is missing required field {field!r}")
            if len(record["tokens"]) < n or len(record["p_trace"]) < n:
                raise ValueError(f"{path} is shorter than the required n={n}")
            probability_prefix = np.asarray(
                record["p_trace"][:n], dtype=np.float64
            )
            if (
                not np.all(np.isfinite(probability_prefix))
                or np.any(probability_prefix < 0.0)
                or np.any(probability_prefix > 1.0)
            ):
                raise ValueError(f"{path} contains invalid partition probabilities")
            record["prompt_idx"] = prompt_idx
            record["watermark"] = watermark
            replay = replay_cached_fixed_map_record(
                artifact, record, false_positive_rate=fpr
            )
            block = replay["blocks"][0]
            hoeffding = replay["calibrations"]["hoeffding"]
            rademacher = replay["calibrations"][
                "weighted_rademacher_chernoff"
            ]
            records.append(
                {
                    "prompt_idx": prompt_idx,
                    "watermark": watermark,
                    "statistic": block["statistic"],
                    "V": block["V"],
                    "hoeffding_decision": hoeffding["decision"],
                    "hoeffding_threshold": hoeffding["threshold"],
                    "rademacher_decision": rademacher["decision"],
                    "rademacher_threshold": rademacher["threshold"],
                    "rademacher_log_pvalue_upper": rademacher[
                        "log_pvalue_upper"
                    ],
                    "threshold_ratio": (
                        rademacher["threshold"] / hoeffding["threshold"]
                    ),
                    "source_path": path,
                    "tokens_sha256": _semantic_array_sha256(
                        record["tokens"][:n], np.int64
                    ),
                    "p_trace_sha256": _semantic_array_sha256(
                        probability_prefix, np.float64
                    ),
                }
            )

    wm = [record for record in records if record["watermark"]]
    null = [record for record in records if not record["watermark"]]

    def _count(rows, field):
        return int(sum(bool(row[field]) for row in rows))

    hoeffding_tp = _count(wm, "hoeffding_decision")
    hoeffding_fp = _count(null, "hoeffding_decision")
    rademacher_tp = _count(wm, "rademacher_decision")
    rademacher_fp = _count(null, "rademacher_decision")
    if (
        hoeffding_tp != expected_hoeffding_tp
        or hoeffding_fp != expected_hoeffding_fp
    ):
        raise AssertionError(
            "Phase 0 did not reproduce the authoritative fixed-run baseline: "
            f"got TP={hoeffding_tp}, FP={hoeffding_fp}; expected "
            f"{expected_hoeffding_tp} and {expected_hoeffding_fp}"
        )
    ratios = np.asarray([record["threshold_ratio"] for record in records])
    record_manifest = [
        {
            "prompt_idx": record["prompt_idx"],
            "watermark": record["watermark"],
            "tokens_sha256": record["tokens_sha256"],
            "p_trace_sha256": record["p_trace_sha256"],
        }
        for record in records
    ]
    records_sha256 = hashlib.sha256(
        json.dumps(
            record_manifest, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    code_sha256 = {
        "modal_low_entropy_replay.py": _source_sha256(__file__),
        "low_entropy_replay.py": _source_sha256(replay_module.__file__),
        "weighted_rademacher.py": _source_sha256(rademacher_module.__file__),
    }
    payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "construction": "fixed_prc",
            "generation_model": _model_display(generation_model_size),
            "generation_model_size": generation_model_size,
            "n": n,
            "T": n,
            "t": t,
            "eta": eta,
            "r": r,
            "target_fpr": fpr,
            "num_prompts": num_prompts,
        },
        "artifact_path": artifact_path,
        "artifact_fingerprint": actual_fingerprint,
        "null_cache_T": null_cache["T"],
        "source_cache_validation": {
            "required_fields": ["tokens", "p_trace"],
            "minimum_length": n,
            "watermarked_records_validated": len(wm),
            "null_records_validated": len(null),
            "record_manifest_sha256": records_sha256,
        },
        "detector_source_sha256": code_sha256,
        "summary": {
            "hoeffding": {
                "tp": hoeffding_tp,
                "tpr": hoeffding_tp / num_prompts,
                "fp": hoeffding_fp,
                "fpr": hoeffding_fp / num_prompts,
            },
            "weighted_rademacher_chernoff": {
                "tp": rademacher_tp,
                "tpr": rademacher_tp / num_prompts,
                "fp": rademacher_fp,
                "fpr": rademacher_fp / num_prompts,
                "additional_true_positives": rademacher_tp - hoeffding_tp,
                "additional_false_positives": rademacher_fp - hoeffding_fp,
            },
            "threshold_ratio_rademacher_to_hoeffding": {
                "mean": float(np.mean(ratios)),
                "median": float(np.median(ratios)),
                "minimum": float(np.min(ratios)),
                "maximum": float(np.max(ratios)),
            },
        },
        "records": records,
    }
    cache_path = _result_cache_path(tag, fpr, generation_model_size)
    payload["remote_result_path"] = cache_path
    if persist:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        temporary_path = f"{cache_path}.tmp-{os.getpid()}"
        with open(temporary_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, cache_path)
        data_vol.commit()
        payload["remote_result_bytes"] = os.path.getsize(cache_path)
    return payload


@app.function(volumes={"/data": data_vol}, timeout=300)
def verify_cached_result(
    n: int = 416,
    t: int = 3,
    eta: float = 0.05,
    r: int = 412,
    fpr: float = 1e-3,
    generation_model_size: str = MODEL_SIZE,
) -> dict:
    data_vol.reload()
    generation_model_size = _normalized_model_size(generation_model_size)
    tag = _tag(n, t, eta, r, generation_model_size)
    path = _result_cache_path(tag, fpr, generation_model_size)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "rb") as handle:
        raw = handle.read()
    payload = json.loads(raw)
    records = payload.get("records", [])
    manifest = [
        {
            "prompt_idx": record["prompt_idx"],
            "watermark": record["watermark"],
            "tokens_sha256": record["tokens_sha256"],
            "p_trace_sha256": record["p_trace_sha256"],
        }
        for record in records
    ]
    manifest_sha256 = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    expected_manifest = payload["source_cache_validation"][
        "record_manifest_sha256"
    ]
    identities = {
        (record["watermark"], record["prompt_idx"]) for record in records
    }
    return {
        "remote_result_path": path,
        "remote_result_bytes": len(raw),
        "remote_result_sha256": hashlib.sha256(raw).hexdigest(),
        "schema_version": payload.get("schema_version"),
        "artifact_fingerprint": payload.get("artifact_fingerprint"),
        "null_cache_T": payload.get("null_cache_T"),
        "record_count": len(records),
        "unique_record_identities": len(identities),
        "manifest_matches": manifest_sha256 == expected_manifest,
        "summary": payload.get("summary"),
        "detector_source_sha256": payload.get("detector_source_sha256"),
    }


@app.local_entrypoint()
def audit() -> None:
    print(json.dumps(audit_fixed_run.remote(), indent=2, sort_keys=True))


@app.local_entrypoint()
def run(
    output: str = "outputs/low_entropy_phase01_fixed_n416_eta005_0p6b.json",
) -> None:
    result = replay_fixed_run.remote()
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(f"cached remote replay -> {result['remote_result_path']}")
    print(f"saved compact replay -> {destination}")


@app.local_entrypoint()
def verify() -> None:
    print(json.dumps(verify_cached_result.remote(), indent=2, sort_keys=True))


@app.local_entrypoint()
def audit_8b_n749() -> None:
    result = audit_fixed_run.remote(
        n=749,
        t=3,
        eta=0.05,
        r=742,
        num_prompts=500,
        generation_model_size="8B",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


@app.local_entrypoint()
def run_8b_n749(
    output: str = "outputs/low_entropy_phase01_fixed_n749_eta005_8b.json",
) -> None:
    result = replay_fixed_run.remote(
        n=749,
        t=3,
        eta=0.05,
        r=742,
        fpr=1e-3,
        num_prompts=500,
        generation_model_size="8B",
    )
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(f"cached remote replay -> {result['remote_result_path']}")
    print(f"saved compact replay -> {destination}")


@app.local_entrypoint()
def verify_8b_n749() -> None:
    result = verify_cached_result.remote(
        n=749,
        t=3,
        eta=0.05,
        r=742,
        fpr=1e-3,
        generation_model_size="8B",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
