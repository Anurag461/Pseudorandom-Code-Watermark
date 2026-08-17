"""Short-lived, CPU-only Modal replay for Phase 2 adaptive bases."""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
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
RESULT_SCHEMA_VERSION = 1

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
app = modal.App("prc-low-entropy-phase2", image=image)


def _normalized_model_size(model_size: str) -> str:
    value = str(model_size).strip().upper()
    return value if value.endswith("B") else f"{value}B"


def _model_tag(model_size: str) -> str:
    size = _normalized_model_size(model_size).lower().replace(".", "p")
    return f"qwen3_{size}_base"


def _model_display(model_size: str) -> str:
    return f"Qwen3-{_normalized_model_size(model_size)}-Base"


def _tag(
    n: int,
    t: int,
    eta: float,
    r: int,
    generation_model_size: str = MODEL_SIZE,
) -> str:
    base = f"n{n}_t{t}_eta{eta:.2f}_T{n}"
    if _normalized_model_size(generation_model_size) != MODEL_SIZE:
        base += f"__gen-{_model_tag(generation_model_size)}"
    return f"{base}_r{r}"


def _known_run(
    n: int,
    t: int,
    eta: float,
    r: int,
    generation_model_size: str,
) -> dict:
    identity = (
        int(n),
        int(t),
        round(float(eta), 12),
        int(r),
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


def _result_cache_path(
    tag: str,
    fpr: float,
    generation_model_size: str,
) -> str:
    return (
        f"/data/{tag}/low_entropy_phase2/fpr-{_fpr_slug(fpr)}/"
        f"fixed_{_model_tag(generation_model_size)}_"
        "adaptive_basis_v1.json"
    )


def _null_candidates(
    required_T: int,
    num_prompts: int,
    generation_model_size: str,
) -> list[dict]:
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


@app.function(volumes={"/data": data_vol}, cpu=2.0, timeout=3600)
def replay_fixed_run_phase2(
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
    erasure_quantiles: tuple[float, ...] = (
        0.0,
        0.01,
        0.025,
        0.05,
        0.10,
        0.15,
        0.20,
        0.30,
    ),
    persist: bool = True,
) -> dict:
    import numpy as np
    import torch
    import adaptive_parity_basis as basis_module
    import low_entropy_replay as replay_module
    import weighted_rademacher as rademacher_module
    from low_entropy_replay import (
        replay_cached_fixed_map_record,
        replay_cached_fixed_map_record_phase2,
    )

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
            "artifact fingerprint mismatch: expected "
            f"{expected_artifact_fingerprint}, got {actual_fingerprint}"
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
    rank_info = artifact.get("parity_check_rank_info") or {}
    if not rank_info.get("full_rank") or int(rank_info.get("rank", -1)) != r:
        raise ValueError("Phase 2 requires the authoritative full-rank basis")

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
    quantiles = tuple(float(value) for value in erasure_quantiles)
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
                raise ValueError(
                    f"{path} contains invalid partition probabilities"
                )
            record["prompt_idx"] = prompt_idx
            record["watermark"] = watermark
            phase1 = replay_cached_fixed_map_record(
                artifact, record, false_positive_rate=fpr
            )
            phase2 = replay_cached_fixed_map_record_phase2(
                artifact,
                record,
                false_positive_rate=fpr,
                erasure_quantiles=quantiles,
            )
            phase1_block = phase1["blocks"][0]
            phase2_block = phase2["blocks"][0]
            hoeffding = phase1["calibrations"]["hoeffding"]
            phase1_rademacher = phase1["calibrations"][
                "weighted_rademacher_chernoff"
            ]
            phase2_rademacher = phase2["calibrations"][
                "weighted_rademacher_chernoff"
            ]
            selection = phase2_block["basis_selection"]
            if selection["basis_rank"] != r:
                raise AssertionError(f"{path} selected a rank-deficient basis")
            records.append(
                {
                    "prompt_idx": prompt_idx,
                    "watermark": watermark,
                    "phase0_hoeffding_decision": hoeffding["decision"],
                    "phase1_rademacher_decision": phase1_rademacher[
                        "decision"
                    ],
                    "phase1_statistic": phase1_block["statistic"],
                    "phase1_V": phase1_block["V"],
                    "phase1_threshold": phase1_rademacher["threshold"],
                    "phase1_log_pvalue_upper": phase1_rademacher[
                        "log_pvalue_upper"
                    ],
                    "phase2_adaptive_decision": phase2_rademacher["decision"],
                    "phase2_statistic": phase2_block["statistic"],
                    "phase2_V": phase2_block["V"],
                    "phase2_threshold": phase2_rademacher["threshold"],
                    "phase2_log_pvalue_upper": phase2_rademacher[
                        "log_pvalue_upper"
                    ],
                    "selected_erasure_quantile": selection[
                        "erasure_quantile"
                    ],
                    "erased_columns": selection["erased_columns"],
                    "erased_column_rank": selection["erased_column_rank"],
                    "erasure_free_rows": selection["erasure_free_rows"],
                    "log_predicted_J": selection["log_predicted_J"],
                    "degree_minimum": selection["degree_minimum"],
                    "degree_mean": selection["degree_mean"],
                    "degree_median": selection["degree_median"],
                    "degree_maximum": selection["degree_maximum"],
                    "basis_sha256": selection["basis_sha256"],
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

    hoeffding_tp = _count(wm, "phase0_hoeffding_decision")
    hoeffding_fp = _count(null, "phase0_hoeffding_decision")
    if (
        hoeffding_tp != expected_hoeffding_tp
        or hoeffding_fp != expected_hoeffding_fp
    ):
        raise AssertionError(
            "Phase 0 did not reproduce the authoritative fixed-run baseline: "
            f"got TP={hoeffding_tp}, FP={hoeffding_fp}; expected "
            f"{expected_hoeffding_tp} and {expected_hoeffding_fp}"
        )
    phase1_tp = _count(wm, "phase1_rademacher_decision")
    phase1_fp = _count(null, "phase1_rademacher_decision")
    phase2_tp = _count(wm, "phase2_adaptive_decision")
    phase2_fp = _count(null, "phase2_adaptive_decision")
    gained_tp = sum(
        row["phase2_adaptive_decision"]
        and not row["phase1_rademacher_decision"]
        for row in wm
    )
    lost_tp = sum(
        row["phase1_rademacher_decision"]
        and not row["phase2_adaptive_decision"]
        for row in wm
    )
    gained_fp = sum(
        row["phase2_adaptive_decision"]
        and not row["phase1_rademacher_decision"]
        for row in null
    )
    removed_fp = sum(
        row["phase1_rademacher_decision"]
        and not row["phase2_adaptive_decision"]
        for row in null
    )
    quantile_counts = Counter(
        f"{record['selected_erasure_quantile']:.12g}" for record in records
    )
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
        "modal_low_entropy_phase2.py": _source_sha256(__file__),
        "low_entropy_replay.py": _source_sha256(replay_module.__file__),
        "adaptive_parity_basis.py": _source_sha256(basis_module.__file__),
        "weighted_rademacher.py": _source_sha256(rademacher_module.__file__),
    }
    payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "construction": "fixed_prc",
            "detector": "reliability_adaptive_basis_v1",
            "generation_model": _model_display(generation_model_size),
            "generation_model_size": generation_model_size,
            "n": n,
            "T": n,
            "t": t,
            "eta": eta,
            "r": r,
            "target_fpr": fpr,
            "num_prompts": num_prompts,
            "erasure_quantiles": list(quantiles),
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
        "soundness_validation": {
            "artifact_basis_full_rank": True,
            "selected_basis_rank": r,
            "selection_uses": [
                "parity_matrix",
                "partition_probability_reliability",
                "noise_rate",
            ],
            "selection_excludes": [
                "one_time_pad",
                "parity_signs",
                "token_bucket_observations",
                "observed_detection_statistic",
            ],
            "calibration": "weighted_rademacher_chernoff",
        },
        "detector_source_sha256": code_sha256,
        "summary": {
            "phase0_hoeffding": {
                "tp": hoeffding_tp,
                "tpr": hoeffding_tp / num_prompts,
                "fp": hoeffding_fp,
                "fpr": hoeffding_fp / num_prompts,
            },
            "phase1_weighted_rademacher": {
                "tp": phase1_tp,
                "tpr": phase1_tp / num_prompts,
                "fp": phase1_fp,
                "fpr": phase1_fp / num_prompts,
            },
            "phase2_adaptive_basis_weighted_rademacher": {
                "tp": phase2_tp,
                "tpr": phase2_tp / num_prompts,
                "fp": phase2_fp,
                "fpr": phase2_fp / num_prompts,
                "gained_true_positives": int(gained_tp),
                "lost_true_positives": int(lost_tp),
                "gained_false_positives": int(gained_fp),
                "removed_false_positives": int(removed_fp),
            },
            "basis_selection": {
                "quantile_counts": dict(sorted(quantile_counts.items())),
                "median_erased_columns": float(
                    np.median([row["erased_columns"] for row in records])
                ),
                "median_erasure_free_rows": float(
                    np.median([row["erasure_free_rows"] for row in records])
                ),
                "median_degree": float(
                    np.median([row["degree_median"] for row in records])
                ),
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
def verify_cached_phase2_result(
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
        "soundness_validation": payload.get("soundness_validation"),
        "detector_source_sha256": payload.get("detector_source_sha256"),
    }


def _save_local_result(result: dict, output: str) -> None:
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
def run(
    output: str = "outputs/low_entropy_phase2_fixed_n416_eta005_0p6b.json",
) -> None:
    _save_local_result(replay_fixed_run_phase2.remote(), output)


@app.local_entrypoint()
def verify() -> None:
    print(
        json.dumps(
            verify_cached_phase2_result.remote(), indent=2, sort_keys=True
        )
    )


@app.local_entrypoint()
def run_8b_n749(
    output: str = "outputs/low_entropy_phase2_fixed_n749_eta005_8b.json",
) -> None:
    result = replay_fixed_run_phase2.remote(
        n=749,
        t=3,
        eta=0.05,
        r=742,
        fpr=1e-3,
        num_prompts=500,
        generation_model_size="8B",
    )
    _save_local_result(result, output)


@app.local_entrypoint()
def verify_8b_n749() -> None:
    result = verify_cached_phase2_result.remote(
        n=749,
        t=3,
        eta=0.05,
        r=742,
        fpr=1e-3,
        generation_model_size="8B",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
