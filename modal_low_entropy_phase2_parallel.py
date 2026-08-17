"""Distributed, resumable Modal replay for Phase 2 adaptive detection."""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

from modal_low_entropy_phase2 import (
    MODEL_SIZE,
    _fpr_slug,
    _known_run,
    _model_display,
    _model_tag,
    _normalized_model_size,
    _null_candidates,
    _tag,
    data_vol,
    image as base_image,
)
from phase2_parallel import (
    PHASE2_PARALLEL_RESULT_SCHEMA_VERSION,
    PHASE2_SHARD_SCHEMA_VERSION,
    merge_phase2_shard_payloads,
    phase2_config_fingerprint,
    phase2_prompt_shards,
    phase2_record_manifest,
    stable_json_sha256,
    summarize_phase2_records,
    validate_phase2_shard_payload,
)


DEFAULT_ERASURE_QUANTILES = (
    0.0,
    0.01,
    0.025,
    0.05,
    0.10,
    0.15,
    0.20,
    0.30,
)
DEFAULT_SHARD_SIZE = 10
DEFAULT_MAX_CONTAINERS = 32

image = base_image.add_local_python_source(
    "phase2_parallel",
    "modal_low_entropy_phase2",
)
app = modal.App("prc-low-entropy-phase2-parallel", image=image)


def _source_sha256(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _known_run_or_none(
    n: int,
    t: int,
    eta: float,
    r: int,
    generation_model_size: str,
) -> dict | None:
    try:
        return _known_run(n, t, eta, r, generation_model_size)
    except ValueError:
        return None


def _grid_sha256(erasure_quantiles) -> str:
    return stable_json_sha256([float(value) for value in erasure_quantiles])


def _parallel_result_cache_path(
    tag: str,
    fpr: float,
    generation_model_size: str,
    num_prompts: int,
    erasure_quantiles,
) -> str:
    grid = _grid_sha256(erasure_quantiles)[:12]
    return (
        f"/data/{tag}/low_entropy_phase2/parallel-v2/"
        f"fpr-{_fpr_slug(fpr)}/prompts-{int(num_prompts)}/grid-{grid}/"
        f"fixed_{_model_tag(generation_model_size)}_adaptive_basis.json"
    )


def _online_parallel_result_cache_path(
    source_tag: str,
    n: int,
    fpr: float,
    generation_model_size: str,
    num_prompts: int,
    erasure_quantiles,
) -> str:
    grid = _grid_sha256(erasure_quantiles)[:12]
    return (
        f"/data/{source_tag}/low_entropy_phase2/online-parallel-v1/"
        f"prefix-n{int(n)}/fpr-{_fpr_slug(fpr)}/"
        f"prompts-{int(num_prompts)}/grid-{grid}/"
        f"{_model_tag(generation_model_size)}_adaptive_basis.json"
    )


def _parallel_shard_cache_path(
    tag: str,
    config_fingerprint: str,
    prompt_indices: list[int],
) -> str:
    if not prompt_indices:
        raise ValueError("cannot name an empty Phase 2 shard")
    prompt_hash = stable_json_sha256(
        [int(index) for index in prompt_indices]
    )[:12]
    label = (
        f"{int(prompt_indices[0]):04d}-{int(prompt_indices[-1]):04d}"
        f"-count{len(prompt_indices)}-{prompt_hash}"
    )
    return (
        f"/data/{tag}/low_entropy_phase2/parallel-v2/"
        f"config-{config_fingerprint[:20]}/shards/prompts-{label}.json"
    )


def _atomic_write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary_path = f"{path}.tmp-{os.getpid()}"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)


@app.function(volumes={"/data": data_vol}, cpu=1.0, timeout=600)
def resolve_parallel_phase2_run(
    n: int,
    t: int,
    eta: float,
    r: int,
    fpr: float = 1e-3,
    num_prompts: int = 500,
    generation_model_size: str = MODEL_SIZE,
    expected_artifact_fingerprint: str = "",
    expected_hoeffding_tp: int = -1,
    expected_hoeffding_fp: int = -1,
    erasure_quantiles: tuple[float, ...] = DEFAULT_ERASURE_QUANTILES,
) -> dict:
    """Resolve and validate shared inputs once before fan-out."""
    import math

    import adaptive_parity_basis as basis_module
    import low_entropy_replay as replay_module
    import modal_low_entropy_phase2 as serial_modal_module
    import phase2_parallel as parallel_module
    import torch
    import weighted_rademacher as rademacher_module
    from prc import parity_check_rank_info

    n = int(n)
    t = int(t)
    eta = float(eta)
    r = int(r)
    fpr = float(fpr)
    num_prompts = int(num_prompts)
    if n <= 0 or t <= 0 or r <= 0 or num_prompts <= 0:
        raise ValueError("n, t, r, and num_prompts must be positive")
    if not 0.0 <= eta <= 0.5:
        raise ValueError("eta must lie in [0, 0.5]")
    if not 0.0 < fpr < 1.0:
        raise ValueError("fpr must lie in (0, 1)")
    quantiles = tuple(float(value) for value in erasure_quantiles)
    if not quantiles or any(not 0.0 <= value <= 1.0 for value in quantiles):
        raise ValueError("erasure quantiles must be nonempty and in [0, 1]")

    data_vol.reload()
    generation_model_size = _normalized_model_size(generation_model_size)
    tag = _tag(n, t, eta, r, generation_model_size)
    artifact_path = f"/data/{tag}/artifacts.pt"
    wm_directory = f"/data/{tag}/wm"
    if not os.path.isfile(artifact_path):
        raise FileNotFoundError(artifact_path)
    artifact = torch.load(
        artifact_path, weights_only=False, map_location="cpu"
    )
    actual_fingerprint = str(artifact.get("artifact_fingerprint", ""))
    if not actual_fingerprint:
        raise ValueError("artifact is missing its fingerprint")
    known = _known_run_or_none(n, t, eta, r, generation_model_size)
    if not expected_artifact_fingerprint and known is not None:
        expected_artifact_fingerprint = known["artifact_fingerprint"]
    if (
        expected_artifact_fingerprint
        and actual_fingerprint != expected_artifact_fingerprint
    ):
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
    decoding_key = artifact.get("decoding_key")
    if not isinstance(decoding_key, (tuple, list)) or len(decoding_key) != 9:
        raise ValueError("artifact is missing the fixed PRC decoding key")
    parity = decoding_key[1]
    if parity.shape != (r, n):
        raise ValueError(
            f"artifact parity matrix has shape {parity.shape}, expected {(r, n)}"
        )
    if int(decoding_key[8]) != t or not math.isclose(
        float(decoding_key[4]), eta, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("artifact t or eta does not match the requested run")
    rank_info = artifact.get("parity_check_rank_info")
    if rank_info is None:
        rank_info = parity_check_rank_info(parity)
    if not rank_info.get("full_rank") or int(rank_info.get("rank", -1)) != r:
        raise ValueError("Phase 2 requires a full-row-rank parity matrix")
    watermarked_present = sum(
        os.path.exists(os.path.join(wm_directory, f"wm_{idx:04d}.pt"))
        for idx in range(num_prompts)
    )
    if watermarked_present != num_prompts:
        raise FileNotFoundError(
            f"watermarked cache has {watermarked_present}/{num_prompts} records"
        )
    null_candidates = [
        candidate
        for candidate in _null_candidates(
            n, num_prompts, generation_model_size
        )
        if candidate["present"] == num_prompts
    ]
    if not null_candidates:
        raise FileNotFoundError(
            f"no complete null cache with T >= {n} for {num_prompts} prompts"
        )
    null_cache = null_candidates[0]
    if known is not None and num_prompts == 500:
        if expected_hoeffding_tp < 0:
            expected_hoeffding_tp = int(known["hoeffding_tp"])
        if expected_hoeffding_fp < 0:
            expected_hoeffding_fp = int(known["hoeffding_fp"])

    code_sha256 = {
        "modal_low_entropy_phase2_parallel.py": _source_sha256(__file__),
        "modal_low_entropy_phase2.py": _source_sha256(
            serial_modal_module.__file__
        ),
        "phase2_parallel.py": _source_sha256(parallel_module.__file__),
        "low_entropy_replay.py": _source_sha256(replay_module.__file__),
        "adaptive_parity_basis.py": _source_sha256(basis_module.__file__),
        "weighted_rademacher.py": _source_sha256(rademacher_module.__file__),
    }
    config = {
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
    }
    run_identity = {
        "parallel_result_schema_version": (
            PHASE2_PARALLEL_RESULT_SCHEMA_VERSION
        ),
        "shard_schema_version": PHASE2_SHARD_SCHEMA_VERSION,
        "config": config,
        "artifact_fingerprint": actual_fingerprint,
        "null_cache_T": int(null_cache["T"]),
        "detector_source_sha256": code_sha256,
    }
    config_fingerprint = phase2_config_fingerprint(run_identity)
    return {
        "config": config,
        "run_identity": run_identity,
        "config_fingerprint": config_fingerprint,
        "tag": tag,
        "artifact_path": artifact_path,
        "artifact_fingerprint": actual_fingerprint,
        "wm_directory": wm_directory,
        "null_directory": null_cache["directory"],
        "null_cache_T": int(null_cache["T"]),
        "detector_source_sha256": code_sha256,
        "expected_hoeffding_tp": int(expected_hoeffding_tp),
        "expected_hoeffding_fp": int(expected_hoeffding_fp),
        "authoritative_baseline_registered": bool(
            known is not None and num_prompts == 500
        ),
        "remote_result_path": _parallel_result_cache_path(
            tag,
            fpr,
            generation_model_size,
            num_prompts,
            quantiles,
        ),
    }


@app.function(volumes={"/data": data_vol}, cpu=1.0, timeout=600)
def resolve_parallel_online_phase2_run(
    source_tag: str,
    n: int,
    t: int,
    eta: float,
    fpr: float = 1e-3,
    num_prompts: int = 500,
    generation_model_size: str = MODEL_SIZE,
    expected_hoeffding_tp: int = -1,
    expected_hoeffding_fp: int = -1,
    erasure_quantiles: tuple[float, ...] = DEFAULT_ERASURE_QUANTILES,
) -> dict:
    """Resolve one saved online ceiling and a shorter detection prefix."""
    import numpy as np
    import torch
    import adaptive_parity_basis as basis_module
    import detectors as detectors_module
    import low_entropy_replay as replay_module
    import online_prc as online_module
    import phase2_parallel as parallel_module
    import weighted_rademacher as rademacher_module
    from online_prc import OnlinePRCKey, materialize_supports, target_row_count

    source_tag = str(source_tag).strip().strip("/")
    if not source_tag or ".." in source_tag.split("/"):
        raise ValueError("source_tag must be a safe relative Modal volume path")
    n = int(n)
    t = int(t)
    eta = float(eta)
    fpr = float(fpr)
    num_prompts = int(num_prompts)
    generation_model_size = _normalized_model_size(generation_model_size)
    quantiles = tuple(float(value) for value in erasure_quantiles)
    if n <= 0 or t <= 0 or num_prompts <= 0:
        raise ValueError("n, t, and num_prompts must be positive")
    if not 0.0 <= eta <= 0.5 or not 0.0 < fpr < 1.0:
        raise ValueError("eta or fpr is outside its valid range")
    if not quantiles or any(not 0.0 <= value <= 1.0 for value in quantiles):
        raise ValueError("erasure quantiles must be nonempty and in [0, 1]")

    data_vol.reload()
    artifact_path = f"/data/{source_tag}/artifacts.pt"
    wm_directory = f"/data/{source_tag}/wm"
    if not os.path.isfile(artifact_path):
        raise FileNotFoundError(artifact_path)
    artifact = torch.load(artifact_path, weights_only=False, map_location="cpu")
    actual_fingerprint = str(artifact.get("artifact_fingerprint", ""))
    if not actual_fingerprint:
        raise ValueError("online artifact is missing its fingerprint")
    source_T = int(artifact.get("T", artifact.get("n", -1)))
    if source_T < n:
        raise ValueError(
            f"online source T={source_T} is shorter than requested n={n}"
        )
    model_size = _normalized_model_size(
        artifact.get(
            "generation_model_size",
            artifact.get("config_sig", {}).get("generation_model_size", MODEL_SIZE),
        )
    )
    if model_size != generation_model_size:
        raise ValueError(
            f"online artifact model is {model_size}, expected "
            f"{generation_model_size}"
        )
    if "online_key" not in artifact or "partition" not in artifact:
        raise ValueError("online artifact is missing its key or partition")
    key = OnlinePRCKey.from_dict(artifact["online_key"])
    if key.check_weight != t or not np.isclose(
        key.noise_rate, eta, rtol=0.0, atol=1e-12
    ):
        raise ValueError("online artifact t or eta does not match the request")
    supports = materialize_supports(n, key)
    r = int(target_row_count(n, key))
    if supports.shape != (r, t):
        raise ValueError("online prefix support table has the wrong shape")
    pivots = supports[:, -1] if r else np.empty(0, dtype=np.int64)
    if r and (
        np.unique(pivots).size != r
        or np.any(supports[:, :-1] >= pivots[:, None])
    ):
        raise ValueError("online causal supports do not form a full-rank basis")

    watermarked_present = sum(
        os.path.exists(os.path.join(wm_directory, f"wm_{idx:04d}.pt"))
        for idx in range(num_prompts)
    )
    if watermarked_present != num_prompts:
        raise FileNotFoundError(
            f"online watermarked cache has {watermarked_present}/"
            f"{num_prompts} records"
        )
    null_candidates = [
        candidate
        for candidate in _null_candidates(n, num_prompts, generation_model_size)
        if candidate["present"] == num_prompts
    ]
    if not null_candidates:
        raise FileNotFoundError(
            f"no complete {generation_model_size} null cache with T >= {n}"
        )
    null_cache = null_candidates[0]

    code_sha256 = {
        "modal_low_entropy_phase2_parallel.py": _source_sha256(__file__),
        "phase2_parallel.py": _source_sha256(parallel_module.__file__),
        "low_entropy_replay.py": _source_sha256(replay_module.__file__),
        "adaptive_parity_basis.py": _source_sha256(basis_module.__file__),
        "weighted_rademacher.py": _source_sha256(rademacher_module.__file__),
        "detectors.py": _source_sha256(detectors_module.__file__),
        "online_prc.py": _source_sha256(online_module.__file__),
    }
    config = {
        "construction": "online_causal_prc_v1",
        "detector": "reliability_adaptive_basis_v1",
        "source_tag": source_tag,
        "source_T": source_T,
        "generation_model": _model_display(generation_model_size),
        "generation_model_size": generation_model_size,
        "n": n,
        "T": n,
        "t": t,
        "eta": eta,
        "r": r,
        "target_fpr": fpr,
        "fpr_policy": "one_shot",
        "num_prompts": num_prompts,
        "erasure_quantiles": list(quantiles),
        "online_key_sha256": key.fingerprint,
    }
    run_identity = {
        "parallel_result_schema_version": PHASE2_PARALLEL_RESULT_SCHEMA_VERSION,
        "shard_schema_version": PHASE2_SHARD_SCHEMA_VERSION,
        "config": config,
        "artifact_fingerprint": actual_fingerprint,
        "null_cache_T": int(null_cache["T"]),
        "detector_source_sha256": code_sha256,
    }
    config_fingerprint = phase2_config_fingerprint(run_identity)
    return {
        "config": config,
        "run_identity": run_identity,
        "config_fingerprint": config_fingerprint,
        "tag": source_tag,
        "artifact_path": artifact_path,
        "artifact_fingerprint": actual_fingerprint,
        "wm_directory": wm_directory,
        "null_directory": null_cache["directory"],
        "null_cache_T": int(null_cache["T"]),
        "detector_source_sha256": code_sha256,
        "expected_hoeffding_tp": int(expected_hoeffding_tp),
        "expected_hoeffding_fp": int(expected_hoeffding_fp),
        "authoritative_baseline_registered": bool(
            expected_hoeffding_tp >= 0 and expected_hoeffding_fp >= 0
        ),
        "remote_result_path": _online_parallel_result_cache_path(
            source_tag,
            n,
            fpr,
            generation_model_size,
            num_prompts,
            quantiles,
        ),
    }


@app.function(volumes={"/data": data_vol}, cpu=1.0, timeout=3600)
def replay_fixed_phase2_prompt_shard(request: dict) -> dict:
    """Score one resumable CPU prompt shard for both source classes."""
    import numpy as np
    import torch
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

    def _finite_or_none(value):
        number = float(value)
        return number if np.isfinite(number) else None

    started = time.perf_counter()
    setup = request["setup"]
    prompt_indices = [int(index) for index in request["prompt_indices"]]
    if (
        not prompt_indices
        or prompt_indices != sorted(prompt_indices)
        or len(set(prompt_indices)) != len(prompt_indices)
    ):
        raise ValueError("prompt shard indices must be nonempty and unique")
    expected_config = phase2_config_fingerprint(setup["run_identity"])
    if expected_config != setup["config_fingerprint"]:
        raise ValueError("parallel Phase 2 setup fingerprint is inconsistent")
    shard_path = _parallel_shard_cache_path(
        setup["tag"], expected_config, prompt_indices
    )
    data_vol.reload()
    if bool(request.get("reuse_shards", True)) and os.path.isfile(shard_path):
        with open(shard_path, "r", encoding="utf-8") as handle:
            cached = json.load(handle)
        cached_indices = validate_phase2_shard_payload(
            cached,
            expected_config_fingerprint=expected_config,
            expected_basis_rank=int(setup["config"]["r"]),
        )
        if cached_indices != prompt_indices:
            raise ValueError("cached Phase 2 shard prompt order changed")
        return {
            "remote_shard_path": shard_path,
            "prompt_indices": prompt_indices,
            "record_count": len(cached["records"]),
            "cached": True,
            "seconds": time.perf_counter() - started,
            "records_sha256": cached["records_sha256"],
            "config_fingerprint": expected_config,
        }

    artifact = torch.load(
        setup["artifact_path"], weights_only=False, map_location="cpu"
    )
    if artifact.get("artifact_fingerprint") != setup["artifact_fingerprint"]:
        raise ValueError("artifact changed after parallel run resolution")
    config = setup["config"]
    n = int(config["n"])
    r = int(config["r"])
    fpr = float(config["target_fpr"])
    quantiles = tuple(float(value) for value in config["erasure_quantiles"])
    generation_model_size = str(config["generation_model_size"])
    records = []
    for watermark, directory, prefix in (
        (True, setup["wm_directory"], "wm"),
        (False, setup["null_directory"], "null"),
    ):
        for prompt_idx in prompt_indices:
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
            phase0_hoeffding = phase1["calibrations"]["hoeffding"]
            phase1_rademacher = phase1["calibrations"][
                "weighted_rademacher_chernoff"
            ]
            phase2_hoeffding = phase2["calibrations"]["hoeffding"]
            phase2_rademacher = phase2["calibrations"][
                "weighted_rademacher_chernoff"
            ]
            selection = phase2_block["basis_selection"]
            if int(selection["basis_rank"]) != r:
                raise AssertionError(f"{path} selected a rank-deficient basis")
            records.append(
                {
                    "prompt_idx": prompt_idx,
                    "watermark": watermark,
                    "phase0_hoeffding_decision": phase0_hoeffding["decision"],
                    "phase0_hoeffding_threshold": _finite_or_none(
                        phase0_hoeffding["threshold"]
                    ),
                    "phase1_rademacher_decision": phase1_rademacher[
                        "decision"
                    ],
                    "phase1_statistic": _finite_or_none(
                        phase1_block["statistic"]
                    ),
                    "phase1_V": _finite_or_none(phase1_block["V"]),
                    "phase1_threshold": _finite_or_none(
                        phase1_rademacher["threshold"]
                    ),
                    "phase1_log_pvalue_upper": _finite_or_none(
                        phase1_rademacher["log_pvalue_upper"]
                    ),
                    "phase2_hoeffding_decision": phase2_hoeffding["decision"],
                    "phase2_hoeffding_threshold": _finite_or_none(
                        phase2_hoeffding["threshold"]
                    ),
                    "phase2_adaptive_decision": phase2_rademacher["decision"],
                    "phase2_statistic": _finite_or_none(
                        phase2_block["statistic"]
                    ),
                    "phase2_V": _finite_or_none(phase2_block["V"]),
                    "phase2_threshold": _finite_or_none(
                        phase2_rademacher["threshold"]
                    ),
                    "phase2_log_pvalue_upper": _finite_or_none(
                        phase2_rademacher["log_pvalue_upper"]
                    ),
                    "basis_rank": int(selection["basis_rank"]),
                    "selected_erasure_quantile": selection[
                        "erasure_quantile"
                    ],
                    "erased_columns": selection["erased_columns"],
                    "erased_column_rank": selection["erased_column_rank"],
                    "erasure_free_rows": selection["erasure_free_rows"],
                    "log_predicted_J": _finite_or_none(
                        selection["log_predicted_J"]
                    ),
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

    payload = {
        "shard_schema_version": PHASE2_SHARD_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_fingerprint": expected_config,
        "artifact_fingerprint": setup["artifact_fingerprint"],
        "detector_source_sha256": setup["detector_source_sha256"],
        "prompt_indices": prompt_indices,
        "record_count": len(records),
        "records_sha256": stable_json_sha256(records),
        "cpu_seconds": time.perf_counter() - started,
        "records": records,
    }
    validate_phase2_shard_payload(
        payload,
        expected_config_fingerprint=expected_config,
        expected_basis_rank=r,
    )
    _atomic_write_json(shard_path, payload)
    data_vol.commit()
    return {
        "remote_shard_path": shard_path,
        "prompt_indices": prompt_indices,
        "record_count": len(records),
        "cached": False,
        "seconds": payload["cpu_seconds"],
        "records_sha256": payload["records_sha256"],
        "config_fingerprint": expected_config,
    }


@app.function(volumes={"/data": data_vol}, cpu=1.0, timeout=3600)
def replay_online_phase2_prompt_shard(request: dict) -> dict:
    """Score one resumable online-prefix prompt shard for all four controls."""
    import numpy as np
    import torch
    from detectors import prepare_online_map_prefix_context, tensor_sha256
    from low_entropy_replay import (
        prepare_online_adaptive_basis_context,
        replay_cached_online_map_record,
        replay_cached_online_map_record_phase2,
    )
    from online_prc import OnlinePRCKey

    def _semantic_array_sha256(values, dtype) -> str:
        if torch.is_tensor(values):
            array = values.detach().cpu().numpy()
        else:
            array = np.asarray(values)
        array = np.ascontiguousarray(array, dtype=dtype)
        header = f"{array.dtype}:{array.shape}:".encode()
        return hashlib.sha256(header + array.tobytes()).hexdigest()

    def _finite_or_none(value):
        number = float(value)
        return number if np.isfinite(number) else None

    started = time.perf_counter()
    setup = request["setup"]
    prompt_indices = [int(index) for index in request["prompt_indices"]]
    if (
        not prompt_indices
        or prompt_indices != sorted(prompt_indices)
        or len(set(prompt_indices)) != len(prompt_indices)
    ):
        raise ValueError("prompt shard indices must be nonempty and unique")
    expected_config = phase2_config_fingerprint(setup["run_identity"])
    if expected_config != setup["config_fingerprint"]:
        raise ValueError("parallel online Phase 2 setup fingerprint is inconsistent")
    shard_path = _parallel_shard_cache_path(
        setup["tag"], expected_config, prompt_indices
    )
    data_vol.reload()
    if bool(request.get("reuse_shards", True)) and os.path.isfile(shard_path):
        with open(shard_path, "r", encoding="utf-8") as handle:
            cached = json.load(handle)
        cached_indices = validate_phase2_shard_payload(
            cached,
            expected_config_fingerprint=expected_config,
            expected_basis_rank=int(setup["config"]["r"]),
        )
        if cached_indices != prompt_indices:
            raise ValueError("cached online Phase 2 shard prompt order changed")
        return {
            "remote_shard_path": shard_path,
            "prompt_indices": prompt_indices,
            "record_count": len(cached["records"]),
            "cached": True,
            "seconds": time.perf_counter() - started,
            "records_sha256": cached["records_sha256"],
            "config_fingerprint": expected_config,
        }

    artifact = torch.load(
        setup["artifact_path"], weights_only=False, map_location="cpu"
    )
    if artifact.get("artifact_fingerprint") != setup["artifact_fingerprint"]:
        raise ValueError("online artifact changed after run resolution")
    config = setup["config"]
    n = int(config["n"])
    r = int(config["r"])
    fpr = float(config["target_fpr"])
    quantiles = tuple(float(value) for value in config["erasure_quantiles"])
    generation_model_size = str(config["generation_model_size"])
    online_key = OnlinePRCKey.from_dict(artifact["online_key"])
    phase1_context = prepare_online_map_prefix_context(online_key, n)
    phase2_context = prepare_online_adaptive_basis_context(online_key, n)
    expected_partition_sha256 = tensor_sha256(artifact["partition"])

    records = []
    for watermark, directory, prefix in (
        (True, setup["wm_directory"], "wm"),
        (False, setup["null_directory"], "null"),
    ):
        for prompt_idx in prompt_indices:
            path = os.path.join(directory, f"{prefix}_{prompt_idx:04d}.pt")
            record = torch.load(path, weights_only=False, map_location="cpu")
            record_model = record.get("generation_model_size")
            if record_model is None and generation_model_size != MODEL_SIZE:
                raise ValueError(f"{path} lacks required 8B model metadata")
            if record_model is not None and _normalized_model_size(
                record_model
            ) != generation_model_size:
                raise ValueError(
                    f"{path} belongs to model {record_model}, expected "
                    f"{generation_model_size}"
                )
            if watermark:
                if record.get("watermark") is not True:
                    raise ValueError(f"{path} is not marked watermarked")
                if int(record.get("prompt_idx", -1)) != prompt_idx:
                    raise ValueError(f"{path} has the wrong prompt index")
                stored_key = record.get("online_key_sha256")
                if stored_key is not None and stored_key != online_key.fingerprint:
                    raise ValueError(f"{path} uses a different online key")
                stored_artifact = record.get("artifact_fingerprint")
                if (
                    stored_artifact is not None
                    and stored_artifact != setup["artifact_fingerprint"]
                ):
                    raise ValueError(f"{path} uses a different online artifact")
            elif record.get("watermark") not in (False, None):
                raise ValueError(f"{path} is not a null record")
            stored_partition = record.get("partition_sha256")
            if (
                stored_partition is not None
                and stored_partition != expected_partition_sha256
            ):
                raise ValueError(f"{path} uses a different token partition")
            stored_prompt = record.get("prompt_token_ids")
            if stored_prompt is not None:
                expected_prompt = torch.as_tensor(
                    artifact["prompt_ids_list"][prompt_idx], dtype=torch.long
                ).reshape(-1)
                observed_prompt = torch.as_tensor(
                    stored_prompt, dtype=torch.long
                ).reshape(-1)
                if not torch.equal(observed_prompt, expected_prompt):
                    raise ValueError(f"{path} has the wrong prompt token IDs")
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
            phase1 = replay_cached_online_map_record(
                artifact,
                record,
                length=n,
                false_positive_rate=fpr,
                fpr_policy="one_shot",
                prepared_context=phase1_context,
            )
            phase2 = replay_cached_online_map_record_phase2(
                artifact,
                record,
                length=n,
                false_positive_rate=fpr,
                fpr_policy="one_shot",
                erasure_quantiles=quantiles,
                prepared_context=phase2_context,
            )
            phase0_hoeffding = phase1["calibrations"]["hoeffding"]
            phase1_rademacher = phase1["calibrations"][
                "weighted_rademacher_chernoff"
            ]
            phase2_hoeffding = phase2["calibrations"]["hoeffding"]
            phase2_rademacher = phase2["calibrations"][
                "weighted_rademacher_chernoff"
            ]
            selection = phase2["basis_selection"]
            if int(selection["basis_rank"]) != r:
                raise AssertionError(f"{path} selected a rank-deficient basis")
            records.append(
                {
                    "prompt_idx": prompt_idx,
                    "watermark": watermark,
                    "phase0_hoeffding_decision": phase0_hoeffding["decision"],
                    "phase0_hoeffding_threshold": _finite_or_none(
                        phase0_hoeffding["threshold"]
                    ),
                    "phase1_rademacher_decision": phase1_rademacher["decision"],
                    "phase1_statistic": _finite_or_none(phase1["statistic"]),
                    "phase1_V": _finite_or_none(phase1["V"]),
                    "phase1_threshold": _finite_or_none(
                        phase1_rademacher["threshold"]
                    ),
                    "phase1_log_pvalue_upper": _finite_or_none(
                        phase1_rademacher["log_pvalue_upper"]
                    ),
                    "phase2_hoeffding_decision": phase2_hoeffding["decision"],
                    "phase2_hoeffding_threshold": _finite_or_none(
                        phase2_hoeffding["threshold"]
                    ),
                    "phase2_adaptive_decision": phase2_rademacher["decision"],
                    "phase2_statistic": _finite_or_none(phase2["statistic"]),
                    "phase2_V": _finite_or_none(phase2["V"]),
                    "phase2_threshold": _finite_or_none(
                        phase2_rademacher["threshold"]
                    ),
                    "phase2_log_pvalue_upper": _finite_or_none(
                        phase2_rademacher["log_pvalue_upper"]
                    ),
                    "basis_rank": int(selection["basis_rank"]),
                    "selected_erasure_quantile": selection["erasure_quantile"],
                    "erased_columns": selection["erased_columns"],
                    "erased_column_rank": selection["erased_column_rank"],
                    "erasure_free_rows": selection["erasure_free_rows"],
                    "log_predicted_J": _finite_or_none(
                        selection["log_predicted_J"]
                    ),
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

    payload = {
        "shard_schema_version": PHASE2_SHARD_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_fingerprint": expected_config,
        "artifact_fingerprint": setup["artifact_fingerprint"],
        "detector_source_sha256": setup["detector_source_sha256"],
        "prompt_indices": prompt_indices,
        "record_count": len(records),
        "records_sha256": stable_json_sha256(records),
        "cpu_seconds": time.perf_counter() - started,
        "records": records,
    }
    validate_phase2_shard_payload(
        payload,
        expected_config_fingerprint=expected_config,
        expected_basis_rank=r,
    )
    _atomic_write_json(shard_path, payload)
    data_vol.commit()
    return {
        "remote_shard_path": shard_path,
        "prompt_indices": prompt_indices,
        "record_count": len(records),
        "cached": False,
        "seconds": payload["cpu_seconds"],
        "records_sha256": payload["records_sha256"],
        "config_fingerprint": expected_config,
    }


@app.function(volumes={"/data": data_vol}, cpu=1.0, timeout=1800)
def aggregate_parallel_phase2_run(
    setup: dict,
    shard_summaries: list[dict],
    preparation_wall_seconds: float,
    shard_size: int,
    max_containers: int,
) -> dict:
    """Validate, merge, summarize, and persist one parallel replay."""
    started = time.perf_counter()
    expected_config = phase2_config_fingerprint(setup["run_identity"])
    if expected_config != setup["config_fingerprint"]:
        raise ValueError("parallel Phase 2 setup fingerprint is inconsistent")
    config = setup["config"]
    expected_prompt_indices = list(range(int(config["num_prompts"])))
    data_vol.reload()
    shard_payloads = []
    inventory = []
    for summary in shard_summaries:
        path = str(summary["remote_shard_path"])
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        prompt_indices = validate_phase2_shard_payload(
            payload,
            expected_config_fingerprint=expected_config,
            expected_basis_rank=int(config["r"]),
        )
        if prompt_indices != [int(index) for index in summary["prompt_indices"]]:
            raise ValueError(f"Phase 2 shard summary disagrees with {path}")
        if payload["records_sha256"] != summary["records_sha256"]:
            raise ValueError(f"Phase 2 shard checksum summary disagrees at {path}")
        shard_payloads.append(payload)
        inventory.append(
            {
                "remote_shard_path": path,
                "prompt_indices": prompt_indices,
                "record_count": len(payload["records"]),
                "cached_this_invocation": bool(summary.get("cached", False)),
                "invocation_seconds": float(summary.get("seconds", 0.0)),
                "original_cpu_seconds": float(payload.get("cpu_seconds", 0.0)),
                "records_sha256": payload["records_sha256"],
            }
        )
    records = merge_phase2_shard_payloads(
        shard_payloads,
        expected_prompt_indices,
        config_fingerprint=expected_config,
        basis_rank=int(config["r"]),
    )
    result_summary = summarize_phase2_records(
        records,
        int(config["num_prompts"]),
        expected_hoeffding_tp=int(setup["expected_hoeffding_tp"]),
        expected_hoeffding_fp=int(setup["expected_hoeffding_fp"]),
    )
    record_manifest = phase2_record_manifest(records)
    payload = {
        "schema_version": PHASE2_PARALLEL_RESULT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            **config,
            "execution_strategy": "cached_prompt_shards_v1",
        },
        "config_fingerprint": expected_config,
        "artifact_path": setup["artifact_path"],
        "artifact_fingerprint": setup["artifact_fingerprint"],
        "null_cache_T": setup["null_cache_T"],
        "source_cache_validation": {
            "required_fields": ["tokens", "p_trace"],
            "minimum_length": int(config["n"]),
            "watermarked_records_validated": int(config["num_prompts"]),
            "null_records_validated": int(config["num_prompts"]),
            "record_manifest_sha256": stable_json_sha256(record_manifest),
        },
        "soundness_validation": {
            "artifact_basis_full_rank": True,
            "selected_basis_rank": int(config["r"]),
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
            "calibrations": [
                "hoeffding",
                "weighted_rademacher_chernoff",
            ],
        },
        "detector_source_sha256": setup["detector_source_sha256"],
        "parallel_execution": {
            "shard_schema_version": PHASE2_SHARD_SCHEMA_VERSION,
            "shard_size": int(shard_size),
            "shard_count": len(inventory),
            "max_containers": int(max_containers),
            "cache_hits_this_invocation": int(
                sum(item["cached_this_invocation"] for item in inventory)
            ),
            "preparation_wall_seconds": float(preparation_wall_seconds),
            "aggregate_cpu_seconds": time.perf_counter() - started,
            "shards": inventory,
        },
        "summary": result_summary,
        "records": records,
        "remote_result_path": setup["remote_result_path"],
    }
    _atomic_write_json(setup["remote_result_path"], payload)
    data_vol.commit()
    payload["remote_result_bytes"] = os.path.getsize(
        setup["remote_result_path"]
    )
    return payload


@app.function(volumes={"/data": data_vol}, cpu=1.0, timeout=300)
def verify_parallel_phase2_result(
    n: int,
    t: int,
    eta: float,
    r: int,
    fpr: float = 1e-3,
    num_prompts: int = 500,
    generation_model_size: str = MODEL_SIZE,
    erasure_quantiles: tuple[float, ...] = DEFAULT_ERASURE_QUANTILES,
) -> dict:
    data_vol.reload()
    generation_model_size = _normalized_model_size(generation_model_size)
    tag = _tag(n, t, eta, r, generation_model_size)
    path = _parallel_result_cache_path(
        tag,
        fpr,
        generation_model_size,
        num_prompts,
        erasure_quantiles,
    )
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, "rb") as handle:
        raw = handle.read()
    payload = json.loads(raw)
    records = payload.get("records", [])
    manifest = phase2_record_manifest(records)
    expected_manifest = payload["source_cache_validation"][
        "record_manifest_sha256"
    ]
    identities = {
        (bool(record["watermark"]), int(record["prompt_idx"]))
        for record in records
    }
    return {
        "remote_result_path": path,
        "remote_result_bytes": len(raw),
        "remote_result_sha256": hashlib.sha256(raw).hexdigest(),
        "schema_version": payload.get("schema_version"),
        "config_fingerprint": payload.get("config_fingerprint"),
        "artifact_fingerprint": payload.get("artifact_fingerprint"),
        "record_count": len(records),
        "unique_record_identities": len(identities),
        "manifest_matches": stable_json_sha256(manifest) == expected_manifest,
        "summary": payload.get("summary"),
        "parallel_execution": payload.get("parallel_execution"),
        "soundness_validation": payload.get("soundness_validation"),
        "detector_source_sha256": payload.get("detector_source_sha256"),
    }


def _parse_quantiles(value: str) -> tuple[float, ...]:
    quantiles = tuple(
        float(item.strip()) for item in str(value).split(",") if item.strip()
    )
    if not quantiles:
        raise ValueError("erasure_quantiles must contain at least one value")
    return quantiles


def _save_local_result(result: dict, output: str) -> None:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    parallel = result["parallel_execution"]
    print(
        "parallel replay: "
        f"shards={parallel['shard_count']}, "
        f"cache_hits={parallel['cache_hits_this_invocation']}, "
        f"wall={parallel['preparation_wall_seconds']:.3f}s"
    )
    print(f"cached remote replay -> {result['remote_result_path']}")
    print(f"saved compact replay -> {destination}")


def _run_parallel(
    *,
    n: int,
    t: int,
    eta: float,
    r: int,
    fpr: float,
    num_prompts: int,
    generation_model_size: str,
    expected_artifact_fingerprint: str,
    expected_hoeffding_tp: int,
    expected_hoeffding_fp: int,
    erasure_quantiles: tuple[float, ...],
    shard_size: int,
    max_containers: int,
    reuse_shards: bool,
    output: str,
) -> None:
    shard_size = int(shard_size)
    max_containers = int(max_containers)
    if max_containers <= 0:
        raise ValueError("max_containers must be positive")
    setup = resolve_parallel_phase2_run.remote(
        n=n,
        t=t,
        eta=eta,
        r=r,
        fpr=fpr,
        num_prompts=num_prompts,
        generation_model_size=generation_model_size,
        expected_artifact_fingerprint=expected_artifact_fingerprint,
        expected_hoeffding_tp=expected_hoeffding_tp,
        expected_hoeffding_fp=expected_hoeffding_fp,
        erasure_quantiles=erasure_quantiles,
    )
    prompt_shards = phase2_prompt_shards(num_prompts, shard_size)
    requests = [
        {
            "setup": setup,
            "prompt_indices": prompt_indices,
            "reuse_shards": bool(reuse_shards),
        }
        for prompt_indices in prompt_shards
    ]
    worker = replay_fixed_phase2_prompt_shard.with_options(
        max_containers=min(max_containers, len(requests))
    )
    started = time.perf_counter()
    shard_summaries = list(worker.map(requests))
    wall_seconds = time.perf_counter() - started
    if len(shard_summaries) != len(prompt_shards):
        raise AssertionError("parallel Phase 2 shard count changed")
    result = aggregate_parallel_phase2_run.remote(
        setup,
        shard_summaries,
        wall_seconds,
        shard_size,
        max_containers,
    )
    _save_local_result(result, output)


def _run_parallel_online(
    *,
    source_tag: str,
    n: int,
    t: int,
    eta: float,
    fpr: float,
    num_prompts: int,
    generation_model_size: str,
    expected_hoeffding_tp: int,
    expected_hoeffding_fp: int,
    erasure_quantiles: tuple[float, ...],
    shard_size: int,
    max_containers: int,
    reuse_shards: bool,
    output: str,
) -> None:
    shard_size = int(shard_size)
    max_containers = int(max_containers)
    if max_containers <= 0:
        raise ValueError("max_containers must be positive")
    setup = resolve_parallel_online_phase2_run.remote(
        source_tag=source_tag,
        n=n,
        t=t,
        eta=eta,
        fpr=fpr,
        num_prompts=num_prompts,
        generation_model_size=generation_model_size,
        expected_hoeffding_tp=expected_hoeffding_tp,
        expected_hoeffding_fp=expected_hoeffding_fp,
        erasure_quantiles=erasure_quantiles,
    )
    prompt_shards = phase2_prompt_shards(num_prompts, shard_size)
    requests = [
        {
            "setup": setup,
            "prompt_indices": prompt_indices,
            "reuse_shards": bool(reuse_shards),
        }
        for prompt_indices in prompt_shards
    ]
    worker = replay_online_phase2_prompt_shard.with_options(
        max_containers=min(max_containers, len(requests))
    )
    started = time.perf_counter()
    shard_summaries = list(worker.map(requests))
    wall_seconds = time.perf_counter() - started
    if len(shard_summaries) != len(prompt_shards):
        raise AssertionError("parallel online Phase 2 shard count changed")
    result = aggregate_parallel_phase2_run.remote(
        setup,
        shard_summaries,
        wall_seconds,
        shard_size,
        max_containers,
    )
    _save_local_result(result, output)


@app.local_entrypoint()
def run(
    n: int,
    t: int,
    eta: float,
    r: int,
    generation_model_size: str = MODEL_SIZE,
    fpr: float = 1e-3,
    num_prompts: int = 500,
    erasure_quantiles: str = "0,0.01,0.025,0.05,0.1,0.15,0.2,0.3",
    shard_size: int = DEFAULT_SHARD_SIZE,
    max_containers: int = DEFAULT_MAX_CONTAINERS,
    reuse_shards: bool = True,
    expected_artifact_fingerprint: str = "",
    expected_hoeffding_tp: int = -1,
    expected_hoeffding_fp: int = -1,
    output: str = "",
) -> None:
    quantiles = _parse_quantiles(erasure_quantiles)
    if not output:
        model = _model_tag(generation_model_size).replace("qwen3_", "")
        output = (
            f"outputs/low_entropy_phase2_parallel_n{n}_eta{eta:.2f}_"
            f"{model}.json"
        )
    _run_parallel(
        n=n,
        t=t,
        eta=eta,
        r=r,
        fpr=fpr,
        num_prompts=num_prompts,
        generation_model_size=generation_model_size,
        expected_artifact_fingerprint=expected_artifact_fingerprint,
        expected_hoeffding_tp=expected_hoeffding_tp,
        expected_hoeffding_fp=expected_hoeffding_fp,
        erasure_quantiles=quantiles,
        shard_size=shard_size,
        max_containers=max_containers,
        reuse_shards=reuse_shards,
        output=output,
    )


@app.local_entrypoint()
def run_online(
    source_tag: str,
    n: int,
    t: int,
    eta: float,
    generation_model_size: str = MODEL_SIZE,
    fpr: float = 1e-3,
    num_prompts: int = 500,
    erasure_quantiles: str = "0,0.01,0.025,0.05,0.1,0.15,0.2,0.3",
    shard_size: int = DEFAULT_SHARD_SIZE,
    max_containers: int = DEFAULT_MAX_CONTAINERS,
    reuse_shards: bool = True,
    expected_hoeffding_tp: int = -1,
    expected_hoeffding_fp: int = -1,
    output: str = "",
) -> None:
    """Replay all four detector controls on a saved online prefix."""
    quantiles = _parse_quantiles(erasure_quantiles)
    if not output:
        model = _model_tag(generation_model_size).replace("qwen3_", "")
        output = (
            f"outputs/low_entropy_phase2_online_parallel_n{n}_"
            f"eta{eta:.2f}_{model}.json"
        )
    _run_parallel_online(
        source_tag=source_tag,
        n=n,
        t=t,
        eta=eta,
        fpr=fpr,
        num_prompts=num_prompts,
        generation_model_size=generation_model_size,
        expected_hoeffding_tp=expected_hoeffding_tp,
        expected_hoeffding_fp=expected_hoeffding_fp,
        erasure_quantiles=quantiles,
        shard_size=shard_size,
        max_containers=max_containers,
        reuse_shards=reuse_shards,
        output=output,
    )


@app.local_entrypoint()
def validate_n416(
    num_prompts: int = 20,
    shard_size: int = 5,
    max_containers: int = 4,
    reuse_shards: bool = True,
    output: str = "outputs/phase2_parallel_validation_n416.json",
) -> None:
    _run_parallel(
        n=416,
        t=3,
        eta=0.05,
        r=412,
        fpr=1e-3,
        num_prompts=num_prompts,
        generation_model_size="0.6B",
        expected_artifact_fingerprint="",
        expected_hoeffding_tp=-1,
        expected_hoeffding_fp=-1,
        erasure_quantiles=DEFAULT_ERASURE_QUANTILES,
        shard_size=shard_size,
        max_containers=max_containers,
        reuse_shards=reuse_shards,
        output=output,
    )
