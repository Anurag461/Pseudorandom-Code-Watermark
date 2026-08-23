"""Scoped scientific diagnostics for the controlled baseline smoke.

The cache phase is deliberately model-free and generation-free.  It consumes
the immutable smoke artifact plus validated PRC/null caches and makes the
quality anomaly explicit at each causal prefix.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Sequence

import numpy as np

from .config import (
    CONTEXT_LENGTH,
    GENERATION_SETTINGS,
    MAX_NEW_TOKENS,
    PREFIX_LENGTHS,
    PRIMARY_SEED,
    SECONDARY_SEED,
    SMOKE_PROMPT_INDICES,
)
from .scoring import deduplicated_positions, quality_metrics


DEFAULT_RAW_SMOKE_PATH = (
    "/data/controlled_baseline_smoke/20260823T020321Z/generated_sequences.pt"
)


def _mean(values: Sequence[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _token_sha256(token_ids: Sequence[int]) -> str:
    array = np.asarray(token_ids, dtype=np.int64)
    return hashlib.sha256(
        f"int64:{array.shape}:".encode() + np.ascontiguousarray(array).tobytes()
    ).hexdigest()


def repeated_ngram_events(token_ids: Sequence[int], n: int = 4) -> dict:
    """Return second-occurrence events for exact token n-grams.

    An event is indexed by the final token of an n-gram that has appeared at an
    earlier start position.  ``first_onset_tokens`` is the one-based generated
    length at the first such event.
    """
    tokens = tuple(map(int, token_ids))
    seen: dict[tuple[int, ...], int] = {}
    event_indices: list[int] = []
    first_gap = None
    for start in range(max(0, len(tokens) - n + 1)):
        ngram = tokens[start : start + n]
        previous = seen.get(ngram)
        if previous is not None:
            event_indices.append(start + n - 1)
            if first_gap is None:
                first_gap = start - previous
        else:
            seen[ngram] = start
    total = max(0, len(tokens) - n + 1)
    return {
        "n": int(n),
        "event_indices": event_indices,
        "event_count": len(event_indices),
        "event_rate": len(event_indices) / total if total else 0.0,
        "first_onset_tokens": event_indices[0] + 1 if event_indices else None,
        "first_recurrence_gap": first_gap,
        "unique_count": len(seen),
        "total_count": total,
    }


def longest_periodic_run(
    token_ids: Sequence[int], *, max_period: int = 64, minimum_cycles: int = 3
) -> dict:
    """Find the longest exact short-period run.

    For period ``p``, a periodic span is a maximal region satisfying
    ``x[i] == x[i-p]``.  A reported loop must contain at least three periods
    and at least eight tokens.  The definition is fixed here so loop onset is
    reproducible and not chosen by inspecting decoded text.
    """
    tokens = tuple(map(int, token_ids))
    candidates: list[dict] = []
    for period in range(1, min(max_period, len(tokens) // minimum_cycles) + 1):
        equality_start = None
        for index in range(period, len(tokens) + 1):
            equal = index < len(tokens) and tokens[index] == tokens[index - period]
            if equal and equality_start is None:
                equality_start = index
            if not equal and equality_start is not None:
                matching = index - equality_start
                span = matching + period
                onset = equality_start - period
                if span >= max(8, minimum_cycles * period):
                    candidates.append(
                        {
                            "onset_index": onset,
                            "onset_tokens": onset + 1,
                            "period": period,
                            "span_tokens": span,
                            "cycles": span / period,
                        }
                    )
                equality_start = None
    if not candidates:
        return {
            "found": False,
            "onset_index": None,
            "onset_tokens": None,
            "period": None,
            "span_tokens": 0,
            "cycles": 0.0,
        }
    longest = min(
        candidates,
        key=lambda item: (-item["span_tokens"], item["onset_index"], item["period"]),
    )
    return {"found": True, **longest}


def analyze_sequence(
    token_ids: Sequence[int],
    entropies: Sequence[float],
    logprobs: Sequence[float],
) -> dict:
    tokens = list(map(int, token_ids))
    entropy = list(map(float, entropies))
    token_logprobs = list(map(float, logprobs))
    if not tokens or len(tokens) != len(entropy) or len(tokens) != len(token_logprobs):
        raise ValueError("tokens, entropies, and log-probabilities must have equal nonzero length")
    if not all(math.isfinite(value) for value in entropy + token_logprobs):
        raise ValueError("diagnostic inputs contain non-finite values")

    quality = quality_metrics(tokens, token_logprobs)
    repeats = repeated_ngram_events(tokens, n=4)
    repeat_indices = set(repeats["event_indices"])
    repeat_entropy = [entropy[index] for index in sorted(repeat_indices)]
    novel_entropy = [
        entropy[index]
        for index in range(3, len(tokens))
        if index not in repeat_indices
    ]
    ordered = sorted(entropy[3:])
    if ordered:
        lower = ordered[int(0.25 * (len(ordered) - 1))]
        upper = ordered[int(0.75 * (len(ordered) - 1))]
    else:
        lower = upper = entropy[0]
    low_positions = [index for index in range(3, len(tokens)) if entropy[index] <= lower]
    high_positions = [index for index in range(3, len(tokens)) if entropy[index] >= upper]
    low_repeat_rate = (
        sum(index in repeat_indices for index in low_positions) / len(low_positions)
        if low_positions
        else 0.0
    )
    high_repeat_rate = (
        sum(index in repeat_indices for index in high_positions) / len(high_positions)
        if high_positions
        else 0.0
    )

    loop = longest_periodic_run(tokens)
    if loop["found"]:
        onset = int(loop["onset_index"])
        end = min(len(tokens), onset + int(loop["span_tokens"]))
        pre = entropy[max(0, onset - 32) : onset]
        during = entropy[onset:end]
    else:
        pre = []
        during = []
    loop.update(
        {
            "pre_loop_entropy_mean_32": _mean(pre),
            "during_loop_entropy_mean": _mean(during),
            "during_minus_pre_entropy": (
                _mean(during) - _mean(pre) if pre and during else None
            ),
        }
    )
    return {
        **quality,
        "base_entropy_mean": _mean(entropy),
        "base_entropy_median": float(statistics.median(entropy)),
        "deduplicated_context_token_count": len(
            deduplicated_positions(tokens, CONTEXT_LENGTH)
        ),
        "repeated_4gram": {key: value for key, value in repeats.items() if key != "event_indices"},
        "repeat_event_entropy_mean": _mean(repeat_entropy),
        "novel_event_entropy_mean": _mean(novel_entropy),
        "repeat_minus_novel_entropy": (
            _mean(repeat_entropy) - _mean(novel_entropy)
            if repeat_entropy and novel_entropy
            else None
        ),
        "entropy_lower_quartile": lower,
        "entropy_upper_quartile": upper,
        "low_entropy_repeat_rate": low_repeat_rate,
        "high_entropy_repeat_rate": high_repeat_rate,
        "low_to_high_entropy_repeat_rate_ratio": (
            low_repeat_rate / high_repeat_rate if high_repeat_rate > 0 else None
        ),
        "longest_periodic_run": loop,
    }


def _load_inputs(data_volume, raw_path: str) -> tuple[list[dict], list[dict]]:
    import torch

    from .smoke_runner import _load_cached_sequences, _numpy_pickle_compat

    data_volume.reload()
    path = Path(raw_path)
    if not path.is_file():
        raise FileNotFoundError(f"committed smoke artifact is missing: {raw_path}")
    _numpy_pickle_compat()
    raw = torch.load(path, weights_only=False, map_location="cpu")
    if raw.get("prompt_indices") != list(SMOKE_PROMPT_INDICES):
        raise AssertionError("raw smoke prompt indices changed")
    if raw.get("generation_settings") != GENERATION_SETTINGS:
        raise AssertionError("raw smoke generation settings changed")
    prompt_rows = [
        json.loads(line) for line in Path("/root/prompts.jsonl").read_text().splitlines() if line
    ]
    if len(prompt_rows) != 500:
        raise AssertionError("canonical prompt corpus is incomplete")
    online, nulls, _ = _load_cached_sequences(prompt_rows)

    sequences: list[dict] = []
    for method in ("textseal", "synthid_text", "gumbel_max"):
        outputs = raw["sequences"][f"{method}/seed{PRIMARY_SEED}"]
        if len(outputs) != len(SMOKE_PROMPT_INDICES):
            raise AssertionError(f"{method} primary smoke coverage changed")
        for row, prompt_index in enumerate(SMOKE_PROMPT_INDICES):
            sequences.append(
                {
                    "method": method,
                    "prompt_index": prompt_index,
                    "prompt_id": f"doc-{prompt_rows[prompt_index]['doc_index']}",
                    "seed": PRIMARY_SEED,
                    **outputs[row],
                }
            )
    for row, prompt_index in enumerate(SMOKE_PROMPT_INDICES):
        for method, record in (("online_prc", online[row]), ("null", nulls[row])):
            sequences.append(
                {
                    "method": method,
                    "prompt_index": prompt_index,
                    "prompt_id": f"doc-{prompt_rows[prompt_index]['doc_index']}",
                    "seed": int(record.get("experiment_seed", PRIMARY_SEED)),
                    "token_ids": list(map(int, np.asarray(record["tokens"])[:MAX_NEW_TOKENS])),
                    "base_token_logprobs": list(
                        map(float, np.asarray(record["base_token_logprob"])[:MAX_NEW_TOKENS])
                    ),
                    # Project caches store entropy in bits; generation stores nats.
                    "base_entropies": list(
                        map(
                            float,
                            np.asarray(record["base_lm_entropy"])[:MAX_NEW_TOKENS]
                            * math.log(2.0),
                        )
                    ),
                }
            )
    return sequences, prompt_rows


def run_cache_diagnostic(data_volume, raw_path: str = DEFAULT_RAW_SMOKE_PATH) -> dict:
    """Analyze committed outputs without loading a model or generating tokens."""
    sequences, _ = _load_inputs(data_volume, raw_path)
    rows = []
    for sequence in sequences:
        if len(sequence["token_ids"]) != MAX_NEW_TOKENS:
            raise AssertionError(f"{sequence['method']} has a non-1024 continuation")
        for prefix in PREFIX_LENGTHS:
            metrics = analyze_sequence(
                sequence["token_ids"][:prefix],
                sequence["base_entropies"][:prefix],
                sequence["base_token_logprobs"][:prefix],
            )
            rows.append(
                {
                    "method": sequence["method"],
                    "prompt_index": sequence["prompt_index"],
                    "prompt_id": sequence["prompt_id"],
                    "seed": sequence["seed"],
                    "prefix_length": prefix,
                    "token_sha256": _token_sha256(sequence["token_ids"][:prefix]),
                    **metrics,
                }
            )

    summary = []
    for prefix in PREFIX_LENGTHS:
        null_rows = [
            row for row in rows if row["method"] == "null" and row["prefix_length"] == prefix
        ]
        if len(null_rows) != 5:
            raise AssertionError("shared-null diagnostic coverage changed")
        null_d2 = statistics.median(row["distinct_2"] for row in null_rows)
        null_d3 = statistics.median(row["distinct_3"] for row in null_rows)
        null_rep = statistics.median(row["repetition_rate"] for row in null_rows)
        for method in ("null", "online_prc", "textseal", "synthid_text", "gumbel_max"):
            selected = [
                row for row in rows if row["method"] == method and row["prefix_length"] == prefix
            ]
            if len(selected) != 5:
                raise AssertionError(f"{method} diagnostic coverage changed at T={prefix}")
            d2 = statistics.median(row["distinct_2"] for row in selected)
            d3 = statistics.median(row["distinct_3"] for row in selected)
            repetition = statistics.median(row["repetition_rate"] for row in selected)
            summary.append(
                {
                    "method": method,
                    "prefix_length": prefix,
                    "prompts": len(selected),
                    "median_distinct_2": d2,
                    "median_distinct_3": d3,
                    "median_repetition_rate": repetition,
                    "median_base_nll": statistics.median(row["base_model_nll"] for row in selected),
                    "median_base_entropy": statistics.median(
                        row["base_entropy_mean"] for row in selected
                    ),
                    "median_deduplicated_context_token_count": statistics.median(
                        row["deduplicated_context_token_count"] for row in selected
                    ),
                    "median_longest_loop_span": statistics.median(
                        row["longest_periodic_run"]["span_tokens"] for row in selected
                    ),
                    "median_first_repeat_4gram_onset": statistics.median(
                        row["repeated_4gram"]["first_onset_tokens"]
                        if row["repeated_4gram"]["first_onset_tokens"] is not None
                        else prefix + 1
                        for row in selected
                    ),
                    "distinct_2_minus_null": d2 - null_d2,
                    "distinct_3_minus_null": d3 - null_d3,
                    "repetition_rate_minus_null": repetition - null_rep,
                    "distinct_2_ratio_to_null": d2 / null_d2 if null_d2 else None,
                    "distinct_3_ratio_to_null": d3 / null_d3 if null_d3 else None,
                }
            )
    return {
        "status": "passed",
        "mode": "cache_only_no_model_no_generation",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "raw_smoke_artifact": raw_path,
        "prompt_indices": list(SMOKE_PROMPT_INDICES),
        "prefix_lengths": list(PREFIX_LENGTHS),
        "generation_attempts": 0,
        "definitions": {
            "distinct_n": "unique token n-grams / all token n-gram positions",
            "repetition_rate": "1 - unique token 4-grams / all token 4-gram positions",
            "repeat_event": "final token of an exact 4-gram seen at an earlier start",
            "loop": (
                "longest exact period<=64 run spanning at least three cycles and eight tokens"
            ),
            "entropy_unit": "nats from base model before sampling each generated token",
        },
        "rows": rows,
        "summary": summary,
    }


def write_cache_diagnostic_artifacts(payload: dict, root: Path) -> list[str]:
    outputs = root / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    json_path = outputs / "controlled_baseline_diagnostic_cache_analysis.json"
    rows_path = outputs / "controlled_baseline_diagnostic_prefix_rows.jsonl"
    summary_path = outputs / "controlled_baseline_diagnostic_prefix_summary.csv"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    with rows_path.open("w") as handle:
        for row in payload["rows"]:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(payload["summary"][0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(payload["summary"])
    return [str(path.relative_to(root)) for path in (json_path, rows_path, summary_path)]


def stable_power_argmax(probabilities: Sequence[float], uniforms: Sequence[float]) -> int:
    """Numerically stable argmax equivalent to ``r ** (1 / p)``.

    This helper is diagnostic-only.  The released Gumbel comparison path
    remains the primary implementation and is never silently replaced.
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    random_scores = np.asarray(uniforms, dtype=np.float64)
    if probs.ndim != 1 or probs.shape != random_scores.shape:
        raise ValueError("probabilities and uniforms must be equal one-dimensional arrays")
    if np.any(probs < 0) or not math.isclose(float(probs.sum()), 1.0, abs_tol=1e-12):
        raise ValueError("probabilities must be nonnegative and normalized")
    scores = np.full_like(probs, -np.inf)
    valid = (probs > 0) & (random_scores > 0)
    scores[valid] = np.log(random_scores[valid]) / probs[valid]
    return int(np.argmax(scores))


def _sample_vanilla(logits, *, temperature: float, top_p: float):
    import torch

    probs = torch.softmax(logits / float(temperature), dim=-1)
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > float(top_p)
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    selected = torch.multinomial(probs_sort, num_samples=1)
    return torch.gather(probs_idx, -1, selected).reshape(-1)


def _sample_stable_gumbel(
    logits,
    context,
    generator,
    *,
    temperature: float,
    top_p: float,
):
    """Log-space equivalent of the pinned comparison path's power argmax."""
    import torch
    from textseal.watermarking.generator import score_all_next_tokens

    probs = torch.softmax(logits / float(temperature), dim=-1)
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > float(top_p)
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    uniforms = score_all_next_tokens(context, generator.wm_args, logits.shape[-1])
    rows = torch.arange(logits.shape[0], device=logits.device).unsqueeze(1)
    sorted_uniforms = uniforms[rows, probs_idx]
    probabilities64 = probs_sort.double()
    uniforms64 = sorted_uniforms.double()
    scores = torch.full_like(probabilities64, float("-inf"))
    valid = (probabilities64 > 0) & (uniforms64 > 0)
    scores[valid] = torch.log(uniforms64[valid]) / probabilities64[valid]
    selected = torch.argmax(scores, dim=-1, keepdim=True)
    return torch.gather(probs_idx, -1, selected).reshape(-1)


def _generate_diagnostic_batch(
    model,
    prompts: Sequence[Sequence[int]],
    *,
    method: str,
    seed: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[list[dict], dict]:
    import time
    import torch

    from qwen import StaticKVCache
    from .official import gumbel_generator, textseal_generator

    if method not in {"vanilla", "textseal", "gumbel_log_stable"}:
        raise ValueError(f"unsupported diagnostic method: {method}")
    if not prompts or any(len(prompt) != 50 for prompt in prompts):
        raise ValueError("diagnostic prompts must be nonempty and exactly 50 tokens")
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    batch_size = len(prompts)
    device = torch.device("cuda")
    all_tokens = torch.empty((batch_size, 50 + max_new_tokens), dtype=torch.long, device=device)
    all_tokens[:, :50] = torch.tensor(prompts, dtype=torch.long, device=device)
    generated = torch.empty((batch_size, max_new_tokens), dtype=torch.long)
    logprobs = torch.empty((batch_size, max_new_tokens), dtype=torch.float32)
    entropies = torch.empty((batch_size, max_new_tokens), dtype=torch.float32)
    cache = StaticKVCache(max_length=50 + max_new_tokens)
    sampler = textseal_generator() if method == "textseal" else None
    if method == "gumbel_log_stable":
        sampler = gumbel_generator()

    torch.cuda.synchronize()
    started = time.perf_counter()
    logits = model(all_tokens[:, :50], cache=cache)[:, -1]
    first_step = {
        "prompt0_logit_sum": float(logits[0].float().sum().item()),
        "prompt0_top10": torch.topk(logits[0].float(), 10).indices.detach().cpu().tolist(),
    }
    for position in range(max_new_tokens):
        base_log_probs = torch.log_softmax(logits.float(), dim=-1)
        base_entropy = -(base_log_probs.exp() * base_log_probs).sum(dim=-1)
        context = all_tokens[:, 50 + position - CONTEXT_LENGTH : 50 + position]
        if method == "vanilla":
            next_token = _sample_vanilla(
                logits, temperature=temperature, top_p=top_p
            )
        elif method == "textseal":
            next_token = sampler.sample_next(
                logits, context, temperature=float(temperature), top_p=float(top_p)
            )
        else:
            next_token = _sample_stable_gumbel(
                logits,
                context,
                sampler,
                temperature=temperature,
                top_p=top_p,
            )
        selected_logprob = base_log_probs.gather(1, next_token[:, None]).squeeze(1)
        generated[:, position] = next_token.detach().cpu()
        logprobs[:, position] = selected_logprob.detach().cpu()
        entropies[:, position] = base_entropy.detach().cpu()
        all_tokens[:, 50 + position] = next_token
        if position + 1 < max_new_tokens:
            logits = model(next_token[:, None], cache=cache)[:, -1]
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    outputs = [
        {
            "token_ids": generated[row].tolist(),
            "base_token_logprobs": logprobs[row].double().tolist(),
            "base_entropies": entropies[row].double().tolist(),
        }
        for row in range(batch_size)
    ]
    return outputs, {
        "method": method,
        "seed": int(seed),
        "batch_size": batch_size,
        "max_new_tokens": int(max_new_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "method_seconds": elapsed,
        "tokens_per_second": batch_size * max_new_tokens / elapsed,
        "first_step": first_step,
    }


def _prefix_rows(label: str, outputs: Sequence[dict], prompt_indices: Sequence[int]) -> list[dict]:
    rows = []
    for output, prompt_index in zip(outputs, prompt_indices):
        for prefix in PREFIX_LENGTHS:
            if prefix > len(output["token_ids"]):
                continue
            rows.append(
                {
                    "label": label,
                    "prompt_index": int(prompt_index),
                    "prefix_length": int(prefix),
                    "token_sha256": _token_sha256(output["token_ids"][:prefix]),
                    **analyze_sequence(
                        output["token_ids"][:prefix],
                        output["base_entropies"][:prefix],
                        output["base_token_logprobs"][:prefix],
                    ),
                }
            )
    return rows


def _median_summary(rows: Sequence[dict], label: str, prefix: int) -> dict:
    selected = [
        row for row in rows if row["label"] == label and row["prefix_length"] == prefix
    ]
    if not selected:
        raise AssertionError(f"no diagnostic rows for {label} T={prefix}")
    return {
        "label": label,
        "prefix_length": int(prefix),
        "prompts": len(selected),
        "median_distinct_2": statistics.median(row["distinct_2"] for row in selected),
        "median_distinct_3": statistics.median(row["distinct_3"] for row in selected),
        "median_repetition_rate": statistics.median(
            row["repetition_rate"] for row in selected
        ),
        "median_base_nll": statistics.median(row["base_model_nll"] for row in selected),
        "median_base_entropy": statistics.median(
            row["base_entropy_mean"] for row in selected
        ),
        "median_first_repeat_4gram_onset": statistics.median(
            row["repeated_4gram"]["first_onset_tokens"]
            if row["repeated_4gram"]["first_onset_tokens"] is not None
            else prefix + 1
            for row in selected
        ),
    }


def _agreement(left: Sequence[int], right: Sequence[int]) -> float:
    if len(left) != len(right):
        raise ValueError("agreement sequences differ in length")
    return sum(int(a) == int(b) for a, b in zip(left, right)) / len(left)


def run_generation_diagnostic(data_volume, raw_path: str = DEFAULT_RAW_SMOKE_PATH) -> dict:
    """Run the authorized minimal five-prompt H100 diagnostic matrix."""
    import gc
    import os
    import time
    import torch

    from .smoke_runner import (
        _integration_fingerprint,
        _model_revision,
        _numpy_pickle_compat,
        load_qwen3_8b,
        preload_official_runtimes,
    )

    overall_started = time.perf_counter()
    preload_official_runtimes()
    data_volume.reload()
    source = Path(raw_path)
    if not source.is_file():
        raise FileNotFoundError(f"committed smoke artifact is missing: {raw_path}")
    _numpy_pickle_compat()
    raw = torch.load(source, weights_only=False, map_location="cpu")
    if raw.get("generation_settings") != GENERATION_SETTINGS:
        raise AssertionError("raw smoke generation settings changed")
    prompt_rows = [
        json.loads(line) for line in Path("/root/prompts.jsonl").read_text().splitlines() if line
    ]
    prompts = [prompt_rows[index]["prompt_tokens"] for index in SMOKE_PROMPT_INDICES]
    model_revision = _model_revision()
    if raw.get("model_revision") != model_revision:
        raise AssertionError("raw smoke model revision changed")

    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    model = load_qwen3_8b()
    model_load_seconds = time.perf_counter() - load_started
    generated: dict[str, list[dict]] = {}
    telemetry = []
    matrix = (
        ("textseal_frozen_seed67890_batch5", "textseal", SECONDARY_SEED, prompts, 1024, 1.0, 1.0),
        ("vanilla_paper_decode_seed12345_batch5", "vanilla", PRIMARY_SEED, prompts, 400, 0.8, 0.9),
        ("textseal_paper_decode_seed12345_batch5", "textseal", PRIMARY_SEED, prompts, 400, 0.8, 0.9),
        ("gumbel_log_frozen_batch5", "gumbel_log_stable", PRIMARY_SEED, prompts, 400, 1.0, 1.0),
        ("gumbel_log_frozen_batch1", "gumbel_log_stable", PRIMARY_SEED, prompts[:1], 400, 1.0, 1.0),
    )
    with torch.no_grad():
        for label, method, seed, batch_prompts, length, temperature, top_p in matrix:
            print(f"[diagnostic] generation start {label}", flush=True)
            outputs, item = _generate_diagnostic_batch(
                model,
                batch_prompts,
                method=method,
                seed=seed,
                max_new_tokens=length,
                temperature=temperature,
                top_p=top_p,
            )
            generated[label] = outputs
            telemetry.append({"label": label, **item})
            print(f"[diagnostic] generation complete {label}", flush=True)

    existing = {
        "textseal_frozen_seed12345_batch5": raw["sequences"][f"textseal/seed{PRIMARY_SEED}"],
        "gumbel_power_frozen_batch5": raw["sequences"][f"gumbel_max/seed{PRIMARY_SEED}"],
        "gumbel_power_frozen_batch1": raw["sequences"][f"gumbel_max/seed{SECONDARY_SEED}"],
    }
    all_rows = []
    for label, outputs in {**existing, **generated}.items():
        indices = SMOKE_PROMPT_INDICES if len(outputs) == 5 else SMOKE_PROMPT_INDICES[:1]
        all_rows.extend(_prefix_rows(label, outputs, indices))

    summaries = []
    for label, outputs in {**existing, **generated}.items():
        for prefix in PREFIX_LENGTHS:
            if prefix <= len(outputs[0]["token_ids"]):
                summaries.append(_median_summary(all_rows, label, prefix))

    primary_textseal = existing["textseal_frozen_seed12345_batch5"]
    secondary_textseal = generated["textseal_frozen_seed67890_batch5"]
    textseal_seed = []
    for row, prompt_index in enumerate(SMOKE_PROMPT_INDICES):
        textseal_seed.append(
            {
                "prompt_index": prompt_index,
                "token_agreement": _agreement(
                    primary_textseal[row]["token_ids"], secondary_textseal[row]["token_ids"]
                ),
                "primary_token_sha256": _token_sha256(primary_textseal[row]["token_ids"]),
                "secondary_token_sha256": _token_sha256(secondary_textseal[row]["token_ids"]),
            }
        )

    paper_vanilla = generated["vanilla_paper_decode_seed12345_batch5"]
    paper_textseal = generated["textseal_paper_decode_seed12345_batch5"]
    paper_comparison = []
    for row, prompt_index in enumerate(SMOKE_PROMPT_INDICES):
        vanilla_metrics = analyze_sequence(
            paper_vanilla[row]["token_ids"],
            paper_vanilla[row]["base_entropies"],
            paper_vanilla[row]["base_token_logprobs"],
        )
        textseal_metrics = analyze_sequence(
            paper_textseal[row]["token_ids"],
            paper_textseal[row]["base_entropies"],
            paper_textseal[row]["base_token_logprobs"],
        )
        paper_comparison.append(
            {
                "prompt_index": prompt_index,
                "vanilla_distinct_3": vanilla_metrics["distinct_3"],
                "textseal_distinct_3": textseal_metrics["distinct_3"],
                "distinct_3_delta": textseal_metrics["distinct_3"]
                - vanilla_metrics["distinct_3"],
                "vanilla_repetition_rate": vanilla_metrics["repetition_rate"],
                "textseal_repetition_rate": textseal_metrics["repetition_rate"],
                "repetition_rate_delta": textseal_metrics["repetition_rate"]
                - vanilla_metrics["repetition_rate"],
            }
        )

    power_batch5 = existing["gumbel_power_frozen_batch5"][0]["token_ids"][:400]
    power_batch1 = existing["gumbel_power_frozen_batch1"][0]["token_ids"][:400]
    log_batch5 = generated["gumbel_log_frozen_batch5"][0]["token_ids"]
    log_batch1 = generated["gumbel_log_frozen_batch1"][0]["token_ids"]
    gumbel = {
        "power_batch1_vs_batch5_agreement": _agreement(power_batch1, power_batch5),
        "log_batch1_vs_batch5_agreement": _agreement(log_batch1, log_batch5),
        "power_vs_log_batch1_agreement": _agreement(power_batch1, log_batch1),
        "power_vs_log_batch5_agreement": _agreement(power_batch5, log_batch5),
        "token_hashes": {
            "power_batch1": _token_sha256(power_batch1),
            "power_batch5": _token_sha256(power_batch5),
            "log_batch1": _token_sha256(log_batch1),
            "log_batch5": _token_sha256(log_batch5),
        },
        "interpretation_guard": (
            "ordinary sampling seed is not a Gumbel replicate; power outputs are reused "
            "from the fixed-shape smoke, and the log-space path is diagnostic-only"
        ),
    }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    remote_root = Path("/data/controlled_baseline_diagnostic") / timestamp
    if remote_root.exists():
        raise FileExistsError(f"diagnostic output already exists: {remote_root}")
    remote_root.mkdir(parents=True)
    artifact = remote_root / "generated_diagnostic_sequences.pt"
    torch.save(
        {
            "source_smoke_artifact": raw_path,
            "model_revision": model_revision,
            "integration_code_fingerprint": _integration_fingerprint(),
            "generated": generated,
            "telemetry": telemetry,
        },
        artifact,
    )
    data_volume.commit()
    torch.cuda.synchronize()
    runtime = {
        "model_load_seconds": model_load_seconds,
        "function_seconds": time.perf_counter() - overall_started,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "requested_gpu": "H100",
        "actual_gpu": torch.cuda.get_device_name(0),
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "status": "passed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "matrix": [item[0] for item in matrix],
        "source_smoke_artifact": raw_path,
        "remote_generation_artifact": {
            "path": str(artifact),
            "size_bytes": artifact.stat().st_size,
        },
        "prefix_rows": all_rows,
        "summaries": summaries,
        "textseal_seed_comparison": textseal_seed,
        "paper_decode_comparison": paper_comparison,
        "gumbel_stability": gumbel,
        "telemetry": telemetry,
        "runtime": runtime,
        "provenance": {
            "model_revision": model_revision,
            "modal_image_id": os.environ.get("MODAL_IMAGE_ID"),
            "modal_task_id": os.environ.get("MODAL_TASK_ID"),
            "integration_code_fingerprint": _integration_fingerprint(),
            "ordinary_gumbel_seed_replicates_generated": 0,
        },
    }


def write_generation_diagnostic_artifacts(payload: dict, root: Path) -> list[str]:
    outputs = root / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    result_path = outputs / "controlled_baseline_diagnostic_generation.json"
    rows_path = outputs / "controlled_baseline_diagnostic_generation_prefix_rows.jsonl"
    summary_path = outputs / "controlled_baseline_diagnostic_generation_summary.csv"
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    with rows_path.open("w") as handle:
        for row in payload["prefix_rows"]:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(payload["summaries"][0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(payload["summaries"])
    return [str(path.relative_to(root)) for path in (result_path, rows_path, summary_path)]


def _logit_distribution_comparison(project_logits, reference_logits) -> dict:
    import torch

    left = project_logits.float()
    right = reference_logits.float()
    difference = torch.abs(left - right)
    left_log = torch.log_softmax(left, dim=-1)
    right_log = torch.log_softmax(right, dim=-1)
    left_prob = left_log.exp()
    right_prob = right_log.exp()
    total_variation = 0.5 * torch.abs(left_prob - right_prob).sum(dim=-1)
    mixture = 0.5 * (left_prob + right_prob)
    mixture_log = torch.log(mixture.clamp_min(torch.finfo(mixture.dtype).tiny))
    js = 0.5 * (
        (left_prob * (left_log - mixture_log)).sum(dim=-1)
        + (right_prob * (right_log - mixture_log)).sum(dim=-1)
    )
    left_top = torch.topk(left, 10, dim=-1).indices
    right_top = torch.topk(right, 10, dim=-1).indices
    rows = []
    for row in range(left.shape[0]):
        left_ids = left_top[row].detach().cpu().tolist()
        right_ids = right_top[row].detach().cpu().tolist()
        rows.append(
            {
                "row": row,
                "top1_equal": left_ids[0] == right_ids[0],
                "project_top1": left_ids[0],
                "reference_top1": right_ids[0],
                "top10_overlap": len(set(left_ids) & set(right_ids)),
                "jensen_shannon_divergence": float(js[row].item()),
                "total_variation_distance": float(total_variation[row].item()),
                "max_abs_logit_difference": float(difference[row].max().item()),
                "mean_abs_logit_difference": float(difference[row].mean().item()),
            }
        )
    return {
        "rows": rows,
        "all_top1_equal": all(row["top1_equal"] for row in rows),
        "minimum_top10_overlap": min(row["top10_overlap"] for row in rows),
        "maximum_jensen_shannon_divergence": max(
            row["jensen_shannon_divergence"] for row in rows
        ),
        "maximum_total_variation_distance": max(
            row["total_variation_distance"] for row in rows
        ),
        "maximum_abs_logit_difference": max(
            row["max_abs_logit_difference"] for row in rows
        ),
        "maximum_mean_abs_logit_difference": max(
            row["mean_abs_logit_difference"] for row in rows
        ),
    }


def run_hf_logits_parity() -> dict:
    """Compare the project Qwen loader with Hugging Face on identical prefixes."""
    import gc
    import os
    import time
    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    from qwen import StaticKVCache
    from .smoke_runner import _model_revision, load_qwen3_8b

    model_root = "/cache/models/Qwen3-8B-Base"
    prompt_rows = [
        json.loads(line) for line in Path("/root/prompts.jsonl").read_text().splitlines() if line
    ]
    if len(prompt_rows) != 500:
        raise AssertionError("canonical prompt corpus is incomplete")
    prompts = torch.tensor(
        [prompt_rows[index]["prompt_tokens"] for index in SMOKE_PROMPT_INDICES],
        dtype=torch.long,
        device="cuda",
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    project_started = time.perf_counter()
    project = load_qwen3_8b()
    project_load_seconds = time.perf_counter() - project_started
    reference_started = time.perf_counter()
    reference = AutoModelForCausalLM.from_pretrained(
        model_root,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to("cuda")
    reference.eval()
    reference_load_seconds = time.perf_counter() - reference_started

    with torch.no_grad():
        project_batch1 = project(prompts[:1])[:, -1]
        reference_batch1 = reference(prompts[:1], use_cache=False).logits[:, -1]
        batch1 = _logit_distribution_comparison(project_batch1, reference_batch1)

        project_batch5 = project(prompts)[:, -1]
        reference_batch5 = reference(prompts, use_cache=False).logits[:, -1]
        batch5 = _logit_distribution_comparison(project_batch5, reference_batch5)

        project_individual = []
        reference_individual = []
        for row in range(len(SMOKE_PROMPT_INDICES)):
            project_individual.append(project(prompts[row : row + 1])[:, -1])
            reference_individual.append(
                reference(prompts[row : row + 1], use_cache=False).logits[:, -1]
            )
        project_individual_logits = torch.cat(project_individual, dim=0)
        reference_individual_logits = torch.cat(reference_individual, dim=0)
        individual_reference = _logit_distribution_comparison(
            project_individual_logits, reference_individual_logits
        )
        project_batch_shape = _logit_distribution_comparison(
            project_individual_logits, project_batch5
        )
        reference_batch_shape = _logit_distribution_comparison(
            reference_individual_logits, reference_batch5
        )

        project_cache = StaticKVCache(max_length=52)
        project_prefill = project(prompts[:1], cache=project_cache)[:, -1]
        reference_prefill = reference(prompts[:1], use_cache=True)
        forced_token = torch.argmax(reference_prefill.logits[:, -1], dim=-1, keepdim=True)
        project_next = project(forced_token, cache=project_cache)[:, -1]
        reference_next = reference(
            forced_token,
            past_key_values=reference_prefill.past_key_values,
            use_cache=True,
        ).logits[:, -1]
        cached_next = _logit_distribution_comparison(project_next, reference_next)

    thresholds = {
        "all_top1_equal": True,
        "minimum_top10_overlap": 9,
        "maximum_jensen_shannon_divergence": 1e-4,
    }
    passed = (
        batch1["all_top1_equal"]
        and batch5["all_top1_equal"]
        and cached_next["all_top1_equal"]
        and batch1["minimum_top10_overlap"] >= thresholds["minimum_top10_overlap"]
        and batch5["minimum_top10_overlap"] >= thresholds["minimum_top10_overlap"]
        and cached_next["minimum_top10_overlap"] >= thresholds["minimum_top10_overlap"]
        and batch1["maximum_jensen_shannon_divergence"]
        <= thresholds["maximum_jensen_shannon_divergence"]
        and batch5["maximum_jensen_shannon_divergence"]
        <= thresholds["maximum_jensen_shannon_divergence"]
        and cached_next["maximum_jensen_shannon_divergence"]
        <= thresholds["maximum_jensen_shannon_divergence"]
    )
    torch.cuda.synchronize()
    result = {
        "status": "passed" if passed else "scientific_discrepancy",
        "passed": passed,
        "model_revision": _model_revision(),
        "prompt_indices": list(SMOKE_PROMPT_INDICES),
        "reference": {
            "implementation": "transformers.AutoModelForCausalLM",
            "transformers_version": transformers.__version__,
            "torch_dtype": "bfloat16",
            "attention_implementation": "eager",
            "local_files_only": True,
        },
        "thresholds": thresholds,
        "batch1_prefill": batch1,
        "batch5_prefill": batch5,
        "individual_prompt_reference": individual_reference,
        "project_batch1_vs_batch5": project_batch_shape,
        "reference_batch1_vs_batch5": reference_batch_shape,
        "batch1_cached_next_token": cached_next,
        "forced_cached_token_id": int(forced_token.item()),
        "runtime": {
            "project_load_seconds": project_load_seconds,
            "reference_load_seconds": reference_load_seconds,
            "function_seconds": time.perf_counter() - started,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "requested_gpu": "H100",
            "actual_gpu": torch.cuda.get_device_name(0),
        },
        "provenance": {
            "modal_image_id": os.environ.get("MODAL_IMAGE_ID"),
            "modal_task_id": os.environ.get("MODAL_TASK_ID"),
            "network_model_downloads": 0,
        },
    }
    del project, reference
    gc.collect()
    torch.cuda.empty_cache()
    return result


def write_logits_parity_artifact(payload: dict, root: Path) -> str:
    path = root / "outputs/controlled_baseline_diagnostic_logits_parity.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return str(path.relative_to(root))
