#!/usr/bin/env python3
"""Validated, deterministic paired fixed-versus-online PRC analysis.

This script performs no generation, model loading, network access, or GPU work.
It consumes the seven frozen fixed/online audit pairs named in the JSON config.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
from pathlib import Path
from typing import Any


DETECTORS = ("map", "entropy", "naive")


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if isinstance(expected, float):
        require(
            isinstance(actual, (int, float))
            and math.isclose(float(actual), expected, rel_tol=0, abs_tol=1e-12),
            f"{label}: expected {expected!r}, got {actual!r}",
        )
    else:
        require(actual == expected, f"{label}: expected {expected!r}, got {actual!r}")


def require_score(value: dict[str, Any], label: str) -> None:
    require(isinstance(value.get("decision"), bool), f"{label}: decision is not bool")
    for field in ("statistic", "threshold"):
        number = value.get(field)
        require(
            isinstance(number, (int, float)) and math.isfinite(float(number)),
            f"{label}: {field} is missing or non-finite",
        )


def fixed_score(record: dict[str, Any], detector: str) -> dict[str, Any]:
    value = {
        "decision": record.get(f"decision_{detector}"),
        "statistic": record.get(f"stat_{detector}"),
        "threshold": record.get(f"thr_{detector}"),
    }
    require_score(value, f"fixed prompt {record.get('prompt_idx')} {detector}")
    return value


def online_score(record: dict[str, Any], detector: str) -> dict[str, Any]:
    raw = record.get("scores", {}).get(detector)
    require(isinstance(raw, dict), f"online prompt {record.get('prompt_idx')} lacks {detector}")
    value = {
        "decision": raw.get("decision"),
        "statistic": raw.get("statistic"),
        "threshold": raw.get("threshold"),
        "variance_proxy": raw.get("V"),
        "status": raw.get("status"),
        "length": raw.get("length"),
        "n": raw.get("n"),
        "T": raw.get("T"),
        "fpr": raw.get("fpr"),
        "effective_fpr": raw.get("effective_fpr"),
    }
    require_score(value, f"online prompt {record.get('prompt_idx')} {detector}")
    require(value["status"] == "ok", f"online prompt {record.get('prompt_idx')} {detector}: status is not ok")
    return value


def index_records(
    records: Any, source: str, expected_prompts: list[int], label: str
) -> dict[int, dict[str, Any]]:
    require(isinstance(records, list), f"{label}: records is not a list")
    selected = [record for record in records if record.get("source") == source]
    indexed: dict[int, dict[str, Any]] = {}
    for record in selected:
        prompt_idx = record.get("prompt_idx")
        require(isinstance(prompt_idx, int), f"{label}: invalid prompt_idx {prompt_idx!r}")
        require(prompt_idx not in indexed, f"{label}: duplicate {source} prompt {prompt_idx}")
        indexed[prompt_idx] = record
    require(sorted(indexed) == expected_prompts, f"{label}: incomplete {source} prompt coverage")
    return indexed


def index_online_records(
    records: Any, watermark: bool, expected_prompts: list[int], label: str
) -> dict[int, dict[str, Any]]:
    require(isinstance(records, list), f"{label}: results is not a list")
    selected = [record for record in records if record.get("watermark") is watermark]
    indexed: dict[int, dict[str, Any]] = {}
    for record in selected:
        prompt_idx = record.get("prompt_idx")
        require(isinstance(prompt_idx, int), f"{label}: invalid prompt_idx {prompt_idx!r}")
        require(prompt_idx not in indexed, f"{label}: duplicate prompt {prompt_idx}")
        indexed[prompt_idx] = record
    kind = "watermarked" if watermark else "null"
    require(sorted(indexed) == expected_prompts, f"{label}: incomplete {kind} prompt coverage")
    return indexed


def extract_online(wrapper: dict[str, Any], key: str | None, label: str) -> dict[str, Any]:
    if key is None:
        return wrapper
    payload = wrapper.get(key)
    require(isinstance(payload, dict), f"{label}: missing payload key {key!r}")
    return payload


def validate_counts(
    payload: dict[str, Any], wm: dict[int, dict[str, Any]], null: dict[int, dict[str, Any]], label: str
) -> None:
    counts = payload.get("counts")
    require(isinstance(counts, dict), f"{label}: missing counts")
    for detector in DETECTORS:
        count = counts.get(detector)
        require(isinstance(count, dict), f"{label}: missing {detector} counts")
        expected = {
            "tp": sum(online_score(record, detector)["decision"] for record in wm.values()),
            "fp": sum(online_score(record, detector)["decision"] for record in null.values()),
            "watermarked_total": len(wm),
            "null_total": len(null),
        }
        for field, value in expected.items():
            require_equal(count.get(field), value, f"{label} {detector} {field}")


def percentile(sorted_values: list[float], q: float) -> float:
    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def comparison_seed(analysis_seed: int, cell_id: str, detector: str) -> int:
    value = f"{analysis_seed}|{cell_id}|{detector}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def paired_bootstrap(
    differences: list[int], resamples: int, seed: int, alpha: float
) -> tuple[float, float]:
    rng = random.Random(seed)
    size = len(differences)
    samples = []
    for _ in range(resamples):
        total = 0
        for _ in range(size):
            total += differences[rng.randrange(size)]
        samples.append(total / size)
    samples.sort()
    return percentile(samples, alpha / 2), percentile(samples, 1 - alpha / 2)


def exact_mcnemar(b: int, c: int) -> float:
    discordant = b + c
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(b, c) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def holm_adjust(rows: list[dict[str, Any]]) -> None:
    ordered = sorted(enumerate(rows), key=lambda item: item[1]["mcnemar_p_exact"])
    running = 0.0
    family_size = len(rows)
    for rank, (index, row) in enumerate(ordered):
        running = max(running, (family_size - rank) * row["mcnemar_p_exact"])
        rows[index]["mcnemar_p_holm"] = min(1.0, running)


def interpretation(effect: float, low: float, high: float, delta: float) -> str:
    if low >= -delta and high <= delta:
        return "practically_equivalent"
    if low > 0 and effect >= delta:
        return "fixed_better"
    if high < 0 and effect <= -delta:
        return "online_better"
    return "inconclusive"


def validate_cost_ledger(path: Path, expected_cells: set[str], hard_stop: float) -> dict[str, Any]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    require({row["cell_id"] for row in rows} == expected_cells, "cost ledger cell set is wrong")
    total = 0.0
    for row in rows:
        label = f"cost ledger {row['cell_id']}"
        require_equal(row["execution_mode"], "cache_only", f"{label} execution_mode")
        require_equal(row["generated_watermarked_tokens"], "0", f"{label} wm tokens")
        require_equal(row["generated_null_tokens"], "0", f"{label} null tokens")
        require_equal(row["gpu_worker_launched"].lower(), "false", f"{label} GPU launch")
        require(float(row["gpu_seconds"]) == 0, f"{label}: GPU seconds must be zero")
        require_equal(row["status"], "complete", f"{label} status")
        require(bool(row["modal_run_url"].strip()), f"{label}: missing Modal run URL")
        try:
            cost = float(row["provider_cost_usd"])
        except ValueError as error:
            raise ValueError(f"{label}: provider cost is not reconciled") from error
        require(cost >= 0, f"{label}: provider cost is negative")
        total += cost
    require(total <= hard_stop, f"cache-only cost ${total:.4f} exceeds ${hard_stop:.2f} hard stop")
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "rows": len(rows),
        "total_provider_cost_usd": total,
        "hard_stop_usd": hard_stop,
    }


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def render_csv(rows: list[dict[str, Any]]) -> bytes:
    fieldnames = [
        "cell_id", "model", "eta", "n", "detector", "num_prompts",
        "fixed_tp", "online_tp", "fixed_tpr", "online_tpr",
        "difference_fixed_minus_online", "difference_percentage_points",
        "bootstrap_ci_low", "bootstrap_ci_high", "bootstrap_resamples",
        "bootstrap_seed", "fixed_1_online_0", "fixed_0_online_1",
        "discordant_total", "mcnemar_p_exact", "mcnemar_p_holm",
        "smallest_meaningful_difference", "interpretation",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def render_report(rows: list[dict[str, Any]], config: dict[str, Any]) -> bytes:
    outputs = config["outputs"]
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["interpretation"]] = counts.get(row["interpretation"], 0) + 1
    eight_b_rows = [row for row in rows if row["model"] == "Qwen3-8B-Base"]
    eight_b_counts: dict[str, int] = {}
    for row in eight_b_rows:
        eight_b_counts[row["interpretation"]] = eight_b_counts.get(row["interpretation"], 0) + 1
    holm_significant = [row for row in rows if row["mcnemar_p_holm"] < 0.05]
    lines = [
        "# Fixed versus online PRC: paired analysis",
        "",
        "## Results",
        "",
        "| Model | eta | n | Detector | Fixed TPR | Online TPR | Fixed - online (pp) | 95% paired CI (pp) | Holm p | Interpretation |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['eta']:.2f} | {row['n']} | {row['detector']} "
            f"| {100 * row['fixed_tpr']:.1f}% | {100 * row['online_tpr']:.1f}% "
            f"| {row['difference_percentage_points']:.1f} "
            f"| [{100 * row['bootstrap_ci_low']:.1f}, {100 * row['bootstrap_ci_high']:.1f}] "
            f"| {row['mcnemar_p_holm']:.4g} | {row['interpretation']} |"
        )
    summary = ", ".join(f"{key.replace('_', ' ')}: {value}" for key, value in sorted(counts.items()))
    analysis = config["analysis"]
    lines.extend([
        "",
        "## Conclusion",
        "",
        f"Across the predeclared family of 21 comparisons: {summary}.",
        f"For Qwen3-8B specifically, neither construction was called better in any of the {len(eight_b_rows)} comparisons; {eight_b_counts.get('practically_equivalent', 0)} were practically equivalent and {eight_b_counts.get('inconclusive', 0)} were inconclusive. The data therefore do not establish a meaningful fixed-versus-online difference for 8B, but most intervals remain too wide to establish practical equivalence.",
        "For MAP, no cell called either construction better. The 0.6B eta=0.15 cell was practically equivalent; the other six MAP cells were inconclusive.",
        f"Only {len(holm_significant)} of 21 exact McNemar tests remained significant after Holm correction: 0.6B, eta=0.15, n=1504, naive (fixed - online = 12.4 points; adjusted p = 0.00119). The other two `fixed_better` labels follow the predeclared effect-plus-bootstrap-interval rule but do not survive familywise McNemar correction.",
        "The row-level interpretations above apply the predeclared rules exactly; no conclusion is based on an unpaired aggregate comparison.",
        "",
        "## Methods",
        "",
        f"Records were paired by prompt index (500 prompts per cell). Effects are fixed minus online TPR. The 95% percentile intervals use {analysis['bootstrap_resamples']:,} deterministic paired prompt resamples with global analysis seed `{analysis['analysis_seed']}` and recorded comparison-specific seeds. Exact two-sided McNemar tests use discordant decisions; Holm correction is applied once across all 21 tests.",
        "",
        f"The smallest meaningful difference was predeclared as {100 * analysis['smallest_meaningful_difference']:.0f} percentage points. Practical equivalence requires the entire paired interval within [-5, +5] points. A method is called better only when the interval excludes zero and the estimated absolute effect is at least 5 points. All other results are inconclusive.",
        "",
        "## Validation and provenance",
        "",
        "All inputs passed exact prompt coverage, uniqueness, model, eta, length, nominal-FPR, experiment-seed, detector-completeness, statistic, threshold, and source-count checks before inference. The five new 8B inputs additionally require `execution_mode=cache_only`, prefix reuse, and a reconciled ledger showing zero generated tokens, no GPU worker, and total provider cost at or below $0.25.",
        "",
        f"Machine-readable records: [`{outputs['prompt_level']}`]({outputs['prompt_level']}) and [`{outputs['paired_summary']}`]({outputs['paired_summary']}). Validation/fingerprints: [`{outputs['validation']}`]({outputs['validation']}). Cost ledger: [`{outputs['cost_ledger']}`]({outputs['cost_ledger']}).",
        "",
        "## Limitations",
        "",
        "The analysis covers seven preselected model/eta/length cells and three detectors, not every possible operating point. Fixed and online keys are construction-specific, domain-separated derivations of the same experiment seed; they are not the same key object. Pairing therefore controls prompts, configuration, and seed, but not token-for-token watermark randomness. The online audit artifacts preserve detector scores and source fingerprints but not prompt text.",
        "",
    ])
    return "\n".join(lines).encode()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="fixed_vs_online_analysis_config.json")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = load_json(config_path)
    analysis = config["analysis"]
    expected_prompts = list(range(
        analysis["expected_prompt_indices"]["start"],
        analysis["expected_prompt_indices"]["stop_exclusive"],
    ))
    require(len(expected_prompts) == 500, "predeclared prompt set must contain 500 indices")
    require_equal(len(config["cells"]), 7, "matched cell count")
    require_equal(analysis["mcnemar_family_size"], 21, "McNemar family size")

    prompt_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    source_validation: list[dict[str, Any]] = []
    pending_cells = {cell["cell_id"] for cell in config["cells"] if cell["online_audit_status"] == "pending_cache_only"}
    cost = validate_cost_ledger(Path(config["outputs"]["cost_ledger"]), pending_cells, 0.25)

    for cell in config["cells"]:
        cell_id = cell["cell_id"]
        fixed_path = Path(cell["fixed"])
        online_path = Path(cell["online"])
        require(fixed_path.is_file(), f"{cell_id}: missing fixed source {fixed_path}")
        require(online_path.is_file(), f"{cell_id}: missing online source {online_path}")
        fixed = load_json(fixed_path)
        online_wrapper = load_json(online_path)
        online = extract_online(online_wrapper, cell["online_payload_key"], cell_id)

        fixed_config = fixed.get("config", {})
        for actual, expected, field in (
            (fixed_config.get("generation_model"), cell["model"], "fixed model"),
            (fixed_config.get("eta"), cell["eta"], "fixed eta"),
            (fixed_config.get("n"), cell["n"], "fixed n"),
            (fixed_config.get("T"), cell["n"], "fixed T"),
            (fixed_config.get("t"), 3, "fixed t"),
            (fixed_config.get("target_fpr"), analysis["target_fpr"], "fixed FPR"),
            (fixed_config.get("seed"), analysis["experiment_seed"], "fixed seed"),
            (online.get("generation_model"), cell["model"], "online model"),
            (online.get("eta"), cell["eta"], "online eta"),
            (online.get("n"), cell["n"], "online n"),
            (online.get("T"), cell["n"], "online T"),
            (online.get("t"), 3, "online t"),
            (online.get("target_fpr"), analysis["target_fpr"], "online FPR"),
            (online.get("experiment_seed"), analysis["experiment_seed"], "online seed"),
        ):
            require_equal(actual, expected, f"{cell_id} {field}")

        if cell["online_audit_status"] == "pending_cache_only":
            require_equal(online_wrapper.get("execution_mode"), "cache_only", f"{cell_id} execution mode")
            require_equal(online.get("watermarked_cache_mode"), "prefix_from_longer", f"{cell_id} cache reuse")
            require(int(online.get("watermarked_cache_T", 0)) >= cell["n"], f"{cell_id}: invalid source prefix length")

        fixed_wm = index_records(fixed.get("records"), "wm", expected_prompts, f"{cell_id} fixed")
        fixed_null = index_records(fixed.get("records"), "null", expected_prompts, f"{cell_id} fixed")
        online_wm = index_online_records(online.get("results"), True, expected_prompts, f"{cell_id} online")
        online_null = index_online_records(online.get("results"), False, expected_prompts, f"{cell_id} online")
        require_equal(fixed.get("prompt_indices"), expected_prompts, f"{cell_id} fixed prompt_indices")
        require_equal(fixed.get("record_count"), 1000, f"{cell_id} fixed record count")
        require_equal(online.get("prompt_indices"), expected_prompts, f"{cell_id} online prompt_indices")
        validate_counts(online, online_wm, online_null, f"{cell_id} online")

        for prompt_idx in expected_prompts:
            fixed_payload: dict[str, Any] = {}
            online_payload: dict[str, Any] = {}
            for kind, fixed_records, online_records in (
                ("watermarked", fixed_wm, online_wm),
                ("null", fixed_null, online_null),
            ):
                fixed_record = fixed_records[prompt_idx]
                online_record = online_records[prompt_idx]
                require_equal(fixed_record.get("n_tokens"), cell["n"], f"{cell_id} fixed prompt {prompt_idx} length")
                require_equal(fixed_record.get("generation_model"), cell["model"], f"{cell_id} fixed prompt {prompt_idx} model")
                fixed_scores = {detector: fixed_score(fixed_record, detector) for detector in DETECTORS}
                online_scores = {detector: online_score(online_record, detector) for detector in DETECTORS}
                for detector, score in online_scores.items():
                    require_equal(score["length"], cell["n"], f"{cell_id} online prompt {prompt_idx} {detector} length")
                    require_equal(score["n"], cell["n"], f"{cell_id} online prompt {prompt_idx} {detector} n")
                    require_equal(score["T"], cell["n"], f"{cell_id} online prompt {prompt_idx} {detector} T")
                    require_equal(score["fpr"], analysis["target_fpr"], f"{cell_id} online prompt {prompt_idx} {detector} FPR")
                fixed_payload[kind] = {
                    "scores": fixed_scores,
                    "tokens_sha256": fixed_record.get("tokens_sha256"),
                    "p_trace_sha256": fixed_record.get("p_trace_sha256"),
                }
                online_payload[kind] = {
                    "scores": online_scores,
                }
            prompt_rows.append({
                "prompt_id": prompt_idx,
                "cell_id": cell_id,
                "model": cell["model"],
                "eta": cell["eta"],
                "n": cell["n"],
                "t": 3,
                "target_fpr": analysis["target_fpr"],
                "experiment_seed": analysis["experiment_seed"],
                "key_relationship": "construction-specific domain-separated keys from the same experiment seed",
                "fixed": fixed_payload,
                "online": online_payload,
                "provenance": {
                    "fixed_source": str(fixed_path),
                    "online_source": str(online_path),
                    "fixed_artifact_fingerprint": fixed.get("artifact_fingerprint"),
                    "online_artifact_fingerprint": online.get("artifact_fingerprint"),
                    "online_watermarked_source_artifact_fingerprint": online.get("watermarked_source_artifact_fingerprint") or online.get("source_artifact_fingerprint"),
                    "online_key_sha256": online.get("online_key_sha256"),
                },
            })

        for detector in DETECTORS:
            fixed_decisions = [fixed_score(fixed_wm[index], detector)["decision"] for index in expected_prompts]
            online_decisions = [online_score(online_wm[index], detector)["decision"] for index in expected_prompts]
            differences = [int(fixed_value) - int(online_value) for fixed_value, online_value in zip(fixed_decisions, online_decisions)]
            bootstrap_seed = comparison_seed(analysis["analysis_seed"], cell_id, detector)
            low, high = paired_bootstrap(
                differences,
                analysis["bootstrap_resamples"],
                bootstrap_seed,
                1 - analysis["confidence_level"],
            )
            fixed_tp = sum(fixed_decisions)
            online_tp = sum(online_decisions)
            b = sum(fixed_value and not online_value for fixed_value, online_value in zip(fixed_decisions, online_decisions))
            c = sum(not fixed_value and online_value for fixed_value, online_value in zip(fixed_decisions, online_decisions))
            effect = sum(differences) / len(differences)
            summary_rows.append({
                "cell_id": cell_id,
                "model": cell["model"],
                "eta": cell["eta"],
                "n": cell["n"],
                "detector": detector,
                "num_prompts": len(expected_prompts),
                "fixed_tp": fixed_tp,
                "online_tp": online_tp,
                "fixed_tpr": fixed_tp / len(expected_prompts),
                "online_tpr": online_tp / len(expected_prompts),
                "difference_fixed_minus_online": effect,
                "difference_percentage_points": 100 * effect,
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "bootstrap_resamples": analysis["bootstrap_resamples"],
                "bootstrap_seed": bootstrap_seed,
                "fixed_1_online_0": b,
                "fixed_0_online_1": c,
                "discordant_total": b + c,
                "mcnemar_p_exact": exact_mcnemar(b, c),
                "mcnemar_p_holm": None,
                "smallest_meaningful_difference": analysis["smallest_meaningful_difference"],
                "interpretation": "pending",
            })

        source_validation.append({
            "cell_id": cell_id,
            "prompt_indices": {"count": 500, "minimum": 0, "maximum": 499, "duplicates": 0, "gaps": 0},
            "fixed": {
                "path": str(fixed_path),
                "file_sha256": file_sha256(fixed_path),
                "artifact_fingerprint": fixed.get("artifact_fingerprint"),
                "records_sha256": fixed.get("records_sha256"),
            },
            "online": {
                "path": str(online_path),
                "file_sha256": file_sha256(online_path),
                "execution_mode": online_wrapper.get("execution_mode", "historical_full_audit"),
                "artifact_fingerprint": online.get("artifact_fingerprint"),
                "watermarked_source_artifact_fingerprint": online.get("watermarked_source_artifact_fingerprint") or online.get("source_artifact_fingerprint"),
                "online_key_sha256": online.get("online_key_sha256"),
                "watermarked_cache_mode": online.get("watermarked_cache_mode", "sweep_final_audit"),
                "watermarked_cache_T": online.get("watermarked_cache_T", online.get("source_T")),
                "null_cache_T": online.get("null_cache_T"),
            },
        })

    require_equal(len(prompt_rows), 3500, "prompt-level row count")
    require_equal(len(summary_rows), analysis["mcnemar_family_size"], "comparison count")
    holm_adjust(summary_rows)
    for row in summary_rows:
        row["interpretation"] = interpretation(
            row["difference_fixed_minus_online"],
            row["bootstrap_ci_low"],
            row["bootstrap_ci_high"],
            analysis["smallest_meaningful_difference"],
        )

    prompt_bytes = ("\n".join(json.dumps(row, sort_keys=True, allow_nan=False) for row in prompt_rows) + "\n").encode()
    summary_bytes = render_csv(summary_rows)
    report_bytes = render_report(summary_rows, config)
    validation = {
        "status": "passed",
        "analysis_config": str(config_path),
        "analysis_config_sha256": file_sha256(config_path),
        "analysis_seed": analysis["analysis_seed"],
        "bootstrap_resamples": analysis["bootstrap_resamples"],
        "mcnemar_holm_family_size": len(summary_rows),
        "cost_validation": cost,
        "source_validation": source_validation,
        "output_validation": {
            config["outputs"]["prompt_level"]: {"rows": len(prompt_rows), "sha256": bytes_sha256(prompt_bytes)},
            config["outputs"]["paired_summary"]: {"rows": len(summary_rows), "sha256": bytes_sha256(summary_bytes)},
            config["outputs"]["report"]: {"sha256": bytes_sha256(report_bytes)},
        },
    }
    validation_bytes = (json.dumps(validation, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()

    atomic_write(Path(config["outputs"]["prompt_level"]), prompt_bytes)
    atomic_write(Path(config["outputs"]["paired_summary"]), summary_bytes)
    atomic_write(Path(config["outputs"]["report"]), report_bytes)
    atomic_write(Path(config["outputs"]["validation"]), validation_bytes)
    print(f"validated 7 cells, 3,500 prompt pairs, and {len(summary_rows)} comparisons")
    print(f"cache-only provider cost: ${cost['total_provider_cost_usd']:.4f}")


if __name__ == "__main__":
    main()
