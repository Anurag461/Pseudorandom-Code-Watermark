"""Finalize the cache-only Qwen3-8B -> Qwen3-0.6B detector study.

This script is deliberately local and streaming: it loads no model, performs
no generation, and writes only compact summaries and a validation manifest.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import statistics
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

from proxy_8b_analysis import (
    COMMON_PREFIXES,
    PRC_AUDITS,
    PROXY_IMAGE_DEFINITION_SHA256,
    PROXY_MODEL_REVISION,
    PROXY_MODEL_WEIGHTS_ETAG,
    PROXY_TOKENIZER_ETAG,
    campaign_code_fingerprint,
)


BOUNDARY_NATIVE_PATHS = {
    0.05: (
        "outputs/online_causal_n640_t3_eta0.05_prompts500_"
        "gen-qwen3_8b_base_sampler-poscdf-v1_kvcache-static-v1_from_n1280.json"
    ),
    0.10: (
        "outputs/online_causal_n1407_t3_eta0.10_prompts500_"
        "gen-qwen3_8b_base_sampler-poscdf-v1_kvcache-static-v1_from_n3072.json"
    ),
    0.15: (
        "outputs/online_causal_n4096_t3_eta0.15_prompts500_"
        "gen-qwen3_8b_base_sampler-poscdf-v1.json"
    ),
    0.20: (
        "outputs/online_causal_n13088_t3_eta0.20_prompts500_"
        "gen-qwen3_8b_base_sampler-poscdf-v1_kvcache-static-v1_from_n14336.json"
    ),
}

# Exact provider-billing rows fetched after the final CPU-only aggregation.
COST_ROWS = (
    ("ap-YkcsbxLcUqRD7YON8jdNkV", "initial image-ordering test", "stopped before GPU", "0.00235101", "0.00005114", "", "0"),
    ("ap-Zyq13J4Rb1xL40H4Mb0HIP", "initial CPU tests", "passed", "0.00027263", "0.00001045", "A10G", "0"),
    ("ap-CvWdAY2PlMXjhp6y6uVt8x", "cache inventory and plan", "passed", "0.00684301", "0.00054421", "", "0"),
    ("ap-gioR8berHP83NJMES9Uzr0", "pinned CPU tests", "passed", "0.00293677", "0.00006913", "", "0"),
    ("ap-VUs3EVOKSXQ1B0gXNUlYRN", "NumPy compatibility preflight", "stopped before GPU", "0.00007880", "0.00000292", "", "0"),
    ("ap-4FRB9LAXBpcjaFibxzb20r", "sequential long-context benchmark", "projection blocked; data retained", "0.05114468", "0.00645700", "A10G", "0.81461105"),
    ("ap-6MhJH7r3KYc5wWfXqApJB0", "chunked exact-equivalence tests", "passed", "0.00042042", "0.00001064", "", "0"),
    ("ap-FUUu2HjZx2ZikhkYvdM3tk", "chunk-64 benchmark and cost gate", "passed", "0.00678580", "0.00052845", "A10G", "0.03911112"),
    ("ap-VX13Dm2pjlL3fwJvcZlzHp", "full PRC/null/TextSeal replay and PRC scoring", "passed", "0.18724712", "0.01952036", "A10G", "1.98799203"),
    ("ap-QzUtJrOC5HAdq5PJZPGmiE", "official TextSeal proxy scoring", "passed", "0.05723953", "0.01936054", "L40S", "0"),
    ("ap-1ty06Yemjf2NzbOVU6m7gG", "native-8B cached quality aggregation", "passed", "0.01388250", "0.00036509", "", "0"),
)


def cost_ledger() -> list[dict]:
    rows = []
    for app_id, purpose, status, cpu, memory, gpu_resource, gpu in COST_ROWS:
        total = Decimal(cpu) + Decimal(memory) + Decimal(gpu)
        rows.append({
            "modal_app_id": app_id,
            "purpose": purpose,
            "status": status,
            "cpu_cost_usd": cpu,
            "memory_cost_usd": memory,
            "gpu_resource": gpu_resource,
            "gpu_cost_usd": gpu,
            "total_cost_usd": str(total),
            "generation_attempts": 0,
            "run_url": f"https://modal.com/apps/new-prc-watermark/main/{app_id}",
        })
    return rows


def read_jsonl(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        with path.open() as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def combined_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty summary {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def neg_log10(value: float) -> float:
    return -math.log10(max(float(value), 1e-300))


def hoeffding_p_upper(score: dict) -> float:
    statistic = float(score["statistic"])
    variance = float(score["V"])
    if statistic <= 0 or variance <= 0 or not math.isfinite(variance):
        return 1.0
    return min(1.0, math.exp(-(statistic * statistic) / (2.0 * variance)))


def binary_summary(rows: list[dict], *, p_getter=None) -> dict:
    watermarked = [row for row in rows if row["sample_type"] == "watermarked"]
    null = [row for row in rows if row["sample_type"] == "null"]
    if len(watermarked) != 500 or len(null) != 500:
        raise ValueError("each detector cell must contain 500 WM and 500 null rows")
    result = {
        "watermarked_n": len(watermarked),
        "true_positives": sum(bool(row["decision"]) for row in watermarked),
        "tpr": sum(bool(row["decision"]) for row in watermarked) / 500,
        "null_n": len(null),
        "false_positives": sum(bool(row["decision"]) for row in null),
        "observed_fpr": sum(bool(row["decision"]) for row in null) / 500,
    }
    if p_getter is not None:
        result.update({
            "median_neg_log10_p_watermarked": statistics.median(
                neg_log10(p_getter(row)) for row in watermarked
            ),
            "median_neg_log10_p_null": statistics.median(
                neg_log10(p_getter(row)) for row in null
            ),
        })
    return result


def validate_proxy_prc(rows: list[dict]) -> None:
    expected_prefixes = {
        float(audit["eta"]): set(COMMON_PREFIXES) | {int(audit["prefix_T"])}
        for audit in PRC_AUDITS
    }
    keys = set()
    coverage = defaultdict(set)
    for row in rows:
        key = (
            float(row["eta"]), int(row["prefix_length"]),
            str(row["sample_type"]), int(row["prompt_index"]),
        )
        if key in keys:
            raise ValueError(f"duplicate proxy PRC row {key}")
        keys.add(key)
        if row.get("generation_attempts") != 0:
            raise ValueError("proxy PRC artifact records a generation attempt")
        if set(row["scores"]) != {"map", "entropy"}:
            raise ValueError("proxy PRC row lacks MAP/entropy scores")
        for score in row["scores"].values():
            if not all(math.isfinite(float(score[field])) for field in (
                "statistic", "threshold", "V"
            )):
                raise ValueError("proxy PRC score is nonfinite")
            expected = hoeffding_p_upper(score) < 1e-3
            if bool(score["decision"]) != expected:
                raise ValueError("proxy PRC decision differs from p<1e-3")
        coverage[(float(row["eta"]), int(row["prefix_length"]), row["sample_type"])].add(
            int(row["prompt_index"])
        )
    expected_rows = sum(len(prefixes) for prefixes in expected_prefixes.values()) * 1000
    if len(rows) != expected_rows:
        raise ValueError(f"proxy PRC row count {len(rows)} != {expected_rows}")
    expected_indices = set(range(500))
    for eta, prefixes in expected_prefixes.items():
        for prefix in prefixes:
            for sample_type in ("watermarked", "null"):
                if coverage[(eta, prefix, sample_type)] != expected_indices:
                    raise ValueError(f"incomplete proxy PRC cell {eta}, {prefix}, {sample_type}")


def prc_proxy_summary(rows: list[dict]) -> list[dict]:
    output = []
    for audit in PRC_AUDITS:
        eta = float(audit["eta"])
        prefixes = sorted(set(COMMON_PREFIXES) | {int(audit["prefix_T"])})
        for prefix in prefixes:
            selected = [
                row for row in rows
                if float(row["eta"]) == eta and int(row["prefix_length"]) == prefix
            ]
            for weight in ("map", "entropy"):
                flattened = [
                    {**row, "decision": bool(row["scores"][weight]["decision"]),
                     "score": row["scores"][weight]}
                    for row in selected
                ]
                summary = binary_summary(
                    flattened, p_getter=lambda row: hoeffding_p_upper(row["score"])
                )
                output.append({
                    "method": "online_prc",
                    "eta": eta,
                    "detector_weight": weight,
                    "generation_model": "Qwen3-8B-Base",
                    "detector_probability_model": "Qwen3-0.6B-Base",
                    "prefix_length": prefix,
                    "length_status": (
                        audit["boundary_status"]
                        if prefix == int(audit["prefix_T"])
                        else "common_controlled_prefix"
                    ),
                    "calibration_type": "Hoeffding p-value upper bound",
                    "nominal_decision_rule": "p_upper < 0.001",
                    **summary,
                })
    return output


def portability_summary(proxy_rows: list[dict], root: Path) -> list[dict]:
    output = []
    for audit in PRC_AUDITS:
        eta = float(audit["eta"])
        prefix = int(audit["prefix_T"])
        native = json.loads((root / BOUNDARY_NATIVE_PATHS[eta]).read_text())
        if int(native["T"]) != prefix or len(native["results"]) != 1000:
            raise ValueError(f"native boundary artifact is incompatible for eta={eta}")
        native_by_key = {
            (int(row["prompt_idx"]), "watermarked" if row["watermark"] else "null"): row
            for row in native["results"]
        }
        proxy_by_key = {
            (int(row["prompt_index"]), row["sample_type"]): row
            for row in proxy_rows
            if float(row["eta"]) == eta and int(row["prefix_length"]) == prefix
        }
        expected_keys = {(index, kind) for index in range(500) for kind in ("watermarked", "null")}
        if set(native_by_key) != expected_keys or set(proxy_by_key) != expected_keys:
            raise ValueError(f"native/proxy pairing failed for eta={eta}")
        for weight in ("map", "entropy"):
            values = {}
            for sample_type in ("watermarked", "null"):
                pairs = [
                    (
                        bool(native_by_key[(index, sample_type)]["scores"][weight]["decision"]),
                        bool(proxy_by_key[(index, sample_type)]["scores"][weight]["decision"]),
                    )
                    for index in range(500)
                ]
                values[sample_type] = {
                    "native_rate": sum(left for left, _ in pairs) / 500,
                    "proxy_rate": sum(right for _, right in pairs) / 500,
                    "agreement": sum(left == right for left, right in pairs) / 500,
                    "native_positive_proxy_negative": sum(left and not right for left, right in pairs),
                    "native_negative_proxy_positive": sum(not left and right for left, right in pairs),
                }
            output.append({
                "eta": eta,
                "prefix_length": prefix,
                "length_status": audit["boundary_status"],
                "detector_weight": weight,
                "native_probability_model": "Qwen3-8B-Base",
                "proxy_probability_model": "Qwen3-0.6B-Base",
                "native_tpr": values["watermarked"]["native_rate"],
                "proxy_tpr": values["watermarked"]["proxy_rate"],
                "proxy_minus_native_tpr": (
                    values["watermarked"]["proxy_rate"]
                    - values["watermarked"]["native_rate"]
                ),
                "watermarked_decision_agreement": values["watermarked"]["agreement"],
                "wm_native_positive_proxy_negative": values["watermarked"]["native_positive_proxy_negative"],
                "wm_native_negative_proxy_positive": values["watermarked"]["native_negative_proxy_positive"],
                "native_observed_fpr": values["null"]["native_rate"],
                "proxy_observed_fpr": values["null"]["proxy_rate"],
                "null_decision_agreement": values["null"]["agreement"],
                "null_native_positive_proxy_negative": values["null"]["native_positive_proxy_negative"],
                "null_native_negative_proxy_positive": values["null"]["native_negative_proxy_positive"],
                "generation_attempts": 0,
            })
    return output


def common_method_summary(
    native_rows: list[dict], textseal_proxy_rows: list[dict], prc_proxy_rows: list[dict]
) -> list[dict]:
    output = []
    for method in ("online_prc", "textseal", "synthid_text", "gumbel_max"):
        for prefix in COMMON_PREFIXES:
            selected = [row for row in native_rows if row["method"] == method and int(row["prefix_length"]) == prefix]
            summary = binary_summary(selected, p_getter=lambda row: row["p_value"])
            dependence = (
                "model_probability_dependent"
                if method in {"online_prc", "textseal"}
                else "model_independent_frequentist_hash_test"
            )
            output.append({
                "method": method,
                "method_setting": "eta=0.05" if method == "online_prc" else "frozen_controlled_setting",
                "detector_variant": "native_8b",
                "detector_probability_model": (
                    "Qwen3-8B-Base" if dependence.startswith("model_probability") else "not_applicable"
                ),
                "detector_model_dependence": dependence,
                "prefix_length": prefix,
                "calibration_type": selected[0]["calibration_type"],
                "nominal_decision_rule": "p < 0.001",
                **summary,
            })

    for prefix in COMMON_PREFIXES:
        selected = [row for row in textseal_proxy_rows if int(row["prefix_length"]) == prefix]
        output.append({
            "method": "textseal",
            "method_setting": "alpha=0.1",
            "detector_variant": "proxy_entropy_sensitivity",
            "detector_probability_model": "Qwen3-0.6B-Base",
            "detector_model_dependence": "model_probability_dependent",
            "prefix_length": prefix,
            "calibration_type": "moment-matched Gamma approximation",
            "nominal_decision_rule": "p < 0.001",
            **binary_summary(selected, p_getter=lambda row: row["p_value"]),
        })

    prc_summary = prc_proxy_summary(prc_proxy_rows)
    for row in prc_summary:
        if row["prefix_length"] not in COMMON_PREFIXES:
            continue
        output.append({
            "method": "online_prc",
            "method_setting": f"eta={row['eta']:.2f}; weight={row['detector_weight']}",
            "detector_variant": "proxy_probability_sensitivity",
            "detector_probability_model": "Qwen3-0.6B-Base",
            "detector_model_dependence": "model_probability_dependent",
            "prefix_length": row["prefix_length"],
            "calibration_type": row["calibration_type"],
            "nominal_decision_rule": row["nominal_decision_rule"],
            **{key: row[key] for key in (
                "watermarked_n", "true_positives", "tpr", "null_n",
                "false_positives", "observed_fpr",
                "median_neg_log10_p_watermarked", "median_neg_log10_p_null",
            )},
        })
    return output


def quality_summary(native_rows: list[dict], quality_rows: list[dict]) -> list[dict]:
    groups = []
    for eta in (0.05, 0.10, 0.15, 0.20):
        groups.append((
            "online_prc", f"eta={eta:.2f}",
            [row for row in quality_rows if row["sample_type"] == "watermarked" and float(row["eta"]) == eta],
        ))
    groups.append(("null", "shared canonical null", [row for row in quality_rows if row["sample_type"] == "null"]))
    for method in ("textseal", "synthid_text", "gumbel_max"):
        groups.append((
            method, "frozen controlled setting",
            [row for row in native_rows if row["method"] == method and row["sample_type"] == "watermarked" and int(row["prefix_length"]) == 1024],
        ))
    metrics = (
        "base_model_nll", "base_model_perplexity", "repetition_rate",
        "distinct_2", "distinct_3",
    )
    output = []
    for method, setting, rows in groups:
        if len(rows) != 500:
            raise ValueError(f"quality group {method}, {setting} has {len(rows)} rows")
        if any(int(row["output_length"]) != 1024 for row in rows):
            raise ValueError("quality summary encountered a non-1024 output")
        output.append({
            "method": method,
            "method_setting": setting,
            "sample_count": len(rows),
            "quality_likelihood_model": "Qwen3-8B-Base",
            **{f"mean_{metric}": statistics.mean(float(row[metric]) for row in rows) for metric in metrics},
            **{f"median_{metric}": statistics.median(float(row[metric]) for row in rows) for metric in metrics},
            "repetition_rate_gt_0p1_count": sum(float(row["repetition_rate"]) > 0.1 for row in rows),
            "distinct_3_lt_0p8_count": sum(float(row["distinct_3"]) < 0.8 for row in rows),
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--native-scored-dir", type=Path,
        default=Path("/private/tmp/qwen3-8b-controlled-full-scored/scored"),
    )
    parser.add_argument(
        "--textseal-proxy-dir", type=Path,
        default=Path("/private/tmp/proxy_ts_shards/textseal"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    root = args.repo_root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir

    prc_proxy_path = root / "outputs/proxy_8b_prc_prompt_level.jsonl"
    quality_path = root / "outputs/proxy_8b_native_quality_prompt_level.jsonl"
    campaign_path = root / "outputs/proxy_8b_campaign_manifest.json"
    pinned_sources_path = root / "baseline_comparison/pinned_sources_manifest.json"
    native_paths = sorted(args.native_scored_dir.glob("shard_*.jsonl"))
    textseal_paths = sorted(args.textseal_proxy_dir.glob("shard_*.jsonl"))
    textseal_validation_paths = sorted(args.textseal_proxy_dir.glob("shard_*_validation.json"))
    if len(native_paths) != 10 or len(textseal_paths) != 10 or len(textseal_validation_paths) != 10:
        raise ValueError("finalization requires exactly ten native and TextSeal proxy shards")

    prc_proxy_rows = read_jsonl([prc_proxy_path])
    validate_proxy_prc(prc_proxy_rows)
    native_rows = read_jsonl(native_paths)
    textseal_proxy_rows = read_jsonl(textseal_paths)
    quality_rows = read_jsonl([quality_path])
    if len(native_rows) != 24_000 or len(textseal_proxy_rows) != 6_000 or len(quality_rows) != 2_500:
        raise ValueError("one or more prompt-level input row counts differ")
    if any(row.get("generation_attempts") != 0 for row in quality_rows):
        raise ValueError("quality inputs record generation")
    validation_payloads = [json.loads(path.read_text()) for path in textseal_validation_paths]
    if any(not payload["passed"] or payload["generation_attempts"] != 0 for payload in validation_payloads):
        raise ValueError("a TextSeal official-reference validation failed")
    campaign = json.loads(campaign_path.read_text())
    pinned_sources = json.loads(pinned_sources_path.read_text())

    summaries = {
        "proxy_8b_prc_summary.csv": prc_proxy_summary(prc_proxy_rows),
        "proxy_8b_prc_portability_summary.csv": portability_summary(prc_proxy_rows, root),
        "proxy_8b_common_method_summary.csv": common_method_summary(native_rows, textseal_proxy_rows, prc_proxy_rows),
        "proxy_8b_quality_summary.csv": quality_summary(native_rows, quality_rows),
        "proxy_8b_cost_ledger.csv": cost_ledger(),
    }
    for name, rows in summaries.items():
        write_csv(output_dir / name, rows)

    inputs = [prc_proxy_path, quality_path, *native_paths, *textseal_paths, *textseal_validation_paths]
    output_paths = [output_dir / name for name in summaries]
    manifest = {
        "status": "passed",
        "campaign": "qwen3_8b_cache_only_proxy_analysis",
        "generation_attempts": 0,
        "model_generation_workers": 0,
        "canonical_prompt_indices": list(range(500)),
        "common_prefixes": list(COMMON_PREFIXES),
        "prc_eta_values": [float(audit["eta"]) for audit in PRC_AUDITS],
        "eta_0p15_status": PRC_AUDITS[2]["boundary_status"],
        "model_provenance": {
            "generation_and_quality_model": "Qwen3-8B-Base",
            "generation_and_quality_model_revision": "49e3418fbbbca6ecbdf9608b4d22e5a407081db4",
            "proxy_probability_model": "Qwen3-0.6B-Base",
            "proxy_probability_model_revision": PROXY_MODEL_REVISION,
            "proxy_model_weights_etag": PROXY_MODEL_WEIGHTS_ETAG,
            "proxy_tokenizer_etag": PROXY_TOKENIZER_ETAG,
            "proxy_modal_image_definition_sha256": PROXY_IMAGE_DEFINITION_SHA256,
        },
        "source_provenance": {
            "textseal": pinned_sources["official_sources"]["textseal"],
            "synthid_text": pinned_sources["official_sources"]["synthid_text"],
            "final_integration_code_fingerprint_sha256": campaign_code_fingerprint([
                str(root / path) for path in (
                    "modal_online_run.py", "online_prc.py", "watermark_expt.py",
                    "detectors.py", "qwen.py", "constants.py",
                    "proxy_8b_analysis.py", "proxy_8b_finalize.py",
                    "baseline_comparison/smoke_runner.py",
                    "baseline_comparison/modal_app.py",
                )
            ]),
            "prc_source_artifact_fingerprints": {
                str(audit["label"]): audit["artifact_fingerprint"]
                for audit in campaign["verified_prc_plan"]["audits"]
            },
        },
        "input_rows": {
            "proxy_prc": len(prc_proxy_rows),
            "native_controlled_baseline": len(native_rows),
            "proxy_textseal": len(textseal_proxy_rows),
            "native_quality": len(quality_rows),
        },
        "official_reference_checks": {
            "textseal_proxy_shards_passed": 10,
            "textseal_official_comparison_count": sum(
                len(payload["validations"]) * 6 for payload in validation_payloads
            ),
            "max_textseal_official_common_abs_p_delta": max(
                validation["max_official_common_p_delta"]
                for payload in validation_payloads
                for validation in payload["validations"]
            ),
            "native_quality_fields_preserved": all(
                payload["native_quality_fields_preserved"]
                for payload in validation_payloads
            ),
        },
        "input_fingerprints": {
            "proxy_prc_jsonl": sha256(prc_proxy_path),
            "native_quality_jsonl": sha256(quality_path),
            "native_controlled_shards_combined": combined_sha256(native_paths),
            "proxy_textseal_shards_combined": combined_sha256(textseal_paths),
            "proxy_textseal_validations_combined": combined_sha256(textseal_validation_paths),
            "campaign_manifest": sha256(campaign_path),
            "pinned_sources_manifest": sha256(pinned_sources_path),
        },
        "output_fingerprints": {path.name: sha256(path) for path in output_paths},
        "notes": [
            "SynthID frequentist and Gumbel exact-Gamma detectors do not consume model probabilities and are carried unchanged.",
            "TextSeal proxy scores replace only entropy weights; all quality likelihoods remain native Qwen3-8B.",
            "PRC eta and TextSeal alpha are unrelated operating parameters.",
            "Five hundred nulls cannot tightly validate nominal 0.1% FPR.",
        ],
        "billing": {
            "status": "exact_provider_reconciled",
            "cpu_cost_usd": "0.32920227",
            "memory_cost_usd": "0.04691993",
            "gpu_cost_usd": "2.84171420",
            "total_cost_usd": "3.21783640",
            "approved_hard_cap_usd": "20.00",
            "cost_ledger": "outputs/proxy_8b_cost_ledger.csv",
        },
    }
    manifest_path = output_dir / "proxy_8b_validation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        "passed": True,
        "generation_attempts": 0,
        "outputs": [str(path) for path in [*output_paths, manifest_path]],
    }, indent=2))


if __name__ == "__main__":
    main()
