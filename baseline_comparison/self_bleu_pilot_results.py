"""Verify replay artifacts and report the frozen Stage A paired analysis."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path, PurePosixPath

import numpy as np

from .self_bleu_config import digest
from .self_bleu_pilot import METHODS
from .self_bleu_pilot_analysis import read_inputs, paired_interval
from .self_bleu_validation import ROOT, save, sha

NAMES = {"online_prc": "PRC", "textseal": "TextSeal", "synthid_text": "SynthID",
         "gumbel_max": "Gumbel-Max", "null": "Ordinary sampling"}
COLORS = {"online_prc": "#087F8C", "textseal": "#D97706", "synthid_text": "#6D4CC4", "gumbel_max": "#CA4564"}


def collect(setup, download=False):
    manifest, inputs = read_inputs(setup)
    by_id = {r["response_id"]: r for r in inputs}
    files = {}
    reports = {}
    for stage in ("prc", "textseal"):
        report = json.loads((setup / f"{stage}_report.json").read_text())
        if not report["passed"] or report["manifest_id"] != manifest["id"] or report["stage"] != stage:
            raise ValueError("pilot replay did not pass under this manifest")
        reports[stage] = report
        for row in report["rows"]:
            if row["completion_sha256"] != by_id[row["response_id"]]["completion_sha256"]:
                raise ValueError("detector scored a different completion")
            if set(row["results"]) != set(map(str, manifest["prefix_lengths"])):
                raise ValueError("detector prefix coverage differs")
        files.update({f"{stage}/{name}": h for name, h in report["files"].items()})
        files[f"{stage}/report.json"] = sha(setup / f"{stage}_report.json")
    prefix = f"self_bleu_pilot/{manifest['id']}"
    raw = setup / "raw/replay"
    if download:
        import modal
        volume = modal.Volume.from_name("prc-completion-only", create_if_missing=False)
        def retrieve(name):
            if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts:
                raise ValueError("invalid artifact path")
            path = raw / name
            if path.exists():
                if sha(path) != files[name]:
                    raise ValueError("local replay artifact changed")
                return
            data = b"".join(volume.read_file(f"{prefix}/{name}"))
            import hashlib
            if hashlib.sha256(data).hexdigest() != files[name]:
                raise ValueError("downloaded artifact changed")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(retrieve, files))
    for name, expected in files.items():
        if sha(raw / name) != expected:
            raise ValueError(f"replay checksum differs: {name}")
    for name in reports["prc"]["files"]:
        payload = json.loads((raw / "prc" / name).read_text())
        if not payload["raw_inputs_verified"] or payload["forward_count"] != 1023:
            raise ValueError("PRC raw-input check missing")
        if len(payload["rows"]) != len(payload["probabilities_2_to_T"]):
            raise ValueError("PRC trace identity count differs")
        for row, trace in zip(payload["rows"], payload["probabilities_2_to_T"]):
            if len(trace) != 1023 or not all(np.isfinite(v) and 0 <= v <= 1 for v in trace):
                raise ValueError("invalid PRC completion-only probabilities")
    for name in reports["textseal"]["files"]:
        payload = json.loads((raw / "textseal" / name).read_text())
        row, data = by_id[payload["response_id"]], payload["data"]
        expected_calls = [n for n in manifest["prefix_lengths"] for _ in range(2 if data["validation"]["performed"] else 1)]
        if (not data["actual_model_inputs_verified"] or data["prefix_strategy"] != "direct"
                or data["forward_lengths"] != expected_calls or data["completion_sha256"] != row["completion_sha256"]):
            raise ValueError("TextSeal raw-input contract differs")
        if data["validation"]["performed"] and not data["validation"]["passed"]:
            raise ValueError("TextSeal exact upstream check failed")
        for n in manifest["prefix_lengths"]:
            if len(data["entropies_by_prefix"][str(n)]) != n-1:
                raise ValueError("TextSeal per-prefix entropy alignment differs")
    measured = sum(r["measured_resource_usd"] for r in reports.values())
    out = {"passed": True, "manifest_id": manifest["id"], "remote_path": prefix, "volume": "prc-completion-only",
           "verified_files": files, "verified_file_count": len(files), "prc_responses": len(reports["prc"]["rows"]),
           "new_textseal_responses": len(reports["textseal"]["rows"]), "reused_textseal_responses": manifest["reused_textseal_records"],
           "new_measured_resource_usd": measured, "total_planning_charge_usd": manifest["cost"]["previous_planning_charge_usd"] + measured,
           "remaining_initial_allocation_usd": 10-manifest["cost"]["previous_planning_charge_usd"]-measured,
           "billing_note": "Timing-derived resource estimates plus the previously declared failure/overhead allowances; not a settled invoice."}
    save(setup / "verification.json", out)
    return out


def score_rows(setup):
    manifest, inputs = read_inputs(setup)
    lookup = {r["response_id"]: r for r in inputs}
    token = json.loads((setup / "token_detection.json").read_text())
    if token["manifest_id"] != manifest["id"]:
        raise ValueError("CPU scoring identity differs")
    out = list(token["rows"])
    prc = json.loads((setup / "prc_report.json").read_text())
    ts = json.loads((setup / "textseal_report.json").read_text())
    reused = json.loads((setup / "reused_textseal.json").read_text())
    if digest(reused) != manifest["reused_textseal_sha256"]:
        raise ValueError("reused TextSeal results changed")
    for detector, rows in (("online_prc", prc["rows"]), ("textseal", ts["rows"] + reused)):
        seen = set()
        for row in rows:
            original = lookup[row["response_id"]]
            if row["response_id"] in seen or row["completion_sha256"] != original["completion_sha256"]:
                raise ValueError("duplicate or mismatched detector response")
            seen.add(row["response_id"])
            for n in manifest["prefix_lengths"]:
                raw = row["results"][str(n)]
                if detector == "online_prc":
                    score = raw
                else:
                    score = {**raw["comparison"], "upstream": raw["upstream"],
                             "calibration_type": "entropy-weighted Gamma approximation"}
                    if raw["completion_sha256"] != digest(original["token_ids"][:n]):
                        raise ValueError("TextSeal result prefix differs")
                out.append({k: original[k] for k in original if k != "token_ids"} | {"detector": detector, "length": n, "score": score})
        if len(seen) != 200:
            raise ValueError("incomplete pilot detector coverage")
    keys = [(r["detector"], r["response_id"], r["length"]) for r in out]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate score row")
    return out


def summarize(setup):
    import scipy.stats
    manifest, inputs = read_inputs(setup)
    verification = collect(setup)
    diversity = json.loads((setup / "diversity.json").read_text())
    if diversity["manifest_id"] != manifest["id"]:
        raise ValueError("diversity identity differs")
    metrics = {(r["method"], r["length"], r["prompt_index"]): r for r in diversity["rows"]}
    scores = score_rows(setup)
    by_key = {(r["detector"], r["method"], r["prompt_index"], r["response_index"], r["length"]): r["score"] for r in scores}
    draws = np.random.default_rng(manifest["analysis"]["bootstrap_seed"]).integers(0, 50, size=(2000, 50))
    results, contrasts, sensitivity, diagnostics = [], [], [], []
    vectors = {}
    for length in manifest["primary_lengths"]:
        null_bleu = np.array([metrics[("null", length, i)]["self_bleu"] for i in range(50)])
        for method in METHODS:
            row_metrics = [metrics[(method, length, i)] for i in range(50)]
            bleu = np.array([r["self_bleu"] for r in row_metrics])
            row = {"method": method, "length": length, "self_bleu": paired_interval(bleu, draws),
                   "self_bleu_minus_null": paired_interval(bleu-null_bleu, draws)}
            if method != "null":
                wm = np.array([[by_key[(method, method, i, r, length)]["decision"] for r in (0, 1)] for i in range(50)], dtype=float)
                null = np.array([[by_key[(method, "null", i, r, length)]["decision"] for r in (0, 1)] for i in range(50)], dtype=float)
                pvalues = [by_key[(method, method, i, r, length)]["p_value"] for i in range(50) for r in (0, 1)]
                available = [-np.log10(max(float(p), 1e-300)) for p in pvalues if p is not None]
                row.update(tpr=paired_interval(wm.mean(1), draws), detected=int(wm.sum()), responses=100,
                           fresh_null_false_positives=int(null.sum()), fresh_null_responses=100,
                           fresh_null_prompt_clusters=50,
                           abstained=sum(p is None for p in pvalues),
                           negative_log10_p_median=float(np.median(available)) if available else None)
                vectors[(method, length)] = (bleu, wm.mean(1))
            row["diagnostics"] = {k: float(np.mean([r[k] for r in row_metrics])) for k in
                                  ("self_bleu_token_ids", "exact_token_duplicate", "exact_text_duplicate", "repetition_rate", "distinct_3", "base_model_nll")}
            row["empty_decoded_responses"] = sum(r["empty_decoded_responses"] for r in row_metrics)
            results.append(row)
        for baseline in ("textseal", "synthid_text", "gumbel_max"):
            a, b = vectors[("online_prc", length)], vectors[(baseline, length)]
            contrasts.append({"comparison": f"PRC minus {NAMES[baseline]}", "length": length,
                              "self_bleu_difference": paired_interval(a[0]-b[0], draws),
                              "tpr_difference": paired_interval(a[1]-b[1], draws)})
        for detector in ("synthid_text", "synthid_historical_mask"):
            wm = np.array([[by_key[(detector, "synthid_text", i, r, length)]["decision"] for r in (0, 1)] for i in range(50)], float)
            null = [by_key[(detector, "null", i, r, length)]["decision"] for i in range(50) for r in (0, 1)]
            shared = [s for s in scores if s["detector"] == detector and s["method"] == "shared_null" and s["length"] == length]
            sensitivity.append({"detector": detector, "length": length, "detected": int(wm.sum()), "tpr": paired_interval(wm.mean(1), draws),
                                "fresh_null_false_positives": sum(null), "shared_null_false_positives": sum(r["score"]["decision"] for r in shared),
                                "mean_effective_watermarked_tokens": float(np.mean([by_key[(detector, "synthid_text", i, r, length)]["effective_tokens"] for i in range(50) for r in (0, 1)]))})
    # Historical shared prompt cohort: context only, not a held-out calibration set.
    old_path = ROOT / "outputs/comparison_redetect/baseline_comparisons.csv"
    old_rows = list(csv.DictReader(old_path.open()))
    shared_fpr = []
    for length in manifest["primary_lengths"]:
        for method in METHODS[:-1]:
            if method == "synthid_text":
                cohort = [r for r in scores if r["method"] == "shared_null" and r["detector"] == method and r["length"] == length]
                count = sum(r["score"]["decision"] for r in cohort)
                total = len(cohort)
            else:
                row = next(r for r in old_rows if r["Method"] == method and int(r["T"]) == length)
                field = row["FPR"].split(" ")[0]
                count, total = map(int, field.split("/"))
            lo = 0 if count == 0 else scipy.stats.beta.ppf(.025, count, total-count+1)
            hi = 1 if count == total else scipy.stats.beta.ppf(.975, count+1, total-count)
            shared_fpr.append({"method": method, "length": length, "false_positives": count, "responses": total, "binomial_ci95": [float(lo), float(hi)]})
    for length in manifest["prefix_lengths"]:
        for method in METHODS[:-1]:
            wm = [[by_key[(method, method, i, r, length)]["decision"] for r in (0, 1)] for i in range(50)]
            diagnostics.append({"method": method, "length": length, "detected": int(np.sum(wm)), "responses": 100,
                                "tpr": paired_interval(np.mean(wm, axis=1), draws)})
    summary = {"manifest_id": manifest["id"], "analysis": manifest["analysis"], "bleu_signature": diversity["bleu_signature"],
               "bootstrap_draws_sha256": digest(draws.tolist()), "results": results, "contrasts": contrasts,
               "synthid_mask_sensitivity": sensitivity, "historical_shared_null_fpr": shared_fpr, "prefix_detection": diagnostics,
               "verification": {k: v for k, v in verification.items() if k != "verified_files"},
               "sources": {name: sha(setup / name) for name in ("manifest.json", "diversity.json", "token_detection.json", "prc_report.json", "textseal_report.json", "reused_textseal.json")},
               "analysis_code_sha256": sha(__file__), "shared_fpr_source_sha256": sha(old_path),
               "metric_and_evidence_code_sha256": sha(Path(__file__).with_name("self_bleu_pilot_analysis.py")),
               "limitations": ["50 prompt clusters, two responses each; deterministic Gumbel duplicates are not independent observations.",
                               "Percentile bootstrap intervals at all-success/all-failure boundaries collapse; they do not establish perfect population detection or zero FPR.",
                               "Nominal analytic/approximate thresholds are not empirically matched false-positive rates.",
                               "The historical null cohort overlaps pilot prompts; it is not a disjoint calibration or audit sample.",
                               "This is an exploratory selected batch and a default-setting comparison, not a parameter-frontier or Bayesian-SynthID comparison."]}
    source_root = setup / "raw/analysis_source"
    source_root.mkdir(parents=True, exist_ok=True)
    for name in ("self_bleu_pilot_analysis.py", "self_bleu_pilot_results.py"):
        source = Path(__file__).with_name(name).read_bytes()
        destination = source_root / name
        if destination.exists() and destination.read_bytes() != source:
            raise ValueError("analysis source snapshot changed")
        destination.write_bytes(source)
    save(setup / "summary.json", summary)
    save(setup / "raw/combined_scores.json", scores)
    return summary


def plot(summary, setup):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False})
    fig = plt.figure(figsize=(12, 6))
    fig.subplots_adjust(left=.065, right=.985, top=.83, bottom=.29)
    outer = fig.add_gridspec(1, 2, wspace=.12)
    for col, length in enumerate((400, 1024)):
        grid = outer[col].subgridspec(1, 2, width_ratios=[5, 1], wspace=.055)
        left = fig.add_subplot(grid[0]); right = fig.add_subplot(grid[1], sharey=left)
        points = [r for r in summary["results"] if r["length"] == length]
        upper = max(r["self_bleu"]["ci95"][1] for r in points if r["method"] != "gumbel_max")
        left.set_xlim(0, max(.06, upper*1.16)); right.set_xlim(.985, 1.015)
        left.set_ylim(0, 1.08); right.set_xticks([1]); right.set_xticklabels(["1.00"])
        left.set_ylabel("Observed detection rate" if col == 0 else "")
        left.set_xlabel("Self-BLEU · lower is more diverse")
        left.set_title(f"{length:,} completion tokens", loc="left", fontweight="bold", pad=14)
        right.set_title("Gumbel", fontsize=9, pad=14)
        for axis in (left, right):
            axis.grid(axis="y", color="#e4e7ea", linewidth=.7)
            axis.set_axisbelow(True)
            axis.spines["right"].set_visible(False)
        right.spines["left"].set_visible(False)
        right.tick_params(axis="y", left=False, labelleft=False)
        left.spines["right"].set_visible(False)
        for axis, x in ((left, 1), (right, 0)):
            axis.plot([x-.015, x+.015], [-.012, .012], transform=axis.transAxes, color="#444", clip_on=False, lw=1)
        null = next(r for r in points if r["method"] == "null")["self_bleu"]
        left.axvspan(*null["ci95"], color="#596579", alpha=.09, zorder=0)
        left.axvline(null["mean"], color="#596579", linestyle="--", lw=1.2)
        if length == 1024:
            left.axhline(.9, color="#9ca3af", ls=":", lw=1)
            left.text(left.get_xlim()[1]*.97, .883, "90% screen", ha="right", color="#6b7280", fontsize=8)
        for row in points:
            method = row["method"]
            if method == "null":
                continue
            x, y = row["self_bleu"], row["tpr"]
            axis = right if method == "gumbel_max" else left
            xe = np.maximum(0, [[x["mean"]-x["ci95"][0]], [x["ci95"][1]-x["mean"]]])
            ye = np.maximum(0, [[y["mean"]-y["ci95"][0]], [y["ci95"][1]-y["mean"]]])
            axis.errorbar(x["mean"], y["mean"], xerr=xe, yerr=ye, fmt="o", ms=7, capsize=3,
                          color=COLORS[method], ecolor=COLORS[method], elinewidth=1.2)
    handles = [Line2D([], [], marker="o", linestyle="none", color=COLORS[m], label=NAMES[m]) for m in METHODS[:-1]]
    handles.append(Line2D([], [], linestyle="--", color="#596579", label="Ordinary-sampling diversity"))
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(.5, .16), ncols=5, frameon=False, fontsize=9)
    fig.suptitle("Stage A · Detectability versus repeated-response diversity", fontsize=15, fontweight="bold", y=.98)
    fig.text(.5, .065, "50 prompt pairs · marginal 95% prompt-bootstrap intervals · broken x-axis\nAll-success bootstrap bounds collapse; they do not establish perfect detection.\nNominal p < 0.001; false-positive rates are not empirically matched.", fontsize=9, ha="center", va="center")
    fig.savefig(setup / "detectability_self_bleu.png", dpi=200, bbox_inches="tight")
    fig.savefig(setup / "detectability_self_bleu.svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()
    verification = collect(args.setup, args.download)
    if args.collect_only:
        print(json.dumps({k: v for k, v in verification.items() if k != "verified_files"}, indent=2))
    else:
        summary = summarize(args.setup)
        plot(summary, args.setup)
        print(json.dumps({k: summary[k] for k in ("results", "contrasts", "synthid_mask_sensitivity", "historical_shared_null_fpr")}, indent=2))
