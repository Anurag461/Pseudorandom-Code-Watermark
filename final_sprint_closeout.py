#!/usr/bin/env python3
"""Build and validate the compact final sprint closeout artifacts.

This script is deliberately cache-only and dependency-light.  It reads already
settled ledgers and summary tables, writes one non-overlapping sprint cost
ledger, and renders the two required scientific figures as vector SVG.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from decimal import Decimal
from html import escape
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
FULL_DIR = OUTPUTS / "controlled_baseline_full" / "qwen3-8b-batch50-validation-20260823-v1"

ETA005_LEDGER = OUTPUTS / "online_8b_eta005_cost_ledger.csv"
ETA010_LEDGER = OUTPUTS / "online_8b_eta010_cost_ledger.csv"
MATCHED_14B_LEDGER = OUTPUTS / "online_14b_lower_eta_full_sweeps_cost_ledger.csv"
FIXED_ONLINE_LEDGER = OUTPUTS / "fixed_vs_online_cache_only_cost_ledger.csv"
SMOKE_LEDGER = OUTPUTS / "controlled_baseline_smoke_cost_ledger.csv"
DIAGNOSTIC_LEDGER = OUTPUTS / "controlled_baseline_diagnostic_cost_ledger.csv"
FULL_BASELINE_LEDGER = FULL_DIR / "controlled_baseline_full_cost_ledger.csv"
PROXY_LEDGER = OUTPUTS / "proxy_8b_cost_ledger.csv"
PREFIX_SUMMARY = FULL_DIR / "controlled_baseline_full_prefix_summary.csv"
QUALITY_SUMMARY = FULL_DIR / "controlled_baseline_full_quality_summary.csv"

FINAL_COST_LEDGER = OUTPUTS / "final_sprint_cost_ledger.csv"
TPR_FIGURE = OUTPUTS / "final_tpr_vs_prefix.svg"
QUALITY_FIGURE = OUTPUTS / "final_detectability_vs_quality.svg"
WORKBOOK_DATA = Path("/private/tmp/prc_final_sprint_closeout_workbook_data.json")
VALIDATION_MANIFEST = OUTPUTS / "final_sprint_validation_manifest.json"

D = Decimal
ZERO = D("0")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def decimal(row: dict[str, str], key: str) -> Decimal:
    value = row.get(key, "").strip()
    if not value:
        return ZERO
    return D(value)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def money(value: Decimal) -> str:
    return f"{value.quantize(D('0.00000001')):.8f}"


def total_row(path: Path, phase_key: str) -> dict[str, str]:
    rows = read_csv(path)
    matches = [row for row in rows if row.get(phase_key, "").strip() == "TOTAL"]
    require(len(matches) == 1, f"{path}: expected exactly one TOTAL row")
    return matches[0]


def sum_resource_rows(
    rows: Iterable[dict[str, str]],
    gpu_key: str,
    cpu_key: str,
    memory_key: str,
    total_key: str,
) -> tuple[Decimal, Decimal, Decimal, Decimal]:
    rows = list(rows)
    gpu = sum((decimal(row, gpu_key) for row in rows), ZERO)
    cpu = sum((decimal(row, cpu_key) for row in rows), ZERO)
    memory = sum((decimal(row, memory_key) for row in rows), ZERO)
    total = sum((decimal(row, total_key) for row in rows), ZERO)
    require(total == gpu + cpu + memory, "resource components do not sum to provider total")
    return gpu, cpu, memory, total


def build_cost_rows() -> list[dict[str, str]]:
    eta005 = total_row(ETA005_LEDGER, "stage")
    eta010 = total_row(ETA010_LEDGER, "stage")

    eta005_costs = tuple(
        decimal(eta005, key)
        for key in (
            "provider_gpu_cost_usd",
            "provider_cpu_cost_usd",
            "provider_memory_cost_usd",
            "provider_total_cost_usd",
        )
    )
    eta010_costs = tuple(
        decimal(eta010, key)
        for key in (
            "provider_gpu_cost_usd",
            "provider_cpu_cost_usd",
            "provider_memory_cost_usd",
            "provider_total_cost_usd",
        )
    )
    require(eta005_costs == (D("1.12447738"), D("0.07216837"), D("0.00994709"), D("1.20659284")), "eta=.05 billing mismatch")
    require(eta010_costs == (D("4.61323391"), D("0.11516370"), D("0.02304711"), D("4.75144472")), "eta=.10 billing mismatch")

    matched_rows = [
        row
        for row in read_csv(MATCHED_14B_LEDGER)
        if row["phase"] == "matched_0p6b_boundary_cache_only"
    ]
    require(len(matched_rows) == 2, "expected two matched 14B cache-only rows")
    matched_costs = sum_resource_rows(
        matched_rows,
        "provider_gpu_cost_usd",
        "provider_cpu_cost_usd",
        "provider_memory_cost_usd",
        "provider_total_cost_usd",
    )
    require(matched_costs == (ZERO, D("0.01610406"), D("0.00144117"), D("0.01754523")), "14B matched-audit billing mismatch")

    fixed_rows = read_csv(FIXED_ONLINE_LEDGER)
    fixed_total = sum((decimal(row, "provider_cost_usd") for row in fixed_rows), ZERO)
    fixed_cpu = D("0.04131978")
    fixed_memory = D("0.00373830")
    require(fixed_total == fixed_cpu + fixed_memory == D("0.04505808"), "fixed-vs-online billing mismatch")
    require(all(row["execution_mode"] == "cache_only" for row in fixed_rows), "fixed-vs-online ledger is not cache-only")
    require(all(row["gpu_worker_launched"] == "false" for row in fixed_rows), "fixed-vs-online ledger launched a GPU")

    smoke = total_row(SMOKE_LEDGER, "timestamp_utc")
    diagnostic = total_row(DIAGNOSTIC_LEDGER, "timestamp_utc")
    smoke_costs = tuple(
        decimal(smoke, key)
        for key in ("provider_gpu_cost_usd", "provider_cpu_cost_usd", "provider_memory_cost_usd", "provider_total_cost_usd")
    )
    diagnostic_costs = tuple(
        decimal(diagnostic, key)
        for key in ("provider_gpu_cost_usd", "provider_cpu_cost_usd", "provider_memory_cost_usd", "provider_total_cost_usd")
    )
    full_rows = [
        row
        for row in read_csv(FULL_BASELINE_LEDGER)
        if row["resource"] != "Total" and row["phase"] != "controlled_full_run"
    ]
    full_gpu = sum((decimal(row, "cost_usd") for row in full_rows if row["resource"] == "H100"), ZERO)
    full_cpu = sum((decimal(row, "cost_usd") for row in full_rows if row["resource"] == "CPU"), ZERO)
    full_memory = sum((decimal(row, "cost_usd") for row in full_rows if row["resource"] == "Memory"), ZERO)
    full_total = full_gpu + full_cpu + full_memory
    require(full_total == D("2.65877380"), "controlled full-run billing mismatch")
    baseline_costs = tuple(smoke_costs[i] + diagnostic_costs[i] + (full_gpu, full_cpu, full_memory, full_total)[i] for i in range(4))
    require(baseline_costs == (D("3.55130394"), D("0.28872017"), D("0.38359930"), D("4.22362341")), "controlled baseline cumulative billing mismatch")

    proxy_rows = read_csv(PROXY_LEDGER)
    proxy_costs = sum_resource_rows(proxy_rows, "gpu_cost_usd", "cpu_cost_usd", "memory_cost_usd", "total_cost_usd")
    require(proxy_costs == (D("2.84171420"), D("0.32920227"), D("0.04691993"), D("3.21783640")), "proxy billing mismatch")

    phases = [
        ("online_8b_eta005_boundary", "8B online PRC eta=.05 boundary campaign", eta005_costs, ETA005_LEDGER, "Generation, exact refinement, and selected-boundary audit; no null generation."),
        ("online_8b_eta010_boundary", "8B online PRC eta=.10 boundary campaign", eta010_costs, ETA010_LEDGER, "Generation, exact refinement, and selected-boundary audit; no null generation."),
        ("matched_14b_cache_only", "14B matched-reference cache-only audits", matched_costs, MATCHED_14B_LEDGER, "Two cache-only rows at n=448 and n=800; zero GPU cost and zero generated tokens."),
        ("fixed_vs_online", "Fixed-versus-online paired analysis", (ZERO, fixed_cpu, fixed_memory, fixed_total), FIXED_ONLINE_LEDGER, "Five cache-only audits plus paired inference; zero GPU cost and zero generated tokens."),
        ("controlled_8b_baselines", "Controlled 8B baseline integration through full run", baseline_costs, FULL_BASELINE_LEDGER, "Integration, five-prompt smoke, diagnostic, 500-prompt generation, and scoring; constituent totals are not double-counted."),
        ("proxy_0p6b_detector", "Cache-only 0.6B proxy detector analysis", proxy_costs, PROXY_LEDGER, "All-eta PRC portability plus common-length sensitivity; zero text generation."),
    ]

    result: list[dict[str, str]] = []
    for phase_id, label, costs, source, note in phases:
        gpu, cpu, memory, total = costs
        require(total == gpu + cpu + memory, f"{phase_id}: components do not sum")
        result.append(
            {
                "phase_id": phase_id,
                "phase": label,
                "billing_status": "exact_provider_reconciled",
                "gpu_cost_usd": money(gpu),
                "cpu_cost_usd": money(cpu),
                "memory_cost_usd": money(memory),
                "total_cost_usd": money(total),
                "source_ledger": str(source.relative_to(ROOT)),
                "notes": note,
            }
        )

    gpu = sum((D(row["gpu_cost_usd"]) for row in result), ZERO)
    cpu = sum((D(row["cpu_cost_usd"]) for row in result), ZERO)
    memory = sum((D(row["memory_cost_usd"]) for row in result), ZERO)
    total = sum((D(row["total_cost_usd"]) for row in result), ZERO)
    require((gpu, cpu, memory, total) == (D("12.13072943"), D("0.86267835"), D("0.46869290"), D("13.46210068")), "grand sprint total mismatch")
    result.append(
        {
            "phase_id": "TOTAL",
            "phase": "Scoped final two-day sprint total",
            "billing_status": "exact_provider_reconciled",
            "gpu_cost_usd": money(gpu),
            "cpu_cost_usd": money(cpu),
            "memory_cost_usd": money(memory),
            "total_cost_usd": money(total),
            "source_ledger": "all phase ledgers above",
            "notes": "Non-overlapping total before workspace credits; excludes prior eta=.15/.20 campaigns and other work outside this final sprint scope.",
        }
    )
    return result


def write_cost_ledger(rows: list[dict[str, str]]) -> None:
    with FINAL_COST_LEDGER.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def line(x1: float, y1: float, x2: float, y2: float, **attrs: object) -> str:
    properties = " ".join(f'{key.replace("_", "-")}="{value}"' for key, value in attrs.items())
    return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" {properties}/>'


def text(x: float, y: float, value: str, **attrs: object) -> str:
    properties = " ".join(f'{key.replace("_", "-")}="{value}"' for key, value in attrs.items())
    return f'<text x="{x:.2f}" y="{y:.2f}" {properties}>{escape(value)}</text>'


def circle(x: float, y: float, radius: float, **attrs: object) -> str:
    properties = " ".join(f'{key.replace("_", "-")}="{value}"' for key, value in attrs.items())
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" {properties}/>'


def svg_document(width: int, height: int, body: list[str], description: str) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
            '<title id="title">Final PRC sprint scientific figure</title>',
            f'<desc id="desc">{escape(description)}</desc>',
            '<rect width="100%" height="100%" fill="#ffffff"/>',
            '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;fill:#202124}.axis{stroke:#4b5563;stroke-width:1.2}.grid{stroke:#d8dee7;stroke-width:1}.label{font-size:14px}.tick{font-size:12px;fill:#4b5563}.title{font-size:20px;font-weight:600}.subtitle{font-size:13px;fill:#4b5563}</style>',
            *body,
            "</svg>",
            "",
        ]
    )


def build_tpr_figure(prefix_rows: list[dict[str, str]]) -> None:
    method_order = ["online_prc", "textseal", "synthid_text", "gumbel_max"]
    labels = {
        "online_prc": "Online PRC eta=.05",
        "textseal": "TextSeal alpha=.1",
        "synthid_text": "SynthID depth 10",
        "gumbel_max": "Gumbel-Max",
    }
    colors = {"online_prc": "#2563eb", "textseal": "#dc2626", "synthid_text": "#059669", "gumbel_max": "#7c3aed"}
    dashes = {"online_prc": "", "textseal": "8 4", "synthid_text": "3 3", "gumbel_max": "12 4 2 4"}
    prefixes = [128, 256, 400, 512, 768, 1024]
    require(len(prefix_rows) == 24, "baseline prefix summary must contain 24 rows")
    require({row["method"] for row in prefix_rows} == set(method_order), "baseline method coverage mismatch")
    require({int(row["prefix_length"]) for row in prefix_rows} == set(prefixes), "baseline prefix coverage mismatch")

    width, height = 960, 560
    left, right, top, bottom = 88, 35, 76, 78
    plot_w, plot_h = width - left - right, height - top - bottom
    xmin, xmax, ymin, ymax = 128, 1024, 0.0, 1.02
    sx = lambda value: left + (value - xmin) / (xmax - xmin) * plot_w
    sy = lambda value: top + (ymax - value) / (ymax - ymin) * plot_h
    body = [text(left, 30, "Detection power by exact causal prefix", class_="title"), text(left, 52, "500 Qwen3-8B prompts; nominal decision rule p < 10⁻³", class_="subtitle")]
    for y_value in (0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0):
        y = sy(y_value)
        body.append(line(left, y, width - right, y, class_="grid"))
        body.append(text(left - 12, y + 4, f"{y_value:.1f}", class_="tick", text_anchor="end"))
    for x_value in prefixes:
        x = sx(x_value)
        body.append(line(x, top, x, height - bottom, class_="grid"))
        body.append(text(x, height - bottom + 22, str(x_value), class_="tick", text_anchor="middle"))
    body.extend(
        [
            line(left, top, left, height - bottom, class_="axis"),
            line(left, height - bottom, width - right, height - bottom, class_="axis"),
            text(left + plot_w / 2, height - 28, "Continuation prefix length (tokens)", class_="label", text_anchor="middle"),
            f'<text x="22" y="{top + plot_h / 2:.2f}" class="label" text-anchor="middle" transform="rotate(-90 22 {top + plot_h / 2:.2f})">True-positive rate</text>',
            line(left, sy(0.9), width - right, sy(0.9), stroke="#111827", stroke_width="1.5", stroke_dasharray="4 5"),
            text(width - right - 5, sy(0.9) - 7, "90%", class_="tick", text_anchor="end"),
        ]
    )
    by_method = {method: [] for method in method_order}
    for row in prefix_rows:
        by_method[row["method"]].append((int(row["prefix_length"]), float(row["tpr"])))
    for method in method_order:
        points = sorted(by_method[method])
        coords = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in points)
        dash = f' stroke-dasharray="{dashes[method]}"' if dashes[method] else ""
        body.append(f'<polyline points="{coords}" fill="none" stroke="{colors[method]}" stroke-width="2.8"{dash}/>' )
        for x_value, y_value in points:
            body.append(circle(sx(x_value), sy(y_value), 4.5, fill="#ffffff", stroke=colors[method], stroke_width="2.4"))
    legend_x, legend_y = left + 18, top + 18
    for index, method in enumerate(method_order):
        y = legend_y + index * 24
        dash = dashes[method]
        body.append(line(legend_x, y, legend_x + 34, y, stroke=colors[method], stroke_width="2.8", stroke_dasharray=dash))
        body.append(circle(legend_x + 17, y, 3.8, fill="#ffffff", stroke=colors[method], stroke_width="2"))
        body.append(text(legend_x + 44, y + 4, labels[method], class_="tick"))
    TPR_FIGURE.write_text(svg_document(width, height, body, "TPR at six exact causal prefixes for online PRC, TextSeal, SynthID-Text, and Gumbel-Max."), encoding="utf-8")


def build_quality_figure(prefix_rows: list[dict[str, str]], quality_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    final_tpr = {row["method"]: float(row["tpr"]) for row in prefix_rows if int(row["prefix_length"]) == 1024}
    watermarked = {row["method"]: row for row in quality_rows if row["sample_type"] == "watermarked"}
    null_rows = [row for row in quality_rows if row["sample_type"] == "null"]
    require(len(watermarked) == 4, "expected four watermarked quality rows")
    require(len(null_rows) == 4, "expected four null quality rows")
    null_distinct = float(null_rows[0]["median_distinct_3"])
    null_repetition = float(null_rows[0]["median_repetition_rate"])
    require(all(float(row["median_distinct_3"]) == null_distinct for row in null_rows), "shared-null distinct-3 mismatch")
    require(all(float(row["median_repetition_rate"]) == null_repetition for row in null_rows), "shared-null repetition mismatch")

    labels = {"online_prc": "PRC", "textseal": "TextSeal", "synthid_text": "SynthID", "gumbel_max": "Gumbel"}
    colors = {"online_prc": "#2563eb", "textseal": "#dc2626", "synthid_text": "#059669", "gumbel_max": "#7c3aed"}
    points: list[dict[str, object]] = []
    for method, row in watermarked.items():
        points.append(
            {
                "method": method,
                "label": labels[method],
                "tpr": final_tpr[method],
                "median_distinct_3": float(row["median_distinct_3"]),
                "median_repetition_rate": float(row["median_repetition_rate"]),
                "median_base_model_nll": float(row["median_base_model_nll"]),
                "color": colors[method],
            }
        )

    width, height = 1100, 520
    top, bottom, panel_w, gap = 82, 75, 430, 95
    left1, left2 = 82, 82 + panel_w + gap
    plot_h = height - top - bottom
    ymin, ymax = 0.94, 1.005
    sy = lambda value: top + (ymax - value) / (ymax - ymin) * plot_h
    body = [text(82, 30, "Detectability must be interpreted with diversity", class_="title"), text(82, 52, "T=1024 medians over 500 Qwen3-8B continuations; null quality shown as a reference line", class_="subtitle")]

    panels = [
        (left1, 0.30, 1.00, "Median distinct-3 (higher is better)", "median_distinct_3", null_distinct),
        (left2, 0.00, 0.68, "Median repeated-4-gram rate (lower is better)", "median_repetition_rate", null_repetition),
    ]
    for left, xmin, xmax, xlabel, key, null_value in panels:
        sx = lambda value, left=left, xmin=xmin, xmax=xmax: left + (value - xmin) / (xmax - xmin) * panel_w
        for y_value in (0.94, 0.96, 0.98, 1.00):
            y = sy(y_value)
            body.append(line(left, y, left + panel_w, y, class_="grid"))
            body.append(text(left - 10, y + 4, f"{y_value:.2f}", class_="tick", text_anchor="end"))
        tick_count = 7
        for index in range(tick_count + 1):
            value = xmin + (xmax - xmin) * index / tick_count
            x = sx(value)
            body.append(line(x, top, x, height - bottom, class_="grid"))
            body.append(text(x, height - bottom + 21, f"{value:.2f}", class_="tick", text_anchor="middle"))
        body.extend(
            [
                line(left, top, left, height - bottom, class_="axis"),
                line(left, height - bottom, left + panel_w, height - bottom, class_="axis"),
                text(left + panel_w / 2, height - 28, xlabel, class_="label", text_anchor="middle"),
                line(sx(null_value), top, sx(null_value), height - bottom, stroke="#4b5563", stroke_width="1.4", stroke_dasharray="4 4"),
                text(
                    sx(null_value) - 6 if key == "median_distinct_3" else sx(null_value) + 6,
                    height - bottom - 10,
                    f"null {null_value:.3f}",
                    class_="tick",
                    text_anchor="end" if key == "median_distinct_3" else "start",
                ),
            ]
        )
        label_offsets = {
            "median_distinct_3": {
                "online_prc": (-10, 20, "end"),
                "textseal": (10, -10, "start"),
                "synthid_text": (-10, 22, "end"),
                "gumbel_max": (12, 22, "start"),
            },
            "median_repetition_rate": {
                "online_prc": (10, 20, "start"),
                "textseal": (10, -10, "start"),
                "synthid_text": (10, -10, "start"),
                "gumbel_max": (-10, 22, "end"),
            },
        }
        for point in points:
            x = sx(float(point[key]))
            y = sy(float(point["tpr"]))
            body.append(circle(x, y, 7, fill=str(point["color"]), stroke="#ffffff", stroke_width="1.8"))
            dx, dy, anchor = label_offsets[key][str(point["method"])]
            body.append(text(x + dx, y + dy, str(point["label"]), class_="tick", text_anchor=anchor))
    body.append(f'<text x="24" y="{top + plot_h / 2:.2f}" class="label" text-anchor="middle" transform="rotate(-90 24 {top + plot_h / 2:.2f})">TPR at p &lt; 10⁻³</text>')
    QUALITY_FIGURE.write_text(svg_document(width, height, body, "At T=1024, SynthID and PRC remain close to null diversity while TextSeal and Gumbel-Max show much higher repetition despite saturated detection."), encoding="utf-8")
    return points


def write_workbook_data(cost_rows: list[dict[str, str]], prefix_rows: list[dict[str, str]], quality_points: list[dict[str, object]]) -> None:
    boundary_rows = []
    expected = {
        ("0.05", "640"): ("exact", "452/500 (90.4%)", "(639,640]"),
        ("0.1", "1407"): ("exact", "451/500 (90.2%)", "(1406,1407]"),
        ("0.15", "4096"): ("censored ceiling", "448/500 (89.6%)", "n90 > 4096"),
        ("0.2", "13088"): ("exact", "451/500 (90.2%)", "(13072,13088]"),
    }
    for (eta, n), (status, map_tpr, boundary) in expected.items():
        boundary_rows.append({"eta": eta, "selected_or_ceiling_n": int(n), "status": status, "map_tpr": map_tpr, "boundary": boundary})
    payload = {
        "cost_rows": cost_rows,
        "prefix_rows": prefix_rows,
        "quality_points": quality_points,
        "boundary_rows": boundary_rows,
    }
    WORKBOOK_DATA.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_dirty(path: Path) -> bool:
    result = subprocess.run(["git", "diff", "--quiet", "--", str(path.relative_to(ROOT))], cwd=ROOT, check=False)
    return result.returncode != 0


def write_manifest() -> None:
    owned = [
        ROOT / "FINAL_TWO_DAY_EXPERIMENT_PLAN.md",
        ROOT / "final_sprint_closeout.py",
        ROOT / "final_sprint_report.md",
        ROOT / "modal_online_8b_eta005_runbook.md",
        ROOT / "modal_online_8b_eta010_runbook.md",
        ETA005_LEDGER,
        ETA010_LEDGER,
        SMOKE_LEDGER,
        DIAGNOSTIC_LEDGER,
        OUTPUTS / "controlled_baseline_smoke_artifact_manifest.json",
        OUTPUTS / "controlled_baseline_diagnostic_artifact_manifest.json",
        FINAL_COST_LEDGER,
        TPR_FIGURE,
        QUALITY_FIGURE,
        OUTPUTS / "final_sprint_closeout.xlsx",
    ]
    sources = [
        ROOT / "hoeffding_results_summary.csv",
        ROOT / "online_causal_results_summary.csv",
        MATCHED_14B_LEDGER,
        FIXED_ONLINE_LEDGER,
        PREFIX_SUMMARY,
        QUALITY_SUMMARY,
        PROXY_LEDGER,
    ]
    require(all(path.exists() for path in owned + sources), "manifest input is missing")
    prompt_summary = read_csv(PREFIX_SUMMARY)
    require(len(prompt_summary) == 24, "final prefix summary row count changed")
    manifest = {
        "status": "passed_exact_billing_and_closeout_validation",
        "generated_utc": "2026-08-23T00:00:00Z",
        "generation_attempts": 0,
        "modal_compute_launched": False,
        "sprint_total_cost_usd": 13.46210068,
        "baseline_prefix_rows": 24,
        "baseline_methods": sorted({row["method"] for row in prompt_summary}),
        "baseline_prefixes": sorted({int(row["prefix_length"]) for row in prompt_summary}),
        "owned_artifacts": {str(path.relative_to(ROOT)): {"sha256": sha256(path), "bytes": path.stat().st_size} for path in owned},
        "source_artifacts": {str(path.relative_to(ROOT)): {"sha256": sha256(path), "bytes": path.stat().st_size} for path in sources},
        "preserved_user_work": {
            "path": "online_causal_results_summary.csv",
            "dirty_before_and_after_closeout": git_dirty(ROOT / "online_causal_results_summary.csv"),
            "treatment": "preserved verbatim; not staged or committed by closeout; fingerprint recorded as provenance only",
        },
        "limitations": [
            "PRC, TextSeal, SynthID-Text, and Gumbel-Max p-values use different calibration constructions.",
            "Five hundred nulls cannot tightly validate a nominal 0.1% false-positive rate.",
            "The eta=.15 PRC boundary is right-censored at 4096 tokens.",
            "TextSeal and Gumbel-Max are not quality-matched detection wins under the frozen Qwen3-8B decoding setting.",
        ],
    }
    VALIDATION_MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    cost_rows = build_cost_rows()
    write_cost_ledger(cost_rows)
    prefix_rows = read_csv(PREFIX_SUMMARY)
    quality_rows = read_csv(QUALITY_SUMMARY)
    build_tpr_figure(prefix_rows)
    quality_points = build_quality_figure(prefix_rows, quality_rows)
    write_workbook_data(cost_rows, prefix_rows, quality_points)
    if args.write_manifest:
        write_manifest()
    print(f"sprint_total_usd={cost_rows[-1]['total_cost_usd']}")
    print(f"prefix_rows={len(prefix_rows)}")
    print("generation_attempts=0")


if __name__ == "__main__":
    main()
