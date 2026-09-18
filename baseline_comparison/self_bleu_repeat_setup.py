"""Freeze repeat ablation inputs, validation gates and budget; never dispatch."""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import subprocess

from .self_bleu_config import digest
from .self_bleu_pilot import load_pairs
from .self_bleu_repeat import ARMS, arm_setting
from .self_bleu_validation import ROOT, RATE, save, sha

PILOT = ROOT / "outputs/self_bleu_pilot/stage_a_v2"
SETUP = ROOT / "outputs/self_bleu_repeat/setup_v1"
TIMEOUTS = {"synthid": 600, "other_generators": 900, "textseal_replay": 300}
NEW_CODE = ("baseline_comparison/self_bleu_repeat.py", "baseline_comparison/self_bleu_repeat_setup.py",
            "baseline_comparison/self_bleu_repeat_modal.py", "baseline_comparison/self_bleu_repeat_analysis.py")


def upstream_hashes():
    from synthid_text import logits_processing, hashing_function
    from .textseal_completion import load_upstream_detector
    import os
    load_upstream_detector(os.environ.get("TEXTSEAL_SOURCE_ROOT"))
    from textseal.watermarking import generator, core
    return {module.__name__: sha(inspect.getsourcefile(module))
            for module in (logits_processing, hashing_function, generator, core)}


def validate(manifest, root=ROOT):
    if digest({k: v for k, v in manifest.items() if k != "id"}) != manifest["id"]:
        raise ValueError("repeat manifest identity differs")
    if (manifest["arms"] != {arm: arm_setting(arm).identity() for arm in ARMS}
            or manifest["prompt_indices"] != list(range(50)) or manifest["seeds"] != [12345, 67890]
            or manifest["length"] != 1024 or manifest["control_tokens"] != 64
            or manifest["protocol"] != "completion_only_raw_abstain_v1"):
        raise ValueError("repeat ablation scope differs")
    cost = manifest["cost"]
    if (cost["timeouts"] != TIMEOUTS or cost["resource_usd_per_second"] != RATE
            or cost["total_reserved_with_prior_usd"] != cost["previous_planning_charge_usd"] + (sum(TIMEOUTS.values())+6)*RATE + .5
            or cost["total_reserved_with_prior_usd"] > 10):
        raise ValueError("repeat ablation exceeds initial allocation")
    for name, expected in manifest["code_sha256"].items():
        if sha(Path(root)/name) != expected:
            raise ValueError(f"ablation source changed: {name}")
    if sha(Path(root)/"prompts.jsonl") != manifest["prompt_sha256"]:
        raise ValueError("canonical prompts changed")


def prepare(output):
    pilot = json.loads((PILOT/"manifest.json").read_text())
    previous = json.loads((PILOT/"verification.json").read_text())
    if not previous["passed"] or previous["manifest_id"] != pilot["id"]:
        raise ValueError("Stage A reference did not pass")
    # Reject accidental modifications to the historical generation/detection paths.
    for name, expected in pilot["code_sha256"].items():
        if sha(ROOT/name) != expected:
            raise ValueError(f"historical implementation changed: {name}")
    batches, _ = load_pairs()
    refs = {}
    for method in ("synthid_text", "textseal", "gumbel_max"):
        for response in (0, 1):
            batch = batches[(method, response)]
            refs[f"{method}/{response}"] = {
                "batch_id": batch["manifest"]["batch_id"],
                "prefix_sha256": [digest(r["token_ids"][:64]) for r in batch["responses"]],
                "completion_sha256": [r["completion_sha256"] for r in batch["responses"]]}
    generation_report = ROOT/"outputs/self_bleu_validation/step3-v4/generation_report.json"
    old_runtime = json.loads(generation_report.read_text())["execution"]
    names = sorted(set(pilot["code_sha256"]) | set(NEW_CODE) |
                   {"baseline_comparison/self_bleu_generation.py", "baseline_comparison/self_bleu_validation_results.py", "watermark_expt.py"})
    previous_cost = previous["total_planning_charge_usd"]
    manifest = {
        "schema_version": 1, "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "protocol": pilot["protocol"], "pilot_id": pilot["id"], "model": pilot["model"],
        "prompt_indices": list(range(50)), "prompt_sha256": sha(ROOT/"prompts.jsonl"),
        "length": 1024, "control_tokens": 64, "seeds": [12345, 67890],
        "primary_lengths": [400, 1024], "prefix_lengths": pilot["prefix_lengths"],
        "arms": {arm: arm_setting(arm).identity() for arm in ARMS}, "references": refs,
        "generation_runtime": {k: v for k, v in old_runtime.items() if k != "modal_image_id"},
        "textseal_runtime": pilot["textseal_runtime"], "analysis": pilot["analysis"],
        "analysis_contract": {
            "primary_contrasts": "new policy minus original policy, paired by prompt",
            "decision_thresholds": "original method-native nominal p < 0.001, unchanged",
            "detector_masks": "unchanged within method; SynthID context mask, TextSeal/Gumbel native tuple mask",
            "nulls": "reuse the 100 fresh ordinary-sampling slots from Stage A; no new calibration",
            "repeat_reporting": "within-response fallback positions/counts, first trigger, and token divergence from saved response",
            "scope": "generation-policy ablation; native context initialization retained, not full implementation harmonization",
        },
        "upstream_sha256": upstream_hashes(), "code_sha256": {name: sha(ROOT/name) for name in names},
        "reference_files": {str(path.relative_to(ROOT)): sha(path) for path in
            (PILOT/"manifest.json", PILOT/"verification.json", PILOT/"inputs.json", PILOT/"summary.json",
             PILOT/"diversity.json", PILOT/"token_detection.json", PILOT/"prc_report.json",
             PILOT/"textseal_report.json", PILOT/"reused_textseal.json", generation_report)},
        "cost": {"timeouts": TIMEOUTS, "resource_usd_per_second": RATE,
                 "previous_planning_charge_usd": previous_cost,
                 "new_timeout_reservation_usd": (sum(TIMEOUTS.values())+6)*RATE,
                 "new_overhead_allowance_usd": .5,
                 "total_reserved_with_prior_usd": previous_cost+(sum(TIMEOUTS.values())+6)*RATE+.5,
                 "initial_allocation_usd": 10, "total_study_ceiling_usd": 200,
                 "pricing_source": "https://modal.com/pricing", "pricing_checked": "2026-09-18",
                 "dispatch": "three explicit sequential stages, one H100 each, retries=0; no automatic dispatch or reruns"},
    }
    manifest["id"] = digest(manifest)
    validate(manifest)
    save(output/"manifest.json", manifest)
    return {"id": manifest["id"], "status": "prepared; no GPU jobs launched",
            "new_responses": {arm: 100 for arm in ARMS}, "cost": manifest["cost"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SETUP)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output), indent=2))
