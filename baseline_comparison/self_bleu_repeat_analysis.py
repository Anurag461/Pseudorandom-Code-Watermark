"""Collect staged ablation artifacts and compare paired policies, without GPUs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath

import numpy as np

from .self_bleu_config import digest
from .self_bleu_pilot_analysis import paired_interval
from .self_bleu_repeat_setup import PILOT, ROOT, SETUP, validate, upstream_hashes
from .self_bleu_validation import save, sha


def collect(setup, stage, download=False):
    manifest = json.loads((setup/"manifest.json").read_text())
    validate(manifest)
    stages = ["synthid"] if stage == "synthid" else ["synthid", "other_generators", "textseal_replay"]
    volume = None
    if download:
        import modal
        volume = modal.Volume.from_name("prc-completion-only", create_if_missing=False)
    files, reports = {}, {}
    for name in stages:
        report_path = setup/f"{name}_report.json"
        remote = f"self_bleu_repeat/{manifest['id']}/{name}"
        if download and not report_path.exists():
            report_path.write_bytes(b"".join(volume.read_file(f"{remote}/report.json")))
        report = json.loads(report_path.read_text())
        if not report["passed"] or report["manifest_id"] != manifest["id"] or report["stage"] != name:
            raise ValueError("repeat stage has not passed")
        reports[name] = report
        for relative, expected in report["files"].items():
            parts = PurePosixPath(relative)
            if parts.is_absolute() or ".." in parts.parts:
                raise ValueError("unsafe artifact path")
            path = setup/"raw"/name/relative
            if download and not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"".join(volume.read_file(f"{remote}/{relative}")))
            if sha(path) != expected:
                raise ValueError("repeat artifact checksum differs")
            files[str(path.relative_to(setup))] = expected
    return manifest, reports, files


def paired_contrast(new_bleu, old_bleu, new_detection, old_detection, draws):
    return {"self_bleu_difference": paired_interval(np.asarray(new_bleu)-old_bleu, draws),
            "tpr_difference": paired_interval(np.asarray(new_detection)-old_detection, draws)}


def analyze(setup, stage, tokenizer_path, download=False):
    import torch
    import sacrebleu
    from sacrebleu.metrics import BLEU
    from tokenizers import Tokenizer
    from .official import synthid_processor, official_gumbel_scores
    from .scoring import deduplicated_positions, synthid_normal_test, gumbel_gamma_test
    from .self_bleu_pilot_results import score_rows
    manifest, reports, files = collect(setup, stage, download)
    if upstream_hashes() != manifest["upstream_sha256"]:
        raise ValueError("upstream token-scoring sources changed")
    for name, expected in manifest["reference_files"].items():
        if sha(ROOT/name) != expected:
            raise ValueError(f"Stage A reference changed: {name}")
    if sha(tokenizer_path) != manifest["model"]["tokenizer_sha256"] or sacrebleu.__version__ != "2.4.3":
        raise ValueError("tokenizer or Self-BLEU implementation differs")
    decoder = Tokenizer.from_file(str(tokenizer_path))
    metric = BLEU(tokenize="13a", smooth_method="exp", effective_order=True, lowercase=False)
    old_metrics = {(r["method"], r["length"], r["prompt_index"]): r for r in json.loads((PILOT/"diversity.json").read_text())["rows"]}
    old_scores = {(r["detector"], r["method"], r["prompt_index"], r["response_index"], r["length"]): r["score"] for r in score_rows(PILOT)}
    old_inputs = {(r["method"], r["prompt_index"], r["response_index"]): r for r in json.loads((PILOT/"inputs.json").read_text())}
    draws = np.random.default_rng(manifest["analysis"]["bootstrap_seed"]).integers(0, 50, (2000, 50))
    arms = ["synthid_off"] if stage == "synthid" else list(manifest["arms"])
    summaries, metric_rows, score_records, diagnostics = [], [], [], []
    ts = {r["response_id"]: r for r in reports.get("textseal_replay", {}).get("rows", [])}
    for arm in arms:
        generation_stage = "synthid" if arm == "synthid_off" else "other_generators"
        method = manifest["arms"][arm]["method"]
        rows = {}
        for response in (0, 1):
            batch = json.loads((setup/"raw"/generation_stage/"batches"/f"{arm}_r{response}.json").read_text())
            if batch["manifest"]["setting"] != manifest["arms"][arm] or len(batch["responses"]) != 50:
                raise ValueError("ablation generation configuration or coverage differs")
            for row in batch["responses"]:
                key = (row["prompt_index"], row["response_index"])
                if key in rows or key[1] != response or digest(row["token_ids"]) != row["completion_sha256"]:
                    raise ValueError("duplicate or mismatched ablation response")
                rows[key] = row
        if set(rows) != {(i, r) for i in range(50) for r in (0, 1)}:
            raise ValueError("missing ablation pair")
        scored = {}
        for key, row in rows.items():
            ids = row["token_ids"]
            if method == "synthid_text":
                processor = synthid_processor("cpu")
                tensor = torch.tensor([ids])
                values = processor.compute_g_values(tensor)[0].numpy()
                mask = processor.compute_context_repetition_mask(tensor)[0].numpy().astype(bool)
            for length in manifest["prefix_lengths"]:
                if method == "synthid_text":
                    score = synthid_normal_test(values[:length-3][mask[:length-3]])
                elif method == "gumbel_max":
                    positions = deduplicated_positions(ids[:length])
                    score = gumbel_gamma_test(official_gumbel_scores(ids, positions))
                else:
                    result = ts[row["response_id"]]
                    if result["completion_sha256"] != row["completion_sha256"]:
                        raise ValueError("TextSeal detector scored different text")
                    raw = result["results"][str(length)]
                    if raw["completion_sha256"] != digest(ids[:length]):
                        raise ValueError("TextSeal detector prefix differs")
                    score = raw["comparison"]
                scored[(*key, length)] = score
                score_records.append({"arm": arm, "response_id": row["response_id"], "length": length, "score": score})
            old_ids = old_inputs[(method, *key)]["token_ids"]
            diagnostics.append({"arm": arm, "prompt_index": key[0], "response_index": key[1],
                "first_token_divergence": next((j for j, (a,b) in enumerate(zip(ids, old_ids)) if a != b), None),
                "first_fallback_position": row["generation_diagnostics"]["first_fallback_position"],
                "first_repeated_context_position": next((j for j, x in enumerate(row["generation_diagnostics"]["repeated_context"]) if x), None),
                "fallback_counts": {str(n): sum(row["generation_diagnostics"]["fallback_applied"][:n]) for n in manifest["primary_lengths"]}})
        for length in manifest["primary_lengths"]:
            bleu, detections = [], []
            for i in range(50):
                texts = [decoder.decode(rows[(i,r)]["token_ids"][:length], skip_special_tokens=True) for r in (0,1)]
                b = (metric.sentence_score(texts[0], [texts[1]]).score + metric.sentence_score(texts[1], [texts[0]]).score)/200
                detected = [bool(scored[(i,r,length)]["decision"]) for r in (0,1)]
                bleu.append(b); detections.append(np.mean(detected))
                metric_rows.append({"arm": arm, "length": length, "prompt_index": i, "self_bleu": b, "detected": detected})
            old_b = np.array([old_metrics[(method,length,i)]["self_bleu"] for i in range(50)])
            old_d = np.array([np.mean([old_scores[(method,method,i,r,length)]["decision"] for r in (0,1)]) for i in range(50)])
            summaries.append({"arm": arm, "length": length, "self_bleu": paired_interval(bleu,draws),
                "tpr": paired_interval(detections,draws), "detected": int(round(sum(detections)*2)),
                "new_minus_original": paired_contrast(bleu,old_b,detections,old_d,draws)})
    cost = manifest["cost"]["previous_planning_charge_usd"] + .5 + sum(r["measured_resource_usd"] for r in reports.values())
    output = {"manifest_id": manifest["id"], "stage": stage, "results": summaries,
              "bootstrap_draws_sha256": digest(draws.tolist()), "bleu_signature": str(metric.get_signature()),
              "planning_charge_usd": cost, "verified_files": files,
              "limitations": ["50 paired prompt clusters; nominal thresholds, not matched empirical FPR.",
                              "All-success bootstrap intervals do not prove perfect detection.",
                              "Within-method generation-policy contrasts; method-native detector masks and context initialization retained."]}
    save(setup/f"{stage}_analysis.json", output)
    save(setup/f"raw/{stage}_metric_rows.json", metric_rows)
    save(setup/f"raw/{stage}_scores.json", score_records)
    save(setup/f"raw/{stage}_diagnostics.json", diagnostics)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", type=Path, default=SETUP)
    parser.add_argument("--stage", choices=("synthid", "all"), required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    result = analyze(args.setup,args.stage,args.tokenizer,args.download)
    print(json.dumps({k:v for k,v in result.items() if k != "verified_files"}, indent=2))
