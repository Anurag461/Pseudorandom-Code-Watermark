"""Local Stage A metrics, official token evidence, and prompt-paired summaries."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path

import numpy as np

from .self_bleu_config import digest
from .self_bleu_pilot import SETUP, METHODS, load_pairs
from .self_bleu_validation import ROOT, save, sha
from .scoring import self_bleu_token_ids, quality_metrics, deduplicated_positions, gumbel_gamma_test, synthid_normal_test


def read_inputs(setup):
    manifest = json.loads((setup / "manifest.json").read_text())
    rows = json.loads((setup / "inputs.json").read_text())
    if digest(rows) != manifest["input_sha256"]:
        raise ValueError("pilot input changed")
    return manifest, rows


def diversity(setup, tokenizer_path):
    import sacrebleu
    from sacrebleu.metrics import BLEU
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast
    manifest, inputs = read_inputs(setup)
    if sacrebleu.__version__ != "2.4.3" or sha(tokenizer_path) != manifest["model"]["tokenizer_sha256"]:
        raise ValueError("metric version or tokenizer bytes differ")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    hf = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path), clean_up_tokenization_spaces=False)
    special = {k for k, token in tokenizer.get_added_tokens_decoder().items() if token.special}
    metric = BLEU(tokenize="13a", smooth_method="exp", effective_order=True, lowercase=False)
    batches, _ = load_pairs()
    result, texts, decoder_checks = [], [], 0
    for method in METHODS:
        for i in range(50):
            pair = [batches[(method, r)]["responses"][i] for r in (0, 1)]
            for length in manifest["primary_lengths"]:
                ids = [r["token_ids"][:length] for r in pair]
                decoded = [tokenizer.decode(row, skip_special_tokens=True) for row in ids]
                if i == 0:
                    for tokens, text in zip(ids, decoded):
                        if hf.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False) != text:
                            raise ValueError("Rust/HF decoding differs")
                        decoder_checks += 1
                score = .5 * (metric.sentence_score(decoded[0], [decoded[1]]).score +
                              metric.sentence_score(decoded[1], [decoded[0]]).score) / 100
                quality = [quality_metrics(tokens, row["generation_diagnostics"]["base_token_logprobs"][:length])
                           for tokens, row in zip(ids, pair)]
                result.append({"method": method, "prompt_index": i, "length": length,
                               "self_bleu": score, "self_bleu_token_ids": self_bleu_token_ids(ids),
                               "exact_token_duplicate": ids[0] == ids[1], "exact_text_duplicate": decoded[0] == decoded[1],
                               "empty_decoded_responses": sum(not s.strip() for s in decoded),
                               "special_tokens": sum(token in special for row in ids for token in row),
                               **{k: float(np.mean([q[k] for q in quality])) for k in ("base_model_nll", "repetition_rate", "distinct_3")}})
                texts.append({"method": method, "prompt_index": i, "length": length, "texts": decoded,
                              "response_ids": [r["response_id"] for r in pair]})
    report = {"manifest_id": manifest["id"], "tokenizer_sha256": sha(tokenizer_path), "bleu_signature": str(metric.get_signature()),
              "runtime_versions": {p: importlib.metadata.version(p) for p in ("sacrebleu", "tokenizers", "transformers", "numpy", "scipy")},
              "decode": {"skip_special_tokens": True, "clean_up_tokenization_spaces": False,
                         "hf_parity_checks": decoder_checks, "special_token_ids": sorted(special)},
              "rows": result}
    save(setup / "diversity.json", report)
    save(setup / "raw/decoded_pairs.json", texts)
    return {"signature": report["bleu_signature"], "rows": len(result),
            "means": [{"method": m, "length": n, "self_bleu": float(np.mean([r["self_bleu"] for r in result if r["method"] == m and r["length"] == n]))}
                      for n in manifest["primary_lengths"] for m in METHODS]}


def token_evidence(setup):
    import torch
    from .config import SYNTHID_COMMIT
    from .textseal_completion import load_upstream_detector
    from .official import official_gumbel_scores, synthid_processor
    load_upstream_detector(os.environ.get("TEXTSEAL_SOURCE_ROOT"))
    distribution = importlib.metadata.distribution("synthid-text")
    direct = json.loads(distribution.read_text("direct_url.json"))
    if direct.get("vcs_info", {}).get("commit_id") != SYNTHID_COMMIT:
        raise ValueError("SynthID source pin differs")
    manifest, inputs = read_inputs(setup)
    old = ROOT / "outputs/controlled_baseline_full/qwen3-8b-batch50-validation-20260823-v1"
    provenance = json.loads((old / "controlled_baseline_full_provenance_manifest.json").read_text())
    historical_path = old / "controlled_baseline_full_prompt_level.jsonl"
    if sha(historical_path) != provenance["compact_artifacts"][historical_path.name]:
        raise ValueError("historical token scores changed")
    historical = {}
    for line in historical_path.open():
        row = json.loads(line)
        if row["method"] in ("synthid_text", "gumbel_max"):
            historical[(row["method"], row["sample_type"], row["prompt_index"], row["prefix_length"])] = row
    shared_path = ROOT / "outputs/comparison_redetect/textseal_setup/direct_prefix/completion_inputs.jsonl"
    shared_manifest = json.loads(shared_path.with_name("native8b_manifest.json").read_text())
    if sha(shared_path) != shared_manifest["inputs"]["sha256"]:
        raise ValueError("shared null corpus changed")
    shared = [{"response_id": f"shared-null/{r['prompt_index']:04d}", "method": "shared_null", "prompt_index": r["prompt_index"],
               "response_index": None, "token_ids": r["token_ids"], "completion_sha256": digest(r["token_ids"])}
              for r in map(json.loads, shared_path.read_text().splitlines()) if r["method"] == "null"]
    candidates = [r for r in inputs if r["method"] in ("synthid_text", "gumbel_max", "null")] + shared
    processor = synthid_processor("cpu")
    all_rows, raw_evidence, parity = [], [], []
    # Batched official integer hashing and official repeated-context mask; no prompts.
    for start in range(0, len(candidates), 50):
        group = candidates[start:start+50]
        tensor = torch.tensor([r["token_ids"] for r in group], dtype=torch.long)
        gvals = processor.compute_g_values(tensor).cpu().numpy()
        masks = processor.compute_context_repetition_mask(tensor).cpu().numpy().astype(bool)
        for i, row in enumerate(group):
            ids = row["token_ids"]
            gumbel_positions = deduplicated_positions(ids)
            gumbel_values = official_gumbel_scores(ids, gumbel_positions)
            by_pos = dict(zip(gumbel_positions, gumbel_values.tolist()))
            raw_evidence.append({"response_id": row["response_id"], "completion_sha256": row["completion_sha256"],
                                 "synthid_g_values_from_position_3": gvals[i].tolist(),
                                 "synthid_context_mask_from_position_3": masks[i].tolist(),
                                 "gumbel_positions": gumbel_positions, "gumbel_scores": gumbel_values.tolist()})
            detectors = []
            if row["method"] in ("gumbel_max", "null", "shared_null"):
                detectors.append("gumbel_max")
            if row["method"] in ("synthid_text", "null", "shared_null"):
                detectors.extend(("synthid_text", "synthid_historical_mask"))
            for length in manifest["prefix_lengths"]:
                positions = [p for p in gumbel_positions if p < length]
                for detector in detectors:
                    if detector == "gumbel_max":
                        score = gumbel_gamma_test([by_pos[p] for p in positions])
                        eligible = positions
                        logp = float(__import__("scipy").stats.gamma.logsf(score["statistic"], a=len(positions))) if positions else 0.
                    else:
                        eligible = (np.flatnonzero(masks[i, :length-3]) + 3).tolist() if detector == "synthid_text" else positions
                        score = synthid_normal_test(gvals[i, np.asarray(eligible, dtype=int)-3])
                        z = score["intermediate"].get("z_score", 0.)
                        logp = float(__import__("scipy").special.log_ndtr(-z)) if eligible else 0.
                    if not math.isfinite(score["threshold"]):
                        score["threshold"] = None
                    score.update(log_p_value=logp if math.isfinite(logp) else None,
                                 log_p_value_underflow=not math.isfinite(logp), effective_tokens=len(eligible))
                    entry = {k: row[k] for k in row if k != "token_ids"} | {"detector": detector, "length": length, "score": score}
                    all_rows.append(entry)
                    # Compare the preserved masks and actual matching primary cached texts.
                    original_method = "synthid_text" if detector == "synthid_historical_mask" else detector
                    if detector in ("gumbel_max", "synthid_historical_mask") and (
                            row["method"] == "shared_null" or (row["method"] == original_method and row["response_index"] == 0)):
                        from .reuse_token_baselines import token_hash
                        prior = historical[(original_method, "null" if row["method"] == "shared_null" else "watermarked", row["prompt_index"], length)]
                        if prior["generated_token_hash"] != token_hash(ids):
                            raise ValueError("CPU parity record tokens differ")
                        difference = abs(score["p_value"] - prior["p_value"])
                        if score["decision"] != prior["decision"] or not math.isclose(score["p_value"], prior["p_value"], rel_tol=1e-9, abs_tol=1e-14):
                            raise ValueError(f"historical CPU decision/p-value differs: {original_method}/{row['prompt_index']}/{length}")
                        parity.append(difference)
        print(f"[pilot] CPU keyed evidence {min(start+50,len(candidates))}/{len(candidates)}", flush=True)
    result = {"manifest_id": manifest["id"], "rows": all_rows, "shared_null_count": len(shared),
              "historical_parity_checks": len(parity), "max_historical_pvalue_abs_difference": max(parity),
              "synthid_source": direct, "runtime_versions": {p: importlib.metadata.version(p) for p in ("torch", "numpy", "scipy", "synthid-text")},
              "historical_source_sha256": sha(historical_path), "shared_null_source_sha256": sha(shared_path)}
    save(setup / "raw/token_evidence.json", raw_evidence)
    save(setup / "token_detection.json", result)
    return {k: v for k, v in result.items() if k not in ("rows", "synthid_source")}


def paired_interval(values, draws):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size != draws.shape[1] or not np.isfinite(values).all():
        raise ValueError("bootstrap requires one finite value per prompt")
    boot = values[draws].mean(axis=1)
    return {"mean": float(values.mean()), "ci95": np.quantile(boot, [.025, .975]).tolist()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", type=Path, default=SETUP)
    parser.add_argument("--stage", choices=("diversity", "token-evidence"), required=True)
    parser.add_argument("--tokenizer", type=Path, default=Path("/private/tmp/self-bleu-stage-a-tokenizer.json"))
    args = parser.parse_args()
    result = diversity(args.setup, args.tokenizer) if args.stage == "diversity" else token_evidence(args.setup)
    print(json.dumps(result, indent=2))
