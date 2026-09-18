"""Prepare, collect and analyze the frozen Stage A completion-only pilot."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import importlib.metadata
import math
import os
import json
from pathlib import Path, PurePosixPath
import subprocess

import numpy as np

from baseline_comparison.scoring import self_bleu_token_ids, quality_metrics, deduplicated_positions, gumbel_gamma_test, synthid_normal_test
from baseline_comparison.config import PREFIX_LENGTHS
from .config import digest, verify_reference
from .validation import ROOT, RATE, save, sha
from .validation import collect as collect_validation

VALIDATION = ROOT / "outputs/self_bleu_validation/step3-v4"
RAW = ROOT / "outputs/self_bleu_validation/raw/743658bc40e7f52c910d9538266bbd0ff461bba949e2a842cc8af36fa32507d8"
SETUP = ROOT / "outputs/self_bleu_pilot/stage_a_v2"
METHODS = ("online_prc", "textseal", "synthid_text", "gumbel_max", "null")
CODE = ("self_bleu/__init__.py", "self_bleu/pilot.py", "self_bleu/pilot_modal.py",
        "self_bleu/validation.py", "self_bleu/validation_modal.py",
        "baseline_comparison/textseal_modal.py", "baseline_comparison/textseal_redetect.py",
        "baseline_comparison/textseal_completion.py", "baseline_comparison/textseal_source_audit.json",
        "baseline_comparison/requirements-textseal.txt", "baseline_comparison/modal_app.py",
        "baseline_comparison/comparison_runner.py", "baseline_comparison/config.py",
        "baseline_comparison/official.py", "baseline_comparison/scoring.py",
        "self_bleu/config.py", "self_bleu/reference.json",
        "qwen.py", "prc.py", "online_prc.py", "detectors.py")


def load_pairs():
    verification = collect_validation(VALIDATION, RAW)
    report = json.loads((VALIDATION / "generation_report.json").read_text())
    batches = {}
    for row in report["settings"]:
        setting = row["setting"]
        if setting["method"] == "textseal" and setting["alpha"] != .1:
            continue
        for bid in row["batches"]:
            batch = json.loads((RAW / "batches" / f"{bid}.json").read_text())
            batches[(setting["method"], batch["manifest"]["response_index"])] = batch
    if set(batches) != {(m, r) for m in METHODS for r in (0, 1)}:
        raise ValueError("Stage A pair coverage differs")
    return batches, verification


def clean_records(batches):
    rows = []
    for (method, replicate), batch in sorted(batches.items()):
        for row in batch["responses"]:
            rows.append({"response_id": row["response_id"], "method": method,
                         "prompt_index": row["prompt_index"], "response_index": replicate,
                         "completion_sha256": row["completion_sha256"], "token_ids": row["token_ids"]})
    return rows


def prepare(output=SETUP, failed_attempt_allowance_usd=0.):
    from baseline_comparison.textseal_results import validate_record
    reference = verify_reference()
    batches, verification = load_pairs()
    rows = clean_records(batches)
    old_setup = ROOT / "outputs/comparison_redetect/textseal_setup/direct_prefix"
    old_manifest = json.loads((old_setup / "native8b_manifest.json").read_text())
    old_report = json.loads((old_setup / "full_report.json").read_text())
    old_by_id = {r["record_id"]: r for r in old_report["rows"]}
    validation_ts = json.loads((VALIDATION / "textseal_report.json").read_text())
    if old_report["runtime"] != validation_ts["execution"]:
        raise ValueError("TextSeal replay runtimes differ")
    by_response = {r["response_id"]: r for r in rows}
    reused = {}
    sources = {str(VALIDATION.relative_to(ROOT) / name): sha(VALIDATION / name)
               for name in ("manifest.json", "verification.json", "generation_report.json", "textseal_report.json")}
    for row in rows:
        if row["method"] != "textseal" or row["response_index"] != 0:
            continue
        rid = f"textseal/{row['prompt_index']:04d}"
        path = old_setup / "cache" / f"{rid}.json"
        if sha(path) != old_report["record_sha256"][rid]:
            raise ValueError("historical TextSeal record changed")
        clean = {k: row[k] for k in ("method", "prompt_index", "token_ids")}
        data = validate_record(json.loads(path.read_text()), clean, old_by_id[rid], old_manifest, old_report["runtime"])
        reused[row["response_id"]] = {"response_id": row["response_id"], "completion_sha256": row["completion_sha256"],
                                      "results": data["results"], "source": str(path.relative_to(ROOT)), "source_sha256": sha(path)}
    for name, expected in validation_ts["files"].items():
        path = RAW / name
        if sha(path) != expected:
            raise ValueError("step-3 TextSeal evidence changed")
        payload = json.loads(path.read_text())
        rid = payload["response_id"]
        if rid not in by_response or payload["alpha"] != .1 or rid in reused:
            continue
        data, row = payload["data"], by_response[rid]
        if not data["actual_model_inputs_verified"] or data["completion_sha256"] != row["completion_sha256"] or not data["validation"]["passed"]:
            raise ValueError("step-3 replay identity or validation differs")
        reused[rid] = {"response_id": rid, "completion_sha256": row["completion_sha256"], "results": data["results"],
                       "source": str(path.relative_to(ROOT)), "source_sha256": expected}
    requests = {"prc": [r for r in rows if r["method"] in ("online_prc", "null")],
                "textseal": [r for r in rows if r["method"] in ("textseal", "null") and r["response_id"] not in reused]}
    if len(rows) != 500 or len(reused) != 53 or [len(requests[m]) for m in ("prc", "textseal")] != [200, 147]:
        raise ValueError("unexpected replay/reuse coverage")
    old_validation = json.loads((VALIDATION / "manifest.json").read_text())
    manifest = {"schema_version": 1, "stage": "A", "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "protocol": reference["protocol"], "model": old_validation["model"], "artifact": reference["prc_generation_artifact"],
                "prefix_lengths": list(PREFIX_LENGTHS), "primary_lengths": [400, 1024], "nominal_fpr": .001,
                "prompts": list(range(50)), "sampling_seeds": [12345, 67890], "input_sha256": digest(rows),
                "requests": {m: {"count": len(r), "sha256": digest(r)} for m, r in requests.items()},
                "reused_textseal_sha256": digest(list(reused.values())), "reused_textseal_records": len(reused),
                "analysis": {"sacrebleu_version": "2.4.3", "tokenize": "13a", "smooth_method": "exp", "effective_order": True,
                             "lowercase": False, "scale": "0-1", "skip_special_tokens": True, "clean_up_tokenization_spaces": False,
                             "bootstrap_resamples": 2000, "bootstrap_seed": 20260918, "bootstrap_unit": "paired prompt cluster",
                             "synthid_primary_mask": "official compute_context_repetition_mask", "synthid_reproduction_mask": "historical unique (context,token), start=4",
                             "operating_point": "nominal only; no empirical tail calibration on pilot"},
                "textseal_runtime": old_report["runtime"], "code_sha256": {p: sha(ROOT / p) for p in CODE},
                "sources": sources, "validation_id": verification["manifest_id"],
                "cost": {"resource_usd_per_second": RATE, "timeout_seconds_per_stage": 600,
                         "maximum_new_resource_reservation_usd": 1204*RATE,
                         "previous_planning_charge_usd": verification["total_planning_charge_usd"] + failed_attempt_allowance_usd,
                         "pilot_failed_attempt_allowance_usd": failed_attempt_allowance_usd,
                         "initial_allocation_usd": 10, "total_study_ceiling_usd": 200,
                         "dispatch": "one H100 per stage, sequential, max_containers=1, retries=0; no generation"}}
    manifest["id"] = digest(manifest)
    for method, records in requests.items():
        validate_request(manifest, records, method, ROOT)
    save(output / "manifest.json", manifest)
    save(output / "inputs.json", rows)
    save(output / "reused_textseal.json", list(reused.values()))
    return {"manifest_id": manifest["id"], "responses": len(rows), "reused_textseal_records": len(reused),
            "requests": manifest["requests"], "reserved_total_with_previous_allowances_usd": manifest["cost"]["previous_planning_charge_usd"]+1204*RATE}


def validate_request(manifest, records, stage, root):
    if digest({k: v for k, v in manifest.items() if k != "id"}) != manifest["id"]:
        raise ValueError("pilot manifest identity differs")
    if manifest["protocol"] != "completion_only_raw_abstain_v1" or manifest["stage"] != "A" or manifest["nominal_fpr"] != .001:
        raise ValueError("pilot protocol differs")
    for name, expected in manifest["code_sha256"].items():
        if not (Path(root) / name).is_file() or sha(Path(root) / name) != expected:
            raise ValueError(f"pilot worker source differs: {name}")
    expected = manifest["requests"][stage]
    if len(records) != expected["count"] or digest(records) != expected["sha256"]:
        raise ValueError("pilot detector request differs")
    seen = set()
    for row in records:
        if (set(row) != {"response_id", "method", "prompt_index", "response_index", "completion_sha256", "token_ids"}
                or row["response_id"] in seen or len(row["token_ids"]) != 1024
                or any(type(t) is not int or not 0 <= t < 151936 for t in row["token_ids"])
                or digest(row["token_ids"]) != row["completion_sha256"]):
            raise ValueError("detector inputs must be unique raw completions with identity metadata only")
        seen.add(row["response_id"])
    cost = manifest["cost"]
    if (cost["timeout_seconds_per_stage"] != 600 or cost["resource_usd_per_second"] != RATE
            or cost["previous_planning_charge_usd"] + 1204*RATE > 10):
        raise ValueError("pilot allocation would be exceeded")



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
    from baseline_comparison.config import SYNTHID_COMMIT
    from baseline_comparison.textseal_completion import load_upstream_detector
    from baseline_comparison.official import official_gumbel_scores, synthid_processor
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
                        from baseline_comparison.reuse_token_baselines import token_hash
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


def summarize(setup, *, output):
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
               "metric_and_evidence_code_sha256": sha(__file__),
               "limitations": ["50 prompt clusters, two responses each; deterministic Gumbel duplicates are not independent observations.",
                               "Percentile bootstrap intervals at all-success/all-failure boundaries collapse; they do not establish perfect population detection or zero FPR.",
                               "Nominal analytic/approximate thresholds are not empirically matched false-positive rates.",
                               "The historical null cohort overlaps pilot prompts; it is not a disjoint calibration or audit sample.",
                               "This is an exploratory selected batch and a default-setting comparison, not a parameter-frontier or Bayesian-SynthID comparison."]}
    source_root = output / "raw/analysis_source"
    source_root.mkdir(parents=True, exist_ok=True)
    for name in ("pilot.py",):
        source = Path(__file__).with_name(name).read_bytes()
        destination = source_root / name
        if destination.exists() and destination.read_bytes() != source:
            raise ValueError("analysis source snapshot changed")
        destination.write_bytes(source)
    save(output / "summary.json", summary)
    save(output / "raw/combined_scores.json", scores)
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

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("prepare", help="Freeze a new replay request")
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--failed-attempt-allowance-usd", type=float, default=0.)
    for name in ("diversity", "token-evidence", "collect", "summarize"):
        command = commands.add_parser(name)
        command.add_argument("--setup", type=Path, default=SETUP)
        if name == "diversity":
            command.add_argument("--tokenizer", type=Path, required=True)
        if name in ("collect", "summarize"):
            command.add_argument("--download", action="store_true")
        if name == "summarize":
            command.add_argument("--output", type=Path, required=True, help="Separate output directory; preserves the original report")
    args = parser.parse_args()
    if args.command == "prepare":
        if not 0 <= args.failed_attempt_allowance_usd <= 5:
            parser.error("failed-attempt allowance must be between 0 and 5")
        result = prepare(args.output, args.failed_attempt_allowance_usd)
    elif args.command == "diversity":
        result = diversity(args.setup, args.tokenizer)
    elif args.command == "token-evidence":
        result = token_evidence(args.setup)
    else:
        result = collect(args.setup, args.download)
        if args.command == "summarize":
            result = summarize(args.setup, output=args.output)
            plot(result, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "verified_files"}, indent=2))


if __name__ == "__main__":
    main()
