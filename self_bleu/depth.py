"""Frozen SynthID depths 2/30 follow-up: local preparation and paired analysis.

The existing generator and detectors are unchanged. Only depth_modal dispatches
the four new full-length batches. Saved depth-10, PRC and null pairs are reused.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path, PurePosixPath
import subprocess

import numpy as np

from .config import StudySetting, digest
from .pilot import paired_interval
from .repeat import upstream_hashes, validate as validate_repeat
from .validation import ROOT, RATE, save, sha

SETUP = ROOT / "outputs/self_bleu_depth/depth2_30_v1"
PILOT = ROOT / "outputs/self_bleu_pilot/stage_a_v2"
REPEAT = ROOT / "outputs/self_bleu_repeat/setup_v4"
PAIRED = ROOT / "outputs/self_bleu_repeat/paired_comparison"
SHARED = ROOT / "outputs/comparison_redetect/textseal_setup/direct_prefix/completion_inputs.jsonl"
TIMEOUT = 900
DEPTHS = (2, 30)


def validate(manifest, root=ROOT):
    root = Path(root)
    if digest({k: v for k, v in manifest.items() if k != "id"}) != manifest["id"]:
        raise ValueError("depth manifest identity differs")
    if (manifest["settings"] != {str(d): StudySetting("synthid_text", depth=d).identity() for d in DEPTHS}
            or manifest["prompt_indices"] != list(range(50)) or manifest["seeds"] != [12345, 67890]
            or manifest["length"] != 1024 or manifest["primary_lengths"] != [400, 1024]
            or manifest["repeat_fallback"] is not True or manifest["control_tokens"] != 64
            or manifest["protocol"] != "completion_only_raw_abstain_v1"):
        raise ValueError("depth follow-up scope changed")
    for name, expected in manifest["code_sha256"].items():
        if sha(root/name) != expected:
            raise ValueError(f"source changed: {name}")
    if sha(root/"prompts.jsonl") != manifest["prompt_sha256"]:
        raise ValueError("prompts changed")
    cost = manifest["cost"]
    if (cost["timeout_seconds"] != TIMEOUT or cost["resource_usd_per_second"] != RATE
            or cost["reserved_total_usd"] != cost["previous_planning_charge_usd"] + (TIMEOUT+2)*RATE + .5
            or cost["reserved_total_usd"] > 10):
        raise ValueError("depth follow-up exceeds initial allocation")


def prepare(setup):
    prior = json.loads((REPEAT/"manifest.json").read_text())
    validate_repeat(prior)
    consolidated = json.loads((PAIRED/"summary.json").read_text())
    if not consolidated["verification"]["passed"]:
        raise ValueError("saved comparison did not pass verification")
    for name, expected in consolidated["sources"].items():
        if sha(ROOT/name) != expected:
            raise ValueError(f"reference changed: {name}")
    generation_path = ROOT/"outputs/self_bleu_validation/step3-v4/generation_report.json"
    generation = json.loads(generation_path.read_text())
    references = {**prior["reference_files"], **consolidated["sources"]}
    for path in (PAIRED/"summary.json", PAIRED/"prompt_metrics.json", REPEAT/"all_analysis.json", SHARED):
        references[str(path.relative_to(ROOT))] = sha(path)
    shared_manifest = json.loads(SHARED.with_name("native8b_manifest.json").read_text())
    if sha(SHARED) != shared_manifest["inputs"]["sha256"]:
        raise ValueError("historical null corpus changed")
    smokes = {}
    raw = ROOT/"outputs/self_bleu_validation/raw/743658bc40e7f52c910d9538266bbd0ff461bba949e2a842cc8af36fa32507d8"
    for depth in DEPTHS:
        entry = next(r for r in generation["parameter_smokes"] if r["setting"].get("depth") == depth)
        path = raw/"batches"/f"{entry['batch']}.json"
        if not entry["passed"] or sha(path) != generation["files"][f"batches/{entry['batch']}.json"]:
            raise ValueError("depth smoke reference changed")
        batch = json.loads(path.read_text())
        if batch["manifest"]["setting"] != StudySetting("synthid_text", depth=depth).identity():
            raise ValueError("smoke keys differ")
        smokes[str(depth)] = [digest(r["token_ids"]) for r in batch["responses"]]
        references[str(path.relative_to(ROOT))] = sha(path)
    previous_cost = json.loads((REPEAT/"all_analysis.json").read_text())["planning_charge_usd"]
    names = sorted(set(prior["code_sha256"]) | {"self_bleu/depth.py", "self_bleu/depth_modal.py"})
    manifest = dict(schema_version=1, source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        protocol=prior["protocol"], model=prior["model"], generation_runtime=prior["generation_runtime"],
        prompt_indices=list(range(50)), prompt_sha256=sha(ROOT/"prompts.jsonl"), seeds=[12345, 67890],
        length=1024, primary_lengths=[400, 1024], control_tokens=64, repeat_fallback=True,
        settings={str(d): StudySetting("synthid_text", depth=d).identity() for d in DEPTHS},
        native_depth10_controls={str(r): prior["references"][f"synthid_text/{r}"]["prefix_sha256"] for r in (0, 1)},
        parameter_smoke_prefixes=smokes, analysis=prior["analysis"],
        analysis_contract={"new_full_responses": 200, "baseline_pairs": ["synthid_depth10", "prc", "null"],
            "detector": "existing weighted normal test, nominal p<.001, explicit per-depth keys and official context mask",
            "detector_inputs": "raw completion tokens only; no prompt or generation diagnostics",
            "nulls": "rescore saved 100 pilot and 500 historical nulls for each depth; no generation or calibration",
            "contrasts": "direct paired prompt differences in Self-BLEU and TPR; 2000 joint prompt resamples",
            "keys": "unchanged predeclared nested key-bank prefixes, fixed across seeds; depth changes key-list length"},
        upstream_sha256=upstream_hashes(), code_sha256={name: sha(ROOT/name) for name in names},
        reference_files=references,
        cost={"previous_planning_charge_usd": previous_cost, "resource_usd_per_second": RATE,
            "rate_basis": "same frozen planning estimate as setup_v4; not a settled bill",
            "timeout_seconds": TIMEOUT, "new_overhead_allowance_usd": .5,
            "reserved_total_usd": previous_cost+(TIMEOUT+2)*RATE+.5,
            "initial_allocation_usd": 10, "study_ceiling_usd": 200,
            "dispatch": "one H100, four sequential generation batches, retries=0"})
    manifest["id"] = digest(manifest)
    validate(manifest)
    save(setup/"manifest.json", manifest)
    return {"id": manifest["id"], "new_responses": 200, "cost": manifest["cost"]}


def check_fallback(depth, device="cpu"):
    """Exercise native ordinary-sampling fallback at each requested depth."""
    import torch
    from baseline_comparison.official import synthid_processor
    from synthid_text.logits_processing import update_scores
    processor = synthid_processor(device, keys=StudySetting("synthid_text", depth=depth).synthid_keys)
    logits = torch.linspace(-3, 3, 32, device=device)[None].repeat(2, 1)
    repeats = firsts = 0
    for token in [1, 2, 3]*4:
        history = (torch.zeros((2, processor.context_history_size), dtype=torch.long, device=device)
                   if processor.state is None else processor.state.context_history.clone())
        output, indices, base = processor.watermarked_call(torch.full((2, 50), token, dtype=torch.long, device=device), logits)
        repeated = (history == processor.state.context_history[:, :1]).any(dim=1)
        keys, _ = processor._compute_keys(processor.state.context, indices)
        expected = update_scores(base, processor.get_gvals(keys))
        if not torch.equal(output[repeated], base[repeated]) or not torch.equal(output[~repeated], expected[~repeated]):
            raise ValueError("native repeat fallback differs")
        repeats += int(repeated.sum()); firsts += int((~repeated).sum())
    if not repeats or not firsts:
        raise ValueError("probe did not exercise both cases")
    return {"passed": True, "depth": depth, "repeated_cases": repeats, "first_occurrences": firsts}


def collect(setup, download=False):
    manifest = json.loads((setup/"manifest.json").read_text())
    validate(manifest)
    volume = None
    remote = f"self_bleu_depth/{manifest['id']}"
    if download:
        import modal
        volume = modal.Volume.from_name("prc-completion-only", create_if_missing=False)
    report_path = setup/"generation_report.json"
    if download and not report_path.exists():
        report_path.write_bytes(b"".join(volume.read_file(f"{remote}/report.json")))
    report = json.loads(report_path.read_text())
    if not report["passed"] or report["manifest_id"] != manifest["id"]:
        raise ValueError("generation did not pass")
    for name, expected in report["files"].items():
        parts = PurePosixPath(name)
        if parts.is_absolute() or ".." in parts.parts:
            raise ValueError("unsafe artifact path")
        path = setup/"raw"/name
        if download and not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"".join(volume.read_file(f"{remote}/{name}")))
        if sha(path) != expected:
            raise ValueError(f"artifact checksum differs: {name}")
    return manifest, report


def score_completions(rows, depth, lengths):
    """The detector receives only raw completion IDs and fixed keys."""
    import torch
    from baseline_comparison.official import synthid_processor
    from baseline_comparison.scoring import synthid_normal_test
    keys = StudySetting("synthid_text", depth=depth).synthid_keys
    processor = synthid_processor("cpu", keys=keys)
    scores = []
    for start in range(0, len(rows), 50):
        batch = rows[start:start+50]
        tensor = torch.tensor([r["token_ids"][:1024] for r in batch], dtype=torch.long)
        values = processor.compute_g_values(tensor).numpy()
        mask = processor.compute_context_repetition_mask(tensor).numpy().astype(bool)
        if values.shape != (len(batch), 1021, depth) or mask.shape != (len(batch), 1021):
            raise ValueError("wrong evidence depth or completion length")
        for n in lengths:
            # Full-trajectory extraction must equal fresh direct-prefix extraction.
            direct = synthid_processor("cpu", keys=keys)
            if not (np.array_equal(direct.compute_g_values(tensor[:1, :n]).numpy(), values[:1, :n-3])
                    and np.array_equal(direct.compute_context_repetition_mask(tensor[:1, :n]).numpy(), mask[:1, :n-3])):
                raise ValueError("detector prefix invariance failed")
            for i, row in enumerate(batch):
                selected = values[i, :n-3][mask[i, :n-3]]
                score = synthid_normal_test(selected)
                if len(score["intermediate"]["layer_weights"]) != depth:
                    raise ValueError("scorer silently used the wrong depth")
                scores.append({"response_id": row["response_id"], "prompt_index": row["prompt_index"],
                    "response_index": row.get("response_index"), "depth": depth, "keys": list(keys), "length": n,
                    "completion_sha256": digest(row["token_ids"][:1024]), "prefix_sha256": digest(row["token_ids"][:n]),
                    "eligible_positions": (np.flatnonzero(mask[i, :n-3])+3).tolist(),
                    "g_values_sha256": digest(selected.tolist()), "effective_tokens": len(selected), "score": score})
    return scores


def analyze(setup, download=False):
    import sacrebleu
    from sacrebleu.metrics import BLEU
    from tokenizers import Tokenizer
    manifest, report = collect(setup, download)
    for name, expected in manifest["reference_files"].items():
        if sha(ROOT/name) != expected:
            raise ValueError(f"saved reference changed: {name}")
    if upstream_hashes() != manifest["upstream_sha256"] or sacrebleu.__version__ != "2.4.3":
        raise ValueError("analysis implementation changed")
    tokenizer_path = PILOT/"raw/tokenizer.json"
    if sha(tokenizer_path) != manifest["model"]["tokenizer_sha256"]:
        raise ValueError("tokenizer changed")
    decoder = Tokenizer.from_file(str(tokenizer_path))
    bleu = BLEU(tokenize="13a", smooth_method="exp", effective_order=True, lowercase=False)
    prior = json.loads((PAIRED/"summary.json").read_text())
    old_rows = json.loads((PAIRED/"prompt_metrics.json").read_text())
    if sha(PAIRED/"prompt_metrics.json") != prior["prompt_metrics_sha256"]:
        raise ValueError("paired reference rows changed")
    inputs = json.loads((PILOT/"inputs.json").read_text())
    originals = {m: sorted([r for r in inputs if r["method"] == m], key=lambda r:(r["prompt_index"], r["response_index"]))
                 for m in ("synthid_text", "null", "online_prc")}
    shared = [{"response_id": f"shared-null/{r['prompt_index']:04d}", "prompt_index": r["prompt_index"],
               "token_ids": r["token_ids"][:1024]} for r in map(json.loads, SHARED.read_text().splitlines()) if r["method"] == "null"]
    if len(shared) != 500 or any(len(x) != 100 for x in originals.values()):
        raise ValueError("saved comparison coverage differs")
    groups = {10: originals["synthid_text"]}
    for depth in DEPTHS:
        rows = []
        for response, seed in enumerate(manifest["seeds"]):
            batch = json.loads((setup/"raw/batches"/f"depth{depth}_r{response}.json").read_text())
            bm = batch["manifest"]
            if (bm["setting"] != manifest["settings"][str(depth)] or bm["sampling_seed"] != seed
                    or bm["prompt_indices"] != manifest["prompt_indices"] or len(batch["responses"]) != 50):
                raise ValueError("generated batch identity differs")
            for row in batch["responses"]:
                i = row["prompt_index"]
                if (row["response_index"] != response or row["sampling_seed"] != seed or len(row["token_ids"]) != 1024
                        or row["completion_sha256"] != digest(row["token_ids"])
                        or row["response_id"] != bm["response_ids"][i] or row["prompt_sha256"] != bm["prompt_sha256"][i]):
                    raise ValueError("generated response identity differs")
            rows.extend(batch["responses"])
        if {(r["prompt_index"], r["response_index"]) for r in rows} != {(i,j) for i in range(50) for j in (0,1)}:
            raise ValueError("generated pairs incomplete")
        groups[depth] = rows
    lengths = manifest["primary_lengths"]
    all_scores, nulls, prompt_rows = [], [], []
    old_scores = json.loads((PILOT/"token_detection.json").read_text())["rows"]
    old_score_map = {(r["response_id"],r["length"]):r["score"] for r in old_scores if r["detector"] == "synthid_text"}
    depth10_checks = 0
    for depth in (2,10,30):
        measured = {}
        for cohort, records in (("watermarked", groups[depth]), ("pilot_null", originals["null"]), ("historical_null", shared)):
            scores = score_completions(records, depth, lengths)
            for row in scores:
                row["cohort"] = cohort
                if depth == 10:
                    old = old_score_map[row["response_id"],row["length"]]
                    if (row["score"]["decision"] != old["decision"] or not np.isclose(row["score"]["p_value"],old["p_value"],rtol=1e-9,atol=1e-14)):
                        raise ValueError("depth-10 detection parity failed")
                    depth10_checks += 1
            all_scores.extend(scores)
            if cohort == "watermarked":
                measured = {(r["prompt_index"],r["response_index"],r["length"]):r for r in scores}
            else:
                for n in lengths:
                    subset = [r for r in scores if r["length"] == n]
                    nulls.append({"depth":depth,"cohort":cohort,"length":n,"false_positives":sum(r["score"]["decision"] for r in subset),"responses":len(subset)})
        by_prompt = {(r["prompt_index"],r["response_index"]):r for r in groups[depth]}
        for n in lengths:
            for i in range(50):
                pair = [by_prompt[i,r] for r in (0,1)]
                texts = [decoder.decode(r["token_ids"][:n],skip_special_tokens=True) for r in pair]
                value = (bleu.sentence_score(texts[0],[texts[1]]).score+bleu.sentence_score(texts[1],[texts[0]]).score)/200
                detected = [measured[i,r,n]["score"]["decision"] for r in (0,1)]
                prompt_rows.append({"setting":f"synthid_depth{depth}","length":n,"prompt_index":i,
                    "response_ids":[r["response_id"] for r in pair],"self_bleu":value,"detected":detected,"tpr":float(np.mean(detected))})
                if depth == 10:
                    old = next(r for r in old_rows if r["setting"]=="synthid_on" and r["length"]==n and r["prompt_index"]==i)
                    if not np.isclose(value,old["self_bleu"],rtol=0,atol=1e-12) or detected != old["detected"]:
                        raise ValueError("depth-10 paired reference differs")
    for r in old_rows:
        if r["setting"] in ("prc","null"):
            prompt_rows.append({k:r[k] for k in ("setting","length","prompt_index","response_ids","self_bleu","detected","tpr")})
    draws = np.random.default_rng(manifest["analysis"]["bootstrap_seed"]).integers(0,50,(2000,50))
    if digest(draws.tolist()) != prior["bootstrap"]["draws_sha256"]:
        raise ValueError("bootstrap draws changed")
    vectors, results = {}, []
    for setting in ("prc","null","synthid_depth2","synthid_depth10","synthid_depth30"):
        for n in lengths:
            rows = sorted([r for r in prompt_rows if r["setting"]==setting and r["length"]==n],key=lambda r:r["prompt_index"])
            if [r["prompt_index"] for r in rows] != list(range(50)):
                raise ValueError("prompt bootstrap pairing differs")
            vectors[setting,n] = {m:np.array([r[m] for r in rows]) for m in (("self_bleu",) if setting=="null" else ("self_bleu","tpr"))}
            results.append({"setting":setting,"length":n,"metrics":{m:paired_interval(v,draws) for m,v in vectors[setting,n].items()},
                            "detected":None if setting=="null" else sum(sum(r["detected"]) for r in rows),"responses":100,"prompts":50})
    contrasts = []
    pairs = [(f"synthid_depth{d}","synthid_depth10") for d in DEPTHS]+[("synthid_depth30","synthid_depth2")]
    pairs += [("prc",f"synthid_depth{d}") for d in (2,10,30)]
    pairs += [(s,"null") for s in ("prc","synthid_depth2","synthid_depth10","synthid_depth30")]
    for left,right in pairs:
        for n in lengths:
            contrasts.append({"left":left,"right":right,"length":n,"metrics":{m:paired_interval(v-vectors[right,n][m],draws)
                              for m,v in vectors[left,n].items() if m in vectors[right,n]}})
    save(setup/"raw/score_records.json",all_scores)
    save(setup/"prompt_metrics.json",prompt_rows)
    summary = {"manifest_id":manifest["id"],"results":results,"contrasts":contrasts,"null_counts":nulls,
        "bootstrap":prior["bootstrap"],"bleu_signature":str(bleu.get_signature()),
        "analysis_versions":{p:importlib.metadata.version(p) for p in ("torch","numpy","scipy","sacrebleu","tokenizers","synthid-text")},
        "verification":{"passed":True,"new_full_responses":200,"depth10_score_parity_checks":depth10_checks,
                        "depth10_self_bleu_pair_checks":100,"prompt_records":len(prompt_rows),"score_records":len(all_scores)},
        "sources":{**manifest["reference_files"],"manifest.json":sha(setup/"manifest.json"),"generation_report.json":sha(setup/"generation_report.json")},
        "prompt_metrics_sha256":sha(setup/"prompt_metrics.json"),"score_records_sha256":sha(setup/"raw/score_records.json"),
        "cost":{"generation_resource_estimate_usd":report["measured_resource_usd"],"new_overhead_allowance_usd":.5,
                "cumulative_planning_charge_usd":manifest["cost"]["previous_planning_charge_usd"]+report["measured_resource_usd"]+.5},
        "limitations":["50 paired prompt clusters; fixed keys and one model; exploratory marginal intervals without multiplicity correction.",
                       "Same weighted frequentist detector family at every depth; not Bayesian SynthID.",
                       "Nominal p<.001, not matched empirical FPR; null cohorts overlap and are not pooled.",
                       "All-success bootstrap intervals do not establish perfect population detection."]}
    save(setup/"summary.json",summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","collect","analyze"))
    parser.add_argument("--setup",type=Path,default=SETUP)
    parser.add_argument("--download",action="store_true")
    args = parser.parse_args()
    result = prepare(args.setup) if args.command=="prepare" else (analyze(args.setup,args.download) if args.command=="analyze" else collect(args.setup,args.download)[1])
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
