"""Score saved fallback-on SynthID depths 2/10/30 at 64/128/256 tokens.

CPU only: no generation, model loading, network, GPU dispatch or calibration.
Use the frozen depth-aware scorer and verify every result on its actual prefix.
"""
from __future__ import annotations

import importlib.metadata
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm

from baseline_comparison.official import synthid_processor
from self_bleu.config import StudySetting, digest
from self_bleu.depth import SETUP, PILOT, SHARED, collect, score_completions
from self_bleu.pilot import paired_interval
from self_bleu.repeat import upstream_hashes
from self_bleu.validation import ROOT, save, sha

OUTPUT = Path(__file__).resolve().parent
DEPTHS = (2, 10, 30)
LENGTHS = (64, 128, 256)


def run():
    generation_manifest, generation_report = collect(SETUP)
    sources = {}

    def read(path, expected=None):
        actual = sha(path)
        if expected is not None and actual != expected:
            raise ValueError(f"source hash differs: {path}")
        sources[str(path.relative_to(ROOT))] = actual
        return json.loads(path.read_text())

    for name, expected in generation_manifest["reference_files"].items():
        if sha(ROOT/name) != expected:
            raise ValueError(f"saved reference differs: {name}")
    if upstream_hashes() != generation_manifest["upstream_sha256"]:
        raise ValueError("upstream detector changed")
    previous = read(SETUP/"summary.json")
    verification = read(SETUP/"verification.json")
    if not verification["passed"] or sha(SETUP/"summary.json") != verification["summary_sha256"]:
        raise ValueError("depth-2/30 run is not verified")
    for name in ("manifest.json", "generation_report.json"):
        read(SETUP/name)
    inputs = read(PILOT/"inputs.json", generation_manifest["reference_files"][str((PILOT/"inputs.json").relative_to(ROOT))])
    old_detection = read(PILOT/"token_detection.json")
    old_scores = {(r["response_id"],r["length"]):r["score"] for r in old_detection["rows"] if r["detector"]=="synthid_text"}
    pilot_null = [r for r in inputs if r["method"]=="null"]
    groups = {10:[r for r in inputs if r["method"]=="synthid_text"]}
    for depth in (2,30):
        groups[depth] = []
        for response in (0,1):
            name = f"batches/depth{depth}_r{response}.json"
            batch = read(SETUP/"raw"/name, generation_report["files"][name])
            if (batch["manifest"]["setting"] != StudySetting("synthid_text",depth=depth).identity()
                    or batch["manifest"]["sampling_seed"] != generation_manifest["seeds"][response]):
                raise ValueError("saved generation setting differs")
            groups[depth].extend(batch["responses"])
    for rows in [*groups.values(),pilot_null]:
        if len(rows)!=100 or {(r["prompt_index"],r["response_index"]) for r in rows}!={(i,j) for i in range(50) for j in (0,1)}:
            raise ValueError("saved prompt/seed pairs incomplete")
        for row in rows:
            if len(row["token_ids"])!=1024 or digest(row["token_ids"])!=row["completion_sha256"]:
                raise ValueError("saved completion changed")
    expected_shared = generation_manifest["reference_files"][str(SHARED.relative_to(ROOT))]
    if sha(SHARED)!=expected_shared:
        raise ValueError("historical null corpus changed")
    sources[str(SHARED.relative_to(ROOT))] = expected_shared
    historical_null = [{"response_id":f"shared-null/{r['prompt_index']:04d}","prompt_index":r["prompt_index"],
        "token_ids":r["token_ids"][:1024]} for r in map(json.loads,SHARED.read_text().splitlines()) if r["method"]=="null"]
    if len(historical_null)!=500 or {r["prompt_index"] for r in historical_null}!=set(range(500)):
        raise ValueError("historical null coverage differs")
    for name in ("self_bleu/depth.py","self_bleu/config.py","self_bleu/pilot.py","baseline_comparison/official.py","baseline_comparison/scoring.py"):
        sources[name] = sha(ROOT/name)
    sources[str(Path(__file__).resolve().relative_to(ROOT))] = sha(Path(__file__))
    request = dict(depths=list(DEPTHS),lengths=list(LENGTHS),generation_manifest_id=generation_manifest["id"],
        setting_keys={str(d):StudySetting("synthid_text",depth=d).identity() for d in DEPTHS},
        repeat_fallback=True,protocol="completion_only_raw_abstain_v1",nominal_threshold=.001,
        detector="unchanged weighted-normal test with official context-repetition mask; explicit per-depth keys",
        cohorts={"watermarked_per_depth":100,"pilot_null":100,"historical_null":500},
        bootstrap={**previous["bootstrap"],"estimator":"Mean of paired prompt-level detection fractions or their direct depth differences"},
        sources=sources,upstream_sha256=generation_manifest["upstream_sha256"],
        generation_attempts=0,model_inference_calls=0,incremental_modal_cost_usd=0)
    request["id"] = digest(request)
    save(OUTPUT/"manifest.json",request)

    all_scores, lookup = [], {}
    checks = dict(direct_prefix_scores=0,independent_weighted_normal_scores=0,depth10_saved_score_parity=0)
    for depth in DEPTHS:
        keys = StudySetting("synthid_text",depth=depth).synthid_keys
        for cohort, rows in (("watermarked",groups[depth]),("pilot_null",pilot_null),("historical_null",historical_null)):
            scores = score_completions(rows,depth,LENGTHS)
            indexed = {(r["response_id"],r["length"]):r for r in scores}
            if len(indexed)!=len(rows)*len(LENGTHS):
                raise ValueError("score coverage differs")
            # Verify ALL truncated completions, not just a representative row.
            for start in range(0,len(rows),50):
                batch = rows[start:start+50]
                for n in LENGTHS:
                    processor = synthid_processor("cpu",keys=keys)
                    prefix = torch.tensor([r["token_ids"][:n] for r in batch],dtype=torch.long)
                    g = processor.compute_g_values(prefix).numpy()
                    masks = processor.compute_context_repetition_mask(prefix).numpy().astype(bool)
                    if g.shape!=(len(batch),n-3,depth) or masks.shape!=(len(batch),n-3):
                        raise ValueError("direct-prefix evidence shape differs")
                    for i,row in enumerate(batch):
                        scored = indexed[row["response_id"],n]
                        selected = g[i][masks[i]]
                        positions = (np.flatnonzero(masks[i])+3).tolist()
                        if scored["eligible_positions"]!=positions or scored["g_values_sha256"]!=digest(selected.tolist()):
                            raise ValueError("later completion tokens affect prefix evidence")
                        checks["direct_prefix_scores"] += 1
                        weights = np.linspace(10.,1.,depth);weights *= depth/weights.sum()
                        count = len(selected)
                        statistic = float(selected.sum(axis=0)@weights)
                        z = (statistic-count*depth/2)/math.sqrt(count*float(weights@weights)/4)
                        pvalue = max(float(norm.sf(z)),1e-300)
                        original = scored["score"]
                        if (not np.isclose(statistic,original["statistic"],atol=1e-10,rtol=1e-12)
                                or not np.isclose(z,original["intermediate"]["z_score"],atol=1e-10,rtol=1e-12)
                                or not np.isclose(pvalue,original["p_value"],atol=1e-14,rtol=1e-10)
                                or bool(pvalue<.001)!=original["decision"]):
                            raise ValueError("independent weighted-normal calculation differs")
                        checks["independent_weighted_normal_scores"] += 1
                        if depth==10 and (row["response_id"],n) in old_scores:
                            old = old_scores[row["response_id"],n]
                            if old["decision"]!=original["decision"] or not np.isclose(old["p_value"],original["p_value"],atol=1e-14,rtol=1e-9):
                                raise ValueError("saved depth-10 score differs")
                            checks["depth10_saved_score_parity"] += 1
            for r in scores:
                r["cohort"] = cohort
                key = (depth,cohort,r["length"],r["prompt_index"],r["response_index"])
                if key in lookup:
                    raise ValueError("duplicate scoring identity")
                lookup[key] = r
            all_scores.extend(scores)
        print(f"Scored depth {depth}: 100 watermarked + 100 pilot null + 500 historical null responses at three lengths",flush=True)

    draws = np.random.default_rng(request["bootstrap"]["seed"]).integers(0,50,(2000,50))
    if digest(draws.tolist())!=request["bootstrap"]["draws_sha256"]:
        raise ValueError("paired bootstrap changed")
    # A separate count-weight implementation verifies all interval arithmetic.
    count_weights = np.stack([np.bincount(draw,minlength=50) for draw in draws])/50
    interval_checks = 0

    def interval(values):
        nonlocal interval_checks
        values = np.asarray(values,dtype=float)
        result = paired_interval(values,draws)
        if not np.isclose(result["mean"],values.sum()/50,atol=1e-12,rtol=0) or not np.allclose(result["ci95"],np.quantile(count_weights@values,[.025,.975]),atol=1e-12,rtol=0):
            raise ValueError("independent prompt-bootstrap interval differs")
        interval_checks += 1
        return result

    prompt_rows, vectors, results = [], {}, []
    for depth in DEPTHS:
        for n in LENGTHS:
            cells = {}
            for cohort in ("watermarked","pilot_null"):
                vals = []
                for i in range(50):
                    pair = [lookup[depth,cohort,n,i,r] for r in (0,1)]
                    decisions = [r["score"]["decision"] for r in pair]
                    vals.append(float(np.mean(decisions)))
                    prompt_rows.append(dict(depth=depth,cohort=cohort,length=n,prompt_index=i,
                        response_ids=[r["response_id"] for r in pair],detected=decisions,detection_fraction=vals[-1]))
                vectors[depth,cohort,n] = np.array(vals)
                selected = [lookup[depth,cohort,n,i,r] for i in range(50) for r in (0,1)]
                cells[cohort] = dict(detected=sum(r["score"]["decision"] for r in selected),responses=100,
                    detection_rate=interval(vals),mean_effective_tokens=float(np.mean([r["effective_tokens"] for r in selected])),
                    median_z_score=float(np.median([r["score"]["intermediate"]["z_score"] for r in selected])))
            selected = [lookup[depth,"historical_null",n,i,None] for i in range(500)]
            cells["historical_null"] = dict(detected=sum(r["score"]["decision"] for r in selected),responses=500)
            results.append(dict(depth=depth,length=n,**cells))
    contrasts = []
    for left,right in ((10,2),(30,2),(30,10)):
        for n in LENGTHS:
            changes = {cohort:interval(vectors[left,cohort,n]-vectors[right,cohort,n]) for cohort in ("watermarked","pilot_null")}
            discordance = {}
            for cohort in ("watermarked","pilot_null"):
                a = [lookup[left,cohort,n,i,r]["score"]["decision"] for i in range(50) for r in (0,1)]
                b = [lookup[right,cohort,n,i,r]["score"]["decision"] for i in range(50) for r in (0,1)]
                discordance[cohort] = dict(left_only=sum(x and not y for x,y in zip(a,b)),right_only=sum(y and not x for x,y in zip(a,b)),
                    both=sum(x and y for x,y in zip(a,b)),neither=sum(not x and not y for x,y in zip(a,b)))
            contrasts.append(dict(left_depth=left,right_depth=right,length=n,left_minus_right=changes,paired_outcomes=discordance))
    if checks!={"direct_prefix_scores":6300,"independent_weighted_normal_scores":6300,"depth10_saved_score_parity":1400}:
        raise ValueError("verification coverage differs")
    save(OUTPUT/"raw/score_records.json",all_scores)
    save(OUTPUT/"prompt_metrics.json",prompt_rows)
    summary = dict(manifest_id=request["id"],lengths=list(LENGTHS),depths=list(DEPTHS),results=results,contrasts=contrasts,
        bootstrap={**request["bootstrap"],"estimator":"Mean of paired prompt-level detection fractions or their direct depth differences"},
        verification=dict(passed=True,**checks,independent_bootstrap_intervals=interval_checks,prompt_records=len(prompt_rows),score_records=len(all_scores)),
        source_manifest_sha256=sha(OUTPUT/"manifest.json"),prompt_metrics_sha256=sha(OUTPUT/"prompt_metrics.json"),score_records_sha256=sha(OUTPUT/"raw/score_records.json"),
        analysis_versions={p:importlib.metadata.version(p) for p in ("torch","numpy","scipy","synthid-text")},
        generation_attempts=0,model_inference_calls=0,incremental_modal_cost_usd=0,
        cumulative_planning_charge_usd=previous["cost"]["cumulative_planning_charge_usd"],
        limitations=["Same frequentist weighted-normal detector; not Bayesian SynthID or matched empirical FPR.",
            "50 paired prompt clusters, fixed keys, one model; exploratory marginal intervals without multiplicity correction.",
            "Pilot nulls have two responses per prompt; historical null prompts overlap. Keep the cohorts separate.",
            "All-success and all-failure bootstrap bounds collapse and do not establish perfect detection or zero population FPR."])
    save(OUTPUT/"summary.json",summary)
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    run()
