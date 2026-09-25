"""Substitution robustness at 4096 tokens: PRC (fixed n=T=4096, eta=0.05) vs EXP, KGW-2.0, SynthID-Text.

Sample sizes: PRC uses all 500 prompts (500 watermarked + 500 unwatermarked texts); each baseline
uses the first 200 prompts (200 watermarked + 200 unwatermarked texts), to fit the budget.

Same design as kth_baselines.py (Qwen3-0.6B-Base, C4 RealNewsLike prompts from prompts.jsonl, paper
code for EXP/KGW, transformers SynthID, identical seeded substitutions for every scheme via
attacks.apply_attack), with three changes forced by the length:
  * completions are M=4096 tokens;
  * the unwatermarked texts are the fixed-PRC null cohort (Qwen3-0.6B-Base samples of the same
    500 prompts, the T=4096 null cache), attacked like the watermarked
    texts, and used for every scheme's empirical FPR;
  * human C4 continuations are too short for the paper's empirical null, so every baseline uses
    an analytic p-value under its own null, thresholded at 1e-3 like PRC's proven Hoeffding bound:
      kgw2    - the detector's one-sided z-test (Kirchenbauer et al.);
      synthid - one-sided z-test of the mean g-value over unmasked (position, depth) pairs,
                g ~ Bernoulli(1/2) under the null;
      exp     - per key shift j, S_j = sum_t -log(1 - xi[(j+t) mod n, y_t]) ~ Gamma(m, 1) under the
                null; p = min(1, n * P[Gamma(m, 1) >= max_j S_j]) (Bonferroni over the n=256 shifts).
PRC rows come from `modal_run.py::redetect --attack` on the manifest written by `prc_manifest`.
"""
import hashlib
import json
from pathlib import Path

import modal

from kth_baselines import (EOS, KEY_LENGTH, KEY_SEED, KGW_DELTA, KGW_GAMMA, MODEL_DIR, NUM_PROMPTS,
                           PROMPT_TOKENS, _cpu_kgw_processor, base_image, key_seeds, load_prompts,
                           synthid_processor)
from modal_run import fixed_image

M = 4096
RATES = (0.05, 0.1, 0.15, 0.2)
SCHEMES = ("exp", "kgw2", "synthid")
ATTACK_VOCAB = 151665
OUT = "kth_long_v1"
CHUNK = 10
BASELINE_PROMPTS = 200   # baselines use prompts 0..199; PRC uses all 500 (prompts 0..499)
PRC_TAG = "n4096_t3_eta0.05_T4096_r4055"
NULL_DIR = "_nulls/T4096"  # the null cohort the fixed-run planner selected for T=4096
MANIFEST = "outputs/attacks/prc_fixed_n4096_eta005_manifest.json"
RESULTS_CSV = "outputs/attacks/kth_long_results.csv"
ALPHA = 1e-3

app = modal.App("prc-kth-long")
hf_cache = modal.Volume.from_name("prc-hf-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("prc-data", create_if_missing=True)
results = modal.Volume.from_name("prc-attacks", create_if_missing=True)
image = base_image.add_local_file("prompts_10k.jsonl", "/root/prompts_10k.jsonl").add_local_python_source(
    "attacks", "kth_baselines", "kth_long", "modal_run", "online_prc")
prc_image = fixed_image.add_local_python_source("attacks", "kth_baselines", "kth_long")


def attack_spec(rate, seed=0):
    return {"kind": "substitution", "rate": float(rate), "seed": seed, "vocab_size": ATTACK_VOCAB}


def attack_id(attack):
    return "clean" if attack is None else f"substitution{attack['rate']:g}_s{attack['seed']}"


def load_null(prompt_idx):
    """First M tokens of the prompt-matched unwatermarked 8192-token sample."""
    import sys
    import numpy as np
    import torch
    sys.modules.setdefault("numpy._core", np.core)  # null records were written under NumPy 2
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
    record = torch.load(f"/data/{NULL_DIR}/null_{prompt_idx:04d}.pt", weights_only=False, map_location="cpu")
    if record["watermark"] or record["prompt_idx"] != prompt_idx or len(record["tokens"]) < M:
        raise ValueError(f"unexpected null record {prompt_idx}")
    return record["tokens"][:M].to(torch.int64)


# ---------------------------------------------------------------- generation

@app.function(image=image, gpu="A10G", memory=32768, timeout=7200, max_containers=10,
              retries=modal.Retries(max_retries=2), volumes={"/cache": hf_cache, "/results": results})
def generate_chunk(scheme, start):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
    path = Path(f"/results/{OUT}/generations/{scheme}/{start:04d}.pt")
    if path.exists():
        return str(path)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float32).cuda().eval()
    vocab_size = model.get_output_embeddings().weight.shape[0]
    idx = list(range(start, min(start + CHUNK, NUM_PROMPTS)))
    prompts = load_prompts()
    prompt_ids = torch.tensor([prompts[i]["prompt_tokens"] for i in idx])
    seeds = key_seeds()[idx]
    torch.manual_seed(1000 + start)
    if scheme == "exp":
        from watermarking.generation import generate
        from watermarking.gumbel.key import gumbel_key_func
        from watermarking.gumbel.sampler import gumbel_sampling
        out = generate(model, prompt_ids, vocab_size, KEY_LENGTH, M, seeds, gumbel_key_func, gumbel_sampling,
                       random_offset=False)
    else:
        processor = (_cpu_kgw_processor(list(tokenizer.get_vocab().values())) if scheme == "kgw2"
                     else synthid_processor(torch.device("cuda")))
        out = model.generate(prompt_ids.cuda(), attention_mask=torch.ones_like(prompt_ids).cuda(), do_sample=True,
                             max_new_tokens=M, min_new_tokens=M, top_k=0, top_p=1.0, temperature=1.0,
                             pad_token_id=EOS, logits_processor=LogitsProcessorList([processor])).cpu()
    tokens = out[:, PROMPT_TOKENS:PROMPT_TOKENS + M]
    if tokens.shape != (len(idx), M):
        raise ValueError(f"expected {M} new tokens, got {tuple(tokens.shape)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"scheme": scheme, "prompt_idx": idx, "seeds": seeds, "tokens": tokens}, path.with_suffix(".partial"))
    path.with_suffix(".partial").replace(path)
    results.commit()
    return str(path)


# ---------------------------------------------------------------- analytic p-values

def exp_pvalue(tokens, seed, vocab_size=151936):
    import torch
    from scipy.special import gammaincc
    from watermarking.gumbel.key import gumbel_key_func
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    xi, _ = gumbel_key_func(generator, KEY_LENGTH, vocab_size)
    m = len(tokens)
    rows = (torch.arange(KEY_LENGTH)[:, None] + torch.arange(m)[None, :]) % KEY_LENGTH
    u = xi[rows, tokens[None, :].expand(KEY_LENGTH, m)].double()
    s_max = float((-torch.log1p(-u)).sum(1).max())
    return min(1.0, KEY_LENGTH * float(gammaincc(m, s_max))), s_max


def synthid_pvalue(processor, tokens):
    import math
    from scipy.stats import norm
    from kth_baselines import SYNTHID
    ids = tokens.unsqueeze(0)
    g = processor.compute_g_values(ids).float()
    mask = processor.compute_context_repetition_mask(ids)
    mask = mask * processor.compute_eos_token_mask(ids, EOS)[:, SYNTHID["ngram_len"] - 1:]
    count = float(mask.sum()) * g.shape[-1]
    if not count:
        return 1.0, 0.5
    mean = float((g * mask[..., None]).sum()) / count
    return float(norm.sf((mean - 0.5) / math.sqrt(0.25 / count))), mean


@app.function(image=image, cpu=2, memory=8192, timeout=7200, max_containers=50,
              retries=modal.Retries(max_retries=3),
              volumes={"/cache": hf_cache, "/results": results, "/data": data_vol})
def score_chunk(scheme, attack, start):
    import torch
    from attacks import apply_attack
    path = Path(f"/results/{OUT}/scores/{scheme}/{attack_id(attack)}_{start:04d}.json")
    if path.exists():
        return str(path)
    results.reload()
    gen = torch.load(f"/results/{OUT}/generations/{scheme}/{start:04d}.pt")
    if scheme == "kgw2":
        from transformers import AutoTokenizer
        from watermarking.kirchenbauer.watermark_processor import WatermarkDetector
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()), gamma=KGW_GAMMA,
                                     seeding_scheme="simple_1", device=torch.device("cpu"), tokenizer=tokenizer,
                                     z_threshold=1.5, normalizers=[], ignore_repeated_bigrams=False)
    elif scheme == "synthid":
        processor = synthid_processor(torch.device("cpu"))
    rows = []
    for row, prompt_idx in enumerate(gen["prompt_idx"]):
        for source, tokens in (("wm", gen["tokens"][row].to(torch.int64)), ("null", load_null(prompt_idx))):
            text = tokens if attack is None else apply_attack(tokens, attack, source, prompt_idx)[:M]
            if scheme == "kgw2":
                score = detector._score_sequence(text)
                p, stat = float(score["p_value"]), float(score["z_score"])
            elif scheme == "synthid":
                p, stat = synthid_pvalue(processor, text)
            else:
                p, stat = exp_pvalue(text, gen["seeds"][row])
            rows.append({"source": source, "prompt_idx": prompt_idx, "p": p, "stat": stat})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"scheme": scheme, "attack": attack, "start": start, "rows": rows}))
    results.commit()
    return str(path)


@app.local_entrypoint()
def run(stage: str = "smoke", schemes: str = ",".join(SCHEMES)):
    """smoke: one generation chunk per scheme scored clean and at the strongest rate; full: all."""
    chosen = schemes.split(",")
    starts = [0] if stage == "smoke" else list(range(0, BASELINE_PROMPTS, CHUNK))
    attacks = [None, attack_spec(max(RATES))] if stage == "smoke" else [None] + [attack_spec(r) for r in RATES]
    for out in generate_chunk.starmap([(s, st) for s in chosen for st in starts], return_exceptions=True):
        print("generated", out if not isinstance(out, Exception) else f"FAILED {out!r}"[:300], flush=True)
    jobs = [(s, a, st) for s in chosen for a in attacks for st in starts]
    done = 0
    for out in score_chunk.starmap(jobs, return_exceptions=True):
        if isinstance(out, Exception):
            print("FAILED", repr(out)[:300], flush=True)
        else:
            done += 1
    print(f"scored {done}/{len(jobs)}")


# ---------------------------------------------------------------- PRC manifest

@app.function(image=prc_image, cpu=4, memory=16384, timeout=3600, volumes={"/data": data_vol, "/cache": hf_cache})
def prc_manifest_case():
    """Frozen redetection case for the new fixed n=4096 eta=0.05 run (wm files + prompt-matched nulls)."""
    import sys
    import numpy as np
    import torch
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)

    def ref(path):
        data = (Path("/data") / path).read_bytes()
        return {"volume": "data", "path": path, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}

    records = []
    for source, pattern in (("wm", f"{PRC_TAG}/wm/wm_{{:04d}}.pt"), ("null", f"{NULL_DIR}/null_{{:04d}}.pt")):
        for i in range(NUM_PROMPTS):
            path = pattern.format(i)
            record = torch.load(Path("/data") / path, weights_only=False, map_location="cpu")
            tokens = record["tokens"][:M].to(torch.int64).contiguous()
            if record["watermark"] != (source == "wm") or record["prompt_idx"] != i or len(tokens) != M:
                raise ValueError(f"unexpected record {path}")
            records.append({"source": source, "prompt_idx": i, "file": ref(path),
                            "tokens_sha256": hashlib.sha256(tokens.numpy().tobytes()).hexdigest()})

    cache = Path("/cache/models/Qwen3-0.6B-Base")
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    model = {"size": "0.6B", "id": "Qwen/Qwen3-0.6B-Base", "dtype": "bfloat16",
             "cache_directory": "models/Qwen3-0.6B-Base",
             "revision": (cache / ".cache/huggingface/download/model.safetensors.metadata").read_text().splitlines()[0],
             "weights_sha256": digest(cache / "model.safetensors"), "tokenizer_sha256": digest(cache / "tokenizer.json")}
    case = {"id": "fixed_0p6b_n4096_eta005", "generation_model": "Qwen3-0.6B-Base", "construction": "fixed",
            "artifact": ref(f"{PRC_TAG}/artifacts.pt"), "lengths": [M], "fpr": ALPHA,
            "fpr_policy": "block_or_bonferroni", "weights": ["map", "entropy"], "batch_size": 100,
            "cache": "static", "records": records}
    return {"protocol": "completion_only_raw_abstain_v1", "schema_version": 1, "model": model, "cases": [case]}


@app.local_entrypoint()
def prc_manifest():
    manifest = prc_manifest_case.remote()
    reference = json.loads(Path("outputs/redetection/.archive/restored/prompt_free/manifests/pilots.json")
                           .read_text())["model"]
    print("detector checkpoint identical to the frozen pilots manifest:", manifest["model"] == reference)
    Path(MANIFEST).write_text(json.dumps(manifest, indent=1))
    print("wrote", MANIFEST, len(manifest["cases"][0]["records"]), "records")


# ================================================================ cloud orchestration
#
# Everything below runs inside Modal, so the run survives the local client going away:
#   modal deploy modal_run.py     (provides FixedGenerationModel and the redetection functions)
#   modal deploy kth_long.py
#   python -c 'import kth_long; kth_long.launch()'
# Every work item is cached per file (wm_XXXX.pt, generation chunks, score chunks, redetection
# traces), so relaunching resumes.

PRC_APP = "prc-watermark"
PRC_BATCH = 16          # 4096-token PRC generation fits 16 rows on an A10G (64 ran out of memory)
REDETECT_GPU = "A100-80GB"


def _wait(calls, label):
    failed = 0
    for call in calls:
        try:
            call.get()
        except Exception as error:  # failed after retries; relaunching resumes from the caches
            failed += 1
            print(f"FAILED {label}: {error!r}"[:400], flush=True)
    print(f"{label}: {len(calls) - failed}/{len(calls)} done", flush=True)
    return failed


def _redetect_attacked(manifest, attack, execution):
    """modal_run.py::redetect for one case, run cloud-side (preparation, validated replay, scoring)."""
    import json as _json
    from modal_run import model_cls_options
    prepare = modal.Function.from_name(PRC_APP, "prepare_redetection")
    finish = modal.Function.from_name(PRC_APP, "finish_redetection")
    spec, case = manifest["model"], dict(manifest["cases"][0])
    if attack is not None:
        case = {**case, "id": f"{case['id']}__{attack_id(attack)}", "attack": attack}
    prepared = prepare.remote(case, spec, execution)
    worker = modal.Cls.from_name(PRC_APP, "RedetectionModel").with_options(
        **{**model_cls_options("0.6B", REDETECT_GPU, 10), "memory": 8192, "scaledown_window": 2})(
        entropy_model_size="0.6B", generation_model_size="0.6B", trace_kv_cache_implementation="static",
        completion_model=_json.dumps(spec, sort_keys=True))
    representatives = {}
    for batch in prepared["batches"]:
        representatives.setdefault(batch["identity"]["count"], batch)
    for batch in representatives.values():  # one independent reference replay per batch shape
        worker.redetect_batch.remote(batch, validate=True)
    list(worker.redetect_batch.map(prepared["batches"]))
    result = finish.remote(prepared)
    return {"case": case["id"], "attack": attack, "root": prepared["root"], "counts": result["counts"]}


@app.function(image=image, cpu=1, memory=4096, timeout=86400,
              volumes={"/data": data_vol, "/results": results})
def orchestrate_long(code_fingerprint: str, execution: dict, schemes: list, n_baseline: int = BASELINE_PROMPTS):
    """PRC generation + baseline generation in parallel, then baseline scoring and PRC redetection."""
    import os
    # 1. Generation: remaining PRC watermarked texts and all baseline chunks, concurrently.
    data_vol.reload()
    wm_dir = Path(f"/data/{PRC_TAG}/wm")
    missing = [i for i in range(NUM_PROMPTS) if not (wm_dir / f"wm_{i:04d}.pt").exists()]
    prc_model = modal.Cls.from_name(PRC_APP, "FixedGenerationModel").with_options(gpu="A10G", max_containers=5)(
        tag=PRC_TAG, model_size="0.6B", code_fingerprint_sha256=code_fingerprint)
    prc_calls = [prc_model.generate_wm.spawn(missing[i:i + PRC_BATCH]) for i in range(0, len(missing), PRC_BATCH)]
    gen_calls = [generate_chunk.spawn(s, st) for s in schemes for st in range(0, n_baseline, CHUNK)]
    _wait(gen_calls, "baseline generation")
    # 2. Baseline scoring (CPU) while PRC generation finishes.
    attacks = [None] + [attack_spec(r) for r in RATES]
    score_calls = [score_chunk.spawn(s, a, st) for s in schemes for a in attacks
                   for st in range(0, n_baseline, CHUNK)]
    if _wait(prc_calls, "PRC generation"):
        raise RuntimeError("PRC generation incomplete; relaunch to resume")
    # 3. PRC: frozen manifest, then attacked redetection per setting (sequential; each uses up to 10 GPUs).
    manifest = prc_manifest_case.remote()
    Path(f"/results/{OUT}/prc_manifest.json").parent.mkdir(parents=True, exist_ok=True)
    Path(f"/results/{OUT}/prc_manifest.json").write_text(json.dumps(manifest))
    results.commit()
    outcomes = []
    for attack in attacks:
        outcomes.append(_redetect_attacked(manifest, attack, execution))
        print(json.dumps(outcomes[-1]), flush=True)
        Path(f"/results/{OUT}/prc_redetection.json").write_text(json.dumps(outcomes, indent=1))
        results.commit()
    _wait(score_calls, "baseline scoring")
    return outcomes


@app.function(image=image, cpu=1, memory=4096, timeout=86400, volumes={"/results": results})
def orchestrate_prc_rates(execution: dict, rates: list):
    """Extra PRC-only substitution rates on the same frozen manifest, appended to prc_redetection.json."""
    results.reload()
    manifest = json.loads(Path(f"/results/{OUT}/prc_manifest.json").read_text())
    path = Path(f"/results/{OUT}/prc_redetection.json")
    outcomes = json.loads(path.read_text())
    done = {None if o["attack"] is None else o["attack"]["rate"] for o in outcomes}
    for rate in rates:
        if rate in done:
            continue
        outcomes.append(_redetect_attacked(manifest, attack_spec(rate), execution))
        print(json.dumps(outcomes[-1]), flush=True)
        path.write_text(json.dumps(outcomes, indent=1))
        results.commit()
    return outcomes


def _execution():
    """Committed code identity recorded with every redetection run."""
    import subprocess
    import modal_run
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    for name in modal_run.EXECUTION_FILES:
        if subprocess.check_output(["git", "show", f"{commit}:{name}"]) != Path(name).read_bytes():
            raise ValueError(f"commit execution code before running: {name}")
    return {"git_commit": commit, "files": {p: modal_run._redetect_sha(p) for p in modal_run.EXECUTION_FILES},
            "gpu": REDETECT_GPU, "allocator": "expandable_segments:True"}


def launch_prc_rates(rates="0.3"):
    """PRC-only extra rates (baselines were not run at these rates)."""
    call = modal.Function.from_name("prc-kth-long", "orchestrate_prc_rates").spawn(
        _execution(), [float(r) for r in rates.split(",")])
    print("spawned", call.object_id)


def launch(schemes=",".join(SCHEMES)):
    """Local launcher: records the committed code identity, then spawns the deployed orchestrator."""
    import modal_run
    execution = _execution()
    fingerprint = modal_run._fixed_local_code_fingerprint()["sha256"]
    call = modal.Function.from_name("prc-kth-long", "orchestrate_long").spawn(fingerprint, execution,
                                                                           schemes.split(","), BASELINE_PROMPTS)
    print("spawned", call.object_id)


# ================================================================ summary

def wilson(k, n, z=1.959964):
    """95% Wilson score interval for k successes in n trials."""
    import math
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    centre, half = (p + z * z / (2 * n)) / (1 + z * z / n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


@app.local_entrypoint()
def summarize():
    """Detection rates at FPR 1e-3 with 95% Wilson intervals; baselines n=200 per cohort, PRC n=500."""
    import csv
    read = lambda p: json.loads(b"".join(results.read_file(p)))
    rows = []

    def add(scheme, rate, wm_hits, wm_n, null_hits, null_n, method):
        lo, hi = wilson(wm_hits, wm_n)
        flo, fhi = wilson(null_hits, null_n)
        rows.append({"scheme": scheme, "substitution rate": rate, "T": M,
                     "watermarked texts": wm_n, "unwatermarked texts": null_n,
                     "TPR@1e-3": f"{wm_hits}/{wm_n} ({wm_hits / wm_n:.1%})", "TPR 95% CI": f"[{lo:.1%}, {hi:.1%}]",
                     "FPR@1e-3": f"{null_hits}/{null_n} ({null_hits / null_n:.1%})", "FPR 95% CI": f"[{flo:.1%}, {fhi:.1%}]",
                     "threshold": method})

    methods = {"exp": "analytic: Gamma tail x 256 shifts (Bonferroni) <= 1e-3",
               "kgw2": "analytic: KGW one-sided z-test p <= 1e-3",
               "synthid": "analytic: mean g-value z-test p <= 1e-3"}
    for scheme in SCHEMES:
        for attack in [None] + [attack_spec(r) for r in RATES]:
            recs = []
            for st in range(0, BASELINE_PROMPTS, CHUNK):
                recs += read(f"{OUT}/scores/{scheme}/{attack_id(attack)}_{st:04d}.json")["rows"]
            wm = [r["p"] <= ALPHA for r in recs if r["source"] == "wm"]
            null = [r["p"] <= ALPHA for r in recs if r["source"] == "null"]
            if len(wm) != BASELINE_PROMPTS or len(null) != BASELINE_PROMPTS:
                raise ValueError(f"{scheme} {attack_id(attack)}: expected {BASELINE_PROMPTS} texts per cohort")
            add(scheme, 0.0 if attack is None else attack["rate"], sum(wm), len(wm), sum(null), len(null),
                methods[scheme])
    for outcome in read(f"{OUT}/prc_redetection.json"):
        rate = 0.0 if outcome["attack"] is None else outcome["attack"]["rate"]
        for weight in ("map", "entropy"):
            counts = outcome["counts"][str(M)][weight]
            add(f"prc_{weight}", rate, counts["wm"]["detected"], counts["wm"]["count"], counts["null"]["detected"],
                counts["null"]["count"],
                f"proven Hoeffding FPR <= 1e-3 (fixed n=T=4096, eta=0.05); run={outcome['root']}")
    rows.sort(key=lambda r: (r["substitution rate"], r["scheme"]))
    with Path(RESULTS_CSV).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(f"{r['scheme']:8s} rate {r['substitution rate']:<5g} TPR {r['TPR@1e-3']:>16s} {r['TPR 95% CI']:>17s}  "
              f"FPR {r['FPR@1e-3']:>14s} {r['FPR 95% CI']}")
