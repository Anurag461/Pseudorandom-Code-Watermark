"""Watermark stealing (Jovanovic et al., ICML 2024) against PRC, KGW-2.0, EXP and SynthID-Text.

E1 (this file, stage `generate`): the attacker's query data. Every scheme is
deployed with ONE fixed key, as a real API would be, on Qwen3-0.6B-Base:
  * prc      - the fixed PRC n=400, eta=0.05 key of the evaluated run (same
               artifact as the substitution study), repo sampler, BF16;
  * kgw2     - KGW gamma=0.25, delta=2.0, previous-token hash, fixed hash key;
  * exp      - Kuditipudi EXP, one fixed key (n=256), offset 0, paper code;
  * synthid  - SynthID-Text, the fixed MarkLLM/DITTO key set;
  * base     - unwatermarked sampling: the attacker's own base responses on
               the query prompts, and the defender's calibration nulls.
Prompts are 50-token prefixes of C4 RealNewsLike *train* documents (the
evaluation prompts come from the validation split), selected with the same
>=562-token rule as get_promtps.py. Completions are 400 tokens, temperature 1,
full vocabulary, EOS suppressed for the HF schemes.
"""
import hashlib
import json
from pathlib import Path

import modal

from kth_baselines import (EOS, M, MODEL_DIR, PROMPT_TOKENS, base_image, _cpu_kgw_processor,
                           synthid_processor)
from modal_run import fixed_image

OUT = "stealing_v1"
NUM_QUERY, NUM_CALIB = 30_000, 5_000
MIN_DOC_TOKENS = 562
CHUNK = 250                       # prompts per work item
HF_BATCH = {"kgw2": 50, "synthid": 25, "exp": 25, "base": 50}
EXP_KEY_SEED = 42                 # the single deployed EXP key
KEY_LENGTH = 256
PRC_ARTIFACT = {"path": "objects/sha256/29/29ec8adc932f6836be978f6e3ff2359c2e53024d5a4865e5c97e61a718f21af2",
                "sha256": "29ec8adc932f6836be978f6e3ff2359c2e53024d5a4865e5c97e61a718f21af2"}
SCHEMES = ("prc", "kgw2", "exp", "synthid", "base")

hf_image = base_image.pip_install("datasets==3.0.1").add_local_python_source(
    "attacks", "kth_baselines", "modal_run", "online_prc")  # modal_run imports online_prc at load
prc_image = fixed_image.add_local_python_source("attacks", "kth_baselines")
app = modal.App("prc-stealing")
hf_cache = modal.Volume.from_name("prc-hf-cache", create_if_missing=False)
archive = modal.Volume.from_name("prc-research-archive", create_if_missing=False)
results = modal.Volume.from_name("prc-attacks", create_if_missing=True)


def _save(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, path.with_suffix(".partial"))
    path.with_suffix(".partial").replace(path)


@app.function(image=hf_image, cpu=4, memory=16384, timeout=7200, volumes={"/cache": hf_cache, "/results": results})
def build_prompts():
    """30k query + 5k calibration prompts from C4 RealNewsLike train, disjoint documents."""
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer
    path = Path(f"/results/{OUT}/prompts.pt")
    if path.exists():
        return str(path)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    stream = load_dataset("allenai/c4", "realnewslike", split="train", streaming=True)
    prompts, docs = [], []
    for doc, example in enumerate(stream):
        tokens = tokenizer.encode(example["text"])
        if len(tokens) >= MIN_DOC_TOKENS:
            prompts.append(tokens[:PROMPT_TOKENS])
            docs.append(doc)
        if len(prompts) == NUM_QUERY + NUM_CALIB:
            break
    _save(path, {"query": torch.tensor(prompts[:NUM_QUERY]), "calib": torch.tensor(prompts[NUM_QUERY:]),
                 "query_docs": docs[:NUM_QUERY], "calib_docs": docs[NUM_QUERY:],
                 "source": "allenai/c4 realnewslike train (streaming order)", "min_doc_tokens": MIN_DOC_TOKENS})
    results.commit()
    return str(path)


def _chunk_path(scheme, split, start):
    return Path(f"/results/{OUT}/generations/{scheme}/{split}_{start:05d}.pt")


@app.function(image=hf_image, gpu="A10G", memory=32768, timeout=5400, max_containers=10,
              retries=modal.Retries(max_retries=2), volumes={"/cache": hf_cache, "/results": results})
def generate_hf(scheme, split, start):
    """KGW-2.0, EXP, SynthID-Text or unwatermarked completions for one chunk of prompts."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
    path = _chunk_path(scheme, split, start)
    if path.exists():
        return str(path)
    results.reload()
    prompts = torch.load(f"/results/{OUT}/prompts.pt")[split][start:start + CHUNK]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float32).cuda().eval()
    vocab_size = model.get_output_embeddings().weight.shape[0]
    torch.manual_seed(int.from_bytes(hashlib.sha256(f"{scheme}:{split}:{start}".encode()).digest()[:4], "little"))
    outputs = []
    for b in range(0, len(prompts), HF_BATCH[scheme]):
        batch = prompts[b:b + HF_BATCH[scheme]]
        if scheme == "exp":
            from watermarking.generation import generate
            from watermarking.gumbel.key import gumbel_key_func
            from watermarking.gumbel.sampler import gumbel_sampling
            # Paper's generate() with every row using the one deployed key.
            seeds = torch.full((len(batch),), EXP_KEY_SEED)
            out = generate(model, batch, vocab_size, KEY_LENGTH, M, seeds, gumbel_key_func, gumbel_sampling,
                           random_offset=False)
        else:
            processors = []
            if scheme == "kgw2":
                processors = [_cpu_kgw_processor(list(tokenizer.get_vocab().values()))]
            elif scheme == "synthid":
                processors = [synthid_processor(torch.device("cuda"))]  # fresh state per batch
            out = model.generate(batch.cuda(), attention_mask=torch.ones_like(batch).cuda(), do_sample=True,
                                 max_new_tokens=M, min_new_tokens=M, top_k=0, top_p=1.0, temperature=1.0,
                                 pad_token_id=EOS, logits_processor=LogitsProcessorList(processors)).cpu()
        outputs.append(out[:, PROMPT_TOKENS:PROMPT_TOKENS + M].to(torch.int32))
    tokens = torch.cat(outputs)
    if tokens.shape != (len(prompts), M):
        raise ValueError(f"expected {(len(prompts), M)}, got {tuple(tokens.shape)}")
    _save(path, {"scheme": scheme, "split": split, "start": start, "tokens": tokens})
    results.commit()
    return str(path)


@app.function(image=prc_image, gpu="A10G", memory=32768, timeout=5400, max_containers=10,
              retries=modal.Retries(max_retries=2),
              volumes={"/cache": hf_cache, "/results": results, "/archive": archive})
def generate_prc(split, start):
    """PRC completions under the evaluated n=400 deployment key, with the repo's own sampler."""
    import hashlib
    import os
    import torch
    path = _chunk_path("prc", split, start)
    if path.exists():
        return str(path)
    results.reload()
    os.environ["PRC_MODEL_SIZE"], os.environ["PRC_MODEL_VARIANT"] = "0.6B", "base"
    from modal_run import _redetect_load
    artifact_path = Path("/archive") / PRC_ARTIFACT["path"]
    if hashlib.sha256(artifact_path.read_bytes()).hexdigest() != PRC_ARTIFACT["sha256"]:
        raise ValueError("PRC deployment artifact changed")
    artifact = _redetect_load(artifact_path)
    import watermark_expt as we
    we.partition = artifact["partition"].to(we.device)
    prompts = torch.load(f"/results/{OUT}/prompts.pt")[split][start:start + CHUNK]
    outputs = []
    for b in range(0, len(prompts), 50):
        batch = prompts[b:b + 50].to(we.device)
        tokens, _ = we.generate_batch_and_collect(we.model, batch, M, artifact["encoding_key"], we.partition,
                                                  watermark=True)
        outputs.append(tokens[:, :M].cpu().to(torch.int32))
    tokens = torch.cat(outputs)
    if tokens.shape != (len(prompts), M):
        raise ValueError(f"expected {(len(prompts), M)}, got {tuple(tokens.shape)}")
    _save(path, {"scheme": "prc", "split": split, "start": start, "tokens": tokens,
                 "artifact_sha256": PRC_ARTIFACT["sha256"]})
    results.commit()
    return str(path)


def _jobs(schemes, n_query):
    jobs = []
    for scheme in schemes:
        for start in range(0, n_query, CHUNK):
            jobs.append((scheme, "query", start))
        if scheme == "base":
            jobs += [("base", "calib", start) for start in range(0, NUM_CALIB, CHUNK)]
    return jobs


@app.function(image=hf_image, cpu=1, memory=2048, timeout=86400)
def orchestrate(schemes: list, n_query: int):
    """Runs in the cloud so a sleeping laptop cannot stop it (deploy the app, then spawn this)."""
    print(build_prompts.remote(), flush=True)
    jobs = _jobs(schemes, n_query)
    calls = [generate_prc.spawn(*j[1:]) if j[0] == "prc" else generate_hf.spawn(*j) for j in jobs]
    failed = []
    for job, call in zip(jobs, calls):
        try:
            call.get()
        except Exception as error:  # failed after retries; relaunching resumes from the cache
            failed.append((job, repr(error)))
    print(f"{len(jobs) - len(failed)}/{len(jobs)} chunks done; failed: {failed}", flush=True)
    return {"done": len(jobs) - len(failed), "failed": failed}


@app.local_entrypoint()
def generate(schemes: str = ",".join(SCHEMES), n_query: int = NUM_QUERY):
    """Foreground E1 run (for smoke tests). Use `launch` below for long runs."""
    print(orchestrate.remote(schemes.split(","), n_query))


def launch(schemes=",".join(SCHEMES), n_query=NUM_QUERY):
    """After `modal deploy stealing.py`: python -c 'import stealing; stealing.launch(...)'."""
    call = modal.Function.from_name("prc-stealing", "orchestrate").spawn(schemes.split(","), n_query)
    print("spawned", call.object_id)


# ================================================================ E2/E3: stealing and spoofing
#
# Scoring follows Jovanovic et al.'s official SpoofedProcessor.get_boosts
# (github.com/eth-sri/watermark-stealing @ b8d207d, MIT licence), with the
# settings of their previous-token (lefthash) spoofing config: w_abcd=2,
# no partial/empty components, min wm count 2 (non-empty context), empty-context
# mass threshold 7e-5, clip c=2, logits += alpha * boost. For context width
# h=1 their CountStore counts reduce to (previous token, token) pairs, so the
# counts are taken with numpy. Two attacker variants:
#   ctx1 - context = previous token (the paper's attack for KGW-2.0-style hashing);
#   pos  - context = position in the completion modulo the scheme's period
#          (EXP key length 256, PRC n=400, 400 otherwise): an adaptive variant
#          for position-keyed schemes, strictly more favourable to the attacker.
# Omitted from the paper's generation: its n-gram repetition penalty and
# graceful-conclusion processor (text quality is measured separately).

SPOOF_ALPHAS = (1.0, 2.0, 4.5, 8.0)
VARIANTS = ("ctx1", "pos")
PERIOD = {"exp": KEY_LENGTH, "prc": 400, "kgw2": 400, "synthid": 400}
JSV = dict(min_wm_count_nonempty=2, min_wm_mass_empty=7e-5, clip_at=2.0)
EVAL_PROMPTS = 500
spoof_image = hf_image.add_local_file("prompts.jsonl", "/root/prompts.jsonl")


def _query_tokens(scheme, n_query):
    import torch
    chunks = [torch.load(_chunk_path(scheme, "query", s))["tokens"] for s in range(0, n_query, CHUNK)]
    return torch.cat(chunks)[:n_query].to(torch.int64)


def pair_counts(tokens, prompts, variant, period):
    """{context: {token: count}} with context = previous token or position (h=1)."""
    import numpy as np
    tok = tokens.numpy()
    if variant == "ctx1":
        prev = np.concatenate([prompts[:, -1:].numpy(), tok[:, :-1]], axis=1)
    else:
        prev = np.broadcast_to(np.arange(tok.shape[1]) % period, tok.shape)
    keys = prev.astype(np.int64) * 1_000_000 + tok
    uniq, counts = np.unique(keys.ravel(), return_counts=True)
    table = {}
    for key, count in zip(uniq.tolist(), counts.tolist()):
        table.setdefault(key // 1_000_000, {})[key % 1_000_000] = count
    return table


def jsv_boosts(wm, base, empty):
    """SpoofedProcessor.get_boosts(normalize=True) for one context, as a sparse {token: boost}."""
    total_wm, total_base = sum(wm.values()) + 1e-6, sum(base.values()) + 1e-6
    threshold = round(JSV["min_wm_mass_empty"] * sum(base.values())) if empty else JSV["min_wm_count_nonempty"]
    enough = [t for t, c in wm.items() if c >= threshold]
    ratios = {t: (wm[t] / total_wm) / (base[t] / total_base) for t in enough if base.get(t, 0) > 0}
    top = max(1.0, max(ratios.values(), default=0.0)) + 1e-3
    ratios.update({t: top for t in enough if base.get(t, 0) == 0})
    clip = JSV["clip_at"]
    boosts = {t: min(r, clip) / clip for t, r in ratios.items() if r >= 1}
    if boosts:
        most = max(wm[t] for t in boosts)
        boosts = {t: b + wm[t] / most * 1e-4 for t, b in boosts.items()}
        peak = max(boosts.values())
        boosts = {t: b / peak for t, b in boosts.items()}
    return boosts


def stolen_table(scheme, variant, n_query):
    """Sparse boosts for every context seen in the attacker's watermarked queries."""
    import torch
    prompts = torch.load(f"/results/{OUT}/prompts.pt")["query"][:n_query]
    wm = pair_counts(_query_tokens(scheme, n_query), prompts, variant, PERIOD[scheme])
    base = pair_counts(_query_tokens("base", n_query), prompts, variant, PERIOD[scheme])
    table = {}
    for ctx, counts in wm.items():
        boosts = jsv_boosts(counts, base.get(ctx, {}), empty=False)
        if boosts:
            table[ctx] = (torch.tensor(list(boosts)), torch.tensor(list(boosts.values()), dtype=torch.float32))
    return table


def _stolen_processor(table, variant, period, alpha):
    import torch
    from transformers import LogitsProcessor

    class StolenBoost(LogitsProcessor):
        def __call__(self, input_ids, scores):
            position = input_ids.shape[1] - PROMPT_TOKENS
            for row in range(input_ids.shape[0]):
                ctx = int(input_ids[row, -1]) if variant == "ctx1" else position % period
                if ctx in table:
                    idx, boost = table[ctx]
                    scores[row, idx.to(scores.device)] += alpha * boost.to(scores.device)
            return scores

    return StolenBoost()


def _eval_prompts():
    import torch
    lines = Path("/root/prompts.jsonl").read_text().splitlines()[:EVAL_PROMPTS]
    return torch.tensor([json.loads(line)["prompt_tokens"] for line in lines])


@app.function(image=spoof_image, gpu="A10G", memory=32768, timeout=7200, max_containers=10,
              retries=modal.Retries(max_retries=2), volumes={"/cache": hf_cache, "/results": results})
def spoof(scheme, variant, n_query, alphas):
    """Learn the stolen table once, then generate 500 spoofed completions per alpha."""
    import torch
    from transformers import AutoModelForCausalLM, LogitsProcessorList
    results.reload()
    paths = {a: Path(f"/results/{OUT}/spoof/{scheme}/{variant}_N{n_query}_a{a:g}.pt") for a in alphas}
    todo = [a for a, p in paths.items() if not p.exists()]
    if not todo:
        return [str(p) for p in paths.values()]
    table = stolen_table(scheme, variant, n_query)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float32).cuda().eval()
    prompts = _eval_prompts()
    for alpha in todo:
        torch.manual_seed(int(alpha * 1000))
        outputs = []
        for b in range(0, len(prompts), 50):
            batch = prompts[b:b + 50].cuda()
            out = model.generate(batch, attention_mask=torch.ones_like(batch), do_sample=True, max_new_tokens=M,
                                 min_new_tokens=M, top_k=0, top_p=1.0, temperature=1.0, pad_token_id=EOS,
                                 logits_processor=LogitsProcessorList(
                                     [_stolen_processor(table, variant, PERIOD[scheme], alpha)])).cpu()
            outputs.append(out[:, PROMPT_TOKENS:PROMPT_TOKENS + M].to(torch.int32))
        _save(paths[alpha], {"scheme": scheme, "variant": variant, "n_query": n_query, "alpha": alpha,
                             "contexts": len(table), "tokens": torch.cat(outputs)})
        results.commit()
    return [str(p) for p in paths.values()]


# ================================================================ detection
#
# Each text set is scored with its scheme's detector under the deployed key:
#   kgw2 / synthid / exp - kth_baselines.Scorer (EXP with key seed EXP_KEY_SEED),
#       thresholded at the 1e-3 quantile of the same statistic on the 5,000
#       unwatermarked calibration completions (non-adversarial text, as in JSV);
#   prc - completion-only BF16 replay and detect_hoeffding (MAP weights) with the
#       deployment decoding key at FPR 1e-3, exactly as `redetect` scores.
# Text sets: calib (null), query500 (first 500 genuine watermarked queries,
# positive control), and every spoof file.

def _text_sets(scheme):
    import torch
    sets = {"calib": torch.cat([torch.load(_chunk_path("base", "calib", s))["tokens"]
                                for s in range(0, NUM_CALIB, CHUNK)]),
            "query500": torch.cat([torch.load(_chunk_path(scheme, "query", s))["tokens"] for s in (0, 250)])}
    spoof_dir = Path(f"/results/{OUT}/spoof/{scheme}")
    for path in sorted(spoof_dir.glob("*.pt")) if spoof_dir.exists() else []:
        sets[path.stem] = torch.load(path)["tokens"]
    return {k: v.to(torch.int64) for k, v in sets.items()}


@app.function(image=spoof_image, cpu=2, memory=8192, timeout=7200, max_containers=50,
              retries=modal.Retries(max_retries=3), volumes={"/cache": hf_cache, "/results": results})
def score_hf(scheme, name, start, stop):
    """Detector statistics (lower = more watermarked) for rows [start, stop) of one text set."""
    import torch
    from transformers import AutoTokenizer
    from kth_baselines import Scorer
    path = Path(f"/results/{OUT}/scores/{scheme}/{name}_{start:05d}.json")
    if path.exists():
        return str(path)
    results.reload()
    tokens = _text_sets(scheme)[name][start:stop]
    scorer = Scorer(scheme, 151936, AutoTokenizer.from_pretrained(MODEL_DIR))
    stats = [scorer(t, EXP_KEY_SEED) for t in tokens]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"scheme": scheme, "name": name, "start": start, "stats": stats}))
    results.commit()
    return str(path)


@app.function(image=prc_image, gpu="A10G", memory=32768, timeout=7200, max_containers=10,
              retries=modal.Retries(max_retries=2),
              volumes={"/cache": hf_cache, "/results": results, "/archive": archive})
def score_prc(name, start, stop):
    """PRC detection of rows [start, stop) of one text set with the deployment key."""
    import hashlib
    import os
    import torch
    path = Path(f"/results/{OUT}/scores/prc/{name}_{start:05d}.json")
    if path.exists():
        return str(path)
    results.reload()
    os.environ["PRC_MODEL_SIZE"], os.environ["PRC_MODEL_VARIANT"] = "0.6B", "base"
    from modal_run import _redetect_load
    from detectors import detect_hoeffding
    from qwen import completion_only_partition_trace_batch
    artifact_path = Path("/archive") / PRC_ARTIFACT["path"]
    if hashlib.sha256(artifact_path.read_bytes()).hexdigest() != PRC_ARTIFACT["sha256"]:
        raise ValueError("PRC deployment artifact changed")
    artifact = _redetect_load(artifact_path)
    import watermark_expt as we
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = we.model.eval().requires_grad_(False)
    if next(model.parameters()).dtype != torch.bfloat16:
        raise ValueError("PRC redetection replays in BF16")
    tokens = _text_sets("prc")[name][start:stop]
    partition = artifact["partition"]
    rows = []
    for b in range(0, len(tokens), 50):
        batch = tokens[b:b + 50]
        trace = completion_only_partition_trace_batch(model, batch.to(we.device),
                                                      partition[1].to(torch.bfloat16).to(we.device), "static")
        for row, p in zip(batch, trace.cpu().numpy()):
            decision, info = detect_hoeffding(artifact["decoding_key"], row, p, partition, fpr=1e-3,
                                              weight="map", return_info=True)
            rows.append({"decision": bool(decision), "statistic": info["statistic"], "V": info["V"]})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"scheme": "prc", "name": name, "start": start, "rows": rows}))
    results.commit()
    return str(path)


@app.function(image=spoof_image, cpu=1, memory=4096, timeout=86400, volumes={"/results": results})
def orchestrate_attack(schemes: list, variants: list, n_query: int, alphas: list):
    """E3 in the cloud: spoof for every scheme/variant, then score calib, positives and spoofs."""
    calls = [spoof.spawn(s, v, n_query, alphas) for s in schemes for v in variants]
    for call in calls:
        try:
            print(call.get(), flush=True)
        except Exception as error:
            print("FAILED spoof", repr(error), flush=True)
    results.reload()
    jobs = []
    for scheme in schemes:
        for name, tokens in _text_sets(scheme).items():
            step = 500 if scheme == "prc" else 100
            jobs += [(scheme, name, s, min(s + step, len(tokens))) for s in range(0, len(tokens), step)]
    calls = [score_prc.spawn(*j[1:]) if j[0] == "prc" else score_hf.spawn(*j) for j in jobs]
    failed = 0
    for call in calls:
        try:
            call.get()
        except Exception as error:
            failed += 1
            print("FAILED score", repr(error), flush=True)
    print(f"scored {len(calls) - failed}/{len(calls)}", flush=True)


def launch_attack(schemes="prc,kgw2,exp,synthid", variants="ctx1,pos", n_query=10_000, alphas=SPOOF_ALPHAS):
    call = modal.Function.from_name("prc-stealing", "orchestrate_attack").spawn(
        schemes.split(","), variants.split(","), n_query, list(alphas))
    print("spawned", call.object_id)
