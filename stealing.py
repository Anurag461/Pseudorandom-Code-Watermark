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
import json
from pathlib import Path

import modal

from kth_baselines import (EOS, M, MODEL_DIR, PROMPT_TOKENS, image as kth_image, _cpu_kgw_processor,
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

hf_image = kth_image.pip_install("datasets==3.0.1").add_local_python_source("kth_baselines", "modal_run")
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
    torch.manual_seed(hash((scheme, split, start)) % 2**31)
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


@app.function(image=fixed_image, gpu="A10G", memory=32768, timeout=5400, max_containers=10,
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


@app.local_entrypoint()
def generate(schemes: str = ",".join(SCHEMES), n_query: int = NUM_QUERY):
    """E1: build prompts, then generate query (and calibration) completions. Cached per chunk."""
    print(build_prompts.remote())
    jobs = _jobs(schemes.split(","), n_query)
    hf = [j for j in jobs if j[0] != "prc"]
    prc = [j[1:] for j in jobs if j[0] == "prc"]
    calls = [generate_hf.spawn(*j) for j in hf] + [generate_prc.spawn(*j) for j in prc]
    failed = 0
    for call in calls:
        try:
            print(call.get(), flush=True)
        except Exception as error:  # a chunk failed after retries; rerunning resumes it
            failed += 1
            print("FAILED", repr(error), flush=True)
    print(f"{len(calls) - failed}/{len(calls)} chunks done")
