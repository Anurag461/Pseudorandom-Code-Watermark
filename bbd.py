"""Black-box watermark detection tests (Gloaguen, Jovanovic, Staab, Vechev, ICLR 2025) against PRC.

Reimplements the paper's three tests from its algorithms and released code
(github.com/eth-sri/watermark-detection @ d4ca66f, which has no licence file, so
nothing is copied verbatim except the prompts):
  * Red-Green      - chat prompt 'Complete the sentence "{prefix} {d*H}" ...' over
                     10 prefixes x 9 digits, <=65 new tokens, answer parsed by
                     identify_fruit (exactly one listed fruit, appearing once);
  * Fixed-Sampling - raw prompt "This is the story of", 100 new tokens,
                     count distinct completions (rarefaction + Mann-Whitney U);
  * Cache          - not applicable to any scheme here (no cache); omitted.
Victim: Qwen3-0.6B (chat model), thinking disabled, temperature 1, full vocabulary.
Responses are generated to a fixed length and cut at the first end-of-turn token,
which equals stopping there because generation is causal.

Schemes: unwatermarked, online PRC (eta=0.05 key), KGW-2.0, SynthID-Text, EXP with
the random key offset. Online PRC draws its sampling randomness from a per-document
seed; `seed_mode="fresh"` gives every response its own document id (a correct
deployment), `seed_mode="prompt"` ties the seed to the prompt (a flawed one).
"""
import hashlib
import json
import os
from pathlib import Path

import modal

from kth_baselines import EOS, base_image, _cpu_kgw_processor, synthid_processor
from modal_run import fixed_image

OUT = "bbd_v1"
CHAT_MODEL = "Qwen/Qwen3-0.6B"
CHAT_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"  # pinned; loaded by revision in every workspace
IM_END = 151645
STOP = (IM_END, EOS)
PRC_ONLINE_ARTIFACT = ("online_causal_prc_v1/qwen3_0p6b_base/"
                       "n400_T400_t3_eta0.05_rr99of100_sampler-poscdf-v1/artifacts.pt")
BBD_SEED = 20260924
EXP_KEY_SEED, EXP_KEY_LENGTH = 42, 256

# Red-Green probe word lists from the paper's released code (per-model lists); FORMAT "" as for Qwen2.
WORD_LISTS = {
    "peaches": (["peaches", "plums", "cherries", "apricots"], "strawberries"),
    "mangoes": (["mangoes", "pineapples", "papayas", "kiwis"], "strawberries"),
    "berries": (["strawberries", "blueberries", "raspberries", "blackberries"], "apples"),
    "apples": (["apples", "bananas", "oranges", "pears"], "strawberries"),
}
FRUITS, EXAMPLE = WORD_LISTS["peaches"]
FORMAT = ""
PREFIXES = ["I ate", "I chose", "I picked", "I selected", "I took", "I went for", "I settled on", "I got",
            "I gathered", "I harvested"]
RG_MAX_NEW = 65
FS_PROMPT, FS_MAX_NEW = "This is the story of", 100

hf_image = base_image.env({"HF_HUB_CACHE": "/cache"}).add_local_python_source(
    "attacks", "kth_baselines", "modal_run", "online_prc")
prc_image = fixed_image.add_local_python_source("attacks", "kth_baselines")
app = modal.App("prc-bbd")
# BBD_GPU=cpu runs generation on CPU containers (small pilots while other jobs hold the GPU quota).
DEVICE = os.environ.get("BBD_GPU", "A10G")
RESOURCES = dict(cpu=8, memory=32768) if DEVICE == "cpu" else dict(gpu=DEVICE, memory=32768)
# Created on first use so the tests can run in any workspace; the online PRC artifact is copied into
# prc-data with `modal volume put` and the chat model is fetched by `fetch_chat_model`.
hf_cache = modal.Volume.from_name("prc-hf-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("prc-data", create_if_missing=True)
results = modal.Volume.from_name("prc-attacks", create_if_missing=True)


def rg_prompt(prefix, digit, H, fruits=None, example=None):
    fruits, example = fruits or FRUITS, example or EXAMPLE
    k = str(digit) * H
    return (f'Complete the sentence "{prefix} {k}" using only and exacty a random word from the list: {fruits}.  '
            f'Answer in this speific format: {FORMAT} {prefix} {k} {example}. (here I chose an other fruit for '
            f'the sake of the example, you have to choose among {fruits})')


def identify_fruit(text, candidates=FRUITS):
    """Index of the single listed fruit that appears exactly once, else None (paper's parser)."""
    found = [(i, text.count(c)) for i, c in enumerate(candidates) if text.count(c) > 0]
    return found[0][0] if len(found) == 1 and found[0][1] == 1 else None


def chat_ids(tokenizer, prompt):
    return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                         enable_thinking=False, tokenize=True)


def cut(tokens):
    """Completion up to (not including) the first end-of-turn / end-of-text token."""
    tokens = [int(t) for t in tokens]
    ends = [i for i, t in enumerate(tokens) if t in STOP]
    return tokens[:ends[0]] if ends else tokens


@app.function(image=hf_image, **RESOURCES, timeout=3600, max_containers=10,
              retries=modal.Retries(max_retries=2), volumes={"/cache": hf_cache})
def generate_hf(scheme, prompt_ids, n, max_new, batch=50, seed=0):
    """n completions of one prompt under an HF-side scheme: none, kgw2, synthid or exp (random offset)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
    tokenizer = AutoTokenizer.from_pretrained(CHAT_MODEL, revision=CHAT_REVISION)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(CHAT_MODEL, revision=CHAT_REVISION, torch_dtype=torch.float32).to(device).eval()
    torch.manual_seed(seed)
    outputs = []
    for b in range(0, n, batch):
        rows = min(batch, n - b)
        ids = torch.tensor([prompt_ids] * rows)
        if scheme == "exp":
            from watermarking.generation import generate
            from watermarking.gumbel.key import gumbel_key_func
            from watermarking.gumbel.sampler import gumbel_sampling
            vocab = model.get_output_embeddings().weight.shape[0]
            out = generate(model, ids, vocab, EXP_KEY_LENGTH, max_new, torch.full((rows,), EXP_KEY_SEED),
                           gumbel_key_func, gumbel_sampling, random_offset=True)
        else:
            processors = {"none": [], "kgw2": [_cpu_kgw_processor(list(tokenizer.get_vocab().values()))],
                          "synthid": [synthid_processor(torch.device(device))]}[scheme]
            # Explicit sampling settings: the chat model's generation_config defaults to top-k 20, T 0.6.
            out = model.generate(ids.to(device), attention_mask=torch.ones_like(ids).to(device), do_sample=True,
                                 max_new_tokens=max_new, top_k=0, top_p=1.0, temperature=1.0,
                                 eos_token_id=list(STOP), pad_token_id=EOS,
                                 logits_processor=LogitsProcessorList(processors)).cpu()
        outputs += [cut(row[len(prompt_ids):]) for row in out]
    return outputs


@app.function(image=prc_image, **RESOURCES, timeout=3600, max_containers=10,
              retries=modal.Retries(max_retries=2), volumes={"/cache": hf_cache, "/data": data_vol})
def generate_prc(prompt_ids, n, max_new, seed_mode="fresh", first_document=0, batch=50):
    """n online-PRC completions of one prompt with the chat model and the eta=0.05 online key."""
    import os
    import torch
    os.environ["PRC_MODEL_SIZE"], os.environ["PRC_MODEL_VARIANT"] = "0.6B", "instruct"
    from modal_run import _redetect_load
    from online_prc import OnlinePRCKey, derive_document_seed
    import watermark_expt as we
    hf_cache.commit()  # the first import downloads the chat weights into /cache/models/Qwen3-0.6B
    artifact = _redetect_load(Path("/data") / PRC_ONLINE_ARTIFACT)
    key = OnlinePRCKey.from_dict(artifact["online_key"])
    we.partition = artifact["partition"].to(we.device)
    if seed_mode == "fresh":
        documents = range(first_document, first_document + n)
    elif seed_mode == "prompt":
        prompt_doc = int.from_bytes(hashlib.sha256(bytes(json.dumps(prompt_ids), "utf8")).digest()[:6], "big")
        documents = [prompt_doc] * n
        batch = 1  # the encoder rejects repeated seeds within a batch; a server reuses them across requests
    else:
        raise ValueError("seed_mode must be fresh or prompt")
    seeds = [derive_document_seed(BBD_SEED, d) for d in documents]
    outputs = []
    for b in range(0, n, batch):
        ids = torch.tensor([prompt_ids] * min(batch, n - b), device=we.device)
        tokens, _ = we.generate_batch_and_collect_online(we.model, ids, max_new, key, we.partition,
                                                         watermark=True, document_seeds=seeds[b:b + len(ids)])
        outputs += [cut(row) for row in tokens.cpu()]
    return outputs


@app.function(image=hf_image, cpu=2, memory=4096, timeout=600, volumes={"/cache": hf_cache})
def prompt_ids():
    """Chat-templated Red-Green prompts (H=1..5) and the raw Fixed-Sampling prompt, as token ids."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(CHAT_MODEL, revision=CHAT_REVISION)
    rg = {f"{p}|{d}|{H}": chat_ids(tokenizer, rg_prompt(p, d, H))
          for p in PREFIXES for d in range(1, 10) for H in range(1, 6)}
    return {"rg": rg, "fs": tokenizer.encode(FS_PROMPT)}


@app.function(image=hf_image, cpu=2, memory=4096, timeout=600, volumes={"/cache": hf_cache})
def list_prompt_ids(lists, Hs):
    """Chat-templated Red-Green prompts keyed 'list|prefix|digit|H'."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(CHAT_MODEL, revision=CHAT_REVISION)
    return {f"{name}|{p}|{d}|{H}": chat_ids(tokenizer, rg_prompt(p, d, H, *WORD_LISTS[name]))
            for name in lists for p in PREFIXES for d in range(1, 10) for H in Hs}


@app.local_entrypoint()
def wordlists():
    """Unwatermarked choice balance and parse rate per candidate word list (2 prefixes x 3 digits x 20)."""
    ids = list_prompt_ids.remote(list(WORD_LISTS), [5])
    cells = [(name, p, d) for name in WORD_LISTS for p in PREFIXES[:2] for d in (1, 5, 9)]
    calls = [generate_hf.spawn("none", ids[f"{n}|{p}|{d}|5"], 20, RG_MAX_NEW, 20, i) for i, (n, p, d) in
             enumerate(cells)]
    texts = decode.remote([c.get() for c in calls])
    report = {}
    for (name, _, _), batch in zip(cells, texts):
        fruits = WORD_LISTS[name][0]
        entry = report.setdefault(name, {"n": 0, "parsed": 0, "counts": dict.fromkeys(fruits, 0)})
        entry["n"] += len(batch)
        for text in batch:
            choice = identify_fruit(text, fruits)
            if choice is not None:
                entry["parsed"] += 1
                entry["counts"][fruits[choice]] += 1
    for entry in report.values():
        entry["max_share"] = round(max(entry["counts"].values()) / max(entry["parsed"], 1), 3)
    Path("outputs/attacks/bbd_wordlists.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


@app.function(image=hf_image, cpu=2, memory=4096, timeout=600, volumes={"/cache": hf_cache})
def decode(batches):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(CHAT_MODEL, revision=CHAT_REVISION)
    return [[tokenizer.decode(t, skip_special_tokens=True) for t in batch] for batch in batches]


@app.local_entrypoint()
def pilot():
    """Small check: instruction following / parse rate, response diversity, PRC on the chat model."""
    ids = prompt_ids.remote()
    cells = [(p, d) for p in PREFIXES[:2] for d in (1, 5, 9)]
    jobs, labels = [], []
    for scheme in ("none", "prc"):
        for p, d in cells:
            prompt = ids["rg"][f"{p}|{d}|5"]
            jobs.append(generate_prc.spawn(prompt, 20, RG_MAX_NEW, "fresh", 1000 * len(jobs)) if scheme == "prc"
                        else generate_hf.spawn("none", prompt, 20, RG_MAX_NEW, 20, len(jobs)))
            labels.append((scheme, "rg", f"{p} {str(d) * 5}"))
    jobs.append(generate_hf.spawn("none", ids["fs"], 100, FS_MAX_NEW, 50, 999))
    labels.append(("none", "fs", "fresh"))
    for mode in ("fresh", "prompt"):
        jobs.append(generate_prc.spawn(ids["fs"], 100, FS_MAX_NEW, mode, 900_000 if mode == "fresh" else 0))
        labels.append(("prc", "fs", mode))
    texts = decode.remote([job.get() for job in jobs])
    report = {}
    for (scheme, test, cell), batch in zip(labels, texts):
        if test == "rg":
            parsed = [identify_fruit(t) for t in batch]
            ok = [x for x in parsed if x is not None]
            entry = report.setdefault(f"{scheme} rg", {"n": 0, "parsed": 0, "counts": [0] * len(FRUITS)})
            entry["n"] += len(batch)
            entry["parsed"] += len(ok)
            for x in ok:
                entry["counts"][x] += 1
            entry.setdefault("examples", []).extend(batch[:2])
        else:
            lengths = sorted(len(t.split()) for t in batch)
            report[f"{scheme} fs {cell}"] = {"n": len(batch), "unique": len(set(batch)),
                                            "median_words": lengths[len(lengths) // 2], "example": batch[0][:200]}
    Path("outputs/attacks").mkdir(parents=True, exist_ok=True)
    Path("outputs/attacks/bbd_pilot.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


@app.function(image=hf_image, cpu=2, memory=8192, timeout=600, volumes={"/cache": hf_cache})
def debug_hf():
    """Report how the chat model resolves from the shared cache inside hf_image."""
    import os
    import traceback
    out = {k: os.environ.get(k) for k in ("HF_HOME", "HF_HUB_CACHE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
    snap = "/cache/models--Qwen--Qwen3-0.6B/snapshots"
    out["snapshots"] = {s: sorted(os.listdir(f"{snap}/{s}")) for s in os.listdir(snap)}
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        AutoTokenizer.from_pretrained(CHAT_MODEL, revision=CHAT_REVISION)
        out["tokenizer"] = "ok"
        AutoModelForCausalLM.from_pretrained(CHAT_MODEL, revision=CHAT_REVISION)
        out["model"] = "ok"
    except Exception:
        out["traceback"] = traceback.format_exc()[-3000:]
    return out


@app.function(image=hf_image, cpu=2, memory=8192, timeout=1800, volumes={"/cache": hf_cache})
def fetch_chat_model():
    """Complete the pinned Qwen3-0.6B hub snapshot (config, weights, chat template) in the shared cache,
    and check its weights equal the copy the PRC loader downloaded to /cache/models/Qwen3-0.6B."""
    import hashlib
    import os
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "0"
    from huggingface_hub import snapshot_download
    path = snapshot_download(CHAT_MODEL, revision=CHAT_REVISION, cache_dir="/cache")
    hf_cache.commit()
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    prc_copy = Path("/cache/models/Qwen3-0.6B/model.safetensors")
    return {"files": sorted(os.listdir(path)), "sha256": digest(f"{path}/model.safetensors"),
            "same_weights_as_prc_loader": digest(prc_copy) == digest(f"{path}/model.safetensors")
            if prc_copy.exists() else "PRC loader copy not downloaded yet"}


@app.local_entrypoint()
def fetch():
    print(fetch_chat_model.remote())
    print(debug_hf.remote())


@app.local_entrypoint()
def debug():
    for key, value in debug_hf.remote().items():
        print(key, ":", value)


# ================================================================ full test (E7)
#
# One Modal call per (scheme, H, prefix) generates the 9 digit cells of that row of the Red-Green
# matrix, re-sampling until each cell has RG_VALID parsed answers (as the paper's rejection loop);
# one call per scheme generates the Fixed-Sampling stories. Counts are saved to the prc-attacks volume
# of the running workspace and analysed locally by `analyze` with the paper's released statistics.

RG_LIST = "apples"          # most balanced list in the word-list pilot (max share 0.46, 93% parsed)
RG_HS = (4, 5)              # SynthID needs H = h = 4 exactly; H = 5 is the released default
RG_VALID, RG_FIRST, RG_TOPUP, RG_ROUNDS = 100, 115, 30, 5
FS_N = 1000
SCHEMES = ("none", "prc", "kgw2", "synthid", "exp")
HF_BATCHES = {"none": 115, "kgw2": 115, "synthid": 60, "exp": 30}


def _label_int(label):
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:6], "big")


class Generator:
    """Loads one scheme's model once and generates n completions of a prompt."""

    def __init__(self, scheme):
        import torch
        self.scheme = scheme
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if scheme == "prc":
            os.environ["PRC_MODEL_SIZE"], os.environ["PRC_MODEL_VARIANT"] = "0.6B", "instruct"
            from modal_run import _redetect_load
            from online_prc import OnlinePRCKey
            import watermark_expt as we
            artifact = _redetect_load(Path("/data") / PRC_ONLINE_ARTIFACT)
            self.we, self.key = we, OnlinePRCKey.from_dict(artifact["online_key"])
            we.partition = artifact["partition"].to(we.device)
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(CHAT_MODEL, revision=CHAT_REVISION)
            self.model = AutoModelForCausalLM.from_pretrained(
                CHAT_MODEL, revision=CHAT_REVISION, torch_dtype=torch.float32).to(self.device).eval()

    def __call__(self, prompt_ids, n, max_new, label):
        """n completions (token lists cut at end of turn); `label` makes seeds and documents unique."""
        import torch
        if self.scheme == "prc":
            from online_prc import derive_document_seed
            base = (_label_int(label) % 2**40) << 20  # fresh document per response; fits the 64-bit document id
            seeds = [derive_document_seed(BBD_SEED, base + i) for i in range(n)]
            out = []
            for b in range(0, n, 60):
                ids = torch.tensor([prompt_ids] * len(seeds[b:b + 60]), device=self.we.device)
                tokens, _ = self.we.generate_batch_and_collect_online(
                    self.we.model, ids, max_new, self.key, self.we.partition, watermark=True,
                    document_seeds=seeds[b:b + 60])
                out += [cut(row) for row in tokens.cpu()]
            return out
        from transformers import LogitsProcessorList
        torch.manual_seed(_label_int(label) % 2**31)
        out, batch = [], HF_BATCHES[self.scheme]
        for b in range(0, n, batch):
            rows = min(batch, n - b)
            ids = torch.tensor([prompt_ids] * rows)
            if self.scheme == "exp":
                from watermarking.generation import generate
                from watermarking.gumbel.key import gumbel_key_func
                from watermarking.gumbel.sampler import gumbel_sampling
                vocab = self.model.get_output_embeddings().weight.shape[0]
                gen = generate(self.model, ids, vocab, EXP_KEY_LENGTH, max_new, torch.full((rows,), EXP_KEY_SEED),
                               gumbel_key_func, gumbel_sampling, random_offset=True)
            else:
                processors = {"none": [],
                              "kgw2": [_cpu_kgw_processor(list(self.tokenizer.get_vocab().values()))],
                              "synthid": [synthid_processor(torch.device(self.device))]}[self.scheme]
                gen = self.model.generate(ids.to(self.device), attention_mask=torch.ones_like(ids).to(self.device),
                                          do_sample=True, max_new_tokens=max_new, top_k=0, top_p=1.0,
                                          temperature=1.0, eos_token_id=list(STOP), pad_token_id=EOS,
                                          logits_processor=LogitsProcessorList(processors)).cpu()
            out += [cut(row[len(prompt_ids):]) for row in gen]
        return out


def _decoder():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(CHAT_MODEL, revision=CHAT_REVISION)
    return lambda tokens: tokenizer.decode(tokens, skip_special_tokens=True)


def _rg_row(scheme, H, prefix, rep, ids):
    path = Path(f"/results/{OUT}/rg/{scheme}/rep{rep}/H{H}/{prefix.replace(' ', '_')}.json")
    if path.exists():
        return str(path)
    generate, decode_one = Generator(scheme), _decoder()
    fruits = WORD_LISTS[RG_LIST][0]
    row = {}
    for d in range(1, 10):
        counts, valid, drawn, examples = [0] * len(fruits), 0, 0, []
        for attempt in range(RG_ROUNDS):
            n = RG_FIRST if attempt == 0 else RG_TOPUP
            texts = [decode_one(t) for t in generate(ids[str(d)], n, RG_MAX_NEW,
                                                     f"rg|{scheme}|{H}|{prefix}|{d}|{rep}|{attempt}")]
            drawn += n
            examples += texts[:2] if attempt == 0 else []
            for text in texts:
                choice = identify_fruit(text, fruits)
                if choice is not None and valid < RG_VALID:
                    counts[choice] += 1
                    valid += 1
            if valid >= RG_VALID:
                break
        row[str(d)] = {"counts": counts, "valid": valid, "drawn": drawn, "examples": examples}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"scheme": scheme, "H": H, "prefix": prefix, "rep": rep, "list": RG_LIST,
                                "fruits": fruits, "cells": row}))
    results.commit()
    return str(path)


def _fs_run(scheme, rep, ids):
    path = Path(f"/results/{OUT}/fs/{scheme}/rep{rep}.json")
    if path.exists():
        return str(path)
    tokens = Generator(scheme)(ids, FS_N, FS_MAX_NEW, f"fs|{scheme}|{rep}")
    decode_one = _decoder()
    digests = [hashlib.sha256(json.dumps(t).encode()).hexdigest()[:16] for t in tokens]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"scheme": scheme, "rep": rep, "n": len(tokens), "digests": digests,
                                "lengths": [len(t) for t in tokens],
                                "examples": [decode_one(t)[:300] for t in tokens[:5]]}))
    results.commit()
    return str(path)


@app.function(image=hf_image, **RESOURCES, timeout=5400, max_containers=10, retries=modal.Retries(max_retries=2),
              volumes={"/cache": hf_cache, "/results": results})
def rg_row_hf(scheme, H, prefix, rep, ids):
    return _rg_row(scheme, H, prefix, rep, ids)


@app.function(image=prc_image, **RESOURCES, timeout=5400, max_containers=10, retries=modal.Retries(max_retries=2),
              volumes={"/cache": hf_cache, "/data": data_vol, "/results": results})
def rg_row_prc(H, prefix, rep, ids):
    return _rg_row("prc", H, prefix, rep, ids)


@app.function(image=hf_image, **RESOURCES, timeout=5400, max_containers=10, retries=modal.Retries(max_retries=2),
              volumes={"/cache": hf_cache, "/results": results})
def fs_run_hf(scheme, rep, ids):
    return _fs_run(scheme, rep, ids)


@app.function(image=prc_image, **RESOURCES, timeout=5400, max_containers=10, retries=modal.Retries(max_retries=2),
              volumes={"/cache": hf_cache, "/data": data_vol, "/results": results})
def fs_run_prc(rep, ids):
    return _fs_run("prc", rep, ids)


@app.function(image=hf_image, cpu=1, memory=2048, timeout=86400, volumes={"/cache": hf_cache})
def orchestrate_bbd(schemes: list, reps: list):
    """Cloud-side driver (deploy the app, then spawn): all Red-Green rows and Fixed-Sampling runs."""
    ids = list_prompt_ids.local([RG_LIST], list(RG_HS))
    from transformers import AutoTokenizer
    fs_ids = AutoTokenizer.from_pretrained(CHAT_MODEL, revision=CHAT_REVISION).encode(FS_PROMPT)
    calls = []
    for rep in reps:
        for scheme in schemes:
            calls.append(fs_run_prc.spawn(rep, fs_ids) if scheme == "prc" else fs_run_hf.spawn(scheme, rep, fs_ids))
            for H in RG_HS:
                for prefix in PREFIXES:
                    row = {str(d): ids[f"{RG_LIST}|{prefix}|{d}|{H}"] for d in range(1, 10)}
                    calls.append(rg_row_prc.spawn(H, prefix, rep, row) if scheme == "prc"
                                 else rg_row_hf.spawn(scheme, H, prefix, rep, row))
    failed = 0
    for call in calls:
        try:
            call.get()
        except Exception as error:
            failed += 1
            print("FAILED", repr(error)[:300], flush=True)
    print(f"{len(calls) - failed}/{len(calls)} jobs done", flush=True)


def launch_bbd(schemes=",".join(SCHEMES), reps="0"):
    call = modal.Function.from_name("prc-bbd", "orchestrate_bbd").spawn(schemes.split(","),
                                                                      [int(r) for r in reps.split(",")])
    print("spawned", call.object_id)
