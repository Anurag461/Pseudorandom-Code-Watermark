"""Kuditipudi et al. (2023) EXP, KGW-2.0 and SynthID-Text baselines under the same substitutions as PRC.

Runs EXP and KGW-2.0 from the Kuditipudi et al. code (cloned at a pinned commit)
and SynthID-Text from Hugging Face transformers on Qwen3-0.6B-Base with the 500 prompts of the fixed PRC n=400 run, and scores
every scheme after the identical substitutions used for PRC (attacks.apply_attack
seeds each candidate by source and prompt index, so the wm and null texts of a
given prompt receive the same edit positions and replacement tokens as in the
PRC sweep).

Settings follow the paper's c4-experiment.py: key length n=256, no random
offset, block size k = text length, KGW gamma=0.25, delta=2.0 with
previous-token (simple_1) seeding, and p-values from the empirical
distribution of test statistics on human C4 continuations (null=True).
SynthID-Text uses the MarkLLM/DITTO configuration (ngram_len=5, 30 keys,
table size 2^16, seed 0, context history 1024), tournament sampling over the
full vocabulary, and the mean g-value score over unmasked positions (the
paper's "mean" detector), with the same empirical p-values.
Deviations, all applied equally to PRC:
  * m=400 tokens (paper: 35/70) to match the PRC block length;
  * substitutions act on tokens with no decode/re-encode;
  * the KGW green-list RNG and the SynthID sampling table are built on CPU
    at generation and detection;
  * SynthID detection sees only the completion, so its first ngram_len-1
    tokens are unscored (as PRC is prompt-free).

Stages (each a separate local entrypoint call):
  generate  -> watermarked completions per scheme (GPU)
  score     -> test statistics for human nulls, clean and attacked texts (CPU)
  summarize -> p-values, median p, TPR/FPR at 1e-3 and 1e-2 (local)
"""
import hashlib
import json
import os
from pathlib import Path

import modal

KTH_REPO = "https://github.com/jthickstun/watermark"
KTH_COMMIT = "80d4ec8f4280da2a2cada03adfc8940593d1964c"
MODEL_DIR = "/cache/models/Qwen3-0.6B-Base"
PRC_MANIFEST = "outputs/redetection/.archive/restored/prompt_free/manifests/pilots.json"
PRC_CASE = "same_0p6b_n400"
OUT = "kth_baselines_v1"
RESULTS_CSV = "outputs/attacks/kth_baseline_results.csv"

M = 400                 # scored completion length (PRC n = T = 400)
PROMPT_TOKENS = 50
NUM_PROMPTS = 500
KEY_LENGTH = 256        # paper's n
KEY_SEED = 1            # paper's --seed for the per-prompt key draw
NULL_SEED = 2           # seeds the random keys of the human null reference
ATTACK_VOCAB = 151665   # identical to the PRC sweep (Qwen3-Base tokenizer)
RATES = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3)
KINDS = ("substitution",)
SCHEMES = ("exp", "kgw2", "synthid")
SYNTHID = dict(ngram_len=5, keys=[654, 400, 836, 123, 340, 443, 597, 160, 57, 29, 590, 639, 13, 715, 468,
                                  990, 966, 226, 324, 585, 118, 504, 421, 521, 129, 669, 732, 225, 90, 960],
               sampling_table_size=65536, sampling_table_seed=0, context_history_size=1024)
EOS = 151643  # Qwen3-Base <|endoftext|>
KGW_GAMMA, KGW_DELTA = 0.25, 2.0
CHUNK = 25

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .pip_install("torch==2.4.0", "transformers==4.51.3", "tokenizers==0.21.1", "safetensors==0.4.5",
                 "huggingface_hub==0.30.2", "numpy==1.26.0", "scipy==1.14.1", "cython==0.29.37",
                 "nltk==3.9.1")
    .run_commands(f"git clone {KTH_REPO} /kth && cd /kth && git checkout {KTH_COMMIT}",
                  # Compile the Cython edit-distance scorer once, into the image.
                  "cd /kth && python -c 'import watermarking.gumbel.score'")
    .env({"PYTHONPATH": "/kth", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
          "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "1"})
    .add_local_file("prompts_10k.jsonl", "/root/prompts_10k.jsonl")
    .add_local_python_source("attacks")
)
app = modal.App("prc-kth-baselines", image=image)
hf_cache = modal.Volume.from_name("prc-hf-cache", create_if_missing=False)
archive = modal.Volume.from_name("prc-research-archive", create_if_missing=False)
results = modal.Volume.from_name("prc-attacks", create_if_missing=True)


# ---------------------------------------------------------------- helpers

def key_seeds():
    """Per-prompt watermark keys, drawn exactly as c4-experiment.py does."""
    import torch
    torch.manual_seed(KEY_SEED)
    return torch.randint(2**32, (NUM_PROMPTS,))


def load_prompts():
    lines = Path("/root/prompts_10k.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def attack_spec(kind, rate, seed=0):
    return {"kind": kind, "rate": float(rate), "seed": seed, "vocab_size": ATTACK_VOCAB}


def attack_id(attack):
    return "clean" if attack is None else f"{attack['kind']}{attack['rate']:g}_s{attack['seed']}"


def corrupt(tokens, attack, source, prompt_idx):
    """Same corruption as the PRC sweep, then keep at most M tokens."""
    from attacks import apply_attack
    if attack is not None:
        tokens = apply_attack(tokens, attack, source, prompt_idx)
    return tokens[:M]


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial")
    tmp.write_text(json.dumps(value) + "\n")
    tmp.replace(path)


def synthid_processor(device):
    """SynthID-Text processor whose sampling table is drawn on CPU, then moved to `device`.

    The table comes from a device-specific torch.Generator, so building it on CPU
    for both generation and detection keeps the two identical.
    """
    from transformers.generation.logits_process import SynthIDTextWatermarkLogitsProcessor
    processor = SynthIDTextWatermarkLogitsProcessor(**SYNTHID, device=torch_device("cpu"))
    processor.keys = processor.keys.to(device)
    processor.sampling_table = processor.sampling_table.to(device)
    processor.device = device
    return processor


def torch_device(name):
    import torch
    return torch.device(name)


def synthid_mean_score(processor, tokens):
    """Mean g-value over positions that are neither repeated contexts nor after EOS."""
    ids = tokens.unsqueeze(0)
    g = processor.compute_g_values(ids).float()                    # [1, T-(ngram_len-1), depth]
    mask = processor.compute_context_repetition_mask(ids)          # [1, T-(ngram_len-1)]
    mask = mask * processor.compute_eos_token_mask(ids, EOS)[:, processor.ngram_len - 1:]
    count = mask.sum() * g.shape[-1]
    return float((g * mask[..., None]).sum() / count) if count else 0.5


class Scorer:
    """The paper's test statistic for one scheme; lower means more watermarked."""

    def __init__(self, scheme, vocab_size, tokenizer=None):
        import torch
        self.scheme, self.vocab_size = scheme, vocab_size
        if scheme == "synthid":
            self.processor = synthid_processor(torch.device("cpu"))
        elif scheme == "kgw2":
            from watermarking.kirchenbauer.watermark_processor import WatermarkDetector
            self.detector = WatermarkDetector(
                vocab=list(tokenizer.get_vocab().values()), gamma=KGW_GAMMA, seeding_scheme="simple_1",
                device=torch.device("cpu"), tokenizer=tokenizer, z_threshold=1.5, normalizers=[],
                ignore_repeated_bigrams=False)
        else:
            from watermarking.detection import adjacency, phi
            from watermarking.gumbel.key import gumbel_key_func
            from watermarking.gumbel.score import gumbel_score
            self.dist = gumbel_score
            self.key_func, self.adjacency = gumbel_key_func, adjacency
            self.phi = lambda tokens, generator, null: phi(
                tokens=tokens, n=KEY_LENGTH, k=len(tokens), generator=generator, key_func=gumbel_key_func,
                vocab_size=vocab_size, dist=self.dist, null=null, normalize=False)

    def keyed_fast(self, tokens, generator):
        """phi(null=False) restricted to the key columns of tokens that occur.

        The key is drawn exactly as phi draws it; the EXP distances only read
        xi[:, token], and pi is the identity, so the result is identical while
        avoiding a (k x vocab) key slice per shift.
        """
        import torch
        xi, pi = self.key_func(generator, KEY_LENGTH, self.vocab_size)
        if not torch.equal(pi, torch.arange(self.vocab_size)):
            raise ValueError("EXP key permutation must be the identity")
        unique, inverse = torch.unique(tokens, return_inverse=True)
        A = self.adjacency(inverse, xi[:, unique].contiguous(), self.dist, len(tokens))
        return torch.min(torch.min(A, axis=1)[0])

    def __call__(self, tokens, seed, null=False, reference=False):
        import torch
        if self.scheme == "kgw2":
            return -float(self.detector._score_sequence(tokens)["z_score"])
        if self.scheme == "synthid":
            return -synthid_mean_score(self.processor, tokens)
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        if null or reference:
            return float(self.phi(tokens, generator, null))
        return float(self.keyed_fast(tokens, generator))


# ---------------------------------------------------------------- generation

def _cpu_kgw_processor(vocab):
    import torch
    from watermarking.kirchenbauer.watermark_processor import WatermarkLogitsProcessor

    class CPUSeededKGW(WatermarkLogitsProcessor):
        """Paper's KGW processor with its green-list RNG on CPU, so detection can run on CPU."""
        def __call__(self, input_ids, scores):
            if self.rng is None:
                self.rng = torch.Generator()
            ids = [self._get_greenlist_ids(row.cpu()).to(scores.device) for row in input_ids]
            mask = self._calc_greenlist_mask(scores=scores, greenlist_token_ids=ids)
            return self._bias_greenlist_logits(scores=scores, greenlist_mask=mask, greenlist_bias=self.delta)

    return CPUSeededKGW(vocab=vocab, gamma=KGW_GAMMA, delta=KGW_DELTA, seeding_scheme="simple_1")


@app.function(gpu="A10G", memory=32768, timeout=3600, max_containers=10,
              volumes={"/cache": hf_cache, "/results": results})
def generate_chunk(scheme, start):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
    path = Path(f"/results/{OUT}/generations/{scheme}/{start:04d}.pt")
    if path.exists():
        return str(path)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    # float32 as in the paper's scripts; probabilities feed the EXP sampler directly.
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float32).cuda().eval()
    vocab_size = model.get_output_embeddings().weight.shape[0]
    idx = list(range(start, min(start + CHUNK, NUM_PROMPTS)))
    prompts = load_prompts()
    prompt_ids = torch.tensor([prompts[i]["prompt_tokens"] for i in idx])
    seeds = key_seeds()[idx]
    torch.manual_seed(1000 + start)
    if scheme in ("kgw2", "synthid"):
        # A fresh processor per call: SynthID keeps per-batch context state.
        processor = (_cpu_kgw_processor(list(tokenizer.get_vocab().values())) if scheme == "kgw2"
                     else synthid_processor(torch.device("cuda")))
        out = model.generate(prompt_ids.cuda(), attention_mask=torch.ones_like(prompt_ids).cuda(),
                             do_sample=True, max_new_tokens=M, min_new_tokens=M, top_k=0, top_p=1.0,
                             temperature=1.0, pad_token_id=tokenizer.eos_token_id,
                             logits_processor=LogitsProcessorList([processor])).cpu()
    else:
        from watermarking.generation import generate
        from watermarking.gumbel.key import gumbel_key_func
        from watermarking.gumbel.sampler import gumbel_sampling
        out = generate(model, prompt_ids, vocab_size, KEY_LENGTH, M, seeds, gumbel_key_func,
                       gumbel_sampling, random_offset=False)
    tokens = out[:, PROMPT_TOKENS:PROMPT_TOKENS + M]
    if tokens.shape != (len(idx), M):
        raise ValueError(f"expected {M} new tokens, got {tuple(tokens.shape)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"scheme": scheme, "prompt_idx": idx, "seeds": seeds, "tokens": tokens,
                "vocab_size": vocab_size, "kth_commit": KTH_COMMIT}, path.with_suffix(".partial"))
    path.with_suffix(".partial").replace(path)
    results.commit()
    return str(path)


# ---------------------------------------------------------------- scoring

def _load_prc_null(ref):
    """PRC null completion, verified against the frozen manifest hashes.

    Records carry no prompt; the run artifact's prompt_ids_list (sha256 29ec8adc...)
    was checked to equal prompts.jsonl, and hence prompts_10k.jsonl[:500], for all 500 prompts.
    """
    import sys
    import numpy as np
    import torch
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
    path = Path("/archive") / ref["file"]["path"]
    data = path.read_bytes()
    if len(data) != ref["file"]["bytes"] or hashlib.sha256(data).hexdigest() != ref["file"]["sha256"]:
        raise ValueError(f"PRC null source changed: {path}")
    record = torch.load(path, weights_only=False, map_location="cpu")
    tokens = record["tokens"][:M].to(torch.int64)
    if (record["watermark"] or record["prompt_idx"] != ref["prompt_idx"]
            or hashlib.sha256(tokens.numpy().tobytes()).hexdigest() != ref["tokens_sha256"]):
        raise ValueError("PRC null record differs from the frozen manifest")
    return tokens


# Retries: a few containers segfault intermittently (Python SystemError in structseq); items are
# cached and idempotent, so a retried item recomputes the same statistics.
@app.function(cpu=2, memory=8192, timeout=7200, max_containers=50, retries=modal.Retries(max_retries=3),
              volumes={"/cache": hf_cache, "/results": results, "/archive": archive})
def score_chunk(item):
    """One work item: null reference texts, or clean/attacked wm+null texts for a prompt range."""
    import torch
    from transformers import AutoTokenizer
    path = Path(f"/results/{OUT}/scores/{item['scheme']}/{item['name']}.json")
    if path.exists():
        return str(path)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    vocab_size = 151936
    scorer = Scorer(item["scheme"], vocab_size, tokenizer)
    rows = []
    if item["kind"] == "null_reference":
        prompts = load_prompts()
        rng = torch.Generator().manual_seed(NULL_SEED)
        seeds = torch.randint(100000, (len(prompts),), generator=rng)
        for doc in range(item["start"], item["stop"]):
            tokens = torch.tensor(tokenizer.encode(prompts[doc]["human_continuation"]))
            if len(tokens) < M:
                continue
            stats = {str(L): scorer(tokens[:L], seeds[doc], null=True) for L in item["lengths"]}
            rows.append({"doc": doc, "stats": stats})
    else:
        gen = torch.load(f"/results/{OUT}/generations/{item['scheme']}/{item['gen_chunk']:04d}.pt")
        prompts = load_prompts()
        for row, prompt_idx in enumerate(gen["prompt_idx"]):
            null_tokens = _load_prc_null(item["null_refs"][row])
            seed = gen["seeds"][row]
            for source, tokens in (("wm", gen["tokens"][row]), ("null", null_tokens)):
                attacked = corrupt(tokens, item["attack"], source, prompt_idx)
                stat = scorer(attacked, seed)
                if item.get("verify") and row < 2 and item["scheme"] != "kgw2":
                    reference = scorer(attacked, seed, reference=True)
                    if stat != reference:
                        raise ValueError(f"fast EXP statistic {stat} != paper phi {reference}")
                rows.append({"source": source, "prompt_idx": prompt_idx, "length": len(attacked), "stat": stat})
    save_json(path, {"item": {k: v for k, v in item.items() if k != "null_refs"},
                     "kth_commit": KTH_COMMIT, "rows": rows})
    results.commit()
    return str(path)


# ---------------------------------------------------------------- orchestration

def _work_items(schemes, attacks, null_docs):
    manifest = json.loads(Path(PRC_MANIFEST).read_text())
    case = next(c for c in manifest["cases"] if c["id"] == PRC_CASE)
    nulls = {r["prompt_idx"]: r for r in case["records"] if r["source"] == "null"}
    if sorted(nulls) != list(range(NUM_PROMPTS)):
        raise ValueError("expected one PRC null per prompt")
    lengths = [M]  # substitution keeps every text at M tokens
    items = []
    for scheme in schemes:
        for start in range(PROMPT_OFFSET_NULL, PROMPT_OFFSET_NULL + null_docs, 100):
            stop = min(start + 100, PROMPT_OFFSET_NULL + null_docs)
            items.append({"scheme": scheme, "kind": "null_reference", "name": f"null_ref_{start:05d}",
                          "start": start, "stop": stop, "lengths": lengths})
        for kind_rate in [None, *attacks]:
            attack = None if kind_rate is None else attack_spec(*kind_rate)
            for start in range(0, NUM_PROMPTS, CHUNK):
                refs = [nulls[i] for i in range(start, min(start + CHUNK, NUM_PROMPTS))]
                items.append({"scheme": scheme, "kind": "eval", "attack": attack, "gen_chunk": start,
                              "name": f"{attack_id(attack)}_{start:04d}", "null_refs": refs})
    return items


# Human null texts come from documents 500.. of prompts_10k.jsonl, disjoint from the eval prompts.
PROMPT_OFFSET_NULL = NUM_PROMPTS


@app.local_entrypoint()
def run(stage: str = "smoke", schemes: str = ",".join(SCHEMES)):
    """smoke: one generation chunk and a few scoring items per scheme; full: everything."""
    if stage not in ("smoke", "generate", "score", "full"):
        raise ValueError("choose smoke, generate, score or full")
    chosen = schemes.split(",")
    if not set(chosen) <= set(SCHEMES):
        raise ValueError(f"schemes must be among {SCHEMES}")
    attacks = [(k, r) for k in KINDS for r in RATES]
    if stage == "smoke":
        for path in generate_chunk.starmap([(s, 0) for s in chosen]):
            print("generated", path)
        items = [{**i, "verify": True} if i["kind"] == "eval" else i
                 for i in _work_items(chosen, [("substitution", 0.3)], null_docs=100)
                 if i["kind"] == "null_reference" or i["gen_chunk"] == 0]
        for path in score_chunk.map(items):
            print("scored", path)
        return
    if stage in ("generate", "full"):
        jobs = [(s, start) for s in chosen for start in range(0, NUM_PROMPTS, CHUNK)]
        for path in generate_chunk.starmap(jobs):
            print("generated", path, flush=True)
    if stage in ("score", "full"):
        null_docs = len(Path("prompts_10k.jsonl").read_text().splitlines()) - PROMPT_OFFSET_NULL
        items = _work_items(chosen, attacks, null_docs)
        done = 0
        for path in score_chunk.map(items, return_exceptions=True):
            if isinstance(path, Exception):
                print("FAILED item:", repr(path), flush=True)
                continue
            done += 1
            if done % 50 == 0:
                print(f"scored {done}/{len(items)}", flush=True)
        print(f"scored {done}/{len(items)}")


# ---------------------------------------------------------------- summary

ALPHAS = (1e-3, 1e-2)
SUMMARY_COLUMNS = ["scheme", "attack", "rate", "scored length", "wm count", "null count", "null reference",
                   "median wm p", "TPR@1e-3", "FPR@1e-3", "TPR@1e-2", "FPR@1e-2", "p-value", "notes"]


def empirical_p(reference, stat):
    """Paper's fast_permutation_test: share of null-reference statistics <= stat."""
    import numpy as np
    return float(np.searchsorted(reference, stat, side="right") / len(reference))


def hoeffding_p(info):
    """PRC detector's p-value bound exp(-S^2 / 2V); valid for any S (1 when S <= 0)."""
    import math
    S, V = info["statistic"], info["V"]
    return 1.0 if S is None or V in (None, 0) or S <= 0 else math.exp(-S * S / (2 * V))


def summary_row(scheme, attack, rows, pvalue, reference_size, notes):
    import numpy as np
    wm = [r for r in rows if r["source"] == "wm"]
    null = [r for r in rows if r["source"] == "null"]
    wm_p = np.array([r["p"] for r in wm])
    null_p = np.array([r["p"] for r in null])
    rate = lambda p, a: f"{int((p <= a).sum())}/{len(p)} ({(p <= a).mean():.1%})"
    return {"scheme": scheme, "attack": "none" if attack is None else attack["kind"],
            "rate": 0.0 if attack is None else attack["rate"],
            "scored length": sorted({r["length"] for r in rows}), "wm count": len(wm), "null count": len(null),
            "null reference": reference_size, "median wm p": f"{np.median(wm_p):.3g}",
            "TPR@1e-3": rate(wm_p, 1e-3), "FPR@1e-3": rate(null_p, 1e-3),
            "TPR@1e-2": rate(wm_p, 1e-2), "FPR@1e-2": rate(null_p, 1e-2), "p-value": pvalue, "notes": notes}


def _prc_rows():
    """PRC rows from the attacked redetection runs of the same case (rate 0 = clean)."""
    import glob
    out = []
    for kind in KINDS:
        for rate in (0.0, *RATES):
            if rate == 0.0 and kind != "substitution":
                continue
            paths = sorted(glob.glob(f"outputs/redetection/.archive/runs/*/{PRC_CASE}__{kind}{rate:g}_s0/full.json"))
            if not paths:
                print(f"missing PRC run for {kind} {rate:g}")
                continue
            report = json.loads(Path(paths[0]).read_text())
            length = report["attacked_length"]
            rows = [{"source": r["source"], "length": length, "p": hoeffding_p(r["scores"][str(M)]["map"])}
                    for r in report["records"]]
            attack = None if rate == 0.0 else attack_spec(kind, rate)
            out.append(summary_row("prc_map", attack, rows, "Hoeffding bound exp(-S^2/2V)", "none (proven FPR)",
                                   f"fixed PRC n=400 eta=0.05; run={Path(paths[0]).parent.parent.name}"))
    return out


@app.local_entrypoint()
def summarize(schemes: str = ",".join(SCHEMES)):
    """Read all score files from the volume and write RESULTS_CSV (baselines and PRC)."""
    import csv
    import numpy as np
    table = []
    for scheme in schemes.split(","):
        files = [e.path for e in results.listdir(f"{OUT}/scores/{scheme}")]
        payloads = [json.loads(b"".join(results.read_file(p))) for p in files]
        refs = {}
        for payload in payloads:
            if payload["item"]["kind"] == "null_reference":
                for row in payload["rows"]:
                    for length, stat in row["stats"].items():
                        refs.setdefault(int(length), []).append(stat)
        refs = {L: np.sort(v) for L, v in refs.items()}
        groups = {}
        for payload in payloads:
            if payload["item"]["kind"] == "eval":
                attack = payload["item"]["attack"]
                groups.setdefault(json.dumps(attack, sort_keys=True), []).extend(payload["rows"])
        for key, rows in sorted(groups.items()):
            attack = json.loads(key)
            for row in rows:
                row["p"] = empirical_p(refs[row["length"]], row["stat"])
            if {r["prompt_idx"] for r in rows} != set(range(NUM_PROMPTS)) or len(rows) != 2 * NUM_PROMPTS:
                print(f"incomplete {scheme} {key}: {len(rows)} rows")
                continue
            size = min(len(refs[r["length"]]) for r in rows)
            table.append(summary_row(scheme, attack, rows, "empirical vs human C4 (paper)", size,
                                     f"KTH commit {KTH_COMMIT[:7]}; Qwen3-0.6B-Base fp32; n={KEY_LENGTH}"))
    table += _prc_rows()
    table.sort(key=lambda r: (r["attack"] != "none", r["attack"], r["rate"], r["scheme"]))
    path = Path(RESULTS_CSV)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)
    for row in table:
        print(f"{row['scheme']:9s} {row['attack']:12s} {row['rate']:<5g} median p {row['median wm p']:>9s}  "
              f"TPR@1e-3 {row['TPR@1e-3']:>16s}  FPR@1e-3 {row['FPR@1e-3']}")
