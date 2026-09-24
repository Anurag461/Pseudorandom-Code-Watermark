"""Substitution robustness at 4096 tokens: PRC (fixed n=T=4096, eta=0.05) vs EXP, KGW-2.0, SynthID-Text.

Same design as kth_baselines.py (Qwen3-0.6B-Base, the first 500 C4 RealNewsLike prompts, paper
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
                           PROMPT_TOKENS, _cpu_kgw_processor, image, key_seeds, load_prompts,
                           synthid_processor)
from modal_run import fixed_image

M = 4096
RATES = (0.05, 0.1, 0.15, 0.2)
SCHEMES = ("exp", "kgw2", "synthid")
ATTACK_VOCAB = 151665
OUT = "kth_long_v1"
CHUNK = 10
PRC_TAG = "n4096_t3_eta0.05_T4096_r4055"
NULL_DIR = "_nulls/T4096"  # the null cohort the fixed-run planner selected for T=4096
MANIFEST = "outputs/attacks/prc_fixed_n4096_eta005_manifest.json"
RESULTS_CSV = "outputs/attacks/kth_long_results.csv"
ALPHA = 1e-3

app = modal.App("prc-kth-long")
hf_cache = modal.Volume.from_name("prc-hf-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("prc-data", create_if_missing=True)
results = modal.Volume.from_name("prc-attacks", create_if_missing=True)
prc_image = fixed_image.add_local_python_source("attacks", "kth_baselines")


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
    starts = [0] if stage == "smoke" else list(range(0, NUM_PROMPTS, CHUNK))
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
