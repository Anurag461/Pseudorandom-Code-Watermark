from __future__ import annotations

from scipy.special import binom, lambertw
from importlib.metadata import version
from qwen import *
import torch
import torch.nn as nn
from transformers import AutoTokenizer
import time
import json
import os
from pathlib import Path
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download, snapshot_download
from constants import test_prompts
import numpy as np
from prc import KeyGen, Encode, Detect
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from benchmarks.task import Task

pkgs = [
    "huggingface_hub",  # to download pretrained weights
    "tokenizers",       # to implement the tokenizer
    "torch",            # to implement the model
]


for p in pkgs:
    print(f"{p} version: {version(p)}")

_variant = os.environ.get("PRC_MODEL_VARIANT", "base")
USE_BASE_MODEL = _variant == "base"
USE_REASONING_MODEL = _variant == "reasoning"
USE_INSTRUCT_MODEL = _variant == "instruct"

if (USE_BASE_MODEL + USE_REASONING_MODEL
    + USE_INSTRUCT_MODEL) != 1:
    raise AttributeError("Only one of the options above can be True.")


CHOOSE_MODEL = os.environ.get("PRC_MODEL_SIZE", "0.6B")
QWEN3_CONFIG = return_qwen_config(CHOOSE_MODEL)

model = Qwen3Model(QWEN3_CONFIG)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model.to(device);

if USE_REASONING_MODEL or USE_INSTRUCT_MODEL:
    repo_id = f"Qwen/Qwen3-{CHOOSE_MODEL}"
else:
    repo_id = f"Qwen/Qwen3-{CHOOSE_MODEL}-Base"

model_cache_root = Path(os.environ.get("PRC_MODEL_CACHE_DIR", "."))
local_dir = model_cache_root / Path(repo_id).parts[-1]

if CHOOSE_MODEL == "0.6B":
    weights_file = hf_hub_download(
        repo_id=repo_id,
        filename="model.safetensors",
        local_dir=local_dir,
    )
    weights_dict = load_file(weights_file)
else:
    repo_dir = snapshot_download(repo_id=repo_id, local_dir=local_dir)
    index_path = os.path.join(repo_dir, "model.safetensors.index.json")
    with open(index_path, "r") as f:
        index = json.load(f)

    weights_dict = {}
    for filename in set(index["weight_map"].values()):
        shard_path = os.path.join(repo_dir, filename)
        shard = load_file(shard_path)
        weights_dict.update(shard)

load_weights_into_qwen(model, QWEN3_CONFIG, weights_dict)
model.to(device)
del weights_dict

tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')


tokenizer_file_path = str(local_dir / "tokenizer.json")

hf_hub_download(
    repo_id=repo_id,
    filename="tokenizer.json",
    local_dir=local_dir,
)

print(tokenizer_file_path)
if USE_REASONING_MODEL or USE_INSTRUCT_MODEL:
    tokenizer = Qwen3Tokenizer(
        tokenizer_file_path=tokenizer_file_path,
        repo_id=repo_id,
        apply_chat_template=True,
        add_generation_prompt=True,
        add_thinking=USE_REASONING_MODEL
    )

else:
    tokenizer = Qwen3Tokenizer(
        tokenizer_file_path=tokenizer_file_path,
        repo_id=repo_id,
        apply_chat_template=False,
        add_generation_prompt=False,
        add_thinking=False
    )


def prompt_to_ids(prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    formatted_text = tok.apply_chat_template(messages, tokenize=False,     add_generation_prompt=True)
    return tokenizer.encode(formatted_text)


def detect(P, vec, z, entropies, entropy_threshold=0.5, fpr=1e-9):
    r, n = P.shape

    # 1. Mark reliable bit positions
    reliable = entropies >= entropy_threshold

    # 2. Drop any parity check that touches an unreliable bit
    P_int = np.asarray(P, dtype=np.int64)
    unreliable_hits = P_int[:, ~reliable].sum(axis=1)   # # of unreliable bits per check
    keep_check = unreliable_hits == 0
    r_eff = int(keep_check.sum())

    if r_eff == 0:
        return False

    # 3. Compute syndrome weight on the surviving checks
    syndrome = np.asarray(P @ (vec + z), dtype=np.int64) % 2
    wt = int(syndrome[keep_check].sum())

    # 4. Hoeffding threshold scaled to r_eff (NOT r)
    threshold = r_eff / 2 - np.sqrt(0.5 * r_eff * np.log(1 / fpr))

    return wt < threshold


vocab_size  = model.tok_emb.weight.shape[0]

v_0 = torch.zeros(vocab_size, dtype=torch.bfloat16).to(device)
indices  = torch.randperm(vocab_size)[:vocab_size//2]
v_0[indices] = 1.0
v_1 = 1-v_0
partition = torch.concat([v_0,v_1]).reshape(2, vocab_size)

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()


"""
PRC text watermarking: generate + detect (entropy-aware, calibrated).

Generation model (Christ-Gunn / Kuditipudi style)
-------------------------------------------------
- Encode(encoding_key) returns a +/-1 PRC codeword; eta noise is baked in
  via KeyGen's noise_rate.
- For each generated token:
    1. p = LM prob of partition 1.
    2. Codeword bit xi -> Bernoulli sampling parameter:
         bern_p = 2*xi*p              if p <= 0.5
         bern_p = 1 - 2*(1-xi)*(1-p)   if p > 0.5
    3. b ~ Bernoulli(bern_p) selects the partition.
    4. argmax token within that partition.
- The realized bit b matches xi only noisily. When p is far from 0.5, the
  observation is uninformative about xi -- this is the LPN noise channel.

Detection model (entropy-aware + null-calibrated)
-------------------------------------------------
- For each generated token we now also need the LM's partition probability
  p that was used to sample it. `generate_text_watermark_prc` returns these
  alongside the token IDs.
- We fold cyclically to length n with weights = H_2(p) / log(2). Tokens
  drawn at near-deterministic LM steps (p ~= 0 or 1) contribute nearly zero
  to the posterior; tokens drawn at high-entropy steps contribute fully.
- The detection threshold is calibrated by sampling the null distribution
  of the test statistic (random codewords pushed through the same channel)
  and setting threshold = null_mean + z * null_std with z = Phi^-1(1 - fpr).
  This Gaussian-tail calibration is principled because the test statistic
  is a CLT-friendly sum, and it sidesteps the Bernstein bound inside
  Detect, which is over-conservative for entropy-weighted posteriors.
"""

import torch
import numpy as np
from scipy.stats import norm
from prc import KeyGen, Encode, Detect
from detectors import (
    build_prc_generation_record,
    binary_entropy,
    fold_naive,
    fold_entropy_weighted,
    fold_soft_token,
    semantic_sha256,
    tensor_sha256,
    tokens_to_bits,
    detect_hoeffding,
)


# -----------------------------------------------------------------------------
# Conversions and entropy
# -----------------------------------------------------------------------------

def signed_to_bits(signed: torch.Tensor) -> torch.Tensor:
    """+/-1 -> {0,1}.  +1 -> 0,  -1 -> 1."""
    return ((1 - signed) / 2).long()


# -----------------------------------------------------------------------------
# Generation: now also returns the LM partition probabilities
# -----------------------------------------------------------------------------

def generate_text_watermark_prc(
    model,
    token_ids,
    max_new_tokens,
    encoding_key,
    partition_map,
    eos_token_id=None,
    watermark=True,
    collect_trace_details=False,
):
    """
    Yields ``(next_token, p1)`` per step by default. With
    ``collect_trace_details=True``, yields ``(next_token, p1, details)`` so the
    caller can persist the exact PRC codeword and compact LM diagnostics.

    The first two values are:
        next_token : (batch, 1) long tensor of generated token IDs
        p1         : (batch,)   float tensor giving LM P[partition 1] at this step

    The caller should accumulate both streams: the token stream becomes the
    generated text, and the p1 stream is used at detection time to weight
    each observation by the LM's entropy.
    """
    model.eval()
    # device = token_ids.device

    n = encoding_key[0].shape[0]                    # codeword length

    # Each length-n block gets its OWN fresh PRC codeword: when the output is
    # longer than n we do NOT reuse the same codeword cyclically. Detection then
    # checks each block independently (block-OR), so every block is an
    # independent watermark under the same key.
    def _fresh_codeword():
        if watermark:
            signed = Encode(encoding_key)            # torch +/-1, length n
            return signed_to_bits(signed).to(device).float()
        # Preserve the historical null RNG consumption even though these bits
        # do not control unwatermarked sampling and are deliberately not saved
        # as a PRC codeword.
        return torch.bernoulli(torch.full((n,), 0.5)).to(device)

    print("Watermark Enabled (PRC)" if watermark else "Watermark Disabled",
          flush=True)
    codeword = _fresh_codeword()

    partition_map = partition_map.to(device)        # (2, vocab)

    with torch.no_grad():
        # Prefill the prompt once; the cache holds K/V so each decode step only
        # processes the single new token instead of re-running the whole prefix.
        cache = KVCache()
        logits = model(token_ids, cache=cache)[:, -1]                   # (batch, vocab)

        for pos in range(max_new_tokens):
            # New block boundary -> sample a fresh PRC codeword for this block.
            if pos > 0 and pos % n == 0:
                codeword = _fresh_codeword()

            probs = torch.softmax(logits, dim=-1)
            p1 = (probs * partition_map[1].to(logits.device)).sum(dim=-1)  # (batch,)

            if collect_trace_details:
                base_log_probs = torch.log_softmax(logits.float(), dim=-1)
                base_probs = torch.exp(base_log_probs)
                base_lm_entropy = -(
                    base_probs * base_log_probs
                ).sum(dim=-1)
                base_lm_entropy = base_lm_entropy / np.log(2)

            if watermark:
                xi = codeword[pos % n]                                  # 0. or 1.
                bern_p = torch.where(
                    p1 <= 0.5,
                    2 * xi * p1,
                    1 - 2 * (1 - xi) * (1 - p1),
                ).clamp(0.0, 1.0)

                b = torch.bernoulli(bern_p).long()                      # (batch,)
                mask = partition_map[b].to(logits.device)               # (batch, vocab)
                sample_logits = logits.masked_fill(mask == 0, float("-inf"))
            else:
                sample_logits = logits

            sample_probs = torch.softmax(sample_logits.float(), dim=-1)
            next_token = torch.multinomial(sample_probs, num_samples=1)
            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break

            if collect_trace_details:
                base_token_logprob = base_log_probs.gather(
                    1, next_token
                ).squeeze(1)
                codeword_bit = None
                if watermark:
                    codeword_bit = xi.expand(token_ids.shape[0]).detach().cpu()
                yield next_token, p1.detach().cpu(), {
                    "prc_codeword_bit": codeword_bit,
                    "base_lm_entropy": base_lm_entropy.detach().cpu(),
                    "base_token_logprob": base_token_logprob.detach().cpu(),
                }
            else:
                yield next_token, p1.detach().cpu()

            # Decode step: feed only the new token; the cache supplies past K/V.
            logits = model(next_token, cache=cache)[:, -1]


# -----------------------------------------------------------------------------
# Folding: equal-weight (legacy) and entropy-aware
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Detector test statistic (matches the internals of Detect)
# -----------------------------------------------------------------------------

def _test_statistic(posteriors: np.ndarray, decoding_key) -> float:
    """
    Compute the centered log-likelihood test statistic from Detect's internals,
    so we can compare it to a calibrated threshold.

    Returns log_plus.sum() - 0.5 * log_prod.sum(), which is the centered
    statistic whose null mean is ~0.
    """
    _, parity_check_matrix, one_time_pad, _, noise_rate, _, _, _, t = decoding_key
    pc = (1 - 2 * noise_rate) * (1 - 2 * np.array(one_time_pad, dtype=float)) * posteriors
    r = parity_check_matrix.shape[0]
    Pi = np.prod(pc[parity_check_matrix.indices.reshape(r, t)], axis=1)
    log_plus = np.log(np.clip((1 + Pi) / 2, 1e-15, 1.0))
    log_minus = np.log(np.clip((1 - Pi) / 2, 1e-15, 1.0))
    log_prod = log_plus + log_minus
    return float(log_plus.sum() - 0.5 * log_prod.sum())


# -----------------------------------------------------------------------------
# Threshold calibration
# -----------------------------------------------------------------------------

def calibrate_threshold(
    decoding_key,
    p_array_for_calibration: np.ndarray,
    fpr: float = 1e-9,
    num_calibration_trials: int = 100,
    seed: int = 90210,
) -> dict:
    """
    Estimate the null distribution of the test statistic and return a
    threshold for the requested FPR.

    The null is generated by:
      1. Sampling a random codeword (uniform bits).
      2. Pushing it through the SAME LM channel the user generated text on
         (we reuse p_array_for_calibration to match the entropy profile).
      3. Computing the entropy-weighted folded posterior and its statistic.

    Args:
        decoding_key: from KeyGen.
        p_array_for_calibration: an array of LM partition probabilities to
            reuse when simulating the null. In practice, pass the p1 trace
            recorded during your watermarked generation -- this guarantees
            the null calibration matches the channel statistics of the
            content under test.
        fpr: target false-positive rate.
        num_calibration_trials: how many null samples to draw.

    Returns:
        Dict with keys: 'threshold', 'null_mean', 'null_std', 'z',
        'fpr', 'num_trials'.
    """
    rng = np.random.default_rng(seed)
    n = decoding_key[0].shape[0]
    p_arr = np.asarray(p_array_for_calibration, dtype=np.float64)
    num_tokens = len(p_arr)
    assert num_tokens >= n, (
        f"Need at least n={n} tokens of p-trace for calibration, got "
        f"{num_tokens}. Generate a longer sequence."
    )

    null_stats = np.empty(num_calibration_trials, dtype=np.float64)
    for trial in range(num_calibration_trials):
        # Random "codeword" bits and the resulting realized bits under the
        # same Bernoulli channel.
        random_codeword = rng.integers(0, 2, size=n)
        xi = random_codeword[np.arange(num_tokens) % n]
        bern_p = np.where(p_arr <= 0.5,
                          2 * xi * p_arr,
                          1 - 2 * (1 - xi) * (1 - p_arr))
        bern_p = np.clip(bern_p, 0.0, 1.0)
        observed = rng.binomial(1, bern_p)
        post = fold_entropy_weighted(observed, p_arr, n)
        null_stats[trial] = _test_statistic(post, decoding_key)

    null_mean = float(null_stats.mean())
    null_std = float(null_stats.std(ddof=1))
    z = float(norm.ppf(1.0 - fpr))
    threshold = null_mean + z * null_std

    return {
        "threshold": threshold,
        "null_mean": null_mean,
        "null_std": null_std,
        "z": z,
        "fpr": fpr,
        "num_trials": num_calibration_trials,
    }


# -----------------------------------------------------------------------------
# Detection: entropy-aware + calibrated
# -----------------------------------------------------------------------------

def detect_watermark_prc(
    decoding_key,
    generated_token_ids: torch.Tensor,
    partition_probs: np.ndarray,
    partition_map: torch.Tensor,
    fpr: float = 1e-9,
    num_calibration_trials: int = 100,
    return_info: bool = False,
):
    """
    Entropy-aware, null-calibrated watermark detection.

    Args:
        decoding_key: from KeyGen.
        generated_token_ids: 1-D long tensor of generated token IDs.
        partition_probs: numpy array of the LM's P[partition 1] at each
            generation step. Must have the same length as
            generated_token_ids. Recorded during generate_text_watermark_prc.
        partition_map: (2, vocab) 0/1 indicator tensor.
        fpr: target false-positive rate.
        num_calibration_trials: null calibration sample size.
        return_info: if True, return (decision, info_dict).

    Returns:
        bool decision, or (decision, info_dict) if return_info=True.
    """
    n = decoding_key[0].shape[0]
    bits = tokens_to_bits(generated_token_ids, partition_map)
    p_arr = np.asarray(partition_probs, dtype=np.float64)
    assert bits.shape == p_arr.shape, (
        f"tokens ({bits.shape}) and partition_probs ({p_arr.shape}) "
        f"must have the same length"
    )

    posteriors = fold_entropy_weighted(bits, p_arr, n)
    statistic = _test_statistic(posteriors, decoding_key)

    cal = calibrate_threshold(decoding_key, p_arr, fpr=fpr,
                              num_calibration_trials=num_calibration_trials)

    decision = bool(statistic > cal["threshold"])
    if return_info:
        info = {**cal, "statistic": statistic,
                "sigmas_above_null": (statistic - cal["null_mean"]) / cal["null_std"]
                                    if cal["null_std"] > 0 else float("inf")}
        return decision, info
    return decision


# -----------------------------------------------------------------------------
# Three-phase workflow helpers (collect -> fit -> detect)
# -----------------------------------------------------------------------------

def generate_and_collect(generator):
    """Drain a generate_text_watermark_prc generator into (tokens, p_trace)."""
    tok_chunks, p_chunks = [], []
    for item in generator:
        next_token, p1 = item[:2]
        tok_chunks.append(next_token)
        p_chunks.append(p1)
    if not tok_chunks:
        return torch.zeros(0, dtype=torch.long), np.zeros(0, dtype=np.float64)
    tokens = torch.cat(tok_chunks, dim=1).flatten()
    p_trace = torch.stack(p_chunks).flatten().float().numpy().astype(np.float64)
    return tokens, p_trace


def generate_and_collect_detailed(generator):
    """Drain detailed PRC generation into tokens, p-trace, and diagnostics."""
    tok_chunks, p_chunks = [], []
    codeword_chunks, entropy_chunks, logprob_chunks = [], [], []
    saw_null_codeword = False
    for item in generator:
        if len(item) != 3:
            raise ValueError(
                "detailed collection requires collect_trace_details=True"
            )
        next_token, p1, details = item
        tok_chunks.append(next_token)
        p_chunks.append(p1)
        entropy_chunks.append(details["base_lm_entropy"])
        logprob_chunks.append(details["base_token_logprob"])
        codeword_bit = details["prc_codeword_bit"]
        if codeword_bit is None:
            saw_null_codeword = True
        else:
            codeword_chunks.append(codeword_bit)

    if not tok_chunks:
        empty = np.zeros(0, dtype=np.float32)
        return (
            torch.zeros(0, dtype=torch.long),
            np.zeros(0, dtype=np.float64),
            {
                "prc_codeword_bits": None,
                "base_lm_entropy": empty,
                "base_token_logprob": empty.copy(),
            },
        )

    if saw_null_codeword and codeword_chunks:
        raise ValueError("generation mixed null and PRC codeword trace steps")
    tokens = torch.cat(tok_chunks, dim=1).flatten()
    p_trace = torch.stack(p_chunks, dim=1).flatten().float().numpy().astype(
        np.float64
    )
    codeword_bits = None
    if codeword_chunks:
        codeword_bits = (
            torch.stack(codeword_chunks, dim=1)
            .flatten()
            .to(dtype=torch.uint8)
            .numpy()
        )
    details = {
        "prc_codeword_bits": codeword_bits,
        "base_lm_entropy": (
            torch.stack(entropy_chunks, dim=1).flatten().float().numpy()
        ),
        "base_token_logprob": (
            torch.stack(logprob_chunks, dim=1).flatten().float().numpy()
        ),
    }
    return tokens, p_trace, details


def generate_batch_and_collect(
    model,
    prompt_ids_batch,
    max_new_tokens,
    encoding_key,
    partition_map,
    watermark=True,
    return_trace_details=False,
):
    """Batched PRC generation over B equal-length prompts.

    Args:
        prompt_ids_batch: (B, L) long tensor; all prompts must share length L
            (RealNews prefixes are all 50 tokens, so no padding is needed).
        others: as in generate_text_watermark_prc.

    Returns by default:
        tokens   : (B, T) long tensor on CPU (T = max_new_tokens).
        p_traces : (B, T) float64 numpy array of LM P[partition 1] per step.

    With ``return_trace_details=True``, also returns a dict containing
    ``prc_codeword_bits``, ``base_lm_entropy``, and
    ``base_token_logprob`` arrays of shape (B, T). The codeword value is None
    for null generations.

    Each sequence gets its OWN fresh PRC codeword per length-n block, so the B
    generations are independent watermark instances under the same key -- byte
    for byte the same channel the per-job path applies, just B at a time. At
    B=1 this is identical to generate_text_watermark_prc + generate_and_collect.
    """
    model.eval()
    B = prompt_ids_batch.shape[0]
    n = encoding_key[0].shape[0]
    pm = partition_map.to(device)                      # (2, vocab)
    part1 = pm[1]                                       # (vocab,)

    def _fresh_codewords():
        # One independent PRC codeword per sequence -> (B, n) of {0.,1.}.
        rows = [signed_to_bits(Encode(encoding_key)).to(device).float()
                for _ in range(B)]
        return torch.stack(rows, dim=0)

    print(f"Watermark Enabled (PRC), batch={B}" if watermark
          else f"Watermark Disabled, batch={B}", flush=True)
    codeword = _fresh_codewords() if watermark else None

    tok_steps, p_steps = [], []
    codeword_steps, lm_entropy_steps, token_logprob_steps = [], [], []
    with torch.no_grad():
        cache = KVCache()
        logits = model(prompt_ids_batch, cache=cache)[:, -1]        # (B, vocab)

        for pos in range(max_new_tokens):
            if watermark and pos > 0 and pos % n == 0:
                codeword = _fresh_codewords()

            probs = torch.softmax(logits, dim=-1)                   # (B, vocab)
            p1 = (probs * part1.to(logits.device)).sum(dim=-1)      # (B,)

            if return_trace_details:
                base_log_probs = torch.log_softmax(logits.float(), dim=-1)
                base_probs = torch.exp(base_log_probs)
                lm_entropy_steps.append(-(
                    base_probs * base_log_probs
                ).sum(dim=-1).div(np.log(2)).detach().cpu())

            if watermark:
                xi = codeword[:, pos % n]                           # (B,)
                bern_p = torch.where(
                    p1 <= 0.5,
                    2 * xi * p1,
                    1 - 2 * (1 - xi) * (1 - p1),
                ).clamp(0.0, 1.0)
                b = torch.bernoulli(bern_p).long()                 # (B,)
                mask = pm[b].to(logits.device)                     # (B, vocab)
                sample_logits = logits.masked_fill(mask == 0, float("-inf"))
            else:
                sample_logits = logits

            sample_probs = torch.softmax(sample_logits.float(), dim=-1)
            next_token = torch.multinomial(sample_probs, num_samples=1)  # (B,1)

            tok_steps.append(next_token)
            p_steps.append(p1.detach().cpu())
            if return_trace_details:
                token_logprob_steps.append(base_log_probs.gather(
                    1, next_token
                ).squeeze(1).detach().cpu())
                if watermark:
                    codeword_steps.append(xi.detach().cpu())

            logits = model(next_token, cache=cache)[:, -1]         # (B, vocab)

    tokens = torch.cat(tok_steps, dim=1).cpu()                     # (B, T)
    p_traces = torch.stack(p_steps, dim=1).float().numpy().astype(np.float64)
    if not return_trace_details:
        return tokens, p_traces
    details = {
        "prc_codeword_bits": (
            torch.stack(codeword_steps, dim=1).to(dtype=torch.uint8).numpy()
            if watermark else None
        ),
        "base_lm_entropy": (
            torch.stack(lm_entropy_steps, dim=1).float().numpy()
        ),
        "base_token_logprob": (
            torch.stack(token_logprob_steps, dim=1).float().numpy()
        ),
    }
    return tokens, p_traces, details


def generate_batch_and_collect_online(
    model,
    prompt_ids_batch,
    max_new_tokens,
    online_key,
    partition_map,
    watermark=True,
    return_trace_details=False,
    document_seeds=None,
    prefix_tokens_batch=None,
    prefix_codeword_bits_batch=None,
    kv_cache_implementation="concat",
):
    """Batched forced-prefix generation, optionally resuming a saved prefix.

    This is intentionally separate from ``generate_batch_and_collect`` so the
    fixed-n experiment remains byte-for-byte untouched.  Each call to the
    online encoder adds at most one check, and that check references only
    earlier generated coordinates.  ``document_seeds`` are tied to prompt IDs
    by the runner, making the latent stream invariant to batch size/order.

    ``max_new_tokens`` is the target total continuation length.  When
    ``prefix_tokens_batch`` is supplied, its tokens are replayed through the
    model to reconstruct the exact incremental KV-cache shape, the causal PRC
    encoder is deterministically advanced and checked against
    ``prefix_codeword_bits_batch``, and only the missing suffix is returned.

    Watermarked bucket/token draws are addressed by document seed and absolute
    position rather than PyTorch's process-global RNG.  Consequently a
    one-shot run and a resumed run use the same random variates for every
    position, independent of batching or worker scheduling.

    ``kv_cache_implementation`` is deliberately opt-in. ``"concat"`` retains
    the historical cache path, while ``"static"`` preallocates per-layer K/V
    storage for the prompt plus the requested continuation length.
    """
    from online_prc import (
        GENERATION_SAMPLER_VERSION,
        OnlinePRCEncoder,
        OnlinePRCKey,
        document_uniform,
    )

    if isinstance(online_key, dict):
        online_key = OnlinePRCKey.from_dict(online_key)
    model.eval()
    batch_size = int(prompt_ids_batch.shape[0])
    target_length = int(max_new_tokens)
    kv_cache_implementation = normalize_kv_cache_implementation(
        kv_cache_implementation
    )
    if target_length < 0:
        raise ValueError("max_new_tokens must be nonnegative")
    if watermark:
        if document_seeds is None:
            raise ValueError("online watermark generation requires document_seeds")
        if len(document_seeds) != batch_size:
            raise ValueError(
                f"got {len(document_seeds)} document seeds for batch {batch_size}"
            )
        encoder = OnlinePRCEncoder(online_key, document_seeds)
    else:
        encoder = None

    if prefix_tokens_batch is None:
        prefix_tokens = torch.empty(
            (batch_size, 0), dtype=torch.long, device=prompt_ids_batch.device
        )
    else:
        if not watermark:
            raise ValueError("saved-prefix resume is only supported for watermark=True")
        prefix_tokens = torch.as_tensor(
            prefix_tokens_batch,
            dtype=torch.long,
            device=prompt_ids_batch.device,
        )
        if prefix_tokens.ndim != 2 or prefix_tokens.shape[0] != batch_size:
            raise ValueError(
                "prefix_tokens_batch must have shape (batch_size, prefix_length)"
            )
    prefix_length = int(prefix_tokens.shape[1])
    if prefix_length > target_length:
        raise ValueError(
            f"saved prefix length {prefix_length} exceeds target {target_length}"
        )
    if prefix_length:
        if prefix_codeword_bits_batch is None:
            raise ValueError(
                "resumed watermark generation requires prefix_codeword_bits_batch"
            )
        prefix_codeword = np.asarray(
            prefix_codeword_bits_batch, dtype=np.uint8
        )
        if prefix_codeword.shape != (batch_size, prefix_length):
            raise ValueError(
                "prefix_codeword_bits_batch must match prefix token shape"
            )
        expected_prefix = encoder.encode_to_length(prefix_length)
        if not np.array_equal(expected_prefix, prefix_codeword):
            raise ValueError(
                "saved prefix PRC bits do not match the requested online key, "
                "document seeds, and positions"
            )
    elif prefix_codeword_bits_batch is not None:
        supplied = np.asarray(prefix_codeword_bits_batch)
        if supplied.size:
            raise ValueError("nonempty prefix codeword supplied without prefix tokens")

    pm = partition_map.to(device)
    part1 = pm[1]
    print(
        f"Watermark Enabled ({online_key.scheme}), batch={batch_size}, "
        f"resume_prefix={prefix_length}"
        if watermark else f"Watermark Disabled, batch={batch_size}",
        flush=True,
    )

    token_steps, p_steps = [], []
    codeword_steps, entropy_steps, logprob_steps = [], [], []
    with torch.no_grad():
        cache = make_kv_cache(
            kv_cache_implementation,
            max_length=int(prompt_ids_batch.shape[1]) + target_length,
        )
        logits = model(prompt_ids_batch, cache=cache)[:, -1]

        # Replaying one token at a time matches the incremental attention path
        # used during one-shot generation and reconstructs the KV cache without
        # storing model-specific GPU state in every generation record.
        for position in range(prefix_length):
            logits = model(
                prefix_tokens[:, position:position + 1], cache=cache
            )[:, -1]

        for position in range(prefix_length, target_length):
            probs = torch.softmax(logits, dim=-1)
            p1 = (probs * part1.to(logits.device)).sum(dim=-1)

            if return_trace_details:
                base_log_probs = torch.log_softmax(logits.float(), dim=-1)
                base_probs = torch.exp(base_log_probs)
                entropy_steps.append(-(
                    base_probs * base_log_probs
                ).sum(dim=-1).div(np.log(2)).detach().cpu())

            if watermark:
                # The bit for this coordinate is constructed only now, after
                # every coordinate referenced by its possible check exists.
                xi = torch.as_tensor(
                    encoder.next_bits(), dtype=torch.float32, device=logits.device
                )
                bern_p = torch.where(
                    p1 <= 0.5,
                    2 * xi * p1,
                    1 - 2 * (1 - xi) * (1 - p1),
                ).clamp(0.0, 1.0)
                bucket_uniform = torch.tensor(
                    [
                        document_uniform(seed, "lm-bucket/v1", position)
                        for seed in document_seeds
                    ],
                    dtype=torch.float64,
                    device=logits.device,
                )
                bucket = (bucket_uniform < bern_p.double()).long()
                mask = pm[bucket].to(logits.device)
                sample_logits = logits.masked_fill(mask == 0, float("-inf"))
            else:
                xi = None
                sample_logits = logits

            sample_probs = torch.softmax(sample_logits.float(), dim=-1)
            if watermark:
                token_uniform = torch.tensor(
                    [
                        document_uniform(seed, "lm-token/v1", position)
                        for seed in document_seeds
                    ],
                    dtype=sample_probs.dtype,
                    device=sample_probs.device,
                ).unsqueeze(1)
                cumulative = torch.cumsum(sample_probs, dim=-1)
                # Guard against a final float32 sum infinitesimally below one.
                cumulative[:, -1] = 1.0
                next_token = torch.searchsorted(
                    cumulative, token_uniform, right=False
                ).clamp_max(sample_probs.shape[1] - 1)
            else:
                next_token = torch.multinomial(sample_probs, num_samples=1)
            token_steps.append(next_token)
            p_steps.append(p1.detach().cpu())
            if return_trace_details:
                logprob_steps.append(base_log_probs.gather(
                    1, next_token
                ).squeeze(1).detach().cpu())
                if watermark:
                    codeword_steps.append(xi.detach().cpu())

            logits = model(next_token, cache=cache)[:, -1]

    suffix_length = target_length - prefix_length
    if suffix_length:
        tokens = torch.cat(token_steps, dim=1).cpu()
        p_traces = (
            torch.stack(p_steps, dim=1).float().numpy().astype(np.float64)
        )
    else:
        tokens = torch.empty((batch_size, 0), dtype=torch.long)
        p_traces = np.empty((batch_size, 0), dtype=np.float64)
    if not return_trace_details:
        return tokens, p_traces
    details = {
        "prc_codeword_bits": (
            torch.stack(codeword_steps, dim=1).to(dtype=torch.uint8).numpy()
            if watermark and suffix_length
            else np.empty((batch_size, 0), dtype=np.uint8)
            if watermark else None
        ),
        "base_lm_entropy": (
            torch.stack(entropy_steps, dim=1).float().numpy()
            if suffix_length else np.empty((batch_size, 0), dtype=np.float32)
        ),
        "base_token_logprob": (
            torch.stack(logprob_steps, dim=1).float().numpy()
            if suffix_length else np.empty((batch_size, 0), dtype=np.float32)
        ),
        "online_sampler_version": (
            GENERATION_SAMPLER_VERSION if watermark else None
        ),
        "kv_cache_implementation": kv_cache_implementation,
        "kv_cache_version": kv_cache_version(kv_cache_implementation),
        "resume_prefix_length": prefix_length,
    }
    return tokens, p_traces, details


def estimate_partition_trace_batch(
    model,
    prompt_ids_batch,
    generated_tokens_batch,
    partition_map,
    kv_cache_implementation="concat",
):
    """Teacher-force generated tokens and estimate P[partition 1] per step.

    This is used when detection estimates entropy with a model different from
    the generator. It does not sample; it only replays cached generations and
    records the detector model's probability mass on partition 1.
    """
    model.eval()
    prompt_ids_batch = prompt_ids_batch.to(device)
    generated_tokens_batch = generated_tokens_batch.to(device)
    pm = partition_map.to(device)
    part1 = pm[1]

    # Teacher forcing has a known final sequence length, so callers doing
    # large offline replays can preallocate the cache and avoid an O(T^2)
    # stream of full-history ``torch.cat`` copies.  Keep concat as the default
    # so existing callers retain their historical behavior.
    trace = teacher_force_partition_trace_batch(
        model,
        prompt_ids_batch,
        generated_tokens_batch,
        part1,
        kv_cache_implementation=kv_cache_implementation,
    )
    return trace.numpy().astype(np.float64)


def estimate_partition_trace(
    model,
    prompt_ids,
    generated_tokens,
    partition_map,
    kv_cache_implementation="concat",
):
    """Single-sequence wrapper around estimate_partition_trace_batch."""
    prompt_batch = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    token_batch = generated_tokens.reshape(1, -1).to(device)
    return estimate_partition_trace_batch(
        model,
        prompt_batch,
        token_batch,
        partition_map,
        kv_cache_implementation=kv_cache_implementation,
    )[0]


def _fold_naive_uniform(bits, p_arr, n):
    return fold_naive(bits, n)


FOLD_FNS = {
    "entropy": fold_entropy_weighted,
    "naive": _fold_naive_uniform,
}


def fit_calibration(
    decoding_key,
    calibration_p_traces,
    fpr=1e-9,
    num_simulated_nulls=2000,
    min_trace_length=None,
    seed=1234,
    fold="entropy",
):
    """Fit a single detection threshold from a batch of p-traces."""
    n = decoding_key[0].shape[0]
    if min_trace_length is None:
        min_trace_length = n
    fold_fn = FOLD_FNS[fold]

    traces = [np.asarray(p, dtype=np.float64)
              for p in calibration_p_traces
              if len(p) >= min_trace_length]
    if not traces:
        raise ValueError(
            f"No calibration traces of length >= {min_trace_length}."
        )

    rng = np.random.default_rng(seed)
    null_stats = np.empty(num_simulated_nulls, dtype=np.float64)

    for i in range(num_simulated_nulls):
        p_arr = traces[rng.integers(0, len(traces))]
        T = len(p_arr)
        codeword = rng.integers(0, 2, size=n)
        xi = codeword[np.arange(T) % n]
        bern_p = np.where(p_arr <= 0.5,
                          2 * xi * p_arr,
                          1 - 2 * (1 - xi) * (1 - p_arr))
        bern_p = np.clip(bern_p, 0.0, 1.0)
        observed = rng.binomial(1, bern_p)
        post = fold_fn(observed, p_arr, n)
        null_stats[i] = _test_statistic(post, decoding_key)

    null_mean = float(null_stats.mean())
    null_std = float(null_stats.std(ddof=1))
    from scipy.stats import norm
    z = float(norm.ppf(1.0 - fpr))
    threshold = null_mean + z * null_std

    return {
        "threshold": threshold,
        "null_mean": null_mean,
        "null_std": null_std,
        "fpr": fpr,
        "z": z,
        "n": n,
        "num_traces_used": len(traces),
        "num_simulated_nulls": num_simulated_nulls,
        "fold": fold,
    }


def save_threshold_state(state, path):
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def load_threshold_state(path):
    with open(path) as f:
        return json.load(f)


def detect_with_threshold(
    decoding_key,
    generated_token_ids,
    partition_probs,
    partition_map,
    threshold_state,
    return_info=False,
):
    """Fast detection using a precomputed threshold."""
    n = decoding_key[0].shape[0]
    if threshold_state["n"] != n:
        raise ValueError(
            f"Calibration was for n={threshold_state['n']}, but key has n={n}."
        )

    fold_fn = FOLD_FNS[threshold_state.get("fold", "entropy")]
    bits = tokens_to_bits(generated_token_ids, partition_map)
    p_arr = np.asarray(partition_probs, dtype=np.float64)
    if bits.shape != p_arr.shape:
        raise ValueError(
            f"tokens length {bits.shape[0]} != p_trace length {p_arr.shape[0]}"
        )

    posteriors = fold_fn(bits, p_arr, n)
    statistic = _test_statistic(posteriors, decoding_key)
    decision = bool(statistic > threshold_state["threshold"])

    if return_info:
        sigmas = ((statistic - threshold_state["null_mean"])
                  / threshold_state["null_std"]) if threshold_state["null_std"] > 0 \
                  else float("inf")
        return decision, {
            "statistic": statistic,
            "threshold": threshold_state["threshold"],
            "sigmas_above_null": sigmas,
        }
    return decision


# -----------------------------------------------------------------------------
# Hoeffding detector (proven FPR, no calibration) -- prc_fpr_proof.pdf
# -----------------------------------------------------------------------------
#
# Folds the observed tokens into length-n soft-tokens S_j = t_j * H_j (via
# fold_soft_token, |S_j| <= 1) and calls prc.Detect, whose threshold
# tau = sqrt(2V * log(1/F)) has a provable FPR <= F over the random OTP. No
# null calibration is needed, so this can run directly on cached generations.
# detect_hoeffding + the fold helpers now live in detectors.py (model-free) and
# are imported at the top of this module.


# -----------------------------------------------------------------------------
# Syndrome-weight detector (PRC paper threshold, Theorem 1)
# -----------------------------------------------------------------------------
#
# Per-block, hard-bit detector: no fold, no entropy weighting, no calibration.
# Generated text is split into consecutive non-overlapping blocks of length n;
# each block gets its own syndrome check. Document is declared watermarked iff
# *any* block passes.  Trailing tokens with T % n != 0 are ignored.
#
# Per-block threshold:  (1/2 - r_eff^{-1/4}) * r_eff   = r_eff/2 - r_eff^{3/4}
# Per-block decision:   weight < threshold
# Document decision:    OR over blocks
#
# entropy_threshold (in bits, H_2), evaluated per block:
#   - None: r_eff = r (use every parity check).
#   - float: drop any check whose t token positions include any token with
#            H_2(p1) < entropy_threshold within that block.

def _syndrome_block(block_bits, block_p, indices, z_rowsum, entropy_threshold):
    syndrome = ((block_bits[indices].sum(axis=1) + z_rowsum) % 2).astype(np.int64)
    if entropy_threshold is None:
        keep = np.ones(syndrome.shape[0], dtype=bool)
    else:
        ent_bits = binary_entropy(block_p) / np.log(2)
        unreliable_pos = ent_bits < entropy_threshold
        keep = ~unreliable_pos[indices].any(axis=1)
    r_eff = int(keep.sum())
    if r_eff == 0:
        return False, 0, 0.0, 0
    weight = int(syndrome[keep].sum())
    threshold = (0.5 - r_eff ** -0.25) * r_eff
    return bool(weight < threshold), weight, float(threshold), r_eff


def detect_syndrome(
    decoding_key,
    generated_token_ids,
    partition_probs,
    partition_map,
    entropy_threshold=None,
    return_info=False,
):
    (_, parity_check_matrix, one_time_pad, _, _, _, _, _, t) = decoding_key
    r, n = parity_check_matrix.shape

    bits = tokens_to_bits(generated_token_ids, partition_map)
    p_arr = np.asarray(partition_probs, dtype=np.float64)
    if bits.shape != p_arr.shape:
        raise ValueError(
            f"tokens length {bits.shape[0]} != p_trace length {p_arr.shape[0]}"
        )

    T = bits.shape[0]
    n_blocks = T // n
    if n_blocks == 0:
        raise ValueError(f"detect_syndrome needs at least n={n} tokens, got T={T}")

    z = np.asarray(one_time_pad, dtype=np.int64)
    indices = parity_check_matrix.indices.reshape(r, t)
    z_rowsum = z[indices].sum(axis=1)

    blocks = []
    decision = False
    for b in range(n_blocks):
        sl = slice(b * n, (b + 1) * n)
        d, w, thr, r_eff = _syndrome_block(
            bits[sl], p_arr[sl], indices, z_rowsum, entropy_threshold
        )
        blocks.append({"block": b, "weight": w, "threshold": thr,
                       "r_eff": r_eff, "passed": d})
        if d:
            decision = True

    if return_info:
        info = {
            "method": "syndrome",
            "entropy_threshold": entropy_threshold,
            "r": int(r),
            "n_blocks": n_blocks,
            "blocks_passed": sum(b["passed"] for b in blocks),
            "blocks": blocks,
        }
        return decision, info
    return decision


def chat_eval_benchmark(benchmark: Task, model: Qwen3Model, tokenizer, log: bool=False):
    scores = []
    for i in range(benchmark.num_examples()):
        conversation = benchmark.get_example(i)
        enc = tokenizer.encode(conversation['messages'])
        if i==0:
            print(tokenizer.decode(enc))
        """ generate_text_watermark_prc(
            model,
            token_ids,
            max_new_tokens,
            encoding_key,
            partition_map,
            eos_token_id=tokenizer.eos_token_id,
            watermark=True)
        """
    return
