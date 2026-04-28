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


pkgs = [
    "huggingface_hub",  # to download pretrained weights
    "tokenizers",       # to implement the tokenizer
    "torch",            # to implement the model
]


for p in pkgs:
    print(f"{p} version: {version(p)}")

USE_BASE_MODEL = True
USE_REASONING_MODEL = False
USE_INSTRUCT_MODEL = False

if (USE_BASE_MODEL + USE_REASONING_MODEL
    + USE_INSTRUCT_MODEL) != 1:
    raise AttributeError("Only one of the options above can be True.")


CHOOSE_MODEL = "0.6B"
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

local_dir = Path(repo_id).parts[-1]

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


if USE_REASONING_MODEL:
    tokenizer_file_path = f"Qwen3-{CHOOSE_MODEL}/tokenizer.json"
else:
    tokenizer_file_path = f"Qwen3-{CHOOSE_MODEL}-Base/tokenizer.json"

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


# -----------------------------------------------------------------------------
# Conversions and entropy
# -----------------------------------------------------------------------------

def signed_to_bits(signed: torch.Tensor) -> torch.Tensor:
    """+/-1 -> {0,1}.  +1 -> 0,  -1 -> 1."""
    return ((1 - signed) / 2).long()


def binary_entropy(p):
    """H_2(p) in nats. Vectorized; safe at p=0 and p=1."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    return -(p * np.log(p) + (1.0 - p) * np.log1p(-p))


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
):
    """
    Yields (next_token, p1) per step where:
        next_token : (batch, 1) long tensor of generated token IDs
        p1         : (batch,)   float tensor giving LM P[partition 1] at this step
                                (equal to None when watermark=False, since the
                                 detector won't use entropy info on unwatermarked
                                 content -- but we still emit None for symmetry)

    The caller should accumulate both streams: the token stream becomes the
    generated text, and the p1 stream is used at detection time to weight
    each observation by the LM's entropy.
    """
    model.eval()
    # device = token_ids.device

    n = encoding_key[0].shape[0]                    # codeword length

    if watermark:
        print("Watermark Enabled (PRC)", flush=True)
        signed = Encode(encoding_key)               # torch +/-1, length n
        codeword = signed_to_bits(signed).to(device).float()
    else:
        print("Watermark Disabled", flush=True)
        codeword = torch.bernoulli(torch.full((n,), 0.5)).to(device)

    partition_map = partition_map.to(device)        # (2, vocab)

    with torch.no_grad():
        # Prefill the prompt once; the cache holds K/V so each decode step only
        # processes the single new token instead of re-running the whole prefix.
        cache = KVCache()
        logits = model(token_ids, cache=cache)[:, -1]                   # (batch, vocab)

        for pos in range(max_new_tokens):
            probs = torch.softmax(logits, dim=-1)
            p1 = (probs * partition_map[1].to(logits.device)).sum(dim=-1)  # (batch,)

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

            yield next_token, p1.detach().cpu()

            # Decode step: feed only the new token; the cache supplies past K/V.
            logits = model(next_token, cache=cache)[:, -1]


# -----------------------------------------------------------------------------
# Folding: equal-weight (legacy) and entropy-aware
# -----------------------------------------------------------------------------

def fold_naive(observed_bits: np.ndarray, n: int) -> np.ndarray:
    """Cyclic fold averaging +/-1 signs at each codeword slot."""
    signs = (1 - 2 * observed_bits.astype(np.int64)).astype(np.float64)
    seq_len = signs.shape[0]
    sums = np.zeros(n, dtype=np.float64)
    counts = np.zeros(n, dtype=np.float64)
    idx = np.arange(seq_len) % n
    np.add.at(sums, idx, signs)
    np.add.at(counts, idx, 1)
    return sums / np.maximum(counts, 1.0)


def fold_entropy_weighted(observed_bits: np.ndarray, p_array: np.ndarray,
                          n: int) -> np.ndarray:
    """
    Entropy-weighted cyclic fold. Each observation contributes its sign
    scaled by H_2(p)/log(2), so deterministic LM steps contribute ~0.

    p_array gives the LM's P[partition 1] at each generation step.
    """
    signs = (1 - 2 * observed_bits.astype(np.int64)).astype(np.float64)
    weights = binary_entropy(p_array) / np.log(2)         # in [0, 1]
    seq_len = signs.shape[0]
    sums = np.zeros(n, dtype=np.float64)
    norms = np.zeros(n, dtype=np.float64)
    idx = np.arange(seq_len) % n
    np.add.at(sums, idx, weights * signs)
    np.add.at(norms, idx, weights)
    return sums / np.maximum(norms, 1e-9)


# -----------------------------------------------------------------------------
# Tokens -> bits (the per-step bit observed by the detector)
# -----------------------------------------------------------------------------

def tokens_to_bits(token_ids: torch.Tensor,
                   partition_map: torch.Tensor) -> np.ndarray:
    """Look up each token's partition (0 or 1) -> length-T int array."""
    if token_ids.dim() != 1:
        token_ids = token_ids.flatten()
    bit_for_token = partition_map[1].long().to(token_ids.device)
    bits = bit_for_token[token_ids].detach().cpu().numpy().astype(np.int64)
    return bits


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
    for next_token, p1 in generator:
        tok_chunks.append(next_token)
        p_chunks.append(p1)
    if not tok_chunks:
        return torch.zeros(0, dtype=torch.long), np.zeros(0, dtype=np.float64)
    tokens = torch.cat(tok_chunks, dim=1).flatten()
    p_trace = torch.stack(p_chunks).flatten().float().numpy().astype(np.float64)
    return tokens, p_trace


def fit_calibration(
    decoding_key,
    calibration_p_traces,
    fpr=1e-9,
    num_simulated_nulls=2000,
    min_trace_length=None,
    seed=1234,
):
    """Fit a single detection threshold from a batch of p-traces."""
    n = decoding_key[0].shape[0]
    if min_trace_length is None:
        min_trace_length = n

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
        post = fold_entropy_weighted(observed, p_arr, n)
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

    bits = tokens_to_bits(generated_token_ids, partition_map)
    p_arr = np.asarray(partition_probs, dtype=np.float64)
    if bits.shape != p_arr.shape:
        raise ValueError(
            f"tokens length {bits.shape[0]} != p_trace length {p_arr.shape[0]}"
        )

    posteriors = fold_entropy_weighted(bits, p_arr, n)
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

