"""
Identify the outlier watermarked prompt (worst naive-fold stat in n=128 run)
and dump per-token / per-slot / per-parity-check diagnostics for an interactive
demo.

Outputs:
  outlier_demo.json  -- structured data, one prompt
  outlier_summary.md -- human-readable summary
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import json
import torch
import numpy as np

from watermark_expt import (
    fold_naive,
    fold_entropy_weighted,
    binary_entropy,
    _test_statistic,
    tokens_to_bits,
    tokenizer,
    test_prompts,
)

WORKDIR = "calib_workdir"


def decode_token(tid):
    """Decode a single token id to a string, escaping for JSON."""
    s = tokenizer.decode([int(tid)])
    return s


def per_check_contributions(decoding_key, posteriors):
    """
    Reproduce the test statistic per parity check.

    Returns:
      indices_per_check : (r, t) int array of slot indices
      Pi                : (r,) product of pc-adjusted posteriors per check
      log_plus          : (r,) per-check log_plus
      log_prod          : (r,) per-check log_plus + log_minus
      contrib           : (r,) per-check contribution to (log_plus - 0.5*log_prod)
    """
    _, parity_check_matrix, one_time_pad, _, noise_rate, _, _, _, t = decoding_key
    pc = (1 - 2 * noise_rate) * (1 - 2 * np.array(one_time_pad, dtype=float)) * posteriors
    r = parity_check_matrix.shape[0]
    indices = parity_check_matrix.indices.reshape(r, t)
    Pi = np.prod(pc[indices], axis=1)
    log_plus = np.log(np.clip((1 + Pi) / 2, 1e-15, 1.0))
    log_minus = np.log(np.clip((1 - Pi) / 2, 1e-15, 1.0))
    log_prod = log_plus + log_minus
    contrib = log_plus - 0.5 * log_prod
    return indices, Pi, log_plus, log_prod, contrib


def main():
    art = torch.load(os.path.join(WORKDIR, "artifacts.pt"),
                     weights_only=False, map_location="cpu")
    decoding_key = art["decoding_key"]
    partition = art["partition"]
    prompt_ids_list = art["prompt_ids_list"]
    encoding_key = art["encoding_key"]
    n = decoding_key[0].shape[0]
    _, parity_check_matrix, _, _, noise_rate, _, _, _, t = decoding_key
    r = parity_check_matrix.shape[0]
    print(f"n={n} r={r} t={t} noise_rate={noise_rate}", flush=True)

    # Load only watermarked traces, sorted by prompt index
    watermarked = []
    for i in range(len(art["jobs"])):
        if not art["jobs"][i]["watermark"]:
            continue
        watermarked.append(torch.load(os.path.join(WORKDIR, f"result_{i:02d}.pt"),
                                      weights_only=False, map_location="cpu"))
    watermarked.sort(key=lambda r: r["job"]["prompt_idx"])

    # Find outlier: worst naive-fold stat
    rows = []
    for r_ in watermarked:
        bits = tokens_to_bits(r_["tokens"], partition)
        p_arr = np.asarray(r_["p_trace"], dtype=np.float64)
        naive_post = fold_naive(bits, n)
        ent_post = fold_entropy_weighted(bits, p_arr, n)
        rows.append({
            "prompt_idx": r_["job"]["prompt_idx"],
            "naive_stat": _test_statistic(naive_post, decoding_key),
            "entropy_stat": _test_statistic(ent_post, decoding_key),
            "trace": r_,
            "bits": bits,
            "p_arr": p_arr,
            "naive_post": naive_post,
            "ent_post": ent_post,
        })
    rows.sort(key=lambda x: x["naive_stat"])
    outlier = rows[0]
    pidx = outlier["prompt_idx"]
    print(f"\noutlier prompt_idx={pidx}  "
          f"naive_stat={outlier['naive_stat']:+.3f}  "
          f"entropy_stat={outlier['entropy_stat']:+.3f}", flush=True)

    bits = outlier["bits"]            # (T,) int 0/1
    p_arr = outlier["p_arr"]          # (T,) float in [0,1]
    naive_post = outlier["naive_post"]
    ent_post = outlier["ent_post"]
    tokens = outlier["trace"]["tokens"].numpy().astype(int)
    T = len(tokens)
    weights = binary_entropy(p_arr) / np.log(2)         # in [0, 1]

    # Per-slot stats
    idx_mod = np.arange(T) % n
    slot_obs_count = np.bincount(idx_mod, minlength=n)
    slot_weight_sum = np.bincount(idx_mod, weights=weights, minlength=n)
    slot_mean_weight = slot_weight_sum / np.maximum(slot_obs_count, 1)

    # Identify "low-entropy" slots by mean weight threshold
    low_thresh = 0.05  # any slot whose tokens are essentially deterministic
    low_slots = np.where(slot_mean_weight < low_thresh)[0]
    high_slots = np.where(slot_mean_weight >= low_thresh)[0]
    print(f"  low-entropy slots (mean weight < {low_thresh}): {len(low_slots)}/{n}",
          flush=True)
    print(f"  high-entropy slots: {len(high_slots)}/{n}", flush=True)

    # Per-parity-check contributions
    indices_n, Pi_n, lp_n, lpr_n, contrib_n = per_check_contributions(
        decoding_key, naive_post)
    indices_e, Pi_e, lp_e, lpr_e, contrib_e = per_check_contributions(
        decoding_key, ent_post)
    assert np.array_equal(indices_n, indices_e)

    # A check is "corrupted" if any of its t slots is low-entropy
    low_set = set(low_slots.tolist())
    is_corrupted = np.array([
        any(int(s) in low_set for s in indices_n[k]) for k in range(r)
    ])
    n_corrupted = int(is_corrupted.sum())
    print(f"  parity checks touching a low-entropy slot: "
          f"{n_corrupted}/{r}", flush=True)

    # Statistic decomposition
    def stat_breakdown(contrib, mask):
        return float(contrib[mask].sum()), float(contrib[~mask].sum())

    n_corr_sum_n, n_clean_sum_n = stat_breakdown(contrib_n, is_corrupted)
    n_corr_sum_e, n_clean_sum_e = stat_breakdown(contrib_e, is_corrupted)
    print(f"\nstatistic decomposition (sum over {r} parity checks)", flush=True)
    print(f"  naive   total = {contrib_n.sum():+.3f}  "
          f"corrupted={n_corr_sum_n:+.3f}  clean={n_clean_sum_n:+.3f}", flush=True)
    print(f"  entropy total = {contrib_e.sum():+.3f}  "
          f"corrupted={n_corr_sum_e:+.3f}  clean={n_clean_sum_e:+.3f}", flush=True)

    # ---------- write structured JSON for the demo ----------
    # Per-token: limit to first N for size; expose all of them as a list
    per_token = []
    for i in range(T):
        per_token.append({
            "i": i,
            "slot": int(idx_mod[i]),
            "token_id": int(tokens[i]),
            "token_str": decode_token(tokens[i]),
            "p1": float(p_arr[i]),
            "weight": float(weights[i]),
            "bit": int(bits[i]),
            "low_entropy": bool(weights[i] < low_thresh),
        })

    per_slot = []
    for s in range(n):
        per_slot.append({
            "slot": s,
            "n_obs": int(slot_obs_count[s]),
            "mean_weight": float(slot_mean_weight[s]),
            "naive_posterior": float(naive_post[s]),
            "entropy_posterior": float(ent_post[s]),
            "low_entropy": bool(slot_mean_weight[s] < low_thresh),
        })

    per_check = []
    for k in range(r):
        per_check.append({
            "k": k,
            "slot_indices": [int(x) for x in indices_n[k]],
            "Pi_naive": float(Pi_n[k]),
            "Pi_entropy": float(Pi_e[k]),
            "contrib_naive": float(contrib_n[k]),
            "contrib_entropy": float(contrib_e[k]),
            "corrupted": bool(is_corrupted[k]),
        })

    prompt_text = test_prompts[pidx]

    payload = {
        "meta": {
            "prompt_idx": pidx,
            "n": int(n),
            "r": int(r),
            "t": int(t),
            "noise_rate": float(noise_rate),
            "tokens_generated": int(T),
            "low_entropy_threshold": low_thresh,
            "n_low_entropy_slots": int(len(low_slots)),
            "n_corrupted_checks": int(n_corrupted),
            "naive_stat": float(outlier["naive_stat"]),
            "entropy_stat": float(outlier["entropy_stat"]),
            "naive_corrupted_contrib": n_corr_sum_n,
            "naive_clean_contrib": n_clean_sum_n,
            "entropy_corrupted_contrib": n_corr_sum_e,
            "entropy_clean_contrib": n_clean_sum_e,
        },
        "prompt": prompt_text,
        "per_token": per_token,
        "per_slot": per_slot,
        "per_check": per_check,
    }
    with open("outlier_demo.json", "w") as f:
        json.dump(payload, f, indent=1)
    print("wrote outlier_demo.json", flush=True)

    # ---------- write a short markdown summary ----------
    md = []
    md.append(f"# Outlier prompt {pidx} — naive vs entropy fold\n")
    md.append("Worst-case watermarked prompt under naive folding from the n=128 run.")
    md.append("Demonstrates how entropy weighting recovers signal that naive fold loses to deterministic tokens.\n")
    md.append("## Setup\n")
    md.append(f"- Prompt: `{prompt_text[:200]}{'...' if len(prompt_text)>200 else ''}`")
    md.append(f"- n = {n}, parity checks r = {r}, weight t = {t}, noise_rate = {noise_rate}")
    md.append(f"- Tokens generated: {T} ({T // n} cycles per slot)\n")
    md.append("## Statistic\n")
    md.append("| fold | total stat | sum over corrupted checks | sum over clean checks |")
    md.append("|---|---|---|---|")
    md.append(f"| naive | {outlier['naive_stat']:+.3f} | {n_corr_sum_n:+.3f} | {n_clean_sum_n:+.3f} |")
    md.append(f"| entropy | {outlier['entropy_stat']:+.3f} | {n_corr_sum_e:+.3f} | {n_clean_sum_e:+.3f} |\n")
    md.append("## Slot diagnostics\n")
    md.append(f"- {len(low_slots)} of {n} slots are **low-entropy** (mean per-token weight < {low_thresh}).")
    md.append(f"- {n_corrupted} of {r} parity checks touch at least one low-entropy slot ({100*n_corrupted/r:.1f}%).\n")
    md.append("## What entropy weighting does\n")
    md.append("Each parity check multiplies adjusted posteriors at t slots. When one slot is")
    md.append("low-entropy, naive fold gives that slot a near-deterministic ±1 posterior")
    md.append("(uncorrelated with the codeword), so the product Π becomes a random ±|signal|")
    md.append("contributing noise to the test statistic. Entropy weighting scales each")
    md.append("posterior by the local entropy, so low-entropy slots have |posterior| ≈ 0,")
    md.append("which collapses Π to ≈ 0 and zeroes that check's contribution — effectively")
    md.append("dropping the corrupted parity checks rather than flipping signs randomly.\n")
    md.append(f"On this prompt the naive statistic loses **{n_corr_sum_n - n_corr_sum_e:+.2f}**")
    md.append(f"to corrupted-check noise; entropy fold recovers a {outlier['entropy_stat'] - outlier['naive_stat']:+.2f}-point")
    md.append("net gain (negative under naive → positive under entropy).\n")
    md.append("## Demo data\n")
    md.append("`outlier_demo.json` has full per-token, per-slot, and per-parity-check arrays.")
    md.append("Visualization ideas:\n")
    md.append("- token strip: color tokens red where `low_entropy=true`")
    md.append("- slot grid (n=128): heatmap of `mean_weight` and the two posteriors")
    md.append("- parity-check list: filter on `corrupted` and show `contrib_naive` vs")
    md.append("  `contrib_entropy` — the sliders/toggles let the user remove corrupted")
    md.append("  checks one at a time and watch the statistic update")
    with open("outlier_summary.md", "w") as f:
        f.write("\n".join(md))
    print("wrote outlier_summary.md", flush=True)


if __name__ == "__main__":
    main()
