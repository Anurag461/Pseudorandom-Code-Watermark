"""
Three-phase PRC watermark calibration with multi-GPU data parallelism.

Layout:
  - Parent builds shared artifacts (encoding/decoding keys, partition map,
    tokenized prompts, job list) ONCE and saves them.
  - Parent spawns one subprocess per GPU; each worker handles one
    (prompt, watermark_flag) job and saves its result.
  - Parent then loads results, fits the threshold from watermarked p-traces,
    and detects on all 8 generations to compute TPR / FPR.

Outputs:
  workdir/artifacts.pt
  workdir/result_NN.pt           (one per job)
  qwen_threshold.json
  workdir/run.log                (concatenated worker stdout)
"""
import json
import os

# Parent only orchestrates and does light numerics; keep it off the GPU so
# none of its imports steal memory from the workers.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import subprocess
import time
import torch
import numpy as np

from watermark_expt import (
    tokenizer,
    tok,
    prompt_to_ids,
    fit_calibration,
    save_threshold_state,
    load_threshold_state,
    detect_with_threshold,
    detect_syndrome,
    KeyGen,
    test_prompts,
)

# -----------------------------------------------------------------------------
# Detection method.  Drives Phase 2 (calibration) and Phase 3 (detect).
#   "entropy_fold"     -- entropy-weighted cyclic fold + null-calibrated
#                         threshold.  Needs T >= n; T = 4n typical.
#   "naive_fold"       -- equal-weight cyclic fold + null-calibrated threshold.
#                         Same calibration / detect pipeline as entropy_fold,
#                         only the per-step weighting differs.
#   "syndrome_all"     -- PRC paper Theorem 1 syndrome detector, OR over
#                         length-n blocks.  No fold, no weighting, no
#                         calibration.
#   "syndrome_entropy" -- syndrome detector restricted to parity checks whose
#                         t token positions all have H_2(p1) >= 0.1 bits
#                         (evaluated per block).
# -----------------------------------------------------------------------------
DETECT_METHOD = os.environ.get("PRC_DETECT_METHOD", "entropy_fold")
SYNDROME_ENTROPY_THRESHOLD = 0.1  # bits, only used by syndrome_entropy
FPR_TARGET = float(os.environ.get("PRC_FPR_TARGET", "2e-5"))

NOISE_RATE = float(os.environ.get("PRC_NOISE_RATE", "0.05"))   # eta
N_CODEWORD = int(os.environ.get("PRC_N", "400"))
T_PARITY = int(os.environ.get("PRC_T", "3"))
MAX_NEW_TOKENS = int(os.environ.get("PRC_MAX_NEW_TOKENS", str(2 * N_CODEWORD)))
G_PARAM = None      # None -> KeyGen default (= secpar)
SEED = int(os.environ.get("PRC_SEED", "12345"))

# Workdir / threshold path default to encode the active config so parallel
# runs at different etas can coexist without collisions.
_eta_tag = f"eta{int(round(NOISE_RATE * 100)):02d}"
_default_workdir = f"calib_workdir_n{N_CODEWORD}_t{T_PARITY}_{_eta_tag}"
_default_threshold = f"qwen_threshold_n{N_CODEWORD}_t{T_PARITY}_{_eta_tag}.json"
WORKDIR = os.environ.get("PRC_WORKDIR", _default_workdir)
ART_PATH = os.path.join(WORKDIR, "artifacts.pt")
THRESHOLD_PATH = os.environ.get("PRC_THRESHOLD_PATH", _default_threshold)

# If set, load prompts from this JSONL file (one entry per line, with key
# `prompt_tokens` as a pre-tokenized list of ints). Bypasses the chat-template
# wrapping in prompt_to_ids -- we want raw text continuation, matching the
# Kirchenbauer/Kuditipudi C4 RealNewsLike protocol.
PROMPTS_JSONL = "prompts.jsonl"
NUM_PROMPTS = 500    # how many entries to take from the head of PROMPTS_JSONL

# Skip Phase 1 if WORKDIR already has matching artifacts + all result files.
# Lets us re-run Phase 3 with a different DETECT_METHOD on the same generations.
REUSE_GENERATIONS = True


def detect_visible_gpus():
    # Allow explicit override (e.g. to skip a GPU shared with another tenant
    # or to run multiple parents on disjoint GPU pools).
    override = os.environ.get("PRC_GPU_IDS", "").strip()
    if override:
        return [int(x) for x in override.split(",") if x.strip()]
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"]
    ).decode().strip().splitlines()
    return [int(x) for x in out]


def load_prompt_ids_list():
    """Return list of token-id lists, one per prompt, in the active config.

    If PROMPTS_JSONL is set, read pre-tokenized `prompt_tokens` from JSONL
    (raw text continuation, no chat template). Otherwise fall back to
    constants.test_prompts wrapped via prompt_to_ids.
    """
    if PROMPTS_JSONL:
        rows = []
        with open(PROMPTS_JSONL) as f:
            for line in f:
                rows.append(json.loads(line))
                if len(rows) >= NUM_PROMPTS:
                    break
        if len(rows) < NUM_PROMPTS:
            raise RuntimeError(
                f"{PROMPTS_JSONL} has only {len(rows)} rows, need {NUM_PROMPTS}"
            )
        prompt_ids_list = [r["prompt_tokens"] for r in rows]
        print(f"[parent] loaded {len(prompt_ids_list)} prompts from "
              f"{PROMPTS_JSONL} (raw token continuation)", flush=True)
        return prompt_ids_list
    prompt_ids_list = [prompt_to_ids(p) for p in test_prompts]
    print(f"[parent] using {len(prompt_ids_list)} chat-template prompts "
          f"from constants.test_prompts", flush=True)
    return prompt_ids_list


def build_artifacts(gpu_ids):
    os.makedirs(WORKDIR, exist_ok=True)

    # Deterministic keys + partition.
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    keygen_kwargs = {
        "n": N_CODEWORD,
        "message_length": 0,
        "false_positive_rate": 0.5,
        "t": T_PARITY,
    }
    if G_PARAM is not None:
        keygen_kwargs["g"] = G_PARAM
    if NOISE_RATE is not None:
        keygen_kwargs["noise_rate"] = NOISE_RATE
    encoding_key, decoding_key = KeyGen(**keygen_kwargs)
    _, parity_check_matrix, _, _, noise_rate, _, g, _, t = decoding_key
    r = parity_check_matrix.shape[0]
    print(f"[parent] PRC params: n={N_CODEWORD} t={t} g={g} r={r} "
          f"noise_rate={noise_rate:.4f}", flush=True)
    print(f"[parent] detect method: {DETECT_METHOD}", flush=True)

    # Build the partition tensor on CPU; we don't need a model for this.
    vocab_size = 151_936  # Qwen3 vocab
    perm = torch.randperm(vocab_size, generator=torch.Generator().manual_seed(SEED))
    v0 = torch.zeros(vocab_size, dtype=torch.bfloat16)
    v0[perm[: vocab_size // 2]] = 1.0
    v1 = 1 - v0
    partition = torch.stack([v0, v1], dim=0)  # (2, vocab)

    prompt_ids_list = load_prompt_ids_list()

    # 2 jobs per prompt: one watermarked, one not.
    jobs = []
    for prompt_idx in range(len(prompt_ids_list)):
        for watermark in (True, False):
            jobs.append({
                "prompt_idx": prompt_idx,
                "watermark": watermark,
                "max_new_tokens": MAX_NEW_TOKENS,
            })

    artifacts = {
        "encoding_key": encoding_key,
        "decoding_key": decoding_key,
        "partition": partition,
        "prompt_ids_list": prompt_ids_list,
        "jobs": jobs,
        "n": N_CODEWORD,
        "seed": SEED,
    }
    torch.save(artifacts, ART_PATH)
    print(f"[parent] saved {len(jobs)} jobs to {ART_PATH}", flush=True)
    return artifacts


def launch_workers(jobs, gpu_ids):
    """Run jobs with a pool of len(gpu_ids) concurrent workers, one per GPU.

    Each GPU runs a sequence of jobs back-to-back. New job is launched on a
    GPU only after that GPU's previous job finished, so we never have more
    than len(gpu_ids) model copies in flight.
    """
    n_jobs = len(jobs)
    n_gpus = len(gpu_ids)
    pending = list(range(n_jobs))           # job indices left to schedule
    in_flight = {}                          # gpu_id -> (Popen, job_idx, log_file)
    failed = []
    completed = 0
    t_start = time.time()

    def launch(gpu, job_idx):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        log_path = os.path.join(WORKDIR, f"worker_{job_idx:02d}.log")
        log_file = open(log_path, "w")
        cmd = ["python", "-u", "worker_generate.py", str(job_idx), ART_PATH, WORKDIR]
        p = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        print(
            f"[parent] launch job={job_idx} gpu={gpu} "
            f"prompt={jobs[job_idx]['prompt_idx']} "
            f"watermark={jobs[job_idx]['watermark']} pid={p.pid}",
            flush=True,
        )
        return (p, job_idx, log_file)

    # Seed each GPU with its first job.
    for gpu in gpu_ids:
        if pending:
            in_flight[gpu] = launch(gpu, pending.pop(0))

    # Drain.
    while in_flight:
        # Poll all in-flight workers; sleep a beat to avoid busy-loop.
        time.sleep(2.0)
        finished_gpus = []
        for gpu, (p, job_idx, log_file) in in_flight.items():
            rc = p.poll()
            if rc is None:
                continue
            log_file.close()
            completed += 1
            elapsed = time.time() - t_start
            if rc != 0:
                failed.append((job_idx, rc))
                print(
                    f"[parent] FAILED job={job_idx} gpu={gpu} rc={rc} "
                    f"({completed}/{n_jobs} done, {elapsed:.0f}s elapsed)",
                    flush=True,
                )
            else:
                print(
                    f"[parent] done   job={job_idx} gpu={gpu} "
                    f"({completed}/{n_jobs} done, {elapsed:.0f}s elapsed)",
                    flush=True,
                )
            finished_gpus.append(gpu)

        # Reschedule freed GPUs onto the next pending job.
        for gpu in finished_gpus:
            del in_flight[gpu]
            if pending:
                in_flight[gpu] = launch(gpu, pending.pop(0))

    if failed:
        for job_idx, _ in failed:
            with open(os.path.join(WORKDIR, f"worker_{job_idx:02d}.log")) as f:
                print(f"\n--- worker {job_idx} log tail ---", flush=True)
                print("".join(f.readlines()[-30:]), flush=True)
        raise RuntimeError(f"{len(failed)} worker(s) failed")


def gather_results(n_jobs):
    results = []
    for job_idx in range(n_jobs):
        path = os.path.join(WORKDIR, f"result_{job_idx:02d}.pt")
        results.append(torch.load(path, weights_only=False, map_location="cpu"))
    return results


def have_complete_workdir(n_jobs):
    if not os.path.isfile(ART_PATH):
        return False
    for j in range(n_jobs):
        if not os.path.isfile(os.path.join(WORKDIR, f"result_{j:02d}.pt")):
            return False
    return True


def fmt_detect_info(info):
    if "statistic" in info:
        return (f"stat={info['statistic']:.2f}  "
                f"sigmas={info['sigmas_above_null']:.1f}")
    if info.get("method") == "syndrome":
        passed = info["blocks_passed"]
        nb = info["n_blocks"]
        wts = "/".join(f"{b['weight']}" for b in info["blocks"])
        thr = "/".join(f"{b['threshold']:.1f}" for b in info["blocks"])
        reff = "/".join(f"{b['r_eff']}" for b in info["blocks"])
        return (f"blocks_passed={passed}/{nb}  "
                f"weights=[{wts}]  thresholds=[{thr}]  r_eff=[{reff}]")
    return str(info)


def run_detect(method, decoding_key, r, partition, threshold_state):
    if method in ("entropy_fold", "naive_fold"):
        return detect_with_threshold(
            decoding_key, r["tokens"], r["p_trace"], partition,
            threshold_state, return_info=True,
        )
    if method == "syndrome_all":
        return detect_syndrome(
            decoding_key, r["tokens"], r["p_trace"], partition,
            entropy_threshold=None, return_info=True,
        )
    if method == "syndrome_entropy":
        return detect_syndrome(
            decoding_key, r["tokens"], r["p_trace"], partition,
            entropy_threshold=SYNDROME_ENTROPY_THRESHOLD, return_info=True,
        )
    raise ValueError(f"unknown DETECT_METHOD={method}")


def main():
    gpu_ids = detect_visible_gpus()
    print(f"[parent] visible GPUs: {gpu_ids}", flush=True)

    # If existing artifacts present, prefer loading them so the keys match
    # any cached result_NN.pt files from a prior run.
    expected_jobs = 2 * NUM_PROMPTS if PROMPTS_JSONL else 2 * len(test_prompts)
    if REUSE_GENERATIONS and os.path.isfile(ART_PATH):
        art = torch.load(ART_PATH, weights_only=False, map_location="cpu")
        n_jobs = len(art["jobs"])
        print(f"[parent] loaded existing artifacts.pt with n_jobs={n_jobs} "
              f"n={art['n']}", flush=True)
        if art["n"] != N_CODEWORD or n_jobs != expected_jobs:
            raise RuntimeError(
                f"existing artifacts.pt mismatches config "
                f"(n={art['n']} vs {N_CODEWORD}, n_jobs={n_jobs} vs "
                f"{expected_jobs}); rm {WORKDIR} or set REUSE_GENERATIONS=False"
            )
    else:
        art = build_artifacts(gpu_ids)
        n_jobs = len(art["jobs"])

    decoding_key = art["decoding_key"]
    partition = art["partition"]

    # ---------------- Phase 1: parallel generation ----------------
    if REUSE_GENERATIONS and have_complete_workdir(n_jobs):
        print(f"\n=== Phase 1 SKIPPED: reusing {n_jobs} existing results "
              f"in {WORKDIR} ===", flush=True)
    else:
        print(f"\n=== Phase 1: {n_jobs} generations across "
              f"{len(gpu_ids)} GPUs ===", flush=True)
        t0 = time.time()
        launch_workers(art["jobs"], gpu_ids)
        print(f"[parent] all generations done in {time.time()-t0:.1f}s",
              flush=True)

    results = gather_results(n_jobs)

    watermarked = [r for r in results if r["job"]["watermark"]]
    unwatermarked = [r for r in results if not r["job"]["watermark"]]
    print(f"[parent] watermarked={len(watermarked)} "
          f"unwatermarked={len(unwatermarked)}", flush=True)

    # ---------------- Phase 2: fit threshold ----------------
    threshold_state = None
    if DETECT_METHOD in ("entropy_fold", "naive_fold"):
        fold_name = "entropy" if DETECT_METHOD == "entropy_fold" else "naive"
        print(f"\n=== Phase 2: fit threshold (fold={fold_name}, "
              f"FPR target {FPR_TARGET:.1e}) ===", flush=True)
        threshold_state = fit_calibration(
            decoding_key,
            [r["p_trace"] for r in watermarked],
            fpr=FPR_TARGET,
            num_simulated_nulls=2000,
            fold=fold_name,
        )
        save_threshold_state(threshold_state, THRESHOLD_PATH)
        print(f"  threshold = {threshold_state['threshold']:.4f}  "
              f"null_mean = {threshold_state['null_mean']:.4f}  "
              f"null_std  = {threshold_state['null_std']:.4f}  "
              f"traces_used = {threshold_state['num_traces_used']}", flush=True)
        print(f"  saved -> {THRESHOLD_PATH}", flush=True)
    else:
        print(f"\n=== Phase 2 SKIPPED: {DETECT_METHOD} uses analytical "
              f"threshold ===", flush=True)

    # ---------------- Phase 3a: TPR ----------------
    print("\n=== Phase 3a: detection on watermarked runs (TPR) ===", flush=True)
    tp = 0
    for r in sorted(watermarked, key=lambda r: r["job"]["prompt_idx"]):
        decision, info = run_detect(DETECT_METHOD, decoding_key, r, partition,
                                    threshold_state)
        tp += int(decision)
        print(f"  prompt {r['job']['prompt_idx']}: detected={decision}  "
              f"{fmt_detect_info(info)}  "
              f"(tokens={r['tokens'].numel()})", flush=True)

    # ---------------- Phase 3b: FPR ----------------
    print("\n=== Phase 3b: detection on UNwatermarked runs (FPR) ===", flush=True)
    fp = 0
    for r in sorted(unwatermarked, key=lambda r: r["job"]["prompt_idx"]):
        decision, info = run_detect(DETECT_METHOD, decoding_key, r, partition,
                                    threshold_state)
        fp += int(decision)
        print(f"  prompt {r['job']['prompt_idx']}: detected={decision}  "
              f"{fmt_detect_info(info)}  "
              f"(tokens={r['tokens'].numel()})", flush=True)

    # ---------------- Summary ----------------
    n = len(watermarked)
    print("\n=== Summary ===", flush=True)
    print(f"  method: {DETECT_METHOD}", flush=True)
    print(f"  TPR: {tp}/{n}  ({tp/n:.1%})", flush=True)
    print(f"  FPR: {fp}/{n}  ({fp/n:.1%})", flush=True)
    if threshold_state is not None:
        print(f"  threshold (FPR target {threshold_state['fpr']:.1e}): "
              f"{threshold_state['threshold']:.4f}", flush=True)


if __name__ == "__main__":
    main()
