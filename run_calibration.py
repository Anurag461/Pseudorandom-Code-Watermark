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
    KeyGen,
    test_prompts,
)

WORKDIR = "calib_workdir"
ART_PATH = os.path.join(WORKDIR, "artifacts.pt")
THRESHOLD_PATH = "qwen_threshold.json"
N_CODEWORD = 4096
MAX_NEW_TOKENS = 4 * N_CODEWORD
T_PARITY = 6        # t = (1/2) log2(n)
G_PARAM = 144       # g = log2(n)^2  -> Omega(log^2 n) per Thm 2
NOISE_RATE = 0.05   # constant eta, paper-style
SEED = 12345


def detect_visible_gpus():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"]
    ).decode().strip().splitlines()
    return [int(x) for x in out]


def build_artifacts(gpu_ids):
    os.makedirs(WORKDIR, exist_ok=True)

    # Deterministic keys + partition.
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    encoding_key, decoding_key = KeyGen(
        n=N_CODEWORD,
        message_length=0,
        false_positive_rate=0.5,
        t=T_PARITY,
        g=G_PARAM,
        noise_rate=NOISE_RATE,
    )
    _, parity_check_matrix, _, _, noise_rate, _, g, _, t = decoding_key
    r = parity_check_matrix.shape[0]
    print(f"[parent] PRC params: n={N_CODEWORD} t={t} g={g} r={r} "
          f"noise_rate={noise_rate:.4f}", flush=True)

    # Build the partition tensor on CPU; we don't need a model for this.
    vocab_size = 151_936  # Qwen3 vocab
    perm = torch.randperm(vocab_size, generator=torch.Generator().manual_seed(SEED))
    v0 = torch.zeros(vocab_size, dtype=torch.bfloat16)
    v0[perm[: vocab_size // 2]] = 1.0
    v1 = 1 - v0
    partition = torch.stack([v0, v1], dim=0)  # (2, vocab)

    prompt_ids_list = [prompt_to_ids(p) for p in test_prompts]

    # 8 jobs: 4 watermarked + 4 unwatermarked, one per visible GPU.
    jobs = []
    for prompt_idx in range(len(test_prompts)):
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


def main():
    gpu_ids = detect_visible_gpus()
    print(f"[parent] visible GPUs: {gpu_ids}", flush=True)

    art = build_artifacts(gpu_ids)
    decoding_key = art["decoding_key"]
    partition = art["partition"]

    # ---------------- Phase 1: parallel generation ----------------
    print(f"\n=== Phase 1: {len(art['jobs'])} generations across "
          f"{len(gpu_ids)} GPUs ===", flush=True)
    t0 = time.time()
    launch_workers(art["jobs"], gpu_ids)
    print(f"[parent] all generations done in {time.time()-t0:.1f}s", flush=True)

    results = gather_results(len(art["jobs"]))

    # Split watermarked vs unwatermarked.
    watermarked = [r for r in results if r["job"]["watermark"]]
    unwatermarked = [r for r in results if not r["job"]["watermark"]]
    print(f"[parent] watermarked={len(watermarked)} "
          f"unwatermarked={len(unwatermarked)}", flush=True)

    # ---------------- Phase 2: fit threshold ----------------
    print("\n=== Phase 2: fit threshold ===", flush=True)
    threshold_state = fit_calibration(
        decoding_key,
        [r["p_trace"] for r in watermarked],
        fpr=1e-9,
        num_simulated_nulls=2000,
    )
    save_threshold_state(threshold_state, THRESHOLD_PATH)
    print(f"  threshold = {threshold_state['threshold']:.4f}  "
          f"null_mean = {threshold_state['null_mean']:.4f}  "
          f"null_std  = {threshold_state['null_std']:.4f}  "
          f"traces_used = {threshold_state['num_traces_used']}", flush=True)
    print(f"  saved -> {THRESHOLD_PATH}", flush=True)

    # ---------------- Phase 3a: TPR ----------------
    print("\n=== Phase 3a: detection on watermarked runs (TPR) ===", flush=True)
    threshold_state = load_threshold_state(THRESHOLD_PATH)
    tp = 0
    for r in sorted(watermarked, key=lambda r: r["job"]["prompt_idx"]):
        decision, info = detect_with_threshold(
            decoding_key, r["tokens"], r["p_trace"], partition,
            threshold_state, return_info=True,
        )
        tp += int(decision)
        print(f"  prompt {r['job']['prompt_idx']}: detected={decision}  "
              f"stat={info['statistic']:.2f}  "
              f"sigmas={info['sigmas_above_null']:.1f}  "
              f"(tokens={r['tokens'].numel()})", flush=True)

    # ---------------- Phase 3b: FPR ----------------
    print("\n=== Phase 3b: detection on UNwatermarked runs (FPR) ===", flush=True)
    fp = 0
    for r in sorted(unwatermarked, key=lambda r: r["job"]["prompt_idx"]):
        decision, info = detect_with_threshold(
            decoding_key, r["tokens"], r["p_trace"], partition,
            threshold_state, return_info=True,
        )
        fp += int(decision)
        print(f"  prompt {r['job']['prompt_idx']}: detected={decision}  "
              f"stat={info['statistic']:.2f}  "
              f"sigmas={info['sigmas_above_null']:.1f}  "
              f"(tokens={r['tokens'].numel()})", flush=True)

    # ---------------- Summary ----------------
    n = len(test_prompts)
    print("\n=== Summary ===", flush=True)
    print(f"  TPR: {tp}/{n}  ({tp/n:.1%})", flush=True)
    print(f"  FPR: {fp}/{n}  ({fp/n:.1%})", flush=True)
    print(f"  threshold (FPR target 1e-9): "
          f"{threshold_state['threshold']:.4f}", flush=True)


if __name__ == "__main__":
    main()
