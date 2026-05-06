"""
Batched-worker version of extend_campaign.py. Loads model once per GPU and
processes all jobs assigned to that GPU serially, eliminating the ~20s
per-job model-load overhead in the original single-job worker design.

Same env vars as extend_campaign.py.
"""
import json
import os
import subprocess
import sys
import time
from glob import glob

import torch


WATERMARK = os.environ["EXT_WATERMARK"].lower()
MODEL_SIZE = os.environ.get("EXT_MODEL_SIZE", "0.6B")
GPU_IDS = [int(x) for x in os.environ["EXT_GPU_IDS"].split(",") if x.strip()]
PROMPTS_JSONL = os.environ.get("EXT_PROMPTS_JSONL", "prompts_10k.jsonl")
N_PROMPTS = int(os.environ.get("EXT_N_PROMPTS", "4983"))
BASE_WORKDIR = os.environ["EXT_BASE_WORKDIR"]
OUT_WORKDIR = os.environ["EXT_OUT_WORKDIR"]
MAX_NEW_TOKENS = int(os.environ.get("EXT_MAX_NEW_TOKENS", "800"))
KGW_GAMMA = float(os.environ.get("EXT_KGW_GAMMA", "0.25"))
KGW_DELTA = float(os.environ.get("EXT_KGW_DELTA", "2.0"))

assert WATERMARK in ("prc", "kgw")
ART_PATH = os.path.join(OUT_WORKDIR, "artifacts.pt")


def result_filename(job_idx: int) -> str:
    if WATERMARK == "prc":
        return f"result_{job_idx:02d}.pt"
    return f"result_{job_idx:04d}.pt"


def load_prompts():
    rows = []
    with open(PROMPTS_JSONL) as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= N_PROMPTS:
                break
    if len(rows) < N_PROMPTS:
        raise RuntimeError(f"{PROMPTS_JSONL} has only {len(rows)} rows, need {N_PROMPTS}")
    return [r["prompt_tokens"] for r in rows]


def build_or_load_artifacts():
    os.makedirs(OUT_WORKDIR, exist_ok=True)
    if os.path.isfile(ART_PATH):
        art = torch.load(ART_PATH, weights_only=False, map_location="cpu")
        if len(art["prompt_ids_list"]) >= N_PROMPTS and len(art["jobs"]) >= 2 * N_PROMPTS:
            print(f"[ext_batched] reusing existing {ART_PATH}", flush=True)
            return art
        print(f"[ext_batched] {ART_PATH} too small, rebuilding", flush=True)

    base_art = torch.load(os.path.join(BASE_WORKDIR, "artifacts.pt"),
                          weights_only=False, map_location="cpu")
    prompt_ids_list = load_prompts()
    jobs = [
        {"prompt_idx": p, "watermark": w, "max_new_tokens": MAX_NEW_TOKENS}
        for p in range(N_PROMPTS) for w in (True, False)
    ]
    if WATERMARK == "prc":
        art = {
            "encoding_key": base_art["encoding_key"],
            "decoding_key": base_art["decoding_key"],
            "partition": base_art["partition"],
            "prompt_ids_list": prompt_ids_list,
            "jobs": jobs,
            "n": base_art.get("n"),
            "seed": base_art.get("seed"),
        }
    else:
        art = {
            "kgw_key": base_art["kgw_key"],
            "gamma": base_art.get("gamma", KGW_GAMMA),
            "delta": base_art.get("delta", KGW_DELTA),
            "prompt_ids_list": prompt_ids_list,
            "jobs": jobs,
        }
    torch.save(art, ART_PATH)
    print(f"[ext_batched] saved extended artifacts: {len(jobs)} jobs, "
          f"{len(prompt_ids_list)} prompts -> {ART_PATH}", flush=True)
    return art


def link_existing_results():
    n_linked = 0
    for src in sorted(glob(os.path.join(BASE_WORKDIR, "result_*.pt"))):
        name = os.path.basename(src)
        dst = os.path.join(OUT_WORKDIR, name)
        if os.path.lexists(dst):
            continue
        os.symlink(os.path.abspath(src), dst)
        n_linked += 1
    if n_linked:
        print(f"[ext_batched] symlinked {n_linked} existing results", flush=True)


def find_pending_jobs(n_jobs: int) -> list:
    pending = []
    for j in range(n_jobs):
        if not os.path.isfile(os.path.join(OUT_WORKDIR, result_filename(j))):
            pending.append(j)
    return pending


def main():
    art = build_or_load_artifacts()
    n_jobs = len(art["jobs"])
    link_existing_results()
    pending = find_pending_jobs(n_jobs)
    print(f"[ext_batched] {WATERMARK.upper()} {MODEL_SIZE}: total={n_jobs} pending={len(pending)}",
          flush=True)
    if not pending:
        print("[ext_batched] nothing to do", flush=True)
        return

    n_gpus = len(GPU_IDS)
    # Round-robin partition (gives each GPU a stride; keeps the load balanced
    # even if early jobs are systematically faster/slower).
    chunks = {gpu: [] for gpu in GPU_IDS}
    for i, j in enumerate(pending):
        chunks[GPU_IDS[i % n_gpus]].append(j)

    # Write per-GPU job-list files.
    list_paths = {}
    for gpu, jobs in chunks.items():
        path = os.path.join(OUT_WORKDIR, f".joblist_gpu{gpu}.txt")
        with open(path, "w") as f:
            for j in jobs:
                f.write(f"{j}\n")
        list_paths[gpu] = path
        print(f"[ext_batched] gpu={gpu} -> {len(jobs)} jobs", flush=True)

    # Spawn one batched worker per GPU.
    procs = {}
    log_files = {}
    t0 = time.time()
    for gpu in GPU_IDS:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        log_path = os.path.join(OUT_WORKDIR, f"batched_worker_gpu{gpu}.log")
        log_files[gpu] = open(log_path, "w")
        cmd = ["python", "-u", "batched_worker.py", WATERMARK, ART_PATH,
               OUT_WORKDIR, f"@{list_paths[gpu]}"]
        procs[gpu] = subprocess.Popen(cmd, env=env, stdout=log_files[gpu],
                                      stderr=subprocess.STDOUT)
        print(f"[ext_batched] launched gpu={gpu} pid={procs[gpu].pid}", flush=True)

    # Wait for all workers to finish, occasionally reporting progress.
    failed = []
    last_report = 0
    while procs:
        time.sleep(30.0)
        finished = []
        for gpu, p in procs.items():
            rc = p.poll()
            if rc is None:
                continue
            log_files[gpu].close()
            if rc != 0:
                failed.append((gpu, rc))
                print(f"[ext_batched] FAIL gpu={gpu} rc={rc}", flush=True)
            else:
                print(f"[ext_batched] DONE gpu={gpu}", flush=True)
            finished.append(gpu)
        for gpu in finished:
            del procs[gpu]
        # Periodic progress
        if time.time() - last_report > 300:
            done_count = n_jobs - len(find_pending_jobs(n_jobs))
            elapsed = time.time() - t0
            print(f"[ext_batched] progress: {done_count}/{n_jobs} done, "
                  f"{elapsed:.0f}s elapsed", flush=True)
            last_report = time.time()

    if failed:
        for gpu, rc in failed:
            log_path = os.path.join(OUT_WORKDIR, f"batched_worker_gpu{gpu}.log")
            try:
                with open(log_path) as f:
                    tail = "".join(f.readlines()[-30:])
                print(f"--- gpu {gpu} log tail ---\n{tail}", flush=True)
            except FileNotFoundError:
                pass
        raise RuntimeError(f"{len(failed)} worker(s) failed: {failed}")

    elapsed = time.time() - t0
    final_pending = find_pending_jobs(n_jobs)
    print(f"[ext_batched] all done in {elapsed:.0f}s. final pending = {len(final_pending)}",
          flush=True)


if __name__ == "__main__":
    main()
