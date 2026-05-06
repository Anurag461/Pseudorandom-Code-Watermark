"""
Driver for the extended (10k-example) PRC and KGW generation campaigns.

Reuses existing watermark keys/partition from a base workdir so cached
result_NN.pt files remain valid. Builds an extended artifacts.pt with
prompt_ids_list of length N_PROMPTS (default 4983) and jobs of length
2 * N_PROMPTS, then re-launches the per-GPU worker pool on whatever
job_indices don't yet have a result file.

Re-runnable: scans the workdir for missing results each time, so two
concurrent invocations on disjoint GPU pools cooperatively drain the queue.

Env vars:
    EXT_WATERMARK       prc | kgw
    EXT_MODEL_SIZE      0.6B (default; 8B not exercised in this run)
    EXT_GPU_IDS         comma-separated, e.g. 1,2,3,4
    EXT_PROMPTS_JSONL   default prompts_10k.jsonl
    EXT_N_PROMPTS       default 4983 (== rows in prompts_10k.jsonl)
    EXT_BASE_WORKDIR    source of existing keys + result_*.pt files
    EXT_OUT_WORKDIR     destination workdir (created if missing)
    EXT_MAX_NEW_TOKENS  default 800 (matches the existing pattern)
    EXT_KGW_GAMMA       default 0.25
    EXT_KGW_DELTA       default 2.0
"""
import json
import os
import subprocess
import sys
import time
from glob import glob

import torch


WATERMARK = os.environ["EXT_WATERMARK"].lower()  # prc | kgw
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

# PRC uses {:02d} (min-2-digit), KGW uses {:04d} (min-4-digit).
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
    """
    Build extended artifacts.pt from the BASE_WORKDIR keys + an extended
    prompt_ids_list of length N_PROMPTS. If OUT_WORKDIR/artifacts.pt already
    exists, just return it.
    """
    os.makedirs(OUT_WORKDIR, exist_ok=True)
    if os.path.isfile(ART_PATH):
        art = torch.load(ART_PATH, weights_only=False, map_location="cpu")
        if len(art["prompt_ids_list"]) >= N_PROMPTS and len(art["jobs"]) >= 2 * N_PROMPTS:
            print(f"[ext] reusing existing {ART_PATH}", flush=True)
            return art
        print(f"[ext] {ART_PATH} too small, rebuilding", flush=True)

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
    print(f"[ext] saved extended artifacts: {len(jobs)} jobs, "
          f"{len(prompt_ids_list)} prompts -> {ART_PATH}", flush=True)
    return art


def link_existing_results():
    """Symlink BASE_WORKDIR/result_*.pt into OUT_WORKDIR (no copy, no move)."""
    n_linked = 0
    for src in sorted(glob(os.path.join(BASE_WORKDIR, "result_*.pt"))):
        name = os.path.basename(src)
        dst = os.path.join(OUT_WORKDIR, name)
        if os.path.lexists(dst):
            continue
        os.symlink(os.path.abspath(src), dst)
        n_linked += 1
    if n_linked:
        print(f"[ext] symlinked {n_linked} existing results from {BASE_WORKDIR} -> {OUT_WORKDIR}", flush=True)


def find_pending_jobs(n_jobs: int) -> list:
    pending = []
    for j in range(n_jobs):
        if not os.path.isfile(os.path.join(OUT_WORKDIR, result_filename(j))):
            pending.append(j)
    return pending


def claim_pending_atomic(n_jobs: int) -> list:
    """
    Acquire pending jobs without colliding with concurrent invocations.
    Uses an O_EXCL .claim file per job. The actual work is done by the
    worker subprocess; the .claim file is removed by this driver after the
    worker exits (whether success or failure) so retries can pick it up.
    """
    claimed = []
    for j in find_pending_jobs(n_jobs):
        claim_path = os.path.join(OUT_WORKDIR, f".claim_{j:06d}")
        try:
            fd = os.open(claim_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            claimed.append((j, claim_path))
        except FileExistsError:
            continue
    return claimed


def launch_workers(claims: list):
    n = len(claims)
    n_gpus = len(GPU_IDS)
    print(f"[ext] launching {n} jobs across {n_gpus} GPUs ({GPU_IDS})", flush=True)
    in_flight = {}        # gpu_id -> (Popen, job_idx, claim_path, log_file)
    pending = list(claims)
    completed = 0
    failed = []
    t_start = time.time()

    worker_script = "worker_generate.py" if WATERMARK == "prc" else "worker_generate_kgw.py"

    def launch(gpu, job_idx, claim_path):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        log_path = os.path.join(OUT_WORKDIR, f"worker_{job_idx:06d}.log")
        log_file = open(log_path, "w")
        cmd = ["python", "-u", worker_script, str(job_idx), ART_PATH, OUT_WORKDIR]
        p = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        return (p, job_idx, claim_path, log_file)

    for gpu in GPU_IDS:
        if pending:
            j, c = pending.pop(0)
            in_flight[gpu] = launch(gpu, j, c)

    while in_flight:
        time.sleep(2.0)
        finished = []
        for gpu, (p, j, c, lf) in in_flight.items():
            rc = p.poll()
            if rc is None:
                continue
            lf.close()
            try:
                os.remove(c)
            except FileNotFoundError:
                pass
            completed += 1
            elapsed = time.time() - t_start
            tag = "ok" if rc == 0 else f"FAIL rc={rc}"
            if rc != 0:
                failed.append(j)
            print(f"[ext] {tag} job={j} gpu={gpu} ({completed}/{n} done, "
                  f"{elapsed:.0f}s elapsed)", flush=True)
            finished.append(gpu)
        for gpu in finished:
            del in_flight[gpu]
            if pending:
                jn, cn = pending.pop(0)
                in_flight[gpu] = launch(gpu, jn, cn)

    if failed:
        for j in failed[:5]:
            log_path = os.path.join(OUT_WORKDIR, f"worker_{j:06d}.log")
            try:
                with open(log_path) as f:
                    print(f"--- worker {j} log tail ---")
                    print("".join(f.readlines()[-30:]))
            except FileNotFoundError:
                pass
        raise RuntimeError(f"{len(failed)} worker(s) failed")


def main():
    art = build_or_load_artifacts()
    n_jobs = len(art["jobs"])
    link_existing_results()
    pending = find_pending_jobs(n_jobs)
    print(f"[ext] {WATERMARK.upper()} {MODEL_SIZE}: total jobs={n_jobs}, "
          f"pending={len(pending)}", flush=True)
    if not pending:
        print("[ext] nothing to do", flush=True)
        return
    claims = claim_pending_atomic(n_jobs)
    print(f"[ext] claimed {len(claims)} jobs (skipping ones already claimed by "
          f"a concurrent driver)", flush=True)
    if claims:
        launch_workers(claims)
    print(f"[ext] driver exiting; remaining unfinished = "
          f"{len(find_pending_jobs(n_jobs))}", flush=True)


if __name__ == "__main__":
    main()
