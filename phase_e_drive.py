"""
Phase E driver: launch one batched worker per GPU in PE_GPU_IDS to process
all pending jobs in a phase_e workdir. Modeled on extend_campaign_batched.py
but skips the artifact-rebuild path (artifacts must already be built by
phase_e_build.py).

Env vars:
  PE_WATERMARK     prc | kgw
  PE_WORKDIR       workdir holding artifacts.pt + result_*.pt
  PE_GPU_IDS       comma-separated GPU ids (default: all visible)
"""
import os
import subprocess
import time
from glob import glob

import torch


WATERMARK = os.environ["PE_WATERMARK"].lower()
WORKDIR = os.environ["PE_WORKDIR"]
GPU_IDS_ENV = os.environ.get("PE_GPU_IDS", "")
if GPU_IDS_ENV:
    GPU_IDS = [int(x) for x in GPU_IDS_ENV.split(",") if x.strip()]
else:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"]
    ).decode().strip().splitlines()
    GPU_IDS = [int(x) for x in out]

assert WATERMARK in ("prc", "kgw")
ART_PATH = os.path.join(WORKDIR, "artifacts.pt")


def find_pending_jobs(n_jobs):
    pending = []
    for j in range(n_jobs):
        p = os.path.join(WORKDIR, f"result_{j:04d}.pt")
        if not os.path.isfile(p):
            pending.append(j)
    return pending


def main():
    art = torch.load(ART_PATH, weights_only=False, map_location="cpu")
    n_jobs = len(art["jobs"])
    pending = find_pending_jobs(n_jobs)
    print(f"[phase_e_drive] {WATERMARK.upper()} {WORKDIR}: total={n_jobs} pending={len(pending)}",
          flush=True)
    if not pending:
        print("[phase_e_drive] nothing to do", flush=True)
        return

    n_gpus = len(GPU_IDS)
    chunks = {gpu: [] for gpu in GPU_IDS}
    for i, j in enumerate(pending):
        chunks[GPU_IDS[i % n_gpus]].append(j)

    list_paths = {}
    for gpu, jobs in chunks.items():
        path = os.path.join(WORKDIR, f".joblist_gpu{gpu}.txt")
        with open(path, "w") as f:
            for j in jobs:
                f.write(f"{j}\n")
        list_paths[gpu] = path
        print(f"[phase_e_drive] gpu={gpu} -> {len(jobs)} jobs", flush=True)

    procs = {}
    log_files = {}
    t0 = time.time()
    for gpu in GPU_IDS:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        log_path = os.path.join(WORKDIR, f"phase_e_worker_gpu{gpu}.log")
        log_files[gpu] = open(log_path, "w")
        cmd = ["python", "-u", "phase_e_worker.py", WATERMARK, ART_PATH,
               WORKDIR, f"@{list_paths[gpu]}"]
        procs[gpu] = subprocess.Popen(cmd, env=env, stdout=log_files[gpu],
                                       stderr=subprocess.STDOUT)
        print(f"[phase_e_drive] launched gpu={gpu} pid={procs[gpu].pid}", flush=True)

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
                print(f"[phase_e_drive] FAIL gpu={gpu} rc={rc}", flush=True)
            else:
                print(f"[phase_e_drive] DONE gpu={gpu}", flush=True)
            finished.append(gpu)
        for gpu in finished:
            del procs[gpu]
        if time.time() - last_report > 300:
            done_count = n_jobs - len(find_pending_jobs(n_jobs))
            elapsed = time.time() - t0
            print(f"[phase_e_drive] progress: {done_count}/{n_jobs} done, "
                  f"{elapsed:.0f}s elapsed", flush=True)
            last_report = time.time()

    if failed:
        for gpu, rc in failed:
            log_path = os.path.join(WORKDIR, f"phase_e_worker_gpu{gpu}.log")
            try:
                with open(log_path) as f:
                    tail = "".join(f.readlines()[-30:])
                print(f"--- gpu {gpu} log tail ---\n{tail}", flush=True)
            except FileNotFoundError:
                pass
        raise RuntimeError(f"{len(failed)} worker(s) failed: {failed}")

    elapsed = time.time() - t0
    final_pending = find_pending_jobs(n_jobs)
    print(f"[phase_e_drive] all done in {elapsed:.0f}s. final pending = {len(final_pending)}",
          flush=True)


if __name__ == "__main__":
    main()
