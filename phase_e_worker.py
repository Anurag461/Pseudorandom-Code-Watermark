"""
Phase E batched worker. Forks `batched_worker.py` but uses uniform 4-digit
result naming (since K=2048) and saves the realized p_trace from PRC
generation so we can later inspect the p1≈0.5 assumption.

Args:
    sys.argv[1]  watermark scheme: "prc" | "kgw"
    sys.argv[2]  path to artifacts.pt
    sys.argv[3]  output workdir
    sys.argv[4]  job-list source: comma-separated ints OR @<path>
"""
import os
import sys
import time
import torch


def parse_job_list(arg):
    if arg.startswith("@"):
        with open(arg[1:]) as f:
            return [int(line.strip()) for line in f if line.strip()]
    return [int(x) for x in arg.split(",") if x.strip()]


def main():
    watermark = sys.argv[1].lower()
    art_path = sys.argv[2]
    out_dir = sys.argv[3]
    job_indices = parse_job_list(sys.argv[4])
    assert watermark in ("prc", "kgw")

    artifacts = torch.load(art_path, weights_only=False, map_location="cpu")
    prompt_ids_list = artifacts["prompt_ids_list"]
    jobs = artifacts["jobs"]

    import watermark_expt as we
    if watermark == "prc":
        encoding_key = artifacts["encoding_key"]
        partition_cpu = artifacts["partition"]
        we.partition = partition_cpu.to(we.device)
    else:
        import watermark_kgw as kgw
        kgw_key = int(artifacts["kgw_key"])
        gamma = float(artifacts.get("gamma", 0.25))
        delta = float(artifacts.get("delta", 2.0))

    n = len(job_indices)
    cuda_dev = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"[phase_e_worker {watermark} cuda={cuda_dev}] starting {n} jobs", flush=True)

    t_global = time.time()
    for k, job_idx in enumerate(job_indices):
        out_path = os.path.join(out_dir, f"result_{job_idx:04d}.pt")
        if os.path.isfile(out_path):
            continue
        job = jobs[job_idx]
        prompt_ids = torch.tensor(
            [prompt_ids_list[job["prompt_idx"]]],
            dtype=torch.long, device=we.device,
        )

        t0 = time.time()
        if watermark == "prc":
            gen = we.generate_text_watermark_prc(
                we.model, prompt_ids,
                max_new_tokens=job["max_new_tokens"],
                encoding_key=encoding_key,
                partition_map=we.partition,
                eos_token_id=None,
                watermark=job["watermark"],
            )
            tokens, p_trace = we.generate_and_collect(gen)
            payload = {
                "job_index": job_idx,
                "job": job,
                "tokens": tokens.cpu(),
                "p_trace": p_trace,
                "duration_sec": time.time() - t0,
                "cuda_device": cuda_dev,
            }
        else:
            gen = kgw.generate_text_watermark_kgw(
                we.model, prompt_ids,
                max_new_tokens=job["max_new_tokens"],
                key=kgw_key, gamma=gamma, delta=delta,
                eos_token_id=None,
                watermark=job["watermark"],
            )
            tokens, _ = we.generate_and_collect(gen)
            payload = {
                "job_index": job_idx,
                "job": job,
                "tokens": tokens.cpu(),
                "duration_sec": time.time() - t0,
                "cuda_device": cuda_dev,
            }
        torch.save(payload, out_path)
        if (k + 1) % 25 == 0 or (k + 1) == n:
            elapsed = time.time() - t_global
            rate = (k + 1) / max(elapsed, 1e-6)
            print(f"[phase_e_worker {watermark} cuda={cuda_dev}] "
                  f"{k+1}/{n} done elapsed={elapsed:.0f}s rate={rate:.2f} jobs/s",
                  flush=True)


if __name__ == "__main__":
    main()
