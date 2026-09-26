"""Replay one kth_long score chunk text by text with faulthandler, to locate a native crash."""
import modal
import kth_long as k

dbg = modal.App("klong-debug")


@dbg.function(image=k.image, cpu=2, memory=16384, timeout=3600,
              volumes={"/cache": k.hf_cache, "/results": k.results, "/data": k.data_vol})
def trace(scheme, start, rate):
    import faulthandler, sys, torch
    faulthandler.enable(file=sys.stderr, all_threads=True)
    from attacks import apply_attack
    gen = torch.load(f"/results/{k.OUT}/generations/{scheme}/{start:04d}.pt", weights_only=False)
    proc = k.synthid_processor(torch.device("cpu")) if scheme == "synthid" else None
    out = []
    for row, idx in enumerate(gen["prompt_idx"]):
        for source in ("wm", "null"):
            print("START", idx, source, flush=True)
            tokens = gen["tokens"][row].to(torch.int64) if source == "wm" else k.load_null(idx)
            text = apply_attack(tokens, k.attack_spec(rate), source, idx)[:k.M]
            print("  attacked", len(text), int(text.max()), flush=True)
            p = k.synthid_pvalue(proc, text)[0] if scheme == "synthid" else k.exp_pvalue(text, gen["seeds"][row])[0]
            print("  p", p, flush=True)
            out.append((idx, source, p))
    return out


@dbg.function(image=k.image, cpu=2, memory=16384, timeout=3600,
              volumes={"/cache": k.hf_cache, "/results": k.results, "/data": k.data_vol})
def score_missing(scheme, start, rate):
    """Same computation and file format as kth_long.score_chunk, for chunks whose container crashes there."""
    import json, torch
    from pathlib import Path
    from attacks import apply_attack
    attack = k.attack_spec(rate)
    path = Path(f"/results/{k.OUT}/scores/{scheme}/{k.attack_id(attack)}_{start:04d}.json")
    if path.exists():
        return "exists"
    gen = torch.load(f"/results/{k.OUT}/generations/{scheme}/{start:04d}.pt", weights_only=False)
    proc = k.synthid_processor(torch.device("cpu")) if scheme == "synthid" else None
    rows = []
    for row, idx in enumerate(gen["prompt_idx"]):
        for source in ("wm", "null"):
            tokens = gen["tokens"][row].to(torch.int64) if source == "wm" else k.load_null(idx)
            text = apply_attack(tokens, attack, source, idx)[:k.M]
            p, stat = k.synthid_pvalue(proc, text) if scheme == "synthid" else k.exp_pvalue(text, gen["seeds"][row])
            rows.append({"source": source, "prompt_idx": idx, "p": p, "stat": stat})
    path.write_text(json.dumps({"scheme": scheme, "attack": attack, "start": start, "rows": rows}))
    k.results.commit()
    return str(path)


if __name__ == "__main__":
    import sys
    if sys.argv[1] == "missing":
        jobs = [("exp", 100), ("exp", 110), ("exp", 170), ("synthid", 10), ("synthid", 30), ("synthid", 60),
                ("synthid", 160)]
        with dbg.run():
            for (s, st), out in zip(jobs, score_missing.starmap([(s, st, 0.3) for s, st in jobs],
                                                                 return_exceptions=True)):
                print(s, st, out if isinstance(out, str) else f"ERR {out!r}"[:300])
        raise SystemExit
    import sys
    with dbg.run():
        try:
            print(trace.remote(sys.argv[1], int(sys.argv[2]), 0.3))
        except Exception as e:
            print("ERR", repr(e)[-300:])
