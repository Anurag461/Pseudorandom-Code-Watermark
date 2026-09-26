import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path

from baselines.config import EOS

M = 400
NUM_QUERY, NUM_CALIB = 30000, 5000
MIN_DOC_TOKENS = 562
CHUNK = 250
HF_BATCH = {"kgw2": 50, "synthid": 25, "exp": 25, "base": 50}
EXP_KEY_SEED = 42
KEY_LENGTH = 256
SCHEMES = ("prc", "kgw2", "exp", "synthid", "base")
SPOOF_ALPHAS = (1.0, 2.0, 4.5, 8.0)
VARIANTS = ("ctx1", "ctx2", "ctx3", "ctx4", "pos")
PERIOD = {"exp": KEY_LENGTH, "prc": 400, "kgw2": 400, "synthid": 400}
EVAL_PROMPTS = 500
E4_QUERIES = (1000, 3000, 10000, 30000)
E4_CELLS = [
    ("kgw2", "ctx1", 4.5),
    ("exp", "pos", 2.0),
    ("synthid", "ctx3", 8.0),
    ("synthid", "ctx4", 8.0),
    ("prc", "ctx1", 8.0),
    ("prc", "ctx2", 8.0),
    ("prc", "ctx3", 8.0),
    ("prc", "ctx4", 8.0),
    ("prc", "pos", 4.5),
]
KEY_SHA256 = "7434d9c229f583ff1f79a754699ebdea0b6bf3e158432ab7ec159fd34ce2621f"


def _load(path):
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def _artifact(settings):
    path = Path(settings.get("key", "data/keys/fixed_eta005_n400.pt"))
    if hashlib.sha256(path.read_bytes()).hexdigest() != KEY_SHA256:
        raise ValueError("PRC deployment key changed")
    return _load(path)


@lru_cache(maxsize=1)
def _prc_model(directory):
    from prc_watermark.qwen import load_model

    return load_model(directory, "0.6B", "base")[0]


PROMPT_TOKENS = 50


def pair_counts(tokens, prompts, variant, period):
    import numpy as np

    tok = tokens.numpy()
    if variant == "pos" or variant == "ctx1":
        if variant == "ctx1":
            prev = np.concatenate([prompts[:, -1:].numpy(), tok[:, :-1]], axis=1)
        else:
            prev = np.broadcast_to(np.arange(tok.shape[1]) % period, tok.shape)
        keys = prev.astype(np.int64) * 1000000 + tok
        uniq, counts = np.unique(keys.ravel(), return_counts=True)
        table = {}
        for key, count in zip(uniq.tolist(), counts.tolist()):
            table.setdefault(key // 1000000, {})[key % 1000000] = count
        return table
    h = int(variant.removeprefix("ctx"))
    full = np.concatenate([prompts[:, -h:].numpy(), tok], axis=1)
    windows = np.lib.stride_tricks.sliding_window_view(full, h + 1, axis=1).reshape(
        -1, h + 1
    )
    uniq, counts = np.unique(windows, axis=0, return_counts=True)
    table = {}
    for row, count in zip(uniq.tolist(), counts.tolist()):
        table.setdefault(tuple(row[:h]), {})[row[h]] = count
    return table


def _stolen_processor(table, variant, period, alpha):
    from transformers import LogitsProcessor

    class StolenBoost(LogitsProcessor):

        def __call__(self, input_ids, scores):
            position = input_ids.shape[1] - PROMPT_TOKENS
            for row in range(input_ids.shape[0]):
                if variant == "pos":
                    ctx = position % period
                elif variant == "ctx1":
                    ctx = int(input_ids[row, -1])
                else:
                    ctx = tuple(input_ids[row, -int(variant[3:]) :].tolist())
                if ctx in table:
                    idx, boost = table[ctx]
                    scores[row, idx.to(scores.device)] += alpha * boost.to(
                        scores.device
                    )
            return scores

    return StolenBoost()


JSV = dict(min_wm_count_nonempty=2, min_wm_mass_empty=7e-05, clip_at=2.0)


def jsv_boosts(wm, base, empty):
    total_wm, total_base = sum(wm.values()) + 1e-6, sum(base.values()) + 1e-6
    threshold = (
        round(JSV["min_wm_mass_empty"] * sum(base.values()))
        if empty
        else JSV["min_wm_count_nonempty"]
    )
    enough = [t for t, c in wm.items() if c >= threshold]
    ratios = {
        t: (wm[t] / total_wm) / (base[t] / total_base)
        for t in enough
        if base.get(t, 0) > 0
    }
    top = max(1.0, max(ratios.values(), default=0.0)) + 1e-3
    ratios.update({t: top for t in enough if base.get(t, 0) == 0})
    clip = JSV["clip_at"]
    boosts = {t: min(r, clip) / clip for t, r in ratios.items() if r >= 1}
    if boosts:
        most = max(wm[t] for t in boosts)
        boosts = {t: b + wm[t] / most * 1e-4 for t, b in boosts.items()}
        peak = max(boosts.values())
        boosts = {t: b / peak for t, b in boosts.items()}
    return boosts


def learn_table(watermarked, unwatermarked, prompts, variant, period=400):
    import torch

    wm = pair_counts(watermarked, prompts, variant, period)
    base = pair_counts(unwatermarked, prompts, variant, period)
    table = {}
    for context, counts in wm.items():
        boosts = jsv_boosts(counts, base.get(context, {}), False)
        if boosts:
            table[context] = (
                torch.tensor(list(boosts), dtype=torch.long),
                torch.tensor(list(boosts.values()), dtype=torch.float32),
            )
    return table


def _save(path, value):
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, path.with_suffix(".partial"))
    path.with_suffix(".partial").replace(path)


def build_prompts(settings, output):
    import os

    os.environ["HF_HUB_OFFLINE"] = os.environ["HF_DATASETS_OFFLINE"] = "0"
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer

    path = Path(f"{output}/prompts.pt")
    if path.exists():
        return str(path)
    tokenizer = AutoTokenizer.from_pretrained(settings["model_directory"])
    stream = load_dataset("allenai/c4", "realnewslike", split="train", streaming=True)
    prompts, docs = ([], [])
    for doc, example in enumerate(stream):
        tokens = tokenizer.encode(example["text"])
        if len(tokens) >= MIN_DOC_TOKENS:
            prompts.append(tokens[:PROMPT_TOKENS])
            docs.append(doc)
        if len(prompts) == NUM_QUERY + NUM_CALIB:
            break
    if len(prompts) != NUM_QUERY + NUM_CALIB:
        raise ValueError("Insufficient C4 prompts")
    _save(
        path,
        {
            "query": torch.tensor(prompts[:NUM_QUERY]),
            "calib": torch.tensor(prompts[NUM_QUERY:]),
            "query_docs": docs[:NUM_QUERY],
            "calib_docs": docs[NUM_QUERY:],
            "source": "allenai/c4 realnewslike train (streaming order)",
            "min_doc_tokens": MIN_DOC_TOKENS,
        },
    )
    return str(path)


def _chunk_path(output, scheme, split, start):
    return Path(f"{output}/generations/{scheme}/{split}_{start:05d}.pt")


def generate_hf(settings, output, scheme, split, start):
    import torch
    from baselines.kgw import _cpu_kgw_processor
    from baselines.synthid import attack_processor as synthid_processor
    from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

    path = _chunk_path(output, scheme, split, start)
    if path.exists():
        return str(path)
    prompts = _load(f"{output}/prompts.pt")[split][start : start + CHUNK]
    tokenizer = AutoTokenizer.from_pretrained(settings["model_directory"])
    model = (
        AutoModelForCausalLM.from_pretrained(
            settings["model_directory"], torch_dtype=torch.float32
        )
        .cuda()
        .eval()
    )
    vocab_size = model.get_output_embeddings().weight.shape[0]
    torch.manual_seed(
        int.from_bytes(
            hashlib.sha256(f"{scheme}:{split}:{start}".encode()).digest()[:4], "little"
        )
    )
    outputs = []
    for b in range(0, len(prompts), HF_BATCH[scheme]):
        batch = prompts[b : b + HF_BATCH[scheme]]
        if scheme == "exp":
            from baselines.exp import generate

            seeds = torch.full((len(batch),), EXP_KEY_SEED)
            out = generate(
                model, batch, vocab_size, KEY_LENGTH, M, seeds, random_offset=False
            )
        else:
            processors = []
            if scheme == "kgw2":
                processors = [_cpu_kgw_processor(list(tokenizer.get_vocab().values()))]
            elif scheme == "synthid":
                processors = [synthid_processor(torch.device("cuda"))]
            out = model.generate(
                batch.cuda(),
                attention_mask=torch.ones_like(batch).cuda(),
                do_sample=True,
                max_new_tokens=M,
                min_new_tokens=M,
                top_k=0,
                top_p=1.0,
                temperature=1.0,
                pad_token_id=EOS,
                logits_processor=LogitsProcessorList(processors),
            ).cpu()
        outputs.append(out[:, PROMPT_TOKENS : PROMPT_TOKENS + M].to(torch.int32))
    tokens = torch.cat(outputs)
    if tokens.shape != (len(prompts), M):
        raise ValueError(f"expected {(len(prompts), M)}, got {tuple(tokens.shape)}")
    _save(path, {"scheme": scheme, "split": split, "start": start, "tokens": tokens})
    return str(path)


def generate_prc(settings, output, split, start):
    import torch
    from prc_watermark.generation import generate_batch_and_collect

    path = _chunk_path(output, "prc", split, start)
    if path.exists():
        return str(path)
    artifact = _artifact(settings)
    model = _prc_model(settings["model_directory"])
    device = next(model.parameters()).device
    partition = artifact["partition"].to(device)
    prompts = _load(Path(output) / "prompts.pt")[split][start : start + CHUNK]
    outputs = []
    for b in range(0, len(prompts), 50):
        batch = prompts[b : b + 50].to(device)
        tokens, _ = generate_batch_and_collect(
            model, batch, M, artifact["encoding_key"], partition, watermark=True
        )
        outputs.append(tokens[:, :M].cpu().to(torch.int32))
    tokens = torch.cat(outputs)
    if tokens.shape != (len(prompts), M):
        raise ValueError(f"expected {(len(prompts), M)}, got {tuple(tokens.shape)}")
    _save(path, {"scheme": "prc", "split": split, "start": start, "tokens": tokens})
    return str(path)


def _query_tokens(output, scheme, n_query):
    import torch

    chunks = [
        _load(_chunk_path(output, scheme, "query", s))["tokens"]
        for s in range(0, n_query, CHUNK)
    ]
    return torch.cat(chunks)[:n_query].to(torch.int64)


def stolen_table(settings, output, scheme, variant, n_query):
    import torch

    prompts = _load(f"{output}/prompts.pt")["query"][:n_query]
    wm = pair_counts(
        _query_tokens(output, scheme, n_query), prompts, variant, PERIOD[scheme]
    )
    base = pair_counts(
        _query_tokens(output, "base", n_query), prompts, variant, PERIOD[scheme]
    )
    table = {}
    for ctx, counts in wm.items():
        boosts = jsv_boosts(counts, base.get(ctx, {}), empty=False)
        if boosts:
            table[ctx] = (
                torch.tensor(list(boosts)),
                torch.tensor(list(boosts.values()), dtype=torch.float32),
            )
    return table


def _eval_prompts(settings):
    import torch

    lines = (
        Path(settings.get("prompts", "data/prompts.jsonl"))
        .read_text()
        .splitlines()[:EVAL_PROMPTS]
    )
    return torch.tensor([json.loads(line)["prompt_tokens"] for line in lines])


def spoof(settings, output, scheme, variant, n_query, alphas):
    import torch
    from transformers import AutoModelForCausalLM, LogitsProcessorList

    paths = {
        a: Path(f"{output}/spoof/{scheme}/{variant}_N{n_query}_a{a:g}.pt")
        for a in alphas
    }
    todo = [a for a, p in paths.items() if not p.exists()]
    if not todo:
        return [str(p) for p in paths.values()]
    table = stolen_table(settings, output, scheme, variant, n_query)
    model = (
        AutoModelForCausalLM.from_pretrained(
            settings["model_directory"], torch_dtype=torch.float32
        )
        .cuda()
        .eval()
    )
    prompts = _eval_prompts(settings)
    for alpha in todo:
        torch.manual_seed(int(alpha * 1000))
        outputs = []
        for b in range(0, len(prompts), 50):
            batch = prompts[b : b + 50].cuda()
            out = model.generate(
                batch,
                attention_mask=torch.ones_like(batch),
                do_sample=True,
                max_new_tokens=M,
                min_new_tokens=M,
                top_k=0,
                top_p=1.0,
                temperature=1.0,
                pad_token_id=EOS,
                logits_processor=LogitsProcessorList(
                    [_stolen_processor(table, variant, PERIOD[scheme], alpha)]
                ),
            ).cpu()
            outputs.append(out[:, PROMPT_TOKENS : PROMPT_TOKENS + M].to(torch.int32))
        _save(
            paths[alpha],
            {
                "scheme": scheme,
                "variant": variant,
                "n_query": n_query,
                "alpha": alpha,
                "contexts": len(table),
                "tokens": torch.cat(outputs),
            },
        )
    return [str(p) for p in paths.values()]


def _text_sets(output, scheme):
    import torch

    sets = {
        "calib": torch.cat(
            [
                _load(_chunk_path(output, "base", "calib", s))["tokens"]
                for s in range(0, NUM_CALIB, CHUNK)
            ]
        ),
        "query500": torch.cat(
            [_load(_chunk_path(output, scheme, "query", s))["tokens"] for s in (0, 250)]
        ),
    }
    spoof_dir = Path(f"{output}/spoof/{scheme}")
    for path in sorted(spoof_dir.glob("*.pt")) if spoof_dir.exists() else []:
        sets[path.stem] = _load(path)["tokens"]
    return {k: v.to(torch.int64) for k, v in sets.items()}


def score_hf(settings, output, scheme, name, start, stop):
    from transformers import AutoTokenizer
    from .substitution import make_scorer

    path = Path(f"{output}/scores/{scheme}/{name}_{start:05d}.json")
    if path.exists():
        return str(path)
    tokens = _text_sets(output, scheme)[name][start:stop]
    scorer = make_scorer(
        scheme, 151936, AutoTokenizer.from_pretrained(settings["model_directory"])
    )
    stats = [scorer(t, EXP_KEY_SEED) for t in tokens]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"scheme": scheme, "name": name, "start": start, "stats": stats})
    )
    return str(path)


def score_prc(settings, output, name, start, stop):
    import torch
    from prc_watermark.detectors import detect_hoeffding
    from prc_watermark.qwen import completion_only_partition_trace_batch

    path = Path(output) / "scores" / "prc" / f"{name}_{start:05d}.json"
    if path.exists():
        return str(path)
    artifact = _artifact(settings)
    model = _prc_model(settings["model_directory"])
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = model.eval().requires_grad_(False)
    if next(model.parameters()).dtype != torch.bfloat16:
        raise ValueError("PRC redetection replays in BF16")
    device = next(model.parameters()).device
    tokens = _text_sets(output, "prc")[name][start:stop]
    partition = artifact["partition"]
    rows = []
    for b in range(0, len(tokens), 50):
        batch = tokens[b : b + 50]
        trace = completion_only_partition_trace_batch(
            model,
            batch.to(device),
            partition[1].to(torch.bfloat16).to(device),
            "static",
        )
        for row, p in zip(batch, trace.cpu().numpy()):
            decision, info = detect_hoeffding(
                artifact["decoding_key"],
                row,
                p,
                partition,
                fpr=1e-3,
                weight="map",
                return_info=True,
            )
            rows.append(
                {
                    "decision": bool(decision),
                    "statistic": info["statistic"],
                    "V": info["V"],
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"scheme": "prc", "name": name, "start": start, "rows": rows})
    )
    return str(path)


def perplexity(settings, output, scheme, name):
    import torch
    from transformers import AutoModelForCausalLM

    path = Path(f"{output}/ppl/{scheme}/{name}.json")
    if path.exists():
        return str(path)
    tokens = _text_sets(output, scheme)[name][:EVAL_PROMPTS]
    if name == "calib":
        prompts = _load(f"{output}/prompts.pt")["calib"][: len(tokens)]
    elif name == "query500":
        prompts = _load(f"{output}/prompts.pt")["query"][: len(tokens)]
    else:
        prompts = _eval_prompts(settings)[: len(tokens)]
    model = (
        AutoModelForCausalLM.from_pretrained(
            settings["perplexity_model_directory"], torch_dtype=torch.bfloat16
        )
        .cuda()
        .eval()
    )
    values = []
    with torch.no_grad():
        for b in range(0, len(tokens), 5):
            ids = torch.cat([prompts[b : b + 5], tokens[b : b + 5]], 1).cuda()
            logits = model(ids).logits[:, PROMPT_TOKENS - 1 : -1].float()
            nll = torch.nn.functional.cross_entropy(
                logits.transpose(1, 2), ids[:, PROMPT_TOKENS:], reduction="none"
            )
            values += nll.mean(1).exp().cpu().tolist()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"scheme": scheme, "name": name, "ppl": values}))
    return str(path)


def _hoeffding_p(row):
    import math

    S, V = (row["statistic"], row["V"])
    return 1.0 if S is None or not V or S <= 0 else math.exp(-S * S / (2 * V))


def summarize_attack(output, schemes="prc,kgw2,exp,synthid"):
    import csv
    import numpy as np

    local = Path(output)
    rows = []
    for scheme in schemes.split(","):
        scores = {}
        for path in (local / "scores" / scheme).glob("*.json"):
            payload = json.loads(path.read_text())
            values = payload["rows"] if scheme == "prc" else payload["stats"]
            scores.setdefault(payload["name"], []).append((payload["start"], values))
        scores = {
            k: [v for _, chunk in sorted(parts) for v in chunk]
            for k, parts in scores.items()
        }
        ppl = {
            json.loads(p.read_text())["name"]: np.array(
                json.loads(p.read_text())["ppl"]
            )
            for p in (local / "ppl" / scheme).glob("*.json")
        }
        proven = None
        if scheme == "prc":
            proven = {
                k: np.array([r["decision"] for r in v]) for k, v in scores.items()
            }
            scores = {k: [_hoeffding_p(r) for r in v] for k, v in scores.items()}
        calib = np.sort(scores["calib"])
        cut = calib[int(0.001 * len(calib)) - 1]
        detected = {k: np.array(v) <= cut for k, v in scores.items()}
        quality_cut = np.quantile(ppl["calib"], 0.95) if "calib" in ppl else np.inf
        genuine = detected["query500"].mean()
        rate = lambda x: f"{x.sum()}/{len(x)} ({x.mean():.1%})"
        for name in sorted(detected):
            d = detected[name][:EVAL_PROMPTS] if name == "calib" else detected[name]
            p = ppl.get(name, np.full(len(d), np.nan))[: len(d)]
            good = p <= quality_cut
            variant, _, rest = name.partition("_N")
            row = {
                "scheme": scheme,
                "set": name,
                "variant": variant if rest else name,
                "n_query": rest.split("_a")[0] if rest else "",
                "alpha": rest.split("_a")[1] if rest else "",
                "texts": len(d),
                "genuine TPR (query500)": f"{genuine:.1%}",
                "detected@1e-3": rate(d),
                "detected & ppl-ok": rate(d & good),
                "median ppl": f"{np.nanmedian(p):.3g}",
                "ppl-ok cut (calib p95)": f"{quality_cut:.3g}",
                "threshold": f"empirical 1e-3 quantile of {len(calib)} unwatermarked (stat<={cut:.4g})",
                "PRC proven-threshold detected": "",
                "PRC proven-threshold & ppl-ok": "",
            }
            if proven is not None:
                q = proven[name][: len(d)]
                row["PRC proven-threshold detected"] = rate(q)
                row["PRC proven-threshold & ppl-ok"] = rate(q & good)
            rows.append(row)
    path = local / "results" / "stealing_results.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


def attack_cells():
    cells = {
        (scheme, variant, 10000, alpha)
        for scheme in SCHEMES[:-1]
        for variant in VARIANTS
        for alpha in SPOOF_ALPHAS
    }
    cells.update(
        (scheme, variant, n, alpha)
        for scheme, variant, alpha in E4_CELLS
        for n in E4_QUERIES
    )
    return sorted(cells)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=("prepare", "generate", "spoof", "score", "perplexity", "summarize"),
    )
    parser.add_argument("--settings", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scheme", choices=SCHEMES)
    parser.add_argument("--split", choices=("query", "calib"))
    parser.add_argument("--start", type=int)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--name")
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--n-query", type=int)
    parser.add_argument("--alpha", type=float)
    args = parser.parse_args()
    settings = json.loads(Path(args.settings).read_text())
    settings = settings.get("stealing", settings)
    output = Path(args.output)
    schemes = [args.scheme] if args.scheme else list(SCHEMES[:-1])
    if args.stage == "prepare":
        print(build_prompts(settings, output))
    elif args.stage == "generate":
        generation_schemes = [args.scheme] if args.scheme else list(SCHEMES)
        for scheme in generation_schemes:
            splits = (
                [args.split]
                if args.split
                else (["query", "calib"] if scheme == "base" else ["query"])
            )
            for split in splits:
                count = NUM_CALIB if split == "calib" else (args.n_query or NUM_QUERY)
                starts = (
                    [args.start] if args.start is not None else range(0, count, CHUNK)
                )
                for start in starts:
                    if scheme == "prc":
                        print(generate_prc(settings, output, split, start))
                    else:
                        print(generate_hf(settings, output, scheme, split, start))
    elif args.stage == "spoof":
        jobs = {}
        for scheme, variant, n_query, alpha in attack_cells():
            if (
                scheme not in schemes
                or (args.variant and variant != args.variant)
                or (args.n_query is not None and n_query != args.n_query)
                or (args.alpha is not None and alpha != args.alpha)
            ):
                continue
            jobs.setdefault((scheme, variant, n_query), []).append(alpha)
        if not jobs:
            parser.error("No paper experiment matches the selected parameters")
        for (scheme, variant, n_query), alphas in jobs.items():
            print(spoof(settings, output, scheme, variant, n_query, alphas))
    elif args.stage in ("score", "perplexity"):
        for scheme in schemes:
            for name, tokens in _text_sets(output, scheme).items():
                if args.name and name != args.name:
                    continue
                if args.stage == "perplexity":
                    print(perplexity(settings, output, scheme, name))
                    continue
                step = 500 if scheme == "prc" else 100
                starts = (
                    [args.start]
                    if args.start is not None
                    else range(0, len(tokens), step)
                )
                for start in starts:
                    stop = (
                        args.stop
                        if args.stop is not None
                        else min(start + step, len(tokens))
                    )
                    if scheme == "prc":
                        print(score_prc(settings, output, name, start, stop))
                    else:
                        print(score_hf(settings, output, scheme, name, start, stop))
    else:
        print(summarize_attack(output, ",".join(schemes)))


if __name__ == "__main__":
    main()
