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
