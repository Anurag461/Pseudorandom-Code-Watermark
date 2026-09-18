"""Bounded diagnosis of TextSeal prefix numerics; never launches full replay.

Calls the frozen upstream detector on the ten original pilot responses, then
isolates shape dependence on null/0000. Diagnostic interventions never enter
production caches or the comparison CSV.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import modal

from .textseal_modal import image, load_model, model_volume, results_volume, runtime_identity
from .textseal_redetect import digest, file_sha, load_request, record_identity, validate_request, write_json

app = modal.App("textseal-prefix-diagnostic", image=image)


def difference(a, b):
    import torch
    a, b = torch.as_tensor(a).float(), torch.as_tensor(b).float()
    if a.shape != b.shape:
        raise ValueError("comparison shapes differ")
    delta = (a-b).abs()
    return {"exact": bool(torch.equal(a, b)), "different_values": int((a != b).sum()),
            "values": a.numel(), "max_abs_difference": float(delta.max()) if delta.numel() else 0,
            "mean_abs_difference": float(delta.mean()) if delta.numel() else 0}


def raw_entropy(detector, ids):
    calls = []

    def observe(module, args, kwargs):
        if kwargs or len(args) != 1 or args[0].tolist() != [ids]:
            raise ValueError("diagnostic model input differs from the specified raw token IDs")
        calls.append(len(ids))

    hook = detector._detector.model.register_forward_pre_hook(observe, with_kwargs=True)
    try:
        entropy = detector._entropies(ids)
    finally:
        hook.remove()
    if calls != [len(ids)]:
        raise ValueError("unexpected diagnostic model call count")
    return entropy


def layer_capture(detector, ids, prefix=128):
    """Read-only forward hooks; preserve upstream's complete computation."""
    import torch
    model = detector._detector.model
    names = ["model.embed_tokens", "model.rotary_emb"]
    for stem in ("input_layernorm", "self_attn.q_proj", "self_attn.q_norm", "self_attn.k_proj",
                 "self_attn.k_norm", "self_attn.v_proj", "self_attn.o_proj", "self_attn",
                 "post_attention_layernorm", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "mlp"):
        names.append("model.layers.0." + stem)
    names += [f"model.layers.{i}" for i in range(model.config.num_hidden_layers)]
    names += ["model.norm", "lm_head"]
    values, order, handles = {}, [], []
    modules = dict(model.named_modules())

    def capture(name):
        def hook(module, args, output):
            outputs = output if isinstance(output, tuple) and name == "model.rotary_emb" else (output[0] if isinstance(output, tuple) else output,)
            values[name] = [v[:, :prefix].detach().cpu().clone() for v in outputs]
            order.append(name)
        return hook

    for name in names:
        handles.append(modules[name].register_forward_hook(capture(name)))
    try:
        entropy = raw_entropy(detector, ids)
    finally:
        for h in handles: h.remove()
    return entropy, values, order


def isolate_linear(model, name, ids, prefix=128):
    """Same frozen weights and identical prefix input, only matrix shape varies."""
    import torch
    import torch.nn.functional as F
    layer = dict(model.named_modules())[name]
    if not isinstance(layer, torch.nn.Linear):
        return {"performed": False, "reason": "first differing module is not Linear"}
    captured = []
    hook = layer.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
    try:
        with torch.no_grad(): model(torch.tensor([ids], device=next(model.parameters()).device))
    finally:
        hook.remove()
    if len(captured) != 1:
        raise ValueError("unexpected Linear invocation count")
    x = captured[0]
    with torch.no_grad():
        full = F.linear(x, layer.weight, layer.bias)[:, :prefix]
        short = F.linear(x[:, :prefix].contiguous(), layer.weight, layer.bias)
        repeated = F.linear(x[:, :prefix].contiguous(), layer.weight, layer.bias)
        # Diagnostic reference only: never changes model weights or production dtype.
        full32 = F.linear(x.float(), layer.weight.float(), None if layer.bias is None else layer.bias.float())[:, :prefix]
        short32 = F.linear(x[:, :prefix].contiguous().float(), layer.weight.float(), None if layer.bias is None else layer.bias.float())
    return {"performed": True, "module": name, "input_prefix_exact": True,
            "input_shape": list(x.shape), "weight_shape": list(layer.weight.shape),
            "bf16_shape_comparison": difference(full.cpu(), short.cpu()),
            "bf16_repeat": difference(short.cpu(), repeated.cpu()),
            "fp32_shape_comparison": difference(full32.cpu(), short32.cpu()),
            "bf16_short_vs_fp32": difference(short.cpu(), short32.cpu()),
            "bf16_full_vs_fp32": difference(full.cpu(), full32.cpu())}


@app.function(gpu="H100", cpu=(4, 4), memory=(65536, 65536), timeout=600,
              max_containers=1, retries=0, scaledown_window=2,
              volumes={"/cache": model_volume, "/results": results_volume})
def diagnose(manifest, records, pilot, source_sha):
    from .textseal_completion import TextSealCompletionDetector
    import torch
    started = time.monotonic()
    approved = digest(manifest)
    validate_request(manifest, records, "pilot", approved)
    if file_sha(__file__) != source_sha:
        raise ValueError("diagnostic source changed")
    runtime = runtime_identity(manifest)
    if runtime != pilot["runtime"] or pilot["manifest_sha256"] != approved:
        raise ValueError("diagnostic runtime must match the failed pilot")
    root = Path("/results/textseal_prefix_diagnostic") / digest({"manifest": approved, "source": source_sha})
    results_volume.reload()
    old_path = Path("/results/textseal_completion_redetect") / approved / "records/null/0000.json"
    if file_sha(old_path) != pilot["record_sha256"]["null/0000"]:
        raise ValueError("original pilot record changed")
    old = json.loads(old_path.read_text())
    load_started = time.monotonic()
    model = load_model(manifest, "/cache")
    detector = TextSealCompletionDetector(model)
    load_seconds = time.monotonic()-load_started
    rows, all_direct = [], []
    first = None
    for record in records:
        ids = record["token_ids"]
        record_started = time.monotonic()
        begin = time.monotonic()
        longest = raw_entropy(detector, ids)
        longest_seconds = time.monotonic()-begin
        direct, checks, lengths_seconds = {}, {}, {}
        for n in manifest["prefix_lengths"]:
            begin = time.monotonic()
            entropies = longest if n == len(ids) else raw_entropy(detector, ids[:n])
            lengths_seconds[str(n)] = longest_seconds if n == len(ids) else time.monotonic()-begin
            actual = detector._result(ids[:n], entropies, .001)
            reused = detector._result(ids[:n], longest[:n-1], .001)
            direct[str(n)] = {"entropy": entropies, "result": actual}
            checks[str(n)] = {**difference(entropies, longest[:n-1]),
                              "upstream_exact": actual["upstream"] == reused["upstream"],
                              "weighted_p_direct": actual["comparison"]["p_value"],
                              "weighted_p_reused": reused["comparison"]["p_value"],
                              "decision_equal": actual["comparison"]["decision"] == reused["comparison"]["decision"],
                              "unweighted_p_exact": actual["upstream"]["p_value_unweighted"] == reused["upstream"]["p_value_unweighted"]}
        identity = record_identity(record)
        if identity["record_id"] == "null/0000":
            first = {"ids": ids, "direct": direct, "longest": longest,
                     "original_longest_exact": longest == old["data"]["entropies_2_to_T"]}
        rows.append({"record": identity, "checks": checks, "model_seconds_by_length": lengths_seconds,
                     "total_seconds": time.monotonic()-record_started})
        all_direct.append({"record": identity, "prefixes": direct})
        print(f"Compared all prefixes for {identity['record_id']}", flush=True)
    ids = first["ids"]
    repeated128 = raw_entropy(detector, ids[:128])
    repeated1024 = raw_entropy(detector, ids)
    reversed_suffix = raw_entropy(detector, ids[:128] + list(reversed(ids[128:])))
    boundary = {str(n): difference(raw_entropy(detector, ids[:n])[:127], first["longest"][:min(n-1, 127)])
                for n in (127, 129, 192, 255)}
    # n=127 has 126 entropy entries; compare only common positions.
    entropy_full, activations_full, order = layer_capture(detector, ids)
    entropy_short, activations_short, _ = layer_capture(detector, ids[:128])
    layers = [{"module": name, "outputs": [difference(a, b) for a, b in zip(activations_full[name], activations_short[name])]}
              for name in order]
    first_difference = next((r["module"] for r in layers if any(not x["exact"] for x in r["outputs"])), None)
    linear = isolate_linear(model, first_difference, ids) if first_difference else {"performed": False}
    controls = {"original_cached_n1024_exact": first["original_longest_exact"],
                "n128_repeat": difference(repeated128, first["direct"]["128"]["entropy"]),
                "n1024_repeat": difference(repeated1024, first["longest"]),
                "suffix_intervention_prefix": difference(reversed_suffix[:127], first["longest"][:127]),
                "hooks_do_not_change_full_entropy": entropy_full == first["longest"],
                "hooks_do_not_change_short_entropy": entropy_short == repeated128}
    report = {"manifest_sha256": approved, "diagnostic_source_sha256": source_sha,
              "runtime": runtime, "records": rows, "controls": controls,
              "boundary_lengths": boundary, "layer_comparisons": layers,
              "first_differing_module": first_difference, "isolated_linear": linear,
              "load_seconds": load_seconds, "total_seconds": time.monotonic()-started,
              "upstream_source": detector.upstream_source,
              "full_replay_launched": False, "csv_modified": False,
              "modal_root": str(root.relative_to('/results'))}
    write_json(root / "report.json", report)
    write_json(root / "direct_entropies.json", all_direct)
    results_volume.commit()
    return report


@app.local_entrypoint()
def run(manifest: str = "outputs/comparison_redetect/textseal_setup/native8b_manifest.json"):
    path = Path(manifest)
    frozen = json.loads(path.read_text())
    frozen, records = load_request(path, "pilot", digest(frozen))
    pilot = json.loads((path.parent / "pilot_report.json").read_text())
    report = diagnose.remote(frozen, records, pilot, file_sha(__file__))
    write_json(path.parent / "prefix_diagnostic/report.json", report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('records','layer_comparisons','upstream_source')}, indent=2))
