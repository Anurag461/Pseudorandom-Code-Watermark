"""Prepare, generate and analyze isolated repeat-fallback policies."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import inspect
import types

import torch
import numpy as np

from .config import StudySetting, digest
from .pilot import CODE as PILOT_CODE, load_pairs, paired_interval, score_rows
from .validation import ROOT, RATE, save, sha, verify_source_hashes

FALLBACK_SEED_DOMAIN = "prc-self-bleu/repeat-fallback/v1"
ARMS = {"synthid_off": ("synthid_text", False), "textseal_on": ("textseal", True),
        "gumbel_on": ("gumbel_max", True)}


@dataclass(frozen=True)
class RepeatSetting(StudySetting):
    repeat_fallback: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.method not in {"synthid_text", "textseal", "gumbel_max"} or type(self.repeat_fallback) is not bool:
            raise ValueError("repeat ablation applies only to the three context-based methods")
        if self.depth != 10 or self.alpha != .1:
            raise ValueError("repeat ablation fixes depth 10 and alpha .1")

    def identity(self):
        return {**super().identity(), "repeat_fallback": self.repeat_fallback,
                "repeat_scope": "within one response; fresh history for each response",
                "context_initialization": "upstream zero context" if self.method == "synthid_text" else "last three prompt tokens",
                "fallback_rng": "existing multinomial stream" if self.method == "synthid_text" else FALLBACK_SEED_DOMAIN}


def arm_setting(arm):
    method, enabled = ARMS[arm]
    return RepeatSetting(method, repeat_fallback=enabled)


def fallback_seed(sampling_seed, prompt_index):
    data = f"{FALLBACK_SEED_DOMAIN}/{sampling_seed}/{prompt_index}".encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big") >> 1


class SynthIDRepeatPolicy:
    """Run the unchanged upstream call; optionally return its pre-fallback scores.

    The private function namespace captures update_scores' return value. No
    upstream module, PRF, key, state update, tensor geometry or RNG is changed.
    Both policies compute precisely the same intermediates for a given history.
    """
    def __init__(self, processor, enabled):
        if processor._num_leaves != 2 or processor.skip_first_ngram_calls or processor.apply_top_k:
            raise ValueError("unexpected SynthID configuration")
        self.processor, self.enabled = processor, enabled
        self.repeated, self.applied = [], []
        self._updated = []
        original = inspect.unwrap(type(processor).watermarked_call)
        upstream_update = original.__globals__["update_scores"]

        def capture(*args, **kwargs):
            value = upstream_update(*args, **kwargs)
            self._updated.append(value)
            return value

        namespace = {**original.__globals__, "update_scores": capture}
        private = types.FunctionType(original.__code__, namespace, original.__name__,
                                     original.__defaults__, original.__closure__)
        private.__kwdefaults__ = original.__kwdefaults__
        self._call = types.MethodType(private, processor)

    @torch.no_grad()
    def watermarked_call(self, ids, scores):
        processor = self.processor
        old = (torch.zeros((scores.shape[0], processor.context_history_size), dtype=torch.long, device=scores.device)
               if processor.state is None else processor.state.context_history.clone())
        self._updated.clear()
        native = self._call(ids, scores)
        if len(self._updated) != 1:
            raise ValueError("upstream SynthID score-update contract changed")
        current_hash = processor.state.context_history[:, :1]
        repeated = (old == current_hash).any(dim=1)
        self.repeated.append(repeated.cpu().tolist())
        self.applied.append((repeated if self.enabled else torch.zeros_like(repeated)).cpu().tolist())
        return (native[0] if self.enabled else self._updated[0]), native[1], native[2]


class SamplerRepeatPolicy:
    """TextSeal/Gumbel native draws plus response-local ordinary-sampling fallback.

    Native sampling runs for the entire batch even at repeated positions. The
    extra fallback draws use private per-prompt generators, so they cannot
    advance the existing key-routing stream or another response's fallback RNG.
    This deliberately preserves native computation rather than optimizing cost.
    """
    def __init__(self, sampler, enabled, sampling_seed, prompt_indices):
        self.sampler, self.enabled = sampler, enabled
        self.seeds = [fallback_seed(sampling_seed, i) for i in prompt_indices]
        self.seen = [set() for _ in prompt_indices]
        self.generators = None
        self.repeated, self.applied = [], []

    @torch.no_grad()
    def sample_next(self, logits, context, *, temperature, top_p):
        if temperature != 1 or top_p != 1 or context.shape != (len(self.seen), 3):
            raise ValueError("fallback ablation requires original temperature/top-p and three-token contexts")
        if self.generators is None:
            self.generators = [torch.Generator(device=logits.device).manual_seed(seed) for seed in self.seeds]
        repeated = []
        for seen, row in zip(self.seen, context.cpu().tolist()):
            key = tuple(row)
            repeated.append(key in seen)
            seen.add(key)
        native = self.sampler.sample_next(logits, context, temperature=temperature, top_p=top_p)
        result = native.clone()
        applied = [self.enabled and value for value in repeated]
        for i, use_fallback in enumerate(applied):
            if use_fallback:
                probs = torch.softmax(logits[i].float(), dim=-1)
                result[i] = torch.multinomial(probs, 1, generator=self.generators[i])[0]
        self.repeated.append(repeated)
        self.applied.append(applied)
        return result


@contextmanager
def install_policy(setting, sampling_seed, prompt_indices):
    """Scope factories to this single-threaded ablation call; always restore them."""
    from baseline_comparison import comparison_runner as runner
    names = ("synthid_processor", "textseal_generator", "gumbel_generator")
    original = {name: getattr(runner, name) for name in names}
    created = []

    def synthid(*args, **kwargs):
        policy = SynthIDRepeatPolicy(original["synthid_processor"](*args, **kwargs), setting.repeat_fallback)
        created.append(policy)
        return policy

    def sampler(name, *args, **kwargs):
        policy = SamplerRepeatPolicy(original[name](*args, **kwargs), setting.repeat_fallback,
                                     sampling_seed, prompt_indices)
        created.append(policy)
        return policy

    if setting.method == "synthid_text":
        runner.synthid_processor = synthid
    elif setting.method == "textseal":
        runner.textseal_generator = lambda **kw: sampler("textseal_generator", **kw)
    else:
        runner.gumbel_generator = lambda: sampler("gumbel_generator")
    try:
        yield created
    finally:
        for name, value in original.items():
            setattr(runner, name, value)


def generate_repeat_batch(model, prompts, indices, *, setting, sampling_seed, response_index,
                          execution, max_new_tokens=1024, device="cuda"):
    from .generation import generate_response_batch
    if not isinstance(setting, RepeatSetting):
        raise ValueError("explicit repeat policy required")
    if max_new_tokens > 1024:
        raise ValueError("ablation exceeds SynthID history capacity")
    execution = {**execution, "repeat_adapter_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest()}
    with install_policy(setting, sampling_seed, indices) as policies:
        batch = generate_response_batch(model, prompts, indices, setting=setting, sampling_seed=sampling_seed,
                                        response_index=response_index, execution=execution,
                                        max_new_tokens=max_new_tokens, device=device)
    policy = policies[0]
    if len(policy.repeated) != max_new_tokens:
        raise ValueError("missing generation-policy trace")
    for i, row in enumerate(batch["responses"]):
        repeated = [bool(step[i]) for step in policy.repeated]
        applied = [bool(step[i]) for step in policy.applied]
        row["generation_diagnostics"].update(repeated_context=repeated, fallback_applied=applied,
            first_fallback_position=next((j for j, value in enumerate(applied) if value), None))
    if "synthid_official_smoke_reference" in batch["telemetry"]:
        batch["telemetry"]["same_policy_single_row_reference"] = batch["telemetry"].pop("synthid_official_smoke_reference")
    batch["manifest"]["namespace"] = f"self_bleu_repeat_v1/{batch['manifest']['batch_id']}"
    return batch


def check_synthid_policy(device="cpu"):
    """Forced repeated contexts exercise the real upstream math, also on H100."""
    from baseline_comparison.official import synthid_processor
    native = synthid_processor(device)
    on, off = (SynthIDRepeatPolicy(synthid_processor(device), flag) for flag in (True, False))
    logits = torch.linspace(-3, 3, 32, device=device)[None].repeat(2, 1)
    observed_difference = False
    for token in [1, 2, 3]*4:
        ids = torch.full((2, 50), token, dtype=torch.long, device=device)
        expected = native.watermarked_call(ids, logits)
        a, b = on.watermarked_call(ids, logits), off.watermarked_call(ids, logits)
        if any(not torch.equal(x, y) for x, y in zip(expected, a)):
            raise ValueError("fallback-on adapter differs from official SynthID")
        # Independent public score-update calculation on the current context.
        from synthid_text.logits_processing import update_scores
        keys, _ = native._compute_keys(native.state.context, expected[1])
        raw = update_scores(expected[2], native.get_gvals(keys))
        if not torch.equal(b[0], raw) or not torch.equal(b[1], expected[1]):
            raise ValueError("fallback-off changed SynthID beyond the fallback")
        repeat = torch.tensor(on.repeated[-1], device=device)
        if not torch.equal(a[0][~repeat], b[0][~repeat]):
            raise ValueError("policies differ on a first occurrence")
        observed_difference |= not torch.equal(a[0][repeat], b[0][repeat])
        if not torch.equal(on.processor.state.context_history, off.processor.state.context_history):
            raise ValueError("fallback toggle changed context history")
    if not observed_difference or not any(any(x) for x in on.applied) or any(any(x) for x in off.applied):
        raise ValueError("forced-repeat probe failed to exercise fallback")
    return {"passed": True, "native_on_exact": True, "unmasked_update_exact": True,
            "first_occurrence_exact": True, "history_exact": True, "forced_repeat_difference": True}


def check_sampler_policy(method, device="cpu"):
    from baseline_comparison.official import textseal_generator, gumbel_generator
    factory = textseal_generator if method == "textseal" else gumbel_generator
    # Released TextSeal's CPU PRF broadcasts only a single context against all
    # candidates; its CUDA helper supports batches. Preserve both upstream paths.
    count = 1 if method == "textseal" and torch.device(device).type == "cpu" else 2
    wrapped = SamplerRepeatPolicy(factory(), True, 12345, list(range(count)))
    native = factory()
    oracle_rng = torch.Generator(device=device).manual_seed(fallback_seed(12345, 0))
    logits = torch.linspace(-3, 3, 32, device=device)[None].repeat(count, 1)
    for step in range(5):
        context = torch.tensor([[1, 2, 3], [step+5, step+6, step+7]][:count], device=device)
        torch.manual_seed(100+step)
        expected = native.sample_next(logits, context, temperature=1., top_p=1.)
        cpu_state = torch.get_rng_state()
        gpu_state = torch.cuda.get_rng_state(device) if logits.is_cuda else None
        torch.manual_seed(100+step)
        actual = wrapped.sample_next(logits, context, temperature=1., top_p=1.)
        if not torch.equal(torch.get_rng_state(), cpu_state) or (logits.is_cuda and not torch.equal(torch.cuda.get_rng_state(device), gpu_state)):
            raise ValueError("fallback advanced the native random stream")
        if step:
            expected[0] = torch.multinomial(torch.softmax(logits[0].float(), -1), 1, generator=oracle_rng)[0]
        if not torch.equal(actual, expected) or wrapped.applied[-1] != [step > 0, False][:count]:
            raise ValueError("ordinary fallback or response-local history differs")
    return {"passed": True, "first_occurrence_exact": True, "ordinary_fallback_exact": True,
            "native_rng_unchanged": True, "repeat_mask_exact": True, "probe_batch_size": count}


PILOT = ROOT / "outputs/self_bleu_pilot/stage_a_v2"
SETUP = ROOT / "outputs/self_bleu_repeat/setup_v4"
TIMEOUTS = {"synthid": 600, "other_generators": 900, "textseal_replay": 300}
NEW_CODE = ("self_bleu/repeat.py", "self_bleu/repeat_modal.py")


def upstream_hashes():
    from synthid_text import logits_processing, hashing_function
    from baseline_comparison.textseal_completion import load_upstream_detector
    import os
    load_upstream_detector(os.environ.get("TEXTSEAL_SOURCE_ROOT"))
    from textseal.watermarking import generator, core
    return {module.__name__: sha(inspect.getsourcefile(module))
            for module in (logits_processing, hashing_function, generator, core)}


def validate(manifest, root=ROOT):
    if digest({k: v for k, v in manifest.items() if k != "id"}) != manifest["id"]:
        raise ValueError("repeat manifest identity differs")
    if (manifest["arms"] != {arm: arm_setting(arm).identity() for arm in ARMS}
            or manifest["prompt_indices"] != list(range(50)) or manifest["seeds"] != [12345, 67890]
            or manifest["length"] != 1024 or manifest["control_tokens"] != 64
            or manifest["protocol"] != "completion_only_raw_abstain_v1"):
        raise ValueError("repeat ablation scope differs")
    cost = manifest["cost"]
    if (cost["timeouts"] != TIMEOUTS or cost["resource_usd_per_second"] != RATE
            or cost["total_reserved_with_prior_usd"] != cost["previous_planning_charge_usd"] + (sum(TIMEOUTS.values())+6)*RATE + .5
            or cost["total_reserved_with_prior_usd"] > 10):
        raise ValueError("repeat ablation exceeds initial allocation")
    for name, expected in manifest["code_sha256"].items():
        if not (Path(root)/name).is_file() or sha(Path(root)/name) != expected:
            raise ValueError(f"ablation source changed: {name}")
    if sha(Path(root)/"prompts.jsonl") != manifest["prompt_sha256"]:
        raise ValueError("canonical prompts changed")


def prepare(output):
    pilot = json.loads((PILOT/"manifest.json").read_text())
    previous = json.loads((PILOT/"verification.json").read_text())
    if not previous["passed"] or previous["manifest_id"] != pilot["id"]:
        raise ValueError("Stage A reference did not pass")
    # Audit original source bytes without requiring the historical file layout.
    # Current worker sources are pinned separately and must pass strict validation.
    verify_source_hashes(pilot["code_sha256"], allow_archived=True)
    batches, _ = load_pairs()
    refs = {}
    for method in ("synthid_text", "textseal", "gumbel_max"):
        for response in (0, 1):
            batch = batches[(method, response)]
            refs[f"{method}/{response}"] = {
                "batch_id": batch["manifest"]["batch_id"],
                "prefix_sha256": [digest(r["token_ids"][:64]) for r in batch["responses"]],
                "completion_sha256": [r["completion_sha256"] for r in batch["responses"]]}
    generation_report = ROOT/"outputs/self_bleu_validation/step3-v4/generation_report.json"
    old_runtime = json.loads(generation_report.read_text())["execution"]
    names = sorted(set(PILOT_CODE) | set(NEW_CODE) |
                   {"self_bleu/generation.py", "watermark_expt.py"})
    previous_cost = previous["total_planning_charge_usd"]
    manifest = {
        "schema_version": 1, "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "supersedes": {"path": "outputs/self_bleu_repeat/setup_v3/manifest.json",
                       "sha256": sha(ROOT/"outputs/self_bleu_repeat/setup_v3/manifest.json"),
                       "reason": "Add paired full-trajectory repeat/fallback diagnostics and causal validation; generation settings unchanged."},
        "protocol": pilot["protocol"], "pilot_id": pilot["id"], "model": pilot["model"],
        "prompt_indices": list(range(50)), "prompt_sha256": sha(ROOT/"prompts.jsonl"),
        "length": 1024, "control_tokens": 64, "seeds": [12345, 67890],
        "primary_lengths": [400, 1024], "prefix_lengths": pilot["prefix_lengths"],
        "arms": {arm: arm_setting(arm).identity() for arm in ARMS}, "references": refs,
        "generation_runtime": {k: v for k, v in old_runtime.items() if k != "modal_image_id"},
        "textseal_runtime": pilot["textseal_runtime"], "analysis": pilot["analysis"],
        "analysis_contract": {
            "primary_contrasts": "new policy minus original policy, paired by prompt",
            "decision_thresholds": "original method-native nominal p < 0.001, unchanged",
            "detector_masks": "unchanged within method; SynthID context mask, TextSeal/Gumbel native tuple mask",
            "nulls": "reuse the 100 fresh ordinary-sampling slots from Stage A; no new calibration",
            "repeat_reporting": "both original and modified responses: native generation-context repeat/fallback positions, counts, fractions and first token divergence",
            "trajectory_check": "all 1024 tokens: no divergence before first repeat; no-repeat reference responses must stay identical; reconstructed traces must match recorded traces",
            "position_convention": "zero-based generated-token positions; null if event never occurs; initial context is native, not detector-mask initialization",
            "scope": "generation-policy ablation; native context initialization retained, not full implementation harmonization",
        },
        "upstream_sha256": upstream_hashes(), "code_sha256": {name: sha(ROOT/name) for name in names},
        "reference_files": {str(path.relative_to(ROOT)): sha(path) for path in
            (PILOT/"manifest.json", PILOT/"verification.json", PILOT/"inputs.json", PILOT/"summary.json",
             PILOT/"diversity.json", PILOT/"token_detection.json", PILOT/"prc_report.json",
             PILOT/"textseal_report.json", PILOT/"reused_textseal.json", generation_report)},
        "cost": {"timeouts": TIMEOUTS, "resource_usd_per_second": RATE,
                 "previous_planning_charge_usd": previous_cost,
                 "new_timeout_reservation_usd": (sum(TIMEOUTS.values())+6)*RATE,
                 "new_overhead_allowance_usd": .5,
                 "total_reserved_with_prior_usd": previous_cost+(sum(TIMEOUTS.values())+6)*RATE+.5,
                 "initial_allocation_usd": 10, "total_study_ceiling_usd": 200,
                 "pricing_source": "https://modal.com/pricing", "pricing_checked": "2026-09-18",
                 "dispatch": "three explicit sequential stages, one H100 each, retries=0; no automatic dispatch or reruns"},
    }
    manifest["id"] = digest(manifest)
    validate(manifest)
    save(output/"manifest.json", manifest)
    return {"id": manifest["id"], "status": "prepared; no GPU jobs launched",
            "new_responses": {arm: 100 for arm in ARMS}, "cost": manifest["cost"]}

def collect(setup, stage, download=False):
    manifest = json.loads((setup/"manifest.json").read_text())
    validate(manifest)
    stages = ["synthid"] if stage == "synthid" else ["synthid", "other_generators", "textseal_replay"]
    volume = None
    if download:
        import modal
        volume = modal.Volume.from_name("prc-completion-only", create_if_missing=False)
    files, reports = {}, {}
    for name in stages:
        report_path = setup/f"{name}_report.json"
        remote = f"self_bleu_repeat/{manifest['id']}/{name}"
        if download and not report_path.exists():
            report_path.write_bytes(b"".join(volume.read_file(f"{remote}/report.json")))
        report = json.loads(report_path.read_text())
        if not report["passed"] or report["manifest_id"] != manifest["id"] or report["stage"] != name:
            raise ValueError("repeat stage has not passed")
        reports[name] = report
        for relative, expected in report["files"].items():
            parts = PurePosixPath(relative)
            if parts.is_absolute() or ".." in parts.parts:
                raise ValueError("unsafe artifact path")
            path = setup/"raw"/name/relative
            if download and not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"".join(volume.read_file(f"{remote}/{relative}")))
            if sha(path) != expected:
                raise ValueError("repeat artifact checksum differs")
            files[str(path.relative_to(setup))] = expected
    return manifest, reports, files


def paired_contrast(new_bleu, old_bleu, new_detection, old_detection, draws):
    return {"self_bleu_difference": paired_interval(np.asarray(new_bleu)-old_bleu, draws),
            "tpr_difference": paired_interval(np.asarray(new_detection)-old_detection, draws)}


def synthid_repeat_masks(token_rows, *, keys):
    """Reconstruct generation-time repeats, including native warmup and sentinel.

    This is not the detector mask. Generation hashes a zero context at position
    zero, then appends generated tokens; its history is initially filled with
    zero hash sentinels. Use upstream hashing and the actual key-derived IV.
    At <= history_size steps no previously observed context has been evicted.
    No logits, model inference, sampling, or prompt tokens are needed.
    """
    from baseline_comparison.official import synthid_processor
    processor = synthid_processor("cpu", keys=keys)
    ids = torch.tensor(token_rows, dtype=torch.long)
    if ids.ndim != 2 or not 0 < ids.shape[1] <= processor.context_history_size:
        raise ValueError("repeat reconstruction requires equal lengths within native history capacity")
    width = processor.ngram_len - 1
    padded = torch.cat((torch.zeros((len(ids), width), dtype=torch.long), ids), dim=1)
    contexts = padded.unfold(1, width, 1)[:, :ids.shape[1]].reshape(-1, width)
    _, hashes = processor._compute_keys(contexts, torch.zeros((len(contexts), 1), dtype=torch.long))
    masks = []
    for row in hashes.reshape(ids.shape).tolist():
        seen, repeated = {0}, []
        for value in row:
            repeated.append(value in seen)
            seen.add(value)
        masks.append(repeated)
    return masks


def trajectory_pair(original, modified, original_repeats, modified_repeats):
    """Describe both SynthID trajectories and test the causal prefix invariant."""
    old_ids, new_ids = original["token_ids"], modified["token_ids"]
    if not len(old_ids) == len(new_ids) == len(original_repeats) == len(modified_repeats):
        raise ValueError("trajectory lengths differ")
    for name in ("prompt_index", "response_index"):
        if original[name] != modified[name]:
            raise ValueError("trajectory pairing differs")
    divergence = next((j for j, (a, b) in enumerate(zip(old_ids, new_ids)) if a != b), None)

    def describe(row, repeats, enabled):
        positions = [j for j, repeated in enumerate(repeats) if repeated]
        applied = positions if enabled else []
        return {"response_id": row["response_id"], "completion_sha256": digest(row["token_ids"]),
                "fallback_enabled": enabled, "repeat_positions": positions, "fallback_positions": applied,
                "repeat_count": len(positions), "fallback_count": len(applied),
                "first_repeat_position": next(iter(positions), None),
                "first_fallback_position": next(iter(applied), None),
                "first_token_divergence": divergence}

    before, after = describe(original, original_repeats, True), describe(modified, modified_repeats, False)
    first_repeat = before["first_repeat_position"]
    through = len(old_ids) if divergence is None else divergence + 1
    recorded = modified["generation_diagnostics"]
    checks = {
        "no_divergence_before_first_repeat": divergence is None or (first_repeat is not None and divergence >= first_repeat),
        "no_repeat_reference_stays_identical": first_repeat is not None or divergence is None,
        "first_repeat_positions_agree": first_repeat == after["first_repeat_position"],
        "repeat_history_agrees_through_divergence": original_repeats[:through] == modified_repeats[:through],
        "modified_repeat_trace_matches_reconstruction": recorded["repeated_context"] == modified_repeats,
        "modified_fallback_trace_is_off": recorded["fallback_applied"] == [False]*len(new_ids)
                                              and recorded["first_fallback_position"] is None,
    }
    return {"prompt_index": original["prompt_index"], "response_index": original["response_index"],
            "first_token_divergence": divergence, "original": before, "modified": after,
            "checks": checks, "passed": all(checks.values())}


def trajectory_summary(pairs, lengths):
    """Counts use response slots; diagnostic means are not independent trials."""
    summaries = []
    for length in lengths:
        variants = {}
        for variant in ("original", "modified"):
            repeats = [sum(p < length for p in row[variant]["repeat_positions"]) for row in pairs]
            fallbacks = [sum(p < length for p in row[variant]["fallback_positions"]) for row in pairs]
            first = [row[variant]["first_repeat_position"] for row in pairs
                     if row[variant]["first_repeat_position"] is not None and row[variant]["first_repeat_position"] < length]
            variants[variant] = {"responses": len(pairs), "responses_with_repeat": sum(n > 0 for n in repeats),
                "fraction_responses_with_repeat": sum(n > 0 for n in repeats)/len(pairs),
                "repeat_count_total": sum(repeats), "repeat_count_mean": float(np.mean(repeats)),
                "repeat_count_median": float(np.median(repeats)), "repeat_count_max": max(repeats),
                "responses_with_fallback": sum(n > 0 for n in fallbacks),
                "fraction_responses_with_fallback": sum(n > 0 for n in fallbacks)/len(pairs),
                "fallback_count_total": sum(fallbacks), "fallback_count_mean": float(np.mean(fallbacks)),
                "fallback_count_median": float(np.median(fallbacks)), "fallback_count_max": max(fallbacks),
                "first_repeat_position_median_among_affected": float(np.median(first)) if first else None}
        divergences = [r["first_token_divergence"] for r in pairs
                       if r["first_token_divergence"] is not None and r["first_token_divergence"] < length]
        summaries.append({"length": length, **variants, "pairs_with_divergence": len(divergences),
            "fraction_pairs_with_divergence": len(divergences)/len(pairs),
            "first_token_divergence_median_among_diverged": float(np.median(divergences)) if divergences else None})
    return summaries


def check_synthid_trajectories(rows, old_inputs, setup, manifest):
    """Save auditable traces before failing a causal or instrumentation check."""
    keys = manifest["arms"]["synthid_off"]["keys"]
    ordered = sorted(rows)
    original = [old_inputs[("synthid_text", *key)] for key in ordered]
    modified = [rows[key] for key in ordered]
    old_masks = synthid_repeat_masks([r["token_ids"] for r in original], keys=keys)
    new_masks = synthid_repeat_masks([r["token_ids"] for r in modified], keys=keys)
    pairs = [trajectory_pair(a, b, am, bm) for a, b, am, bm in zip(original, modified, old_masks, new_masks)]
    # Native-on prefixes have recorded GPU traces; independently verify the
    # reconstruction on them as well as the entire modified off trajectories.
    control_checks = 0
    for response in (0, 1):
        path = setup/"raw/synthid/controls"/f"synthid_off_r{response}.json"
        control = json.loads(path.read_text())
        for row in control["responses"]:
            i = ordered.index((row["prompt_index"], response))
            if row["token_ids"] != original[i]["token_ids"][:manifest["control_tokens"]]:
                raise ValueError("native control tokens differ from reference")
            expected = old_masks[i][:manifest["control_tokens"]]
            trace = row["generation_diagnostics"]
            if (trace["repeated_context"] != expected or trace["fallback_applied"] != expected
                    or trace["first_fallback_position"] != next((j for j, value in enumerate(expected) if value), None)):
                raise ValueError("native control repeat/fallback trace differs from reconstruction")
            control_checks += 1
    if control_checks != 100:
        raise ValueError("native control trajectory coverage differs")
    checks = {name: sum(row["checks"][name] for row in pairs) for name in pairs[0]["checks"]}
    report = {"passed": all(row["passed"] for row in pairs), "response_pairs": len(pairs),
        "checks_passed": checks, "native_prefix_traces_verified": control_checks,
        "position_convention": "zero-based generated-token positions; null if event never occurs",
        "original_trace_provenance": "Reconstructed from saved tokens with upstream generation-context hashing, zero-context warmup and zero-filled history; cross-checked against native GPU prefix traces.",
        "modified_trace_provenance": "Full recorded generation traces independently checked against token-based reconstruction.",
        "summaries": trajectory_summary(pairs, manifest["primary_lengths"]),
        "failed_pairs": [{"prompt_index": row["prompt_index"], "response_index": row["response_index"],
                          "checks": row["checks"]} for row in pairs if not row["passed"]]}
    compact = []
    for row in pairs:
        record = {key: row[key] for key in ("prompt_index", "response_index", "first_token_divergence", "passed")}
        record["sampling_seed"] = manifest["seeds"][row["response_index"]]
        for variant in ("original", "modified"):
            values = row[variant]
            record[variant] = {k: v for k, v in values.items() if k not in ("repeat_positions", "fallback_positions")}
            record[variant]["counts_by_prefix"] = {str(n): {
                "repeat_count": sum(p < n for p in values["repeat_positions"]),
                "fallback_count": sum(p < n for p in values["fallback_positions"])} for n in manifest["primary_lengths"]}
        compact.append(record)
    save(setup/"raw/synthid_trajectory_pairs.json", pairs)
    save(setup/"synthid_response_diagnostics.json", {"manifest_id": manifest["id"],
         "position_convention": report["position_convention"], "rows": compact})
    save(setup/"synthid_trajectory.json", report)
    if not report["passed"]:
        raise ValueError("full-trajectory causal or trace check failed; inspect synthid_trajectory.json before interpreting results")
    return report


def analyze(setup, stage, tokenizer_path, download=False):
    import torch
    import sacrebleu
    from sacrebleu.metrics import BLEU
    from tokenizers import Tokenizer
    from baseline_comparison.official import synthid_processor, official_gumbel_scores
    from baseline_comparison.scoring import deduplicated_positions, synthid_normal_test, gumbel_gamma_test
    manifest, reports, files = collect(setup, stage, download)
    if upstream_hashes() != manifest["upstream_sha256"]:
        raise ValueError("upstream token-scoring sources changed")
    for name, expected in manifest["reference_files"].items():
        if sha(ROOT/name) != expected:
            raise ValueError(f"Stage A reference changed: {name}")
    if sha(tokenizer_path) != manifest["model"]["tokenizer_sha256"] or sacrebleu.__version__ != "2.4.3":
        raise ValueError("tokenizer or Self-BLEU implementation differs")
    decoder = Tokenizer.from_file(str(tokenizer_path))
    metric = BLEU(tokenize="13a", smooth_method="exp", effective_order=True, lowercase=False)
    old_metrics = {(r["method"], r["length"], r["prompt_index"]): r for r in json.loads((PILOT/"diversity.json").read_text())["rows"]}
    old_scores = {(r["detector"], r["method"], r["prompt_index"], r["response_index"], r["length"]): r["score"] for r in score_rows(PILOT)}
    old_inputs = {(r["method"], r["prompt_index"], r["response_index"]): r for r in json.loads((PILOT/"inputs.json").read_text())}
    draws = np.random.default_rng(manifest["analysis"]["bootstrap_seed"]).integers(0, 50, (2000, 50))
    arms = ["synthid_off"] if stage == "synthid" else list(manifest["arms"])
    summaries, metric_rows, score_records, diagnostics = [], [], [], []
    trajectory = None
    ts = {r["response_id"]: r for r in reports.get("textseal_replay", {}).get("rows", [])}
    for arm in arms:
        generation_stage = "synthid" if arm == "synthid_off" else "other_generators"
        method = manifest["arms"][arm]["method"]
        rows = {}
        for response in (0, 1):
            batch = json.loads((setup/"raw"/generation_stage/"batches"/f"{arm}_r{response}.json").read_text())
            if (batch["manifest"]["setting"] != manifest["arms"][arm] or len(batch["responses"]) != 50
                    or batch["manifest"]["sampling_seed"] != manifest["seeds"][response]):
                raise ValueError("ablation generation configuration or coverage differs")
            for row in batch["responses"]:
                key = (row["prompt_index"], row["response_index"])
                if (key in rows or key[1] != response or len(row["token_ids"]) != manifest["length"]
                        or row["sampling_seed"] != manifest["seeds"][response]
                        or digest(row["token_ids"]) != row["completion_sha256"]):
                    raise ValueError("duplicate or mismatched ablation response")
                rows[key] = row
        if set(rows) != {(i, r) for i in range(50) for r in (0, 1)}:
            raise ValueError("missing ablation pair")
        if method == "synthid_text":
            trajectory = check_synthid_trajectories(rows, old_inputs, setup, manifest)
        scored = {}
        for key, row in rows.items():
            ids = row["token_ids"]
            if method == "synthid_text":
                processor = synthid_processor("cpu", keys=manifest["arms"][arm]["keys"])
                tensor = torch.tensor([ids])
                values = processor.compute_g_values(tensor)[0].numpy()
                mask = processor.compute_context_repetition_mask(tensor)[0].numpy().astype(bool)
            for length in manifest["prefix_lengths"]:
                if method == "synthid_text":
                    score = synthid_normal_test(values[:length-3][mask[:length-3]])
                elif method == "gumbel_max":
                    positions = deduplicated_positions(ids[:length])
                    score = gumbel_gamma_test(official_gumbel_scores(ids, positions))
                else:
                    result = ts[row["response_id"]]
                    if result["completion_sha256"] != row["completion_sha256"]:
                        raise ValueError("TextSeal detector scored different text")
                    raw = result["results"][str(length)]
                    if raw["completion_sha256"] != digest(ids[:length]):
                        raise ValueError("TextSeal detector prefix differs")
                    score = raw["comparison"]
                scored[(*key, length)] = score
                score_records.append({"arm": arm, "response_id": row["response_id"], "length": length, "score": score})
            old_ids = old_inputs[(method, *key)]["token_ids"]
            diagnostics.append({"arm": arm, "prompt_index": key[0], "response_index": key[1],
                "first_token_divergence": next((j for j, (a,b) in enumerate(zip(ids, old_ids)) if a != b), None),
                "first_fallback_position": row["generation_diagnostics"]["first_fallback_position"],
                "first_repeated_context_position": next((j for j, x in enumerate(row["generation_diagnostics"]["repeated_context"]) if x), None),
                "fallback_counts": {str(n): sum(row["generation_diagnostics"]["fallback_applied"][:n]) for n in manifest["primary_lengths"]}})
        for length in manifest["primary_lengths"]:
            bleu, detections = [], []
            for i in range(50):
                texts = [decoder.decode(rows[(i,r)]["token_ids"][:length], skip_special_tokens=True) for r in (0,1)]
                b = (metric.sentence_score(texts[0], [texts[1]]).score + metric.sentence_score(texts[1], [texts[0]]).score)/200
                detected = [bool(scored[(i,r,length)]["decision"]) for r in (0,1)]
                bleu.append(b); detections.append(np.mean(detected))
                metric_rows.append({"arm": arm, "length": length, "prompt_index": i, "self_bleu": b, "detected": detected})
            old_b = np.array([old_metrics[(method,length,i)]["self_bleu"] for i in range(50)])
            old_d = np.array([np.mean([old_scores[(method,method,i,r,length)]["decision"] for r in (0,1)]) for i in range(50)])
            summaries.append({"arm": arm, "length": length, "self_bleu": paired_interval(bleu,draws),
                "tpr": paired_interval(detections,draws), "detected": int(round(sum(detections)*2)),
                "new_minus_original": paired_contrast(bleu,old_b,detections,old_d,draws)})
    cost = manifest["cost"]["previous_planning_charge_usd"] + .5 + sum(r["measured_resource_usd"] for r in reports.values())
    output = {"manifest_id": manifest["id"], "stage": stage, "results": summaries,
              "bootstrap_draws_sha256": digest(draws.tolist()), "bleu_signature": str(metric.get_signature()),
              "planning_charge_usd": cost, "verified_files": files, "synthid_trajectory": trajectory,
              "limitations": ["50 paired prompt clusters; nominal thresholds, not matched empirical FPR.",
                              "All-success bootstrap intervals do not prove perfect detection.",
                              "Within-method generation-policy contrasts; method-native detector masks and context initialization retained."]}
    save(setup/f"{stage}_analysis.json", output)
    save(setup/f"raw/{stage}_metric_rows.json", metric_rows)
    save(setup/f"raw/{stage}_scores.json", score_records)
    save(setup/f"raw/{stage}_diagnostics.json", diagnostics)
    return output

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("prepare", help="Freeze a new repeat ablation request; no GPU")
    create.add_argument("--output", type=Path, default=SETUP)
    analyze_parser = commands.add_parser("analyze", help="Collect and compare repeat-policy arms; no GPU")
    analyze_parser.add_argument("--setup", type=Path, default=SETUP)
    analyze_parser.add_argument("--stage", choices=("synthid", "all"), required=True)
    analyze_parser.add_argument("--tokenizer", type=Path, required=True)
    analyze_parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    result = prepare(args.output) if args.command == "prepare" else analyze(args.setup, args.stage, args.tokenizer, args.download)
    print(json.dumps({k: v for k, v in result.items() if k != "verified_files"}, indent=2))


if __name__ == "__main__":
    main()
