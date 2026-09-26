import inspect
import types
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import torch
from .config import StudySetting

FALLBACK_SEED_DOMAIN = "prc-self-bleu/repeat-fallback/v1"


@dataclass(frozen=True)
class RepeatSetting(StudySetting):
    repeat_fallback: bool = True

    def __post_init__(self):
        super().__post_init__()
        if (
            self.method not in {"synthid_text", "textseal", "gumbel_max"}
            or type(self.repeat_fallback) is not bool
        ):
            raise ValueError(
                "repeat ablation applies only to the three context-based methods"
            )
        if self.depth != 10 or self.alpha != 0.1:
            raise ValueError("repeat ablation fixes depth 10 and alpha .1")

    def identity(self):
        return {
            **super().identity(),
            "repeat_fallback": self.repeat_fallback,
            "repeat_scope": "within one response; fresh history for each response",
            "context_initialization": (
                "upstream zero context"
                if self.method == "synthid_text"
                else "last three prompt tokens"
            ),
            "fallback_rng": (
                "existing multinomial stream"
                if self.method == "synthid_text"
                else FALLBACK_SEED_DOMAIN
            ),
        }


def fallback_seed(sampling_seed, prompt_index):
    data = f"{FALLBACK_SEED_DOMAIN}/{sampling_seed}/{prompt_index}".encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big") >> 1


class SynthIDRepeatPolicy:

    def __init__(self, processor, enabled):
        if (
            processor._num_leaves != 2
            or processor.skip_first_ngram_calls
            or processor.apply_top_k
        ):
            raise ValueError("unexpected SynthID configuration")
        (self.processor, self.enabled) = (processor, enabled)
        (self.repeated, self.applied) = ([], [])
        self._updated = []
        original = inspect.unwrap(type(processor).watermarked_call)
        upstream_update = original.__globals__["update_scores"]

        def capture(*args, **kwargs):
            value = upstream_update(*args, **kwargs)
            self._updated.append(value)
            return value

        namespace = {**original.__globals__, "update_scores": capture}
        private = types.FunctionType(
            original.__code__,
            namespace,
            original.__name__,
            original.__defaults__,
            original.__closure__,
        )
        private.__kwdefaults__ = original.__kwdefaults__
        self._call = types.MethodType(private, processor)

    @torch.no_grad()
    def watermarked_call(self, ids, scores):
        processor = self.processor
        old = (
            torch.zeros(
                (scores.shape[0], processor.context_history_size),
                dtype=torch.long,
                device=scores.device,
            )
            if processor.state is None
            else processor.state.context_history.clone()
        )
        self._updated.clear()
        native = self._call(ids, scores)
        if len(self._updated) != 1:
            raise ValueError("upstream SynthID score-update contract changed")
        current_hash = processor.state.context_history[:, :1]
        repeated = (old == current_hash).any(dim=1)
        self.repeated.append(repeated.cpu().tolist())
        self.applied.append(
            (repeated if self.enabled else torch.zeros_like(repeated)).cpu().tolist()
        )
        return (native[0] if self.enabled else self._updated[0], native[1], native[2])


class SamplerRepeatPolicy:

    def __init__(self, sampler, enabled, sampling_seed, prompt_indices):
        (self.sampler, self.enabled) = (sampler, enabled)
        self.seeds = [fallback_seed(sampling_seed, i) for i in prompt_indices]
        self.seen = [set() for _ in prompt_indices]
        self.generators = None
        (self.repeated, self.applied) = ([], [])

    @torch.no_grad()
    def sample_next(self, logits, context, *, temperature, top_p):
        if temperature != 1 or top_p != 1 or context.shape != (len(self.seen), 3):
            raise ValueError(
                "fallback ablation requires original temperature/top-p and three-token contexts"
            )
        if self.generators is None:
            self.generators = [
                torch.Generator(device=logits.device).manual_seed(seed)
                for seed in self.seeds
            ]
        repeated = []
        for seen, row in zip(self.seen, context.cpu().tolist()):
            key = tuple(row)
            repeated.append(key in seen)
            seen.add(key)
        native = self.sampler.sample_next(
            logits, context, temperature=temperature, top_p=top_p
        )
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
    from baselines import generation as runner

    names = ("synthid_processor", "textseal_generator", "gumbel_generator")
    original = {name: getattr(runner, name) for name in names}
    created = []

    def synthid(*args, **kwargs):
        policy = SynthIDRepeatPolicy(
            original["synthid_processor"](*args, **kwargs), setting.repeat_fallback
        )
        created.append(policy)
        return policy

    def sampler(name, *args, **kwargs):
        policy = SamplerRepeatPolicy(
            original[name](*args, **kwargs),
            setting.repeat_fallback,
            sampling_seed,
            prompt_indices,
        )
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


def generate_repeat_batch(
    model,
    prompts,
    indices,
    *,
    setting,
    sampling_seed,
    response_index,
    execution,
    max_new_tokens=1024,
    device="cuda",
):
    from .generation import generate_response_batch

    if not isinstance(setting, RepeatSetting):
        raise ValueError("explicit repeat policy required")
    if max_new_tokens > 1024:
        raise ValueError("ablation exceeds SynthID history capacity")
    execution = {
        **execution,
        "repeat_adapter_sha256": hashlib.sha256(
            open(__file__, "rb").read()
        ).hexdigest(),
    }
    with install_policy(setting, sampling_seed, indices) as policies:
        batch = generate_response_batch(
            model,
            prompts,
            indices,
            setting=setting,
            sampling_seed=sampling_seed,
            response_index=response_index,
            execution=execution,
            max_new_tokens=max_new_tokens,
            device=device,
        )
    policy = policies[0]
    if len(policy.repeated) != max_new_tokens:
        raise ValueError("missing generation-policy trace")
    for i, row in enumerate(batch["responses"]):
        repeated = [bool(step[i]) for step in policy.repeated]
        applied = [bool(step[i]) for step in policy.applied]
        row["generation_diagnostics"].update(
            repeated_context=repeated,
            fallback_applied=applied,
            first_fallback_position=next(
                (j for (j, value) in enumerate(applied) if value), None
            ),
        )
    if "synthid_official_smoke_reference" in batch["telemetry"]:
        batch["telemetry"]["same_policy_single_row_reference"] = batch["telemetry"].pop(
            "synthid_official_smoke_reference"
        )
    batch["manifest"][
        "namespace"
    ] = f"self_bleu_repeat_v1/{batch['manifest']['batch_id']}"
    return batch
