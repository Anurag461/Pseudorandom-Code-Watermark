from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from baselines.config import (
    GENERATION_SETTINGS,
    GUMBEL_KEY,
    MODEL_ID,
    PRIMARY_SEED,
    SECONDARY_SEED,
    SYNTHID_KEYS,
    TEXTSEAL_KEY_A,
    TEXTSEAL_KEY_B,
)

PROTOCOL = "completion_only_raw_abstain_v1"
REFERENCE_PATH = Path(__file__).with_name("reference.json")
SYNTHID_KEY_DOMAIN = "prc-self-bleu/synthid-key-bank/v1/"
SYNTHID_KEY_BANK = SYNTHID_KEYS + tuple(
    (
        int.from_bytes(
            hashlib.sha256(f"{SYNTHID_KEY_DOMAIN}{i}".encode()).digest()[:4], "big"
        )
        >> 1
        for i in range(len(SYNTHID_KEYS), 30)
    )
)
assert len(set(SYNTHID_KEY_BANK)) == 30
SAMPLING_SEEDS = (PRIMARY_SEED, SECONDARY_SEED)


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def validate_seed(seed):
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be a nonnegative 63-bit integer")
    return seed


@dataclass(frozen=True)
class StudySetting:
    method: str
    eta: float = 0.05
    key_seed: int = PRIMARY_SEED
    alpha: float = 0.1
    depth: int = 10

    def __post_init__(self):
        if self.method not in {
            "online_prc",
            "textseal",
            "synthid_text",
            "gumbel_max",
            "null",
        }:
            raise ValueError("unknown study method")
        validate_seed(self.key_seed)
        if not math.isfinite(self.eta) or not 0 <= self.eta < 0.5:
            raise ValueError("PRC eta must be in [0, .5)")
        if not math.isfinite(self.alpha) or not 0 <= self.alpha <= 0.5:
            raise ValueError("study TextSeal alpha must be in [0, .5]")
        if type(self.depth) is not int or not 1 <= self.depth <= len(SYNTHID_KEY_BANK):
            raise ValueError("SynthID depth must be an integer in [1, 30]")

    @property
    def synthid_keys(self):
        return SYNTHID_KEY_BANK[: self.depth]

    def online_key(self):
        from prc_watermark.prc import OnlinePRCKey

        if self.method != "online_prc":
            raise ValueError("only PRC has an online key")
        return OnlinePRCKey.from_seed(
            self.key_seed,
            check_weight=3,
            noise_rate=self.eta,
            row_rate_numerator=99,
            row_rate_denominator=100,
        )

    def identity(self):
        value = {"method": self.method}
        if self.method == "online_prc":
            value.update(
                eta=self.eta,
                key_seed=self.key_seed,
                check_weight=3,
                row_rate=[99, 100],
                key_fingerprint=self.online_key().fingerprint,
            )
        elif self.method == "textseal":
            value.update(
                alpha=self.alpha,
                keys=[TEXTSEAL_KEY_A, TEXTSEAL_KEY_B],
                ngram=3,
                scoring_method="v2",
                score_field="p_value_weighted",
            )
        elif self.method == "synthid_text":
            value.update(
                depth=self.depth,
                keys=list(self.synthid_keys),
                ngram_len=4,
                key_bank_domain=SYNTHID_KEY_DOMAIN,
            )
        elif self.method == "gumbel_max":
            value.update(key=GUMBEL_KEY, ngram=3)
        return value

    @property
    def fingerprint(self):
        return digest(self.identity())


def batch_manifest(
    setting,
    prompt_indices,
    prompts,
    *,
    sampling_seed,
    response_index,
    execution,
    max_new_tokens=1024,
):
    validate_seed(sampling_seed)
    if type(response_index) is not int or response_index not in (0, 1):
        raise ValueError("response_index must be 0 or 1")
    if type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer")
    indices, prompts = (list(prompt_indices), [list(row) for row in prompts])
    if (
        not indices
        or len(indices) != len(prompts)
        or len(set(indices)) != len(indices)
        or any((type(i) is not int or not 0 <= i < 500 for i in indices))
        or any(
            (
                len(row) != 50
                or any((type(t) is not int or not 0 <= t < 151936 for t in row))
                for row in prompts
            )
        )
    ):
        raise ValueError(
            "expected unique canonical prompt indices and 50-token prompts"
        )
    if not isinstance(execution, dict) or not execution:
        raise ValueError("execution identity is required")
    reference = json.loads(REFERENCE_PATH.read_text())
    payload = {
        "schema_version": 1,
        "setting": setting.identity(),
        "setting_sha256": setting.fingerprint,
        "sampling_seed": sampling_seed,
        "response_index": response_index,
        "prompt_indices": indices,
        "prompt_sha256": [digest(row) for row in prompts],
        "batch_size": len(indices),
        "generation": {**GENERATION_SETTINGS, "max_new_tokens": max_new_tokens},
        "model_id": MODEL_ID,
        "model_revision": reference["model_revision"],
        "detector_protocol": PROTOCOL,
        "reference_commit": reference["reference_commit"],
        "reference_sha256": hashlib.sha256(REFERENCE_PATH.read_bytes()).hexdigest(),
        "execution": execution,
    }
    payload["batch_id"] = digest(payload)
    payload["namespace"] = f"self_bleu_v1/{payload['batch_id']}"
    payload["response_ids"] = [
        f"{payload['batch_id']}/p{i:04d}/r{response_index}" for i in indices
    ]
    return payload
