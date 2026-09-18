"""Local experiment identities; no model loading, Modal calls or file writes."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess

from .config import (
    GENERATION_SETTINGS, GUMBEL_KEY, MODEL_ID, PRIMARY_SEED, SECONDARY_SEED,
    SYNTHID_KEYS, TEXTSEAL_KEY_A, TEXTSEAL_KEY_B,
)

PROTOCOL = "completion_only_raw_abstain_v1"
REFERENCE_PATH = Path(__file__).with_name("self_bleu_reference.json")
# Extend the historical ten keys once, independently of any experiment output.
# Each added key is the first 31 bits of SHA256(domain + zero-based layer index).
SYNTHID_KEY_DOMAIN = "prc-self-bleu/synthid-key-bank/v1/"
SYNTHID_KEY_BANK = SYNTHID_KEYS + tuple(
    int.from_bytes(hashlib.sha256(f"{SYNTHID_KEY_DOMAIN}{i}".encode()).digest()[:4], "big") >> 1
    for i in range(len(SYNTHID_KEYS), 30)
)
assert len(set(SYNTHID_KEY_BANK)) == 30
SAMPLING_SEEDS = (PRIMARY_SEED, SECONDARY_SEED)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def validate_seed(seed):
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be a nonnegative 63-bit integer")
    return seed


def verify_reference(root: Path | None = None) -> dict:
    """Verify committed source blobs and local artifact records, read-only.

    Current source may contain deliberate study changes. Verify the frozen git
    blobs, not equality of modified working files to the earlier implementation.
    The historical result/provenance records must still match byte for byte.
    """
    root = Path(root) if root is not None else REFERENCE_PATH.parent.parent
    reference = json.loads((root / "baseline_comparison/self_bleu_reference.json").read_text())
    if reference["protocol"] != PROTOCOL:
        raise ValueError("reference detector protocol differs")
    for name, expected in reference["source_sha256"].items():
        blob = subprocess.check_output(
            ["git", "show", f"{reference['reference_commit']}:{name}"], cwd=root,
        )
        if hashlib.sha256(blob).hexdigest() != expected:
            raise ValueError(f"frozen source differs: {name}")
    for name, expected in reference["artifact_record_sha256"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"frozen artifact record differs: {name}")
    return reference


@dataclass(frozen=True)
class StudySetting:
    method: str
    eta: float = .05
    key_seed: int = PRIMARY_SEED
    alpha: float = .1
    depth: int = 10

    def __post_init__(self):
        if self.method not in {"online_prc", "textseal", "synthid_text", "gumbel_max", "null"}:
            raise ValueError("unknown study method")
        validate_seed(self.key_seed)
        if not math.isfinite(self.eta) or not 0 <= self.eta < .5:
            raise ValueError("PRC eta must be in [0, .5)")
        if not math.isfinite(self.alpha) or not 0 <= self.alpha <= .5:
            raise ValueError("study TextSeal alpha must be in [0, .5]")
        if type(self.depth) is not int or not 1 <= self.depth <= len(SYNTHID_KEY_BANK):
            raise ValueError("SynthID depth must be an integer in [1, 30]")

    @property
    def synthid_keys(self):
        return SYNTHID_KEY_BANK[:self.depth]

    def online_key(self):
        from online_prc import OnlinePRCKey

        if self.method != "online_prc":
            raise ValueError("only PRC has an online key")
        return OnlinePRCKey.from_seed(self.key_seed, check_weight=3, noise_rate=self.eta,
                                      row_rate_numerator=99, row_rate_denominator=100)

    def identity(self):
        value = {"method": self.method}
        if self.method == "online_prc":
            value.update(eta=self.eta, key_seed=self.key_seed, check_weight=3,
                         row_rate=[99, 100], key_fingerprint=self.online_key().fingerprint)
        elif self.method == "textseal":
            value.update(alpha=self.alpha, keys=[TEXTSEAL_KEY_A, TEXTSEAL_KEY_B],
                         ngram=3, scoring_method="v2", score_field="p_value_weighted")
        elif self.method == "synthid_text":
            value.update(depth=self.depth, keys=list(self.synthid_keys), ngram_len=4,
                         key_bank_domain=SYNTHID_KEY_DOMAIN)
        elif self.method == "gumbel_max":
            value.update(key=GUMBEL_KEY, ngram=3)
        return value

    @property
    def fingerprint(self):
        return digest(self.identity())


def pilot_settings(stage="A"):
    if stage == "A":
        return tuple(StudySetting(method) for method in
                     ("online_prc", "textseal", "synthid_text", "gumbel_max", "null"))
    if stage == "B":
        return (StudySetting("textseal", alpha=0), StudySetting("textseal", alpha=.5),
                StudySetting("synthid_text", depth=2), StudySetting("synthid_text", depth=20))
    if stage == "depth30":
        return (StudySetting("synthid_text", depth=30),)
    raise ValueError("stage must be A, B or depth30")


def batch_manifest(setting, prompt_indices, prompts, *, sampling_seed, response_index,
                   execution, max_new_tokens=1024):
    """Separate watermark identity from sample identity and bind batch geometry.

    `execution` must describe the actual loaded runtime. No frozen historical
    runtime is asserted merely because a caller passes the same model name.
    """
    validate_seed(sampling_seed)
    if type(response_index) is not int or response_index not in (0, 1):
        raise ValueError("response_index must be 0 or 1")
    if type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer")
    indices, prompts = list(prompt_indices), [list(row) for row in prompts]
    if (not indices or len(indices) != len(prompts) or len(set(indices)) != len(indices)
            or any(type(i) is not int or not 0 <= i < 500 for i in indices)
            or any(len(row) != 50 or any(type(t) is not int or not 0 <= t < 151936 for t in row)
                   for row in prompts)):
        raise ValueError("expected unique canonical prompt indices and 50-token prompts")
    if not isinstance(execution, dict) or not execution:
        raise ValueError("execution identity is required")
    reference = json.loads(REFERENCE_PATH.read_text())
    payload = {
        "schema_version": 1, "setting": setting.identity(), "setting_sha256": setting.fingerprint,
        "sampling_seed": sampling_seed, "response_index": response_index,
        "prompt_indices": indices, "prompt_sha256": [digest(row) for row in prompts],
        "batch_size": len(indices), "generation": {**GENERATION_SETTINGS, "max_new_tokens": max_new_tokens},
        "model_id": MODEL_ID, "model_revision": reference["model_revision"],
        "detector_protocol": PROTOCOL, "reference_commit": reference["reference_commit"],
        "reference_sha256": hashlib.sha256(REFERENCE_PATH.read_bytes()).hexdigest(),
        "execution": execution,
    }
    payload["batch_id"] = digest(payload)
    payload["namespace"] = f"self_bleu_v1/{payload['batch_id']}"
    payload["response_ids"] = [f"{payload['batch_id']}/p{i:04d}/r{response_index}" for i in indices]
    return payload
