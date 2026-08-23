"""Strict, JSON-safe shared prompt-level result schema."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from typing import Any

from .config import SCHEMA_VERSION


@dataclass(frozen=True)
class PromptLevelResult:
    prompt_index: int
    prompt_id: str
    prompt_fingerprint: str
    sample_type: str
    method: str
    method_configuration: dict[str, Any]
    model_id: str
    model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    generation_seed: int
    key_seed: int | None
    key_domain: str
    generation_settings: dict[str, Any]
    generated_token_count: int
    generated_token_hash: str
    prefix_length: int
    deduplicated_sample_count: int
    statistic: float
    p_value: float
    calibration_type: str
    threshold: float
    decision: bool
    base_model_nll: float
    base_model_perplexity: float
    output_length: int
    repetition_rate: float
    repetition_metric: str
    distinct_2: float
    distinct_3: float
    source_repository_url: str
    source_repository_commit: str
    prc_code_fingerprint: str
    integration_code_fingerprint: str
    image_fingerprint: str
    artifact_fingerprint: str
    cache_or_generation_provenance: dict[str, Any]
    runtime_seconds: float
    intermediate_values: dict[str, Any] = field(default_factory=dict)
    diversity_fields: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unexpected schema version")
        if self.prompt_index < 0 or not self.prompt_id or not self.prompt_fingerprint:
            raise ValueError("prompt identity is incomplete")
        if self.sample_type not in {"watermarked", "null", "seed_validation"}:
            raise ValueError("invalid sample_type")
        if self.generated_token_count <= 0 or self.output_length != self.generated_token_count:
            raise ValueError("generated token counts are inconsistent")
        if not 0 < self.prefix_length <= self.generated_token_count:
            raise ValueError("prefix length is outside the generated continuation")
        if not 0 <= self.deduplicated_sample_count <= self.prefix_length:
            raise ValueError("invalid deduplicated sample count")
        if not 0.0 <= self.p_value <= 1.0:
            raise ValueError("p-value is outside [0, 1]")
        for name in (
            "statistic",
            "base_model_nll",
            "base_model_perplexity",
            "repetition_rate",
            "distinct_2",
            "distinct_3",
            "runtime_seconds",
        ):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} is not finite")
        if not (math.isfinite(self.threshold) or math.isinf(self.threshold)):
            raise ValueError("threshold must be finite or infinite")
        if not self.model_revision or not self.tokenizer_revision:
            raise ValueError("model and tokenizer revisions must be explicit")
        if not self.source_repository_commit or not self.image_fingerprint:
            raise ValueError("source/image provenance is incomplete")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, allow_nan=False)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PromptLevelResult":
        result = cls(**payload)
        result.validate()
        return result
