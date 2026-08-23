"""Common adapter contracts, including fail-closed PRC cache reuse."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class GeneratedContinuation:
    prompt_index: int
    method: str
    seed: int
    token_ids: tuple[int, ...]
    base_token_logprobs: tuple[float, ...]
    base_entropies: tuple[float, ...]
    provenance: dict[str, Any]


class GenerationAdapter(ABC):
    method: str

    @abstractmethod
    def generate(
        self, prompt_indices: Sequence[int], seed: int, max_new_tokens: int
    ) -> list[GeneratedContinuation]:
        raise NotImplementedError


GenerationBackend = Callable[
    [Sequence[int], int, int, str], list[GeneratedContinuation]
]


class OfficialGenerationAdapter(GenerationAdapter):
    """Thin method-specific wrapper around a Modal-only official backend.

    The backend owns model I/O and invokes the pinned upstream sampler.  This
    adapter only enforces the shared continuation contract; it does not
    reimplement or alter any watermark formula.
    """

    def __init__(self, *, method: str, backend: GenerationBackend) -> None:
        self.method = str(method)
        self._backend = backend

    def generate(
        self, prompt_indices: Sequence[int], seed: int, max_new_tokens: int
    ) -> list[GeneratedContinuation]:
        indices = [int(index) for index in prompt_indices]
        outputs = self._backend(indices, int(seed), int(max_new_tokens), self.method)
        if [output.prompt_index for output in outputs] != indices:
            raise ValueError(f"{self.method} backend changed prompt ordering/coverage")
        for output in outputs:
            if output.method != self.method or output.seed != int(seed):
                raise ValueError(f"{self.method} backend returned inconsistent method/seed")
            if not (
                len(output.token_ids)
                == len(output.base_token_logprobs)
                == len(output.base_entropies)
                == int(max_new_tokens)
            ):
                raise ValueError(f"{self.method} backend returned an incomplete continuation")
        return outputs


class TextSealAdapter(OfficialGenerationAdapter):
    def __init__(self, backend: GenerationBackend) -> None:
        super().__init__(method="textseal", backend=backend)


class SynthIDTextAdapter(OfficialGenerationAdapter):
    def __init__(self, backend: GenerationBackend) -> None:
        super().__init__(method="synthid_text", backend=backend)


class GumbelMaxAdapter(OfficialGenerationAdapter):
    def __init__(self, backend: GenerationBackend) -> None:
        super().__init__(method="gumbel_max", backend=backend)


class CachedResultAdapter:
    """Read-only adapter with no generation fallback by construction."""

    def __init__(
        self,
        *,
        method: str,
        loader: Callable[[int], GeneratedContinuation | None],
        exact_length: int,
    ) -> None:
        self.method = str(method)
        self._loader = loader
        self.exact_length = int(exact_length)
        self.generation_attempts = 0

    def load(self, prompt_indices: Sequence[int]) -> list[GeneratedContinuation]:
        records = []
        for index in prompt_indices:
            record = self._loader(int(index))
            if record is None:
                raise FileNotFoundError(
                    f"{self.method} cache missing prompt {int(index)}; regeneration is disabled"
                )
            if len(record.token_ids) < self.exact_length:
                raise ValueError(
                    f"{self.method} prompt {int(index)} has {len(record.token_ids)} tokens; "
                    f"need {self.exact_length}"
                )
            records.append(record)
        return records

    def generate(self, *_args, **_kwargs):
        self.generation_attempts += 1
        raise RuntimeError(f"{self.method} is cache-only; generation is forbidden")


class OnlinePRCCachedAdapter(CachedResultAdapter):
    pass


class FixedPRCCachedAdapter(CachedResultAdapter):
    pass
