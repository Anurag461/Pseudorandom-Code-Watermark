"""Raw-completion detection using the unmodified, hash-pinned TextSeal code.

Only token IDs and a model enter detection. No cached generation entropies,
prompt argument, synthetic prefix, or reimplementation of TextSeal scoring.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.machinery
import json
import math
from pathlib import Path
import sys
import threading
import time
import types
from typing import Sequence

from .config import NOMINAL_FPR, PREFIX_LENGTHS, TEXTSEAL_COMMIT


PROTOCOL = "completion_only_raw_abstain_v1"
AUDIT_PATH = Path(__file__).with_name("textseal_source_audit.json")
_IMPORT_LOCK = threading.RLock()


def verify_upstream_source(source_root: str | Path | None = None) -> dict:
    """Check actual upstream file bytes before importing any TextSeal module.

    source_root, when supplied, is a checkout root containing textseal/.
    Otherwise use the installed TextSeal distribution, not an arbitrary import.
    """
    if source_root is None:
        source_root = importlib.metadata.distribution("textseal").locate_file("")
    root = Path(source_root).resolve()
    audit = json.loads(AUDIT_PATH.read_text())
    if audit["pinned_commit"] != TEXTSEAL_COMMIT:
        raise ValueError("TextSeal audit and configured commit disagree")
    hashes = {}
    for relative, expected in audit["source_files_sha256"].items():
        path = root / relative
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"TextSeal upstream source differs: {relative}")
        hashes[relative] = digest
    return {"root": str(root), "commit": TEXTSEAL_COMMIT, "sha256": hashes}


def load_upstream_detector(source_root: str | Path | None = None):
    """Load original config/core/detector files without the optional UI stack.

    Upstream __init__ eagerly imports PostHocWatermarker and evaluation tools.
    Namespace packages bypass those initializers; the checked numerical source
    files execute unchanged under their original absolute module names.
    """
    provenance = verify_upstream_source(source_root)
    root = Path(provenance["root"])
    with _IMPORT_LOCK:
        # Reject an already-loaded different release rather than mixing sources.
        for name in ("config", "core", "detector"):
            module = sys.modules.get(f"textseal.watermarking.{name}")
            if module is not None:
                path = Path(module.__file__).resolve()
                if path != root / "textseal" / "watermarking" / f"{name}.py":
                    raise ValueError(f"Different TextSeal module already loaded: {name}")
        for name, relative in (("textseal", "textseal"),
                               ("textseal.watermarking", "textseal/watermarking")):
            path = str(root / relative)
            existing = sys.modules.get(name)
            if existing is not None:
                if path not in [str(Path(p).resolve()) for p in existing.__path__]:
                    raise ValueError(f"Different TextSeal package already loaded: {name}")
                continue
            module = types.ModuleType(name)
            module.__path__ = [path]
            module.__package__ = name
            module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
            sys.modules[name] = module
        detector = importlib.import_module("textseal.watermarking.detector")
    return detector.TextSealDetector, provenance


def _completion_ids(tokens: Sequence[int]) -> list[int]:
    # A mapping carrying token_ids/prompt/entropies is not a detector input.
    if isinstance(tokens, (str, bytes, dict)):
        raise TypeError("pass raw completion token IDs only")
    ids = list(tokens)
    if any(type(token) is not int or token < 0 for token in ids):
        raise ValueError("completion token IDs must be nonnegative Python integers")
    return ids


class TextSealCompletionDetector:
    """Thin token-level entry point to upstream detect, with fresh model input.

    detect_prefixes replays each requested length independently. Longest-trace
    reuse is an explicit diagnostic path, since BF16 outputs depend on shape.
    """

    def __init__(self, model, *, source_root: str | Path | None = None):
        if model is None:
            raise ValueError("a model is required for completion-only entropy")
        detector_type, self.upstream_source = load_upstream_detector(source_root)
        from .official import textseal_config

        model.eval()
        self._detector = detector_type(None, textseal_config(), model=model, scoring_method="v2")

    def detect(self, completion_tokens: Sequence[int], *, nominal_fpr: float = NOMINAL_FPR) -> dict:
        if not math.isfinite(nominal_fpr) or not 0 < nominal_fpr < 1:
            raise ValueError("nominal_fpr must lie strictly between zero and one")
        ids = _completion_ids(completion_tokens)
        # Exactly the two numerical calls made by upstream detect(text), using
        # stored token IDs in place of decode/re-encode. No cache is supplied.
        entropies = self._entropies(ids)
        return self._result(ids, entropies, nominal_fpr)

    def _entropies(self, ids):
        entropies = self._detector._compute_entropies(ids)
        if len(entropies) != max(len(ids) - 1, 0):
            raise ValueError("upstream completion entropy alignment differs")
        if any(not math.isfinite(h) or h < 0 for h in entropies):
            raise ValueError("upstream completion entropy is invalid")
        return entropies

    def _result(self, ids, entropies, nominal_fpr):
        upstream = self._detector._score_text(ids, entropies, "v2")
        weighted = upstream.get("p_value_weighted")
        return {
            "protocol": PROTOCOL,
            "upstream_commit": TEXTSEAL_COMMIT,
            "completion_length": len(ids),
            "completion_sha256": hashlib.sha256(
                json.dumps(ids, separators=(",", ":")).encode()
            ).hexdigest(),
            "entropy_count": len(entropies),
            "upstream": upstream,
            "comparison": {
                "score_field": "p_value_weighted",
                "nominal_fpr": nominal_fpr,
                "p_value": weighted,
                "decision": weighted is not None and weighted < nominal_fpr,
                "abstained": weighted is None,
            },
        }

    def detect_prefixes(self, completion_tokens: Sequence[int], *,
                        prefix_lengths=PREFIX_LENGTHS, nominal_fpr=NOMINAL_FPR,
                        validate_prefixes=False) -> dict:
        """Run unmodified upstream computation at every actual prefix length.

        Pilot validation calls upstream's public detect() independently. Its
        tokenizer only passes the stored IDs through; no decode/re-encode or
        additional model context is involved.
        """
        ids = _completion_ids(completion_tokens)
        lengths = list(prefix_lengths)
        if (not lengths or any(type(n) is not int or not 1 < n <= len(ids) for n in lengths)
                or len(set(lengths)) != len(lengths) or max(lengths) != len(ids)):
            raise ValueError("unique prefixes longer than one token must include the full length")
        if not math.isfinite(nominal_fpr) or not 0 < nominal_fpr < 1:
            raise ValueError("nominal_fpr must lie strictly between zero and one")

        class TokenIdTokenizer:
            def encode(self, text, *, add_special_tokens):
                if add_special_tokens:
                    raise ValueError("special tokens are forbidden")
                return list(map(int, text.split()))

        entropies, results, checks = {}, {}, {}
        production_seconds = 0.0
        for n in sorted(lengths):
            prefix = ids[:n]
            started = time.monotonic()
            h = self._entropies(prefix)
            results[str(n)] = self._result(prefix, h, nominal_fpr)
            entropies[str(n)] = h
            production_seconds += time.monotonic() - started
            if validate_prefixes:
                reference = type(self._detector)(TokenIdTokenizer(), self._detector.wm_config,
                                                model=self._detector.model, scoring_method="v2")
                original_compute, captured = reference._compute_entropies, []

                def observe(tokens):
                    value = original_compute(tokens)
                    captured.append(value)
                    return value

                reference._compute_entropies = observe
                expected = reference.detect(" ".join(map(str, prefix)))
                if len(captured) != 1:
                    raise ValueError("unexpected upstream reference entropy call count")
                checks[str(n)] = {
                    "entropy_exact": h == captured[0],
                    "upstream_result_exact": results[str(n)]["upstream"] == expected,
                    "decision_equal": results[str(n)]["comparison"]["decision"] ==
                                      (expected.get("p_value_weighted") is not None and expected["p_value_weighted"] < nominal_fpr),
                }
        return {
            "protocol": PROTOCOL, "prefix_strategy": "direct", "completion_length": len(ids),
            "completion_sha256": results[str(len(ids))]["completion_sha256"],
            "entropies_by_prefix": entropies, "results": results,
            "production_seconds": production_seconds,
            "validation": {"performed": bool(validate_prefixes),
                           "reference": "upstream public detect on identical raw IDs at each length",
                           "passed": all(c["entropy_exact"] and c["upstream_result_exact"] for c in checks.values()) if validate_prefixes else None,
                           "prefixes": checks},
        }

    def detect_prefixes_reusing_longest(self, completion_tokens: Sequence[int], *,
                        prefix_lengths=PREFIX_LENGTHS, nominal_fpr=NOMINAL_FPR,
                        validate_prefixes=False) -> dict:
        """One longest-prefix replay; each prefix calls upstream scoring afresh.

        This accepts no externally supplied entropy. validate_prefixes adds
        direct model passes for pilot checks and reports exact equality.
        """
        ids = _completion_ids(completion_tokens)
        lengths = list(prefix_lengths)
        if (not lengths or any(type(n) is not int or not 1 <= n <= len(ids) for n in lengths)
                or len(set(lengths)) != len(lengths) or max(lengths) != len(ids)):
            raise ValueError("unique positive prefixes must include the full completion length")
        if not math.isfinite(nominal_fpr) or not 0 < nominal_fpr < 1:
            raise ValueError("nominal_fpr must lie strictly between zero and one")
        lengths.sort()
        started = time.monotonic()
        entropies = self._entropies(ids)
        results = {str(n): self._result(ids[:n], entropies[:n-1], nominal_fpr) for n in lengths}
        production_seconds = time.monotonic() - started
        checks = {}
        if validate_prefixes:
            for n in lengths[:-1]:
                direct = self._entropies(ids[:n])
                expected = self._result(ids[:n], direct, nominal_fpr)
                checks[str(n)] = {
                    "entropy_exact": direct == entropies[:n-1],
                    "max_abs_entropy_difference": max((abs(a-b) for a, b in zip(direct, entropies)), default=0.0),
                    "upstream_result_exact": expected["upstream"] == results[str(n)]["upstream"],
                    "decision_equal": expected["comparison"]["decision"] == results[str(n)]["comparison"]["decision"],
                }
        return {
            "protocol": PROTOCOL, "prefix_strategy": "reuse", "completion_length": len(ids),
            "completion_sha256": results[str(len(ids))]["completion_sha256"],
            "entropies_2_to_T": entropies, "results": results,
            "production_seconds": production_seconds,
            "validation": {
                "performed": bool(validate_prefixes),
                "passed": all(c["entropy_exact"] and c["upstream_result_exact"] for c in checks.values()) if validate_prefixes else None,
                "prefixes": checks,
            },
        }
