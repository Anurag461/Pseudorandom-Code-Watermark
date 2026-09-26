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
import types
from typing import Sequence
import numpy as np
import torch
from .config import CONTEXT_LENGTH, NOMINAL_FPR
from .synthid import SYNTHID_KEYS
from .scoring import _empty_test, gamma_survival, gamma_threshold, _windows_targets

TEXTSEAL_ALPHA = 0.1
TEXTSEAL_KEY_A = 42
TEXTSEAL_KEY_B = TEXTSEAL_KEY_A + 12345
TEXTSEAL_COMMIT = "c60d0d1da2e59f09a698438e218a07ee779b4616"

PROTOCOL = "completion_only_raw_abstain_v1"
AUDIT_PATH = Path(__file__).with_name("textseal_source.json")
_IMPORT_LOCK = threading.RLock()


def verify_upstream_source(source_root: str | Path | None = None) -> dict:
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
    provenance = verify_upstream_source(source_root)
    root = Path(provenance["root"])
    with _IMPORT_LOCK:
        for name in ("config", "core", "detector"):
            module = sys.modules.get(f"textseal.watermarking.{name}")
            if module is not None:
                path = Path(module.__file__).resolve()
                if path != root / "textseal" / "watermarking" / f"{name}.py":
                    raise ValueError(
                        f"Different TextSeal module already loaded: {name}"
                    )
        for name, relative in (
            ("textseal", "textseal"),
            ("textseal.watermarking", "textseal/watermarking"),
        ):
            path = str(root / relative)
            existing = sys.modules.get(name)
            if existing is not None:
                if path not in [str(Path(p).resolve()) for p in existing.__path__]:
                    raise ValueError(
                        f"Different TextSeal package already loaded: {name}"
                    )
                continue
            module = types.ModuleType(name)
            module.__path__ = [path]
            module.__package__ = name
            module.__spec__ = importlib.machinery.ModuleSpec(
                name, loader=None, is_package=True
            )
            sys.modules[name] = module
        detector = importlib.import_module("textseal.watermarking.detector")
    return (detector.TextSealDetector, provenance)


def _completion_ids(tokens: Sequence[int]) -> list[int]:
    if isinstance(tokens, (str, bytes, dict)):
        raise TypeError("pass raw completion token IDs only")
    ids = list(tokens)
    if any((type(token) is not int or token < 0 for token in ids)):
        raise ValueError("completion token IDs must be nonnegative Python integers")
    return ids


class TextSealCompletionDetector:

    def __init__(
        self,
        model,
        *,
        source_root: str | Path | None = None,
        alpha: float = TEXTSEAL_ALPHA,
    ):
        if model is None:
            raise ValueError("a model is required for completion-only entropy")
        detector_type, self.upstream_source = load_upstream_detector(source_root)

        model.eval()
        self._detector = detector_type(
            None, textseal_config(alpha=alpha), model=model, scoring_method="v2"
        )

    def detect(
        self, completion_tokens: Sequence[int], *, nominal_fpr: float = NOMINAL_FPR
    ) -> dict:
        if not math.isfinite(nominal_fpr) or not 0 < nominal_fpr < 1:
            raise ValueError("nominal_fpr must lie strictly between zero and one")
        ids = _completion_ids(completion_tokens)
        entropies = self._entropies(ids)
        return self._result(ids, entropies, nominal_fpr)

    def _entropies(self, ids):
        entropies = self._detector._compute_entropies(ids)
        if len(entropies) != max(len(ids) - 1, 0):
            raise ValueError("upstream completion entropy alignment differs")
        if any((not math.isfinite(h) or h < 0 for h in entropies)):
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


import platform


def runtime_identity(manifest):
    import importlib.metadata
    import torch

    versions = {}
    for requirement in manifest["runtime"]["dependencies"]:
        name, expected = requirement.split("==")
        actual = importlib.metadata.version(name)
        if actual.split("+")[0] != expected:
            raise ValueError(
                f"runtime dependency differs: {name}={actual}, expected {expected}"
            )
        versions[name] = actual
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name():
        raise ValueError("setup requires an H100 CUDA worker")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    return {
        "versions": versions,
        "python": platform.python_version(),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "dtype": "bfloat16",
        "attention": "eager",
        "use_cache": False,
        "tf32": False,
        "bf16_reduced_precision_reduction": False,
    }


def load_model(manifest, cache_root):
    import torch
    from transformers import AutoModelForCausalLM

    spec = manifest["model"]
    root = Path(cache_root) / spec["cache_directory"]
    files = {
        **spec["weight_files"],
        "config.json": spec["config_sha256"],
        "model.safetensors.index.json": spec["index_sha256"],
        "tokenizer.json": spec["tokenizer_sha256"],
    }
    for name, expected in files.items():
        if file_sha(root / name) != expected:
            raise ValueError(f"cached checkpoint differs: {name}")
        metadata = root / ".cache/huggingface/download" / f"{name}.metadata"
        if metadata.read_text().splitlines()[0] != spec["revision"]:
            raise ValueError(f"checkpoint revision differs: {name}")
    index = json.loads((root / "model.safetensors.index.json").read_text())
    if set(index["weight_map"].values()) != set(spec["weight_files"]):
        raise ValueError("checkpoint shard index differs")
    model = (
        AutoModelForCausalLM.from_pretrained(
            str(root),
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            local_files_only=True,
            trust_remote_code=False,
        )
        .eval()
        .to("cuda")
    )
    model.config.use_cache = False
    if model.config.model_type != "qwen3" or model.config.vocab_size != 151936:
        raise ValueError("unexpected model configuration")
    torch.cuda.synchronize()
    return model


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def textseal_config(*, watermark_type: str = "textseal", alpha: float = TEXTSEAL_ALPHA):
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("TextSeal alpha must be finite and in [0, 1]")
    from textseal.watermarking.config import WatermarkConfig

    return WatermarkConfig(
        secret_key=TEXTSEAL_KEY_A,
        secret_key_b=TEXTSEAL_KEY_B,
        ngram=CONTEXT_LENGTH,
        watermark_type=watermark_type,
        method="uniform",
        mixing_alpha=float(alpha),
        scoring_method="v2",
        depth=len(SYNTHID_KEYS),
    )


def textseal_generator(*, alpha: float = TEXTSEAL_ALPHA):
    import textseal.watermarking.generator as generator_module
    from textseal.watermarking.generator import TextSealGenerator

    def eager_fast_prf_dual(w, token_ids, sk_a, sk_b):
        from textseal.watermarking.core import _prf_dual_compiled, _weighted_sum

        original = getattr(
            _prf_dual_compiled, "_torchdynamo_orig_callable", _prf_dual_compiled
        )
        weighted = _weighted_sum(w)
        key_a = torch.tensor(sk_a, dtype=torch.long, device=w.device)
        key_b = torch.tensor(sk_b, dtype=torch.long, device=w.device)
        return original(weighted, token_ids, key_a, key_b)

    generator_module.fast_prf_dual = eager_fast_prf_dual
    config = textseal_config(alpha=alpha)
    generator = TextSealGenerator.__new__(TextSealGenerator)
    generator.wm_args = config
    generator.ngram = config.ngram
    generator.secret_key = config.secret_key
    generator.key_a = config.key_a
    generator.key_b = config.key_b
    generator.mixing_alpha = config.mixing_alpha
    return generator


def official_textseal_fused_scores(
    token_ids: Sequence[int], positions: Sequence[int], *, alpha: float = TEXTSEAL_ALPHA
) -> np.ndarray:
    from textseal.watermarking.core import prf_dual

    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("TextSeal alpha must be finite and in [0, 1]")
    if not positions:
        return np.empty(0, dtype=np.float64)
    windows, targets = _windows_targets(token_ids, positions, CONTEXT_LENGTH)
    r_a, r_b = prf_dual(windows, targets, TEXTSEAL_KEY_A, TEXTSEAL_KEY_B)
    score_a = -torch.log1p(-r_a)
    score_b = -torch.log1p(-r_b)
    fused = alpha * score_a + (1.0 - alpha) * score_b
    return fused.double().cpu().numpy()


def textseal_gamma_test(
    fused_scores: Sequence[float],
    entropies: Sequence[float],
    alpha: float = 0.1,
    nominal_fpr: float = 0.001,
) -> dict:
    scores = np.asarray(fused_scores, dtype=np.float64)
    entropy = np.asarray(entropies, dtype=np.float64)
    if scores.size == 0:
        return _empty_test("moment-matched Gamma approximation")
    if scores.shape != entropy.shape:
        raise ValueError("fused scores and entropies must align")
    if not np.all(np.isfinite(scores)) or not np.all(np.isfinite(entropy)):
        raise ValueError("TextSeal inputs must be finite")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    entropy_min = float(entropy.min())
    entropy_max = float(entropy.max())
    if entropy_max - entropy_min < 1e-06:
        entropy_min, entropy_max = (0.0, 5.0)
    ratio = np.clip((entropy - entropy_min) / (entropy_max - entropy_min), 0.0, 1.0)
    weights = 0.1 + 0.9 * ratio
    statistic = float(np.sum(weights * scores))
    routing_variance = float(alpha**2 + (1.0 - alpha) ** 2)
    mean = float(weights.sum())
    variance = float(np.sum(weights**2) * routing_variance)
    shape = mean**2 / variance
    scale = variance / mean
    p_value = gamma_survival(statistic, shape, scale)
    threshold = gamma_threshold(shape, scale, nominal_fpr)
    return {
        "statistic": statistic,
        "p_value": p_value,
        "threshold": threshold,
        "decision": bool(p_value < nominal_fpr),
        "calibration_type": "moment-matched Gamma approximation",
        "intermediate": {
            "gamma_shape": shape,
            "gamma_scale": scale,
            "routing_variance": routing_variance,
            "entropy_min": entropy_min,
            "entropy_max": entropy_max,
            "weight_sum": mean,
            "weight_squared_sum": float(np.sum(weights**2)),
            "unweighted_statistic": float(scores.sum()),
            "unweighted_p_value": gamma_survival(
                float(scores.sum()), scores.size / routing_variance, routing_variance
            ),
        },
    }
