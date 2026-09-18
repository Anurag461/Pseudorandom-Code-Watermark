"""Generation controls for the study, using already-loaded project models.

No Modal entrypoint or dispatch is provided here. Target-GPU validation and
cache-reuse decisions precede the later pilot launch. Generated model traces
are diagnostics; completion-only detection must replay the saved token IDs.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from .self_bleu_config import StudySetting, batch_manifest, digest


def implementation_identity():
    root = Path(__file__).resolve().parents[1]
    names = ("baseline_comparison/self_bleu_config.py", "baseline_comparison/self_bleu_generation.py",
             "baseline_comparison/official.py", "baseline_comparison/comparison_runner.py",
             "baseline_comparison/config.py", "watermark_expt.py", "online_prc.py", "qwen.py")
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def generate_response_batch(model, prompts, prompt_indices, *, setting: StudySetting,
                            sampling_seed: int, response_index: int, execution: dict,
                            max_new_tokens=1024, device="cuda", prc_artifact=None,
                            online_sampler=None):
    """Generate one replicate without changing a key or writing historical caches.

    For PRC pass the original artifact and the already-loaded runtime's
    `generate_batch_and_collect_online`. That function already accepts an
    independent document-seed stream; we do not route sampling_seed through
    modal_run's legacy experiment_seed/artifact builder.
    """
    if not isinstance(execution, dict) or not execution:
        raise ValueError("actual execution identity is required")
    prompts, indices = [list(row) for row in prompts], list(prompt_indices)
    runtime = {**execution, "device": str(device), "code_sha256": implementation_identity()}
    manifest = batch_manifest(setting, indices, prompts, sampling_seed=sampling_seed,
                              response_index=response_index, execution=runtime,
                              max_new_tokens=max_new_tokens)
    document_seeds = None
    if setting.method == "online_prc":
        from detectors import tensor_sha256
        from online_prc import OnlinePRCKey, derive_document_seed

        if prc_artifact is None or online_sampler is None:
            raise ValueError("PRC requires its original artifact and loaded online sampler")
        key = OnlinePRCKey.from_dict(prc_artifact["online_key"])
        if key != setting.online_key():
            raise ValueError("artifact key/configuration differs from the requested fixed key")
        partition = prc_artifact["partition"]
        if (not isinstance(partition, torch.Tensor) or partition.ndim != 2 or partition.shape[0] != 2
                or not torch.all((partition == 0) | (partition == 1))
                or not torch.all(partition.sum(dim=0) == 1)):
            raise ValueError("invalid two-bucket PRC partition")
        if any(max(row) >= partition.shape[1] for row in prompts):
            raise ValueError("prompt token outside PRC partition vocabulary")
        document_seeds = [derive_document_seed(sampling_seed, index) for index in indices]
        # Partition identity is part of the request namespace as well as the
        # saved key identity, so a different artifact cannot alias this batch.
        runtime["partition_sha256"] = tensor_sha256(partition)
        manifest = batch_manifest(setting, indices, prompts, sampling_seed=sampling_seed,
                                  response_index=response_index, execution=runtime,
                                  max_new_tokens=max_new_tokens)
        tokens, _prompted_probabilities, details = online_sampler(
            model, torch.tensor(prompts, dtype=torch.long, device=device), max_new_tokens,
            key, partition, watermark=True, return_trace_details=True,
            document_seeds=document_seeds, kv_cache_implementation="static",
        )
        sequences = [
            {"token_ids": list(map(int, tokens[row].tolist())),
             "base_token_logprobs": details["base_token_logprob"][row].tolist(),
             "prc_codeword_bits": details["prc_codeword_bits"][row].tolist()}
            for row in range(len(indices))
        ]
        telemetry = {"online_sampler_version": details["online_sampler_version"],
                     "kv_cache_implementation": details["kv_cache_implementation"]}
    else:
        if prc_artifact is not None or online_sampler is not None:
            raise ValueError("PRC artifact/sampler is only valid for online_prc")
        from .comparison_runner import generate_method

        sequences, telemetry = generate_method(
            model, prompts, method=setting.method, seed=sampling_seed,
            textseal_alpha=setting.alpha, synthid_keys=setting.synthid_keys,
            max_new_tokens=max_new_tokens, device=device,
        )
    if len(sequences) != len(indices):
        raise ValueError("generation returned the wrong response count")
    responses = []
    for row, sequence in enumerate(sequences):
        tokens = sequence["token_ids"]
        if len(tokens) != max_new_tokens or any(type(t) is not int or t < 0 for t in tokens):
            raise ValueError("generation returned invalid completion tokens")
        responses.append({
            "response_id": manifest["response_ids"][row], "prompt_index": indices[row],
            "prompt_sha256": manifest["prompt_sha256"][row], "response_index": response_index,
            "setting_sha256": setting.fingerprint, "sampling_seed": sampling_seed,
            "document_seed": None if document_seeds is None else document_seeds[row],
            "token_ids": tokens, "completion_sha256": digest(tokens),
            "generation_diagnostics": {k: v for k, v in sequence.items() if k != "token_ids"},
        })
    return {"manifest": manifest, "responses": responses, "telemetry": telemetry}
