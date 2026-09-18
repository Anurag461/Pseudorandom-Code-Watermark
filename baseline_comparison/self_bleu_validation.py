"""Local preparation and pure checks for the bounded step-3 GPU validation."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

from .config import ONLINE_PRC_SOURCE_TAG, PREFIX_LENGTHS
from .self_bleu_config import digest, verify_reference

ROOT = Path(__file__).resolve().parents[1]
RUN = "qwen3-8b-batch50-validation-20260823-v1"
RATE = .001097 + 4 * .0000131 + 64 * .00000222


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != data:
            raise FileExistsError(f"immutable artifact already exists: {path}")
        return
    path.write_text(data)


def load_online_sampler(path, *, device):
    """Execute the unchanged function without the notebook's model-load side effects."""
    import numpy as np
    import torch
    from qwen import make_kv_cache, normalize_kv_cache_implementation, kv_cache_version
    tree = ast.parse(Path(path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "generate_batch_and_collect_online")
    namespace = dict(torch=torch, np=np, device=torch.device(device),
                     make_kv_cache=make_kv_cache, kv_cache_version=kv_cache_version,
                     normalize_kv_cache_implementation=normalize_kv_cache_implementation)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[node.name]


def audit_sources(cache):
    import torch
    from .comparison_runner import _numpy_pickle_compat
    _numpy_pickle_compat()
    reference = verify_reference()
    preflight = json.loads((ROOT / "outputs/comparison_redetect/preflight/preflight.json").read_text())
    known = {r["path"]: r["sha256"] for r in preflight["source_files"]}
    artifact = reference["prc_generation_artifact"]
    known[artifact["path"]] = artifact["sha256"]
    prompts = [json.loads(line)["prompt_tokens"] for line in (ROOT / "prompts.jsonl").read_text().splitlines()]
    if sha(ROOT / "prompts.jsonl") != reference["prompt_sha256"]:
        raise ValueError("canonical prompt file changed")
    files = []
    def read(relative):
        if sha(cache / relative) != known[relative]:
            raise ValueError(f"source bytes changed: {relative}")
        files.append({"path": relative, "sha256": known[relative]})
        return torch.load(cache / relative, map_location="cpu", weights_only=False)
    key = read(artifact["path"])
    if key["prompt_ids_list"][:50] != prompts[:50]:
        raise ValueError("PRC artifact prompt order differs")
    shard = read(f"controlled_baseline_full/{RUN}/generated/shard_00.pt")
    if shard["prompt_indices"] != list(range(50)) or shard["generation_batch_size"] != 50 or shard["seed"] != 12345:
        raise ValueError("historical baseline geometry or seed differs")
    expected = {m: [digest(list(map(int, r["token_ids"]))) for r in shard["sequences"][m]]
                for m in ("textseal", "synthid_text", "gumbel_max")}
    for method, prefix in (("online_prc", f"{ONLINE_PRC_SOURCE_TAG}/wm/wm"),
                           ("null", "_nulls/qwen3_8b_base/T13088/null")):
        expected[method] = []
        for i in range(50):
            record = read(f"{prefix}_{i:04d}.pt")
            if record["prompt_idx"] != i or list(map(int, record["prompt_token_ids"])) != prompts[i]:
                raise ValueError(f"prompt mismatch: {method}/{i}")
            if method == "online_prc" and record["online_key_sha256"] != artifact["online_key_fingerprint"]:
                raise ValueError("cached PRC key differs")
            expected[method].append(digest(list(map(int, record["tokens"][:1024]))))
    return {"passed": True, "files": files, "expected_completion_sha256": expected,
            "prompts": 50, "baseline_batch_size": 50, "baseline_seed": 12345,
            "prc_generation_horizon": 1280, "null_generation_horizon": 13088,
            "reuse_status": "Input identity verified; exact GPU reproduction still required."}


def validate_manifest(manifest, root):
    identity = {k: v for k, v in manifest.items() if k != "id"}
    if digest(identity) != manifest["id"]:
        raise ValueError("validation manifest identity differs")
    if (manifest["prompt_indices"] != list(range(50)) or manifest["length"] != 1024
            or manifest["seeds"] != [12345, 67890] or manifest["budget_usd"] != 10):
        raise ValueError("validation scope differs")
    if manifest["generation_timeout"] > 3000 or manifest["textseal_timeout"] > 600:
        raise ValueError("validation timeout exceeds budget reservation")
    for name, expected in manifest["code_sha256"].items():
        if sha(Path(root) / name) != expected:
            raise ValueError(f"validation source changed: {name}")
    if sha(Path(root) / "prompts.jsonl") != manifest["prompt_sha256"]:
        raise ValueError("validation prompts changed")


def compare_replicates(first, second, replay, *, deterministic):
    a, b, c = ([r["token_ids"] for r in result["responses"]] for result in (first, second, replay))
    aligned = (len(a) == len(b) == len(c) > 0 and
               [r["prompt_index"] for r in first["responses"]] ==
               [r["prompt_index"] for r in second["responses"]] ==
               [r["prompt_index"] for r in replay["responses"]])
    fixed = a == c
    different = sum(x != y for x, y in zip(a, b))
    same_key = first["manifest"]["setting"] == second["manifest"]["setting"] == replay["manifest"]["setting"]
    return {"passed": aligned and fixed and same_key and (different == 0 if deterministic else different > 0),
            "responses_aligned": aligned,
            "same_seed_exact": fixed, "fixed_key_configuration": same_key,
            "responses_changed_across_seeds": different, "responses": len(a),
            "deterministic_expected": deterministic}


def prepare(cache, output, resume=None):
    reference = verify_reference()
    audit = audit_sources(cache)
    save(output / "source_audit.json", audit)
    textseal = json.loads((ROOT / "outputs/comparison_redetect/textseal_setup/direct_prefix/native8b_manifest.json").read_text())
    code = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "baseline_comparison").glob("*.py"))
    code += ["baseline_comparison/self_bleu_reference.json", "baseline_comparison/textseal_source_audit.json",
             "baseline_comparison/requirements-textseal.txt", "qwen.py", "prc.py", "online_prc.py",
             "detectors.py", "watermark_expt.py"]
    # The legacy paper-figure drafts are unrelated to this worker.
    code = [p for p in code if "/make_paper_" not in p]
    manifest = {"schema_version": 1, "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "protocol": reference["protocol"], "model": textseal["model"],
                "prompt_sha256": reference["prompt_sha256"], "prompt_indices": list(range(50)),
                "length": 1024, "prefix_lengths": list(PREFIX_LENGTHS), "seeds": [12345, 67890],
                "code_sha256": {p: sha(ROOT / p) for p in code}, "source_audit": audit,
                "artifact": reference["prc_generation_artifact"], "budget_usd": 10,
                "generation_timeout": 3000, "textseal_timeout": 600,
                "resource_usd_per_second": RATE,
                "maximum_worker_resource_reservation_usd": (3000 + 600 + 4)*RATE,
                "overhead_reserve_usd": 2, "pricing_source": "https://modal.com/pricing",
                "dispatch": "One generation H100, then one HF-replay H100; max_containers=1, retries=0. No pilot analysis or full sweep."}
    if resume is not None:
        previous = json.loads(Path(resume).read_text())
        if digest({k: v for k, v in previous.items() if k != "id"}) != previous["id"]:
            raise ValueError("resume manifest identity differs")
        for field in ("model", "prompt_sha256", "prompt_indices", "length", "prefix_lengths", "seeds", "artifact", "source_audit"):
            if previous[field] != manifest[field]:
                raise ValueError(f"resume would change study inputs: {field}")
        allowed = {"baseline_comparison/self_bleu_validation.py", "baseline_comparison/self_bleu_validation_modal.py"}
        if {p for p, h in previous["code_sha256"].items() if manifest["code_sha256"].get(p) != h} - allowed:
            raise ValueError("resume would change generation or detector implementation")
        manifest.update(resume_from_manifest=previous, generation_timeout=600,
                        maximum_worker_resource_reservation_usd=(600+600+4)*RATE,
                        dispatch="Detector-only repair from verified saved pairs, then HF replay; no generation. Each H100 timeout 600s, max_containers=1, retries=0.")
    manifest["id"] = digest(manifest)
    validate_manifest(manifest, ROOT)
    save(output / "manifest.json", manifest)
    return {"id": manifest["id"], "source_files_verified": len(audit["files"]),
            "maximum_reserved_usd": manifest["maximum_worker_resource_reservation_usd"] + 2}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("/private/tmp/comparison-redetect-cache"))
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/self_bleu_validation/step3")
    parser.add_argument("--resume-generation-manifest", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.cache, args.output, args.resume_generation_manifest), indent=2))
