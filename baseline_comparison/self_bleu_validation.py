"""Prepare, verify and collect the bounded step-3 GPU validation."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess

from .config import ONLINE_PRC_SOURCE_TAG, PREFIX_LENGTHS
from .self_bleu_config import digest, verify_reference

ROOT = Path(__file__).resolve().parents[1]
RUN = "qwen3-8b-batch50-validation-20260823-v1"
RATE = .001097 + 4 * .0000131 + 64 * .00000222
# Exact source tree for the completed validation/pilot and superseded setup_v1.
LEGACY_SOURCE_COMMIT = "7bde5c6d54dee444db3b69d96bfce6b09c79ba4c"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_source_hashes(hashes, root=ROOT, *, allow_archived=False):
    """Verify source bytes; only historical artifact readers may opt into Git.

    Worker/dispatch validation requires the current files. Reading old results
    can instead verify their original bytes at the fixed pre-consolidation
    commit; archived code is never executed by this helper.
    """
    for name, expected in hashes.items():
        path = Path(root) / name
        if path.is_file() and sha(path) == expected:
            continue
        if allow_archived:
            source = subprocess.run(
                ["git", "show", f"{LEGACY_SOURCE_COMMIT}:{name}"],
                cwd=root, capture_output=True, check=False,
            )
            if source.returncode == 0 and hashlib.sha256(source.stdout).hexdigest() == expected:
                continue
        raise ValueError(f"source changed: {name}")


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


def validate_manifest(manifest, root, *, allow_archived=False):
    identity = {k: v for k, v in manifest.items() if k != "id"}
    if digest(identity) != manifest["id"]:
        raise ValueError("validation manifest identity differs")
    if (manifest["prompt_indices"] != list(range(50)) or manifest["length"] != 1024
            or manifest["seeds"] != [12345, 67890] or manifest["budget_usd"] != 10):
        raise ValueError("validation scope differs")
    if manifest["generation_timeout"] > 3000 or manifest["textseal_timeout"] > 600:
        raise ValueError("validation timeout exceeds budget reservation")
    verify_source_hashes(manifest["code_sha256"], root, allow_archived=allow_archived)
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



def collect(setup, raw, *, download=False):
    manifest = json.loads((setup / "manifest.json").read_text())
    validate_manifest(manifest, ROOT, allow_archived=True)
    generation = json.loads((setup / "generation_report.json").read_text())
    textseal = json.loads((setup / "textseal_report.json").read_text())
    for report in (generation, textseal):
        if not report["passed"] or report["manifest_id"] != manifest["id"]:
            raise ValueError("requires matching passing validation reports")
    prefix = f"self_bleu_validation/{manifest['id']}"
    files = {**generation["files"], **textseal["files"]}
    for name in ("manifest.json", "generation_report.json", "textseal_report.json"):
        files[name] = sha(setup / name)
    for name in files:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("invalid artifact path")
    if download:
        import modal
        volume = modal.Volume.from_name("prc-completion-only", create_if_missing=False)
        def retrieve(name):
            path = raw / name
            if path.exists():
                if sha(path) != files[name]:
                    raise ValueError(f"existing artifact differs: {name}")
                return
            data = b"".join(volume.read_file(f"{prefix}/{name}"))
            import hashlib
            if hashlib.sha256(data).hexdigest() != files[name]:
                raise ValueError(f"download checksum differs: {name}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(retrieve, files))
    for name, expected in files.items():
        if sha(raw / name) != expected:
            raise ValueError(f"saved artifact checksum differs: {name}")
    batches, response_ids = {}, set()
    full = short = synthid_checks = 0
    for name in generation["files"]:
        if not name.startswith("batches/"):
            continue
        batch = json.loads((raw / name).read_text())
        identity = batch["manifest"]
        base = {k: v for k, v in identity.items() if k not in ("batch_id", "namespace", "response_ids")}
        if (digest(base) != identity["batch_id"] or digest(identity["setting"]) != identity["setting_sha256"]
                or identity["prompt_indices"] != list(range(50)) or len(batch["responses"]) != 50):
            raise ValueError("batch identity or coverage differs")
        for i, row in enumerate(batch["responses"]):
            if (row["response_id"] in response_ids or row["response_id"] != identity["response_ids"][i]
                    or row["prompt_index"] != i or row["response_index"] != identity["response_index"]
                    or row["sampling_seed"] != identity["sampling_seed"]
                    or row["setting_sha256"] != identity["setting_sha256"]
                    or len(row["token_ids"]) != identity["generation"]["max_new_tokens"]
                    or digest(row["token_ids"]) != row["completion_sha256"]):
                raise ValueError("response identity or token coverage differs")
            response_ids.add(row["response_id"])
        if identity["generation"]["max_new_tokens"] == 1024:
            full += len(batch["responses"])
        else:
            short += len(batch["responses"])
        if identity["setting"]["method"] == "synthid_text":
            check = batch["telemetry"]["synthid_official_smoke_reference"]
            if not check["indices_equal"] or check["max_abs_score_difference"] != 0:
                raise ValueError("SynthID official update parity failed")
            synthid_checks += 1
        batches[identity["batch_id"]] = batch
    if (full, short, len(batches), len(textseal["records"]), synthid_checks) != (600, 200, 16, 7, 5):
        raise ValueError("validation coverage differs")
    reuse = []
    for row in generation["settings"]:
        first, second = (batches[identifier] for identifier in row["batches"])
        if first["manifest"]["sampling_seed"] != 12345 or second["manifest"]["sampling_seed"] != 67890:
            raise ValueError("replicate seed differs")
        if first["manifest"]["setting"] != second["manifest"]["setting"]:
            raise ValueError("watermark configuration changed across replicates")
        matches = row["historical_first_response_matches"]
        if matches is not None:
            expected = manifest["source_audit"]["expected_completion_sha256"][row["setting"]["method"]]
            actual = [r["completion_sha256"] == old for r, old in zip(first["responses"], expected)]
            if actual != matches:
                raise ValueError("historical reuse audit differs")
        reuse.append({"setting": row["setting"], "historical_exact_matches": None if matches is None else sum(matches),
                      "responses_changed_across_seeds": row["responses_changed_across_seeds"]})
    measured = generation["measured_resource_usd"] + generation.get("repair_measured_resource_usd", 0) + textseal["measured_resource_usd"]
    result = {"passed": True, "manifest_id": manifest["id"], "volume": "prc-completion-only",
              "remote_path": prefix, "verified_files": files, "verified_file_count": len(files),
              "full_length_response_records": full, "short_response_records": short,
              "synthid_official_update_checks": synthid_checks, "reuse": reuse,
              "measured_resource_usd": measured, "failed_startup_allowance_usd": .5,
              "image_startup_storage_allowance_usd": 2, "total_planning_charge_usd": measured + 2.5,
              "remaining_initial_allocation_after_allowances_usd": 10-measured-2.5,
              "billing_note": "Resource time is measured for generation, PRC repair and TextSeal replay. Separate allowances are conservative reservations, not invoice amounts."}
    save(setup / "verification.json", result)
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare", help="Freeze a new GPU validation request")
    prepare_parser.add_argument("--cache", type=Path, default=Path("/private/tmp/comparison-redetect-cache"))
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--resume-generation-manifest", type=Path)
    collect_parser = commands.add_parser("collect", help="Verify saved validation artifacts; no GPU")
    collect_parser.add_argument("--setup", type=Path, required=True)
    collect_parser.add_argument("--raw", type=Path, required=True)
    collect_parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.cache, args.output, args.resume_generation_manifest)
    else:
        result = collect(args.setup, args.raw, download=args.download)
    print(json.dumps({k: v for k, v in result.items() if k not in ("verified_files", "reuse")}, indent=2))


if __name__ == "__main__":
    main()
