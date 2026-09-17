"""Share a completed batch validation across workers with identical inference."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

from prompt_free.manifest import digest_json, file_sha, relative_path

NUMERICAL_FILES = ("prompt_free/core.py", "prompt_free/storage.py", "qwen.py",
                   "detectors.py", "online_prc.py", "prc.py", "prompt_free/requirements.txt")


def stable_ast(node):
    """Ignore the empty type-parameter field added by local Python 3.12.

    Modal uses 3.11. Nonempty type parameters still participate in identity.
    """
    if isinstance(node, ast.AST):
        return [type(node).__name__, [[name, stable_ast(value)] for name, value in ast.iter_fields(node)
                if not (name == "type_params" and value == [])]]
    if isinstance(node, list):
        return [stable_ast(value) for value in node]
    return node


def numerical_profile(files, modal_source):
    """Orchestration may change; model loading and actual replay must not."""
    tree = ast.parse(modal_source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Detector")
    loader = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "load")
    batch = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "batch")
    names = {"inputs", "tokens", "part", "cache", "trace"}
    replay = [n for n in batch.body if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)]
    if len(replay) != len(names):
        raise ValueError("unexpected inference statement structure")
    return {"files": {name: files[name] for name in NUMERICAL_FILES},
            "model_loader": stable_ast(loader),
            "replay": [stable_ast(n) for n in replay]}


def current_profile(root):
    root = Path(root)
    return numerical_profile({name: file_sha(root/name) for name in NUMERICAL_FILES},
                             (root/"prompt_free/modal_redetect.py").read_text())


def reference_proof(root, source):
    """Check a prior run's immutable Git source before trusting its certificate."""
    files = {}
    modal_source = None
    for name in (*NUMERICAL_FILES, "prompt_free/modal_redetect.py"):
        raw = subprocess.check_output(["git", "show", f"{source['git_commit']}:{name}"], cwd=root)
        files[name] = hashlib.sha256(raw).hexdigest()
        if files[name] != source["files"][name]:
            raise ValueError("prior validation source differs from its Git commit")
        if name == "prompt_free/modal_redetect.py":
            modal_source = raw.decode()
    profile = numerical_profile(files, modal_source)
    if profile != current_profile(root):
        raise ValueError("prior validation used different numerical code")
    return {"profile": profile, "modal_source_sha256": files["prompt_free/modal_redetect.py"]}


def configuration(identity):
    return {"gpu_type": identity.get("gpu_type", "A10G"),
            "allocator_config": identity.get("allocator_config", "unset"), **{k: identity[k] for k in (
             "protocol", "model", "partition_sha256", "maximum_length",
             "cache", "token_step", "actual_batch_size", "prepended_token_count", "first_coordinate_score")}}


def memory_report(peak_allocated, peak_reserved, total):
    """Gate live tensor allocations; allocator reservation is diagnostic only."""
    if total <= 0 or not 0 <= peak_allocated <= peak_reserved:
        raise ValueError("invalid CUDA memory measurements")
    return {"peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved,
            "total_gpu_bytes": total, "memory_gate": "peak_allocated_below_85_percent",
            "within_allocated_memory_margin": peak_allocated < .85*total}


def gpu_family(name):
    if name in ("NVIDIA A10", "NVIDIA A10G"):
        return "A10G"
    if name.startswith("NVIDIA A100") and "80GB" in name:
        return "A100-80GB"
    raise ValueError("unsupported validation GPU family")


def check_certificate(reference, identity, root, profile, gpu_name="NVIDIA A10"):
    path = Path(root)/relative_path(reference["path"])
    if file_sha(path) != reference["sha256"]:
        raise ValueError("shared validation certificate hash changed")
    certificate = json.loads(path.read_text())
    if (not certificate["passed"] or certificate["numerical_profile"] != profile
            or certificate["configuration"] != configuration(identity)
            or gpu_family(gpu_name) != identity.get("gpu_type", "A10G")
            or gpu_family(certificate["gpu"]) != gpu_family(gpu_name)):
        raise ValueError("shared validation does not match this inference configuration")
    return certificate


def publish_certificates(prepared, references, proofs, root, profile):
    """CPU: verify the original trace/proof once, then publish a small certificate."""
    from prompt_free.storage import load_pt, validate_trace, json_write
    root = Path(root)
    available = {}
    for prior, proof in zip(references, proofs, strict=True):
        persisted = json.loads((root/relative_path(prior["root"])/"prepared.json").read_text())
        if persisted["identity"] != prior["identity"]:
            raise ValueError("prior prepared identity changed")
        source = prior["identity"]["source"]
        if (proof["profile"] != profile or proof["modal_source_sha256"] != source["files"]["prompt_free/modal_redetect.py"]
                or profile["files"] != {name: source["files"][name] for name in NUMERICAL_FILES}):
            raise ValueError("prior numerical provenance differs")
        for batch in prior["batches"]:
            directory = root/relative_path(batch["root"])
            if not (directory/"validation.json").exists():
                continue
            validation = json.loads((directory/"validation.json").read_text())
            if not validation.get("validation_run_on_this_batch") and not validation.get("shared_validation"):
                continue
            if (not validation["passed"] or not validation["raw_inputs_only"]
                    or validation["inference_dtype"] != "bfloat16"):
                raise ValueError("prior full validation failed")
            if gpu_family(validation["gpu"]) != batch["identity"].get("gpu_type", "A10G"):
                raise ValueError("validated GPU differs from batch identity")
            if batch["identity"]["code_sha256"] != source["sha256"]:
                raise ValueError("validated batch and source identity differ")
            validate_trace(load_pt(directory/"trace.pt"), batch["identity"])
            if file_sha(directory/"trace.pt") != validation["trace_sha256"]:
                raise ValueError("validated reference trace hash changed")
            config = configuration(batch["identity"])
            if not validation.get("validation_run_on_this_batch"):
                available[digest_json(config)] = check_certificate(
                    validation["shared_validation"], batch["identity"], root, profile, validation["gpu"])
                continue
            available[digest_json(config)] = {"passed": True, "configuration": config,
                "numerical_profile": profile, "gpu": validation["gpu"],
                "reference_root": batch["root"], "reference_source": source,
                "reference_validation": validation}
    for case in prepared:
        for batch in case["batches"]:
            signature = digest_json(configuration(batch["identity"]))
            if signature not in available:
                raise ValueError("no full validation for this batch shape/configuration")
            path = root/case["root"]/"validation"/(signature+".json")
            json_write(path, available[signature])
            batch["validation_reference"] = {"path": str(path.relative_to(root)), "sha256": file_sha(path)}
        json_write(root/case["root"]/"prepared.json", case)
    return prepared
