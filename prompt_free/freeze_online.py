"""Freeze existing same-model online audit inputs; CPU only, no inference.

The historical probabilities are used only to verify the prompted reference.
Only file references and raw-completion token hashes enter the new manifest.
"""
import json
from pathlib import Path

import modal

from prompt_free.modal_redetect import image, data, ROOT
from prompt_free.manifest import file_sha, validate
from prompt_free.storage import json_write

image = image.add_local_file(__file__, "/root/prompt_free/freeze_online.py", copy=True)
app = modal.App("prc-freeze-online-redetection", image=image)


@app.function(cpu=4, memory=8192, timeout=1800, retries=0, include_source=False,
              volumes={"/data": data})
def freeze(reference, model, case_id, batch_size):
    import io
    import hashlib
    from concurrent.futures import ThreadPoolExecutor
    import numpy as np
    import torch
    from detectors import map_soft_token, tokens_to_bits, weights_from_p
    from online_prc import OnlinePRCKey, materialize_supports, otp_prefix, support_sha256
    from prompt_free.core import PROTOCOL, _checks
    from prompt_free.storage import load_pt, token_sha

    torch.set_num_threads(1)
    if (reference["generation_model_size"], reference["fpr_policy"], reference["execution_mode"]) != ("0.6B", "one_shot", "cache_only"):
        raise ValueError("expected a same-model, one-shot cached online audit")
    length = reference["T"]

    def read(path):
        from prompt_free.manifest import relative_path
        raw = (Path("/data")/relative_path(path)).read_bytes()
        return load_pt(io.BytesIO(raw)), {"volume": "data", "path": path,
                "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}

    artifact, artifact_ref = read(reference["tag"]+"/artifacts.pt")
    source, source_ref = read(reference["watermarked_cache_tag"]+"/artifacts.pt")
    if artifact["artifact_fingerprint"] != reference["artifact_fingerprint"] or source["artifact_fingerprint"] != reference["watermarked_source_artifact_fingerprint"]:
        raise ValueError("original artifact fingerprint differs")
    key = OnlinePRCKey.from_dict(artifact["online_key"])
    if key.fingerprint != reference["online_key_sha256"] or support_sha256(length, key) != reference["online_support_sha256"]:
        raise ValueError("original online key/supports differ")
    if source["online_key"] != artifact["online_key"] or not torch.equal(source["partition"], artifact["partition"]) or source["prompt_ids_list"] != artifact["prompt_ids_list"]:
        raise ValueError("longer watermarked source is incompatible")
    supports = materialize_supports(length, key)
    otp = otp_prefix(length, key).astype(np.int64)
    if supports.shape != (reference["r"], reference["t"]):
        raise ValueError("parity support dimensions differ")
    original = {(r["watermark"], r["prompt_idx"]): r for r in reference["results"]}
    expected = [(watermark, i) for watermark in (True, False) for i in reference["prompt_indices"]]
    if len(original) != len(reference["results"]) or set(original) != set(expected):
        raise ValueError("original candidate membership differs")

    def extract(ident):
        watermark, index = ident
        label = "wm" if watermark else "null"
        directory = (reference["watermarked_cache_tag"]+"/wm" if watermark
                     else f"_nulls/T{reference['null_cache_T']}")
        record, ref = read(f"{directory}/{label}_{index:04d}.pt")
        if record["prompt_idx"] != index or record["watermark"] != watermark:
            raise ValueError("cached candidate identity differs")
        tokens = record["tokens"][:length]
        if len(tokens) != length:
            raise ValueError("cached completion is too short")
        bits = tokens_to_bits(tokens, artifact["partition"]).astype(np.int64)
        probabilities = np.asarray(record["p_trace"], dtype=np.float64)[:length]
        if probabilities.shape != (length,):
            raise ValueError("historical probability trace is too short")
        maximum_delta = 0.
        decisions = {}
        for weight in ("map", "entropy"):
            soft = (map_soft_token(bits, probabilities) if weight == "map"
                    else (1-2*bits)*weights_from_p(probabilities, weight))
            current = _checks(soft, supports, otp, reference["target_fpr"], 1e-15)
            old = original[ident]["scores"][weight]
            if current["decision"] != old["decision"]:
                raise ValueError("cached prompted decision differs from saved audit")
            for field in ("statistic", "V", "threshold"):
                delta = abs(current[field]-old[field])
                maximum_delta = max(maximum_delta, delta)
                if delta > 1e-10:
                    raise ValueError("cached prompted statistic differs from saved audit")
            decisions[weight] = current["decision"]
        return ({"source": label, "prompt_idx": index, "file": ref,
                 "tokens_sha256": token_sha(tokens)}, decisions, maximum_delta)

    with ThreadPoolExecutor(max_workers=8) as pool:
        checked = list(pool.map(extract, expected))
    manifest = {"schema_version": 1, "protocol": PROTOCOL, "model": model, "cases": [{
        "id": case_id, "generation_model": reference["generation_model"], "construction": "online",
        "artifact": artifact_ref, "lengths": [length], "fpr": reference["target_fpr"],
        "fpr_policy": reference["fpr_policy"], "weights": ["map", "entropy"],
        "batch_size": batch_size, "cache": "static", "records": [x[0] for x in checked]}]}
    validate(manifest)
    counts = {w: {label: sum(x[1][w] for x in checked if x[0]["source"] == label)
                  for label in ("wm", "null")} for w in ("map", "entropy")}
    evidence = {"passed": True, "candidate_count": len(checked), "prompted_counts": counts,
                "maximum_absolute_prompted_score_difference": max(x[2] for x in checked),
                "watermarked_source_artifact": source_ref, "online_key_sha256": key.fingerprint,
                "online_support_sha256": support_sha256(length, key),
                "parity_checks": len(supports), "checks_containing_coordinate_one": int(np.any(supports == 0, axis=1).sum()),
                "eta": reference["eta"], "t": reference["t"], "n": length,
                "batch_size": batch_size, "inference_launched": False}
    return manifest, evidence


@app.local_entrypoint()
def main(reference: str, output: str, case_id: str, batch_size: int = 20):
    source = Path(reference)
    model = json.loads((ROOT/"prompt_free/manifests/pilots.json").read_text())["model"]
    manifest, evidence = freeze.remote(json.loads(source.read_text()), model, case_id, batch_size)
    # Keep each candidate reference on one line for reviewable manifest diffs.
    rows = manifest["cases"][0].pop("records")
    marker = "__FROZEN_CANDIDATE_ROWS__"
    manifest["cases"][0]["records"] = marker
    rendered = json.dumps(manifest, indent=2).replace(json.dumps(marker), "[\n"+",\n".join(
        "        "+json.dumps(row, separators=(",", ":")) for row in rows)+"\n      ]")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered+"\n")
    evidence["prompted_reference"] = {"path": source.as_posix(), "sha256": file_sha(source)}
    json_write(path.with_suffix(".audit.json"), evidence)
    print(json.dumps(evidence, indent=2))
