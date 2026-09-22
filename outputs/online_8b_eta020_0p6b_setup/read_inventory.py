"""Read-only Modal storage/billing inventory; no cloud function dispatch."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path

from modal import Volume, Workspace

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
native = json.loads((OUT.parent / "online_8b_eta020_batch50_setup/setup.json").read_text())
prior = json.loads((OUT.parent / "online_8b_eta015_remaining400_execution/setup.json").read_text())
model = prior["models"]["0.6B"]
results = Volume.from_name("prc-completion-only", create_if_missing=False)
data = Volume.from_name("prc-data", create_if_missing=False)
hf = Volume.from_name("prc-hf-cache", create_if_missing=False)
now = datetime.now(timezone.utc)


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, default=str) + "\n")


def inspect_manifest(entry):
    path = entry.path.rstrip("/") + "/manifest.json"
    raw = b"".join(results.read_file(path))
    value = json.loads(raw)
    case = value.get("case", {})
    if value.get("model", {}).get("id") != model["id"]:
        return None
    if case.get("artifact", {}).get("sha256") != native["artifact"]["sha256"]:
        return None
    return {"manifest_path": path, "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "protocol": value.get("protocol"), "lengths": case.get("lengths"),
            "model": value["model"], "records": case.get("records", []),
            "files": [{"path": e.path, "bytes": e.size}
                      for e in results.iterdir(entry.path, recursive=True)]}


roots = list(results.iterdir("completion_only_raw_abstain_v1/integrated", recursive=False))
with ThreadPoolExecutor(max_workers=8) as pool:
    matches = [r for r in pool.map(inspect_manifest, roots) if r]
source_files = [{"path": e.path, "bytes": e.size}
                for e in data.iterdir(native["source_tag"] + "/wm", recursive=False)]
artifact = b"".join(data.read_file(native["artifact"]["path"]))
assert hashlib.sha256(artifact).hexdigest() == native["artifact"]["sha256"]
checkpoint = []
configs = {}
for size in ("8B", "0.6B"):
    spec = prior["models"][size]
    for name in ("config.json", "tokenizer.json"):
        path = spec["cache_directory"] + "/" + name
        raw = b"".join(hf.read_file(path))
        sha = hashlib.sha256(raw).hexdigest()
        expected = spec.get("metadata_sha256", {}).get(name)
        if expected:
            assert sha == expected, path
        checkpoint.append({"path": path, "sha256": sha, "bytes": len(raw)})
        if name == "config.json":
            configs[size] = json.loads(raw)
assert checkpoint[1]["sha256"] == checkpoint[3]["sha256"]
assert configs["8B"]["vocab_size"] == configs["0.6B"]["vocab_size"] == 151936
weights_listing = [{"path": e.path, "bytes": e.size}
                   for e in hf.iterdir(model["cache_directory"], recursive=False)]
save("inventory.json", {"checked_utc": now.isoformat(), "read_only": True,
     "source_tag": native["source_tag"], "source_count": len(source_files),
     "source_files": source_files, "artifact": native["artifact"],
     "artifact_hash_verified": True, "integrated_manifests_inspected": len(roots),
     "matching_0p6b_source_caches": matches, "checkpoint_metadata": checkpoint,
     "configs": configs, "tokenizer_ids_compatible": True,
     "detector_checkpoint_listing": weights_listing, "paid_compute_launched": False})

folders = ["online_8b_eta020_batch50_setup", "online_8b_eta020_remaining450_setup",
           "online_8b_eta020_prefixes_setup", "online_8b_eta015_remaining400_execution",
           "online_8b_eta015_prefixes_setup"]
known = {}
for name in folders:
    bill = json.loads((OUT.parent / name / "billing_final.json").read_text())
    known[name] = bill.get("app_ids") or sorted({r["object_id"] for r in bill["task_rows"]})
ids = {app for group in known.values() for app in group}
rows = [asdict(r) for r in Workspace.from_context().billing.report(
    start=datetime(2026, 9, 20, tzinfo=timezone.utc),
    end=now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1), resolution="h")
    if r.object_id in ids]
totals = {name: str(sum((Decimal(str(r["cost"])) for r in rows if r["object_id"] in apps), Decimal(0)))
          for name, apps in known.items()}
save("billing_before.json", {"checked_utc": now.isoformat(), "read_only": True,
     "known_app_ids": known, "known_task_totals_usd": totals, "task_rows": rows,
     "account_credit_balance": "not exposed by usage report"})
print(json.dumps({"source_count": len(source_files), "manifests_inspected": len(roots),
                  "matching_0p6b_caches": len(matches), "tokenizer_ids_compatible": True,
                  "known_task_totals_usd": totals, "paid_compute_launched": False}))
