"""Read-only cache and billing checks; never starts cloud computation."""
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from modal import Volume, Workspace
from modal.exception import NotFoundError

OUT = Path(__file__).resolve().parent
plan = json.loads((OUT / "setup.json").read_text())
source = plan["sources"]["8B"]
volume = Volume.from_name("prc-completion-only", create_if_missing=False)
now = datetime.now(timezone.utc)


def read_json(ref):
    raw = b"".join(volume.read_file(ref["path"]))
    if len(raw) != ref["bytes"] or hashlib.sha256(raw).hexdigest() != ref["sha256"]:
        raise ValueError("source JSON changed")
    return json.loads(raw)


prepared = read_json(source["prepared"])
report = read_json(source["report"])
if prepared["component_prepared"] != source["component_prepared"]:
    raise ValueError("source components changed")
if report["counts"]["14336"] != source["reused_T14336_counts"]:
    raise ValueError("baseline counts changed")
files = []
for component in prepared["component_prepared"]:
    entries = list(volume.iterdir(component["root"], recursive=True))
    names = {e.path for e in entries}
    for batch in component["batches"]:
        for suffix in ("inputs.pt", "trace.pt"):
            path = batch["root"] + "/" + suffix
            if path not in names:
                raise ValueError("missing saved input or trace: " + path)
    files.extend({"path": e.path, "bytes": e.size} for e in entries)
try:
    existing = [e.path for e in volume.iterdir(plan["run_id"], recursive=True)]
except (FileNotFoundError, NotFoundError):
    existing = []
if existing:
    raise ValueError("prefix output already exists; inspect/reuse it before paid scoring")
inventory = {"read_only": True, "checked_utc": now.isoformat(), "N": 500,
             "native_trace_shards": len(report["trace_shard_sha256"]),
             "source_json_hashes_verified": True, "component_files": files,
             "existing_target_prefix_files": existing, "paid_compute_launched": False}
(OUT / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
old = OUT.parent / "online_8b_eta020_batch50_setup"
new = OUT.parent / "online_8b_eta020_remaining450_setup"
ids = {a for p in (old, new) for a in json.loads((p / "billing_final.json").read_text())["app_ids"]}
rows = [asdict(r) for r in Workspace.from_context().billing.report(
    start=datetime(2026, 9, 20, tzinfo=timezone.utc),
    end=now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1), resolution="h")
    if r.object_id in ids]
cost = sum((Decimal(str(r["cost"])) for r in rows), Decimal(0))
billing = {"read_only": True, "checked_utc": now.isoformat(), "app_ids": sorted(ids),
           "reported_native_N500_cost_usd": str(cost), "task_rows": rows,
           "account_credit_balance": "not available from usage API"}
(OUT / "billing_before.json").write_text(json.dumps(billing, indent=2, default=str) + "\n")
print(json.dumps({"N": 500, "saved_native_trace_shards": inventory["native_trace_shards"],
                  "baseline_counts": report["counts"]["14336"],
                  "native_N500_cost_usd": str(cost), "paid_compute_launched": False}))
