"""Read-only Modal inventory and billing; never dispatches cloud functions."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from modal import Volume, Workspace

OUT = Path(__file__).resolve().parent
PRIOR = OUT.parent / "online_8b_eta020_batch50_setup"
plan = json.loads((PRIOR / "setup.json").read_text())
data = Volume.from_name("prc-data", create_if_missing=False)
results = Volume.from_name("prc-completion-only", create_if_missing=False)
now = datetime.now(timezone.utc)


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, default=str) + "\n")


def read_manifest(entry):
    path = entry.path.rstrip("/") + "/manifest.json"
    value = json.loads(b"".join(results.read_file(path)))
    case = value.get("case", {})
    if (value.get("protocol") == plan["protocol"]
            and value.get("model", {}).get("id") == plan["model"]["id"]
            and max(case.get("lengths", [0])) >= 14336
            and case.get("artifact", {}).get("sha256") == plan["artifact"]["sha256"]):
        refs = [r for r in case.get("records", []) if r["source"] == "wm"]
        files = [e.path for e in results.iterdir(entry.path, recursive=True)
                 if e.path.endswith("/trace.pt")]
        return {"manifest": path, "prompt_ids": [r["prompt_idx"] for r in refs],
                "trace_files": files, "case_id": case.get("id"),
                "model": value["model"]}
    return None


listing = [{"path": e.path, "size": e.size, "type": str(e.type)}
           for e in data.iterdir(plan["source_tag"] + "/wm")]
roots = list(results.iterdir(plan["protocol"] + "/integrated", recursive=False))
with ThreadPoolExecutor(max_workers=8) as pool:
    matches = [m for m in pool.map(read_manifest, roots) if m]
pilot_refs = []
for name in ("collected_replay.json", "collected_score.json"):
    for ref in json.loads((PRIOR / name).read_text())["files"]:
        if ref["path"].endswith(("/trace.pt", "/full.json", "/summary.json")):
            raw = b"".join(results.read_file(ref["path"]))
            if len(raw) != ref["bytes"] or hashlib.sha256(raw).hexdigest() != ref["sha256"]:
                raise ValueError("completed first-batch cache changed")
            pilot_refs.append(ref)
save("inventory.json", {"read_only": True, "checked_utc": now.isoformat(),
     "source_listing": listing, "source_count": len(listing),
     "manifests_inspected": len(roots), "matching_caches": matches,
     "verified_pilot_files": pilot_refs, "paid_compute_launched": False})
rows = [asdict(r) for r in Workspace.from_context().billing.report(
    start=datetime(2026, 9, 20, tzinfo=timezone.utc),
    end=now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1), resolution="h")]
known = {}
for folder in [PRIOR, OUT.parent / "online_8b_eta015_remaining400_execution",
               OUT.parent / "online_8b_eta015_prefixes_setup"]:
    record = json.loads((folder / "billing_final.json").read_text())
    ids = record.get("app_ids", [])
    if not ids:
        ids = sorted({r["object_id"] for r in record.get("task_rows", [])})
    known[folder.name] = ids
totals = {name: str(sum((Decimal(str(r["cost"])) for r in rows if r["object_id"] in ids), Decimal(0)))
          for name, ids in known.items()}
save("billing_before.json", {"read_only": True, "checked_utc": now.isoformat(),
     "known_app_ids": known, "known_task_totals_usd": totals,
     "rows": [r for r in rows if r["object_id"] in {a for ids in known.values() for a in ids}],
     "account_credit_balance": "not exposed by usage report"})
apps = json.loads(subprocess.check_output([sys.executable, "-m", "modal", "app", "list", "--json"], text=True))
save("apps_before.json", {"read_only": True, "checked_utc": now.isoformat(),
     "apps": [a for a in apps if a["state"] != "stopped" or a["app_id"] in known[PRIOR.name]]})
print(json.dumps({"source_count": len(listing), "matching_caches": matches,
                  "billing": totals, "paid_compute_launched": False}))
