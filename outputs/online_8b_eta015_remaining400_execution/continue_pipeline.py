"""Local dispatch/collection only; model work and scoring remain on Modal."""
from datetime import datetime, timezone, timedelta
from dataclasses import asdict
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from modal import Workspace

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
os.chdir(ROOT)
PLAN = json.loads((OUT / "setup.json").read_text())
STAGES = ("generate", "freeze", "detect", "score")
PREVIOUS_APPS = {"ap-jKSj0iAhFtHqpsvyenejqY", "ap-74oPtw92UaCPwKJQzaIGEo"}
CSV = ROOT / "outputs/redetection/redetection_results_summary.csv"


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, default=str) + "\n")


def log(message):
    print(datetime.now(timezone.utc).isoformat(), message, flush=True)


def billing(label):
    now = datetime.now(timezone.utc)
    rows = [asdict(row) for row in Workspace.from_context().billing.report(
        start=datetime(2026, 9, 20, tzinfo=timezone.utc),
        end=now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1), resolution="h")]
    apps = set(PREVIOUS_APPS)
    for path in OUT.glob("launch_*.log"):
        apps.update(re.findall(r"/apps/[^/]+/main/(ap-[A-Za-z0-9]+)", path.read_text()))
    own = [row for row in rows if row["object_id"] in apps]
    spent = sum((Decimal(str(row["cost"])) for row in own), Decimal(0))
    result = {"checked_utc": now.isoformat(), "read_only": True, "app_ids": sorted(apps),
              "reported_task_cost_usd": spent, "task_rows": own, "workspace_rows": rows}
    save("billing_" + label + ".json", result)
    return spent


def checkpoint(stage):
    collected = json.loads((OUT / ("collected_" + stage + ".json")).read_text())
    save("checkpoint_" + stage + ".json", {
        "stage": stage, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": hashlib.sha256((OUT / "setup.json").read_bytes()).hexdigest(),
        "files": collected["files"],
        "storage": "Raw binaries committed to Modal volumes and downloaded with SHA256 verification; Git retains manifests and evidence, respecting existing binary ignores."})
    paths = [path for path in OUT.rglob("*.json") if "execution_sources" not in path.parts]
    paths += [OUT / "PLAN.md", OUT / "setup.sha256", OUT / "continue_pipeline.py",
              ROOT / "online_8b_eta015_remaining400.py", ROOT / "tests/test_eta015_worker_log.py",
              ROOT / "tests/test_eta015_remaining400_metadata.py"]
    if stage == "score":
        before = (OUT / "csv_before_execution.csv").read_bytes()
        if not CSV.read_bytes().startswith(before):
            raise RuntimeError("CSV changed outside append-only experiment output; preserve and review before commit")
        paths.append(CSV)
    relative = sorted({str(path.relative_to(ROOT)) for path in paths if path.is_file()})
    if subprocess.check_output(["git", "branch", "--show-current"], text=True).strip() != "redetection":
        raise RuntimeError("branch changed; refusing commit")
    subprocess.run(["git", "add", "--", *relative], check=True)
    subprocess.run(["git", "commit", "--only", "-m",
                    f"Save online 8B eta0.15 remaining400 {stage} outputs", "--", *relative], check=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    save("commit_" + stage + ".json", {"commit": commit})
    log(f"Saved and committed {stage}: {commit}")


def main():
    if not (OUT / "csv_before_execution.csv").exists():
        (OUT / "csv_before_execution.csv").write_bytes(CSV.read_bytes())
    log("Waiting for the already-running four generation batches; no duplicate dispatch.")
    deadline = time.monotonic() + 3600
    while not (OUT / "collected_generate.json").exists():
        if (OUT / "failures_generate.json").exists():
            raise RuntimeError("generation needs recovery; preserve successful batches")
        if time.monotonic() > deadline:
            raise RuntimeError("generation collection is delayed; inspect existing app, never redispatch blindly")
        time.sleep(10)
    billing("after_generate")
    checkpoint("generate")
    for stage in STAGES[1:]:
        if (OUT / ("attempt_" + stage + ".json")).exists():
            raise RuntimeError(f"{stage} already attempted; inspect saved outputs")
        spent = billing("before_" + stage)
        remaining_allowances = sum(Decimal(str(PLAN["stages"][name]["allowance_usd"]))
                                   for name in STAGES[STAGES.index(stage):])
        if spent + remaining_allowances > Decimal("35.11"):
            raise RuntimeError("Reconcile actual cost before more dispatch; no blind extra spending")
        log(f"Starting approved {stage}; reported task spending ${spent}")
        with (OUT / ("launch_" + stage + ".log")).open("x") as output:
            subprocess.run([sys.executable, "-m", "modal", "run", "--detach",
                            "online_8b_eta015_remaining400.py::run", "--stage", stage,
                            "--approval-reference", "user-3511-full-run-20260920"],
                           stdout=output, stderr=subprocess.STDOUT, check=True)
        billing("after_" + stage)
        checkpoint(stage)
    spent = billing("completed")
    save("pipeline_completed.json", {"completed_utc": datetime.now(timezone.utc).isoformat(),
                                    "reported_task_cost_usd": spent, "N": 500, "null_N": 0})
    log(f"Full experiment complete; provider-reported task cost ${spent}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        save("pipeline_attention.json", {"error": str(exc), "utc": datetime.now(timezone.utc).isoformat()})
        raise
