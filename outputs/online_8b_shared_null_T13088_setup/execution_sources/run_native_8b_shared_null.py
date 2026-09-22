"""Local orchestration for the approved $65 shared native-null run.

All model execution and scoring run in the frozen cloud driver. This controller
reads billing, records stage approvals from the user's go-ahead, and stops on
failure or a budget conflict. It never retries a paid stage.
"""
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import time

from modal import Workspace
import online_8b_shared_null as work
from fixed_4b_comparison import write_json

OUT = work.OUT
REFERENCE = "user-20260922-go-ahead-native8b-shared-null-total65"
RATE = {"prepare": 4*.0000131 + 16*.00000222,
        "replay": .001261 + 4*.0000131 + 64*.00000222,
        "score": 4*.0000131 + 8*.00000222}


def read(path):
    return json.loads(Path(path).read_text())


def update(status, **fields):
    p = read(OUT / "progress.json")
    p.update(status=status, updated_utc=datetime.now(timezone.utc).isoformat(), **fields)
    p["app_ids"] = {f.stem.removeprefix("app_"): read(f)["app_id"] for f in OUT.glob("app_*.json")}
    write_json(OUT / "progress.json", p)
    print(json.dumps({"status": status, **fields}), flush=True)


def billing(label):
    now = datetime.now(timezone.utc)
    ids = {read(p)["app_id"] for p in OUT.glob("app_*.json")}
    prior = set(read(OUT.parent / "online_8b_eta020_0p6b_setup/billing_final.json")["app_ids"])
    rows = [asdict(r) for r in Workspace.from_context().billing.report(
        start=datetime(2026, 9, 22, tzinfo=timezone.utc),
        end=now.replace(minute=0, second=0, microsecond=0)+timedelta(hours=1), resolution="h")
        if r.object_id in ids | prior]
    amount = sum((Decimal(str(r["cost"])) for r in rows if r["object_id"] in ids), Decimal(0))
    old = sum((Decimal(str(r["cost"])) for r in rows if r["object_id"] in prior), Decimal(0))
    value = {"checked_utc": now.isoformat(), "app_ids": sorted(ids), "reported_task_cost_usd": str(amount),
             "prior_separate_run_cost_usd": str(old), "rows": rows, "billing_may_lag": True,
             "account_credit_balance": "not exposed by usage API", "authorized_new_run_budget_usd": 65}
    (OUT / f"billing_{label}.json").write_text(json.dumps(value, indent=2, default=str)+"\n")
    return float(amount)


def measured_envelope(stage):
    result = read(OUT / f"result_{stage}.json")
    refs = [r for r in result["files"] if r["path"].endswith("/timing.json")]
    if len(refs) != (10 if stage == "replay" else 1):
        raise ValueError("incomplete timing evidence")
    seconds = 0
    for ref in refs:
        path = OUT / "cache/results" / ref["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != ref["sha256"]:
            raise ValueError("timing evidence checksum changed")
        seconds += read(path)["wall_seconds"] + 92
    return seconds * RATE[stage]


def main():
    auth = read(OUT / "authorization.json")
    plan = read(OUT / "setup.json")
    sha = hashlib.sha256((OUT / "setup.json").read_bytes()).hexdigest()
    if (auth["approval_reference"] != REFERENCE or auth["explicit_user_approval"] is not True
            or auth["total_budget_usd"] != 65 or auth["plan_sha256"] != sha
            or auth["approved_stages"] != list(work.STAGES)):
        raise ValueError("require matching approval for this65-dollar run")
    with (OUT / "pipeline_attempt.json").open("x") as f:
        json.dump({"approval_reference": REFERENCE, "plan_sha256": sha, "created_unix": time.time()}, f)
    spent_upper, completed = 0, []
    for stage in work.STAGES:
        paid = billing("before_"+stage)
        consumed = max(paid, spent_upper)
        future = work.STAGES[work.STAGES.index(stage):]
        remaining_allowances = sum(plan["stages"][s]["allowance_usd"] for s in future)
        if consumed+remaining_allowances > 65+1e-8:
            raise ValueError("remaining approved allowances do not fit total65; no next stage launched")
        write_json(OUT / f"approval_{stage}.json", {
            "stage": stage, "plan_sha256": sha, "approval_reference": REFERENCE,
            "explicit_user_approval": True, "user_instruction": "go ahead",
            "authorization_evidence": "authorization.json", "authorized_total_budget_usd": 65,
            "authorized_spend_usd": plan["stages"][stage]["allowance_usd"],
            "confirmed_available_budget_usd": 65-consumed,
            "budget_basis": "user-approved allocation; account credit balance is not exposed",
            "recorded_utc": datetime.now(timezone.utc).isoformat()})
        update(stage+"_running", paid_compute_launched=True, completed_stages=completed,
               reported_cost_usd=paid, spent_upper_estimate_usd=spent_upper,
               budget_remaining_using_upper_estimate_usd=65-consumed)
        work.run(stage, REFERENCE)
        spent_upper += measured_envelope(stage)
        completed.append(stage)
        update(stage+"_complete", completed_stages=completed, spent_upper_estimate_usd=spent_upper)
    paid = billing("at_completion")
    result = read(OUT / "result_score.json")
    summary = {k: {"root": v["root"], "point_count": len(v["counts"]),
                  "maximum_length_counts": v["counts"][str(max(map(int, v["counts"]))) ]}
               for k, v in result["summaries"].items()}
    update("complete", completed_stages=completed, null_records=500, completed_reporting_points=173,
           reported_cost_usd=paid, spent_upper_estimate_usd=spent_upper,
           budget_remaining_using_upper_estimate_usd=65-max(paid, spent_upper), summary=summary)
    lines = ["# Completed native8B shared-null FPR", "",
             "All500 saved nulls were replayed once throughT13088 on ten H200s with batch50. "
             "Both original keys were scored at all173 frozen reporting points. Saved watermarked scores were reused.", "",
             f"Modal currently reports ${paid:.4f}; billing can lag. The resource-cost estimate including lifecycle reserves is ${spent_upper:.4f}, within the $65 allocation.", "",
             "| Key | Longest reported length | Posterior FPR | Entropy FPR |", "|---|---:|---:|---:|"]
    for key, values in result["summaries"].items():
        n = str(max(map(int, values["counts"])))
        c = values["counts"][n]
        lines.append(f"| {key} | {n} | {c['map']['null']['detected']}/500 | {c['entropy']['null']['detected']}/500 |")
    lines += ["", "All173 combined TPR/FPR rows were appended to the existing redetection CSV. "
                   "T14336 FPR remains outside this run. No new text generation, extra validation pass or automatic retry was performed.",
              "", "See `progress.json`, `result_score.json`, the cached per-record reports, and `billing_at_completion.json`."]
    (OUT / "RESULTS.md").write_text("\n".join(lines)+"\n")
    (OUT / "RUN_STATUS.md").write_text("# Completed\n\nAll500 nulls, two keys,173 reporting points. See [results](RESULTS.md) and [status](progress.json).\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        update("stopped_needs_inspection", error=str(exc), automatic_retry=False)
        raise
