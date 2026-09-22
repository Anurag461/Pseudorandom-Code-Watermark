"""Continue the user's authorized eta=.20 run only within $35 total.

Local orchestration, metadata arithmetic and read-only billing calls only.
All model work and scoring stay in the existing cloud routines.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import subprocess
import time

from modal import Workspace
import online_8b_eta020_0p6b as primary
import eta020_a100_fallback as fallback
from fixed_4b_comparison import write_json

OUT = primary.OUT
REFERENCE = "user-20260922-continue-total-under35-else-original-a100"


def read(name):
    return json.loads((OUT / name).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state(status, **fields):
    value = read("progress.json")
    value.update(status=status, updated_utc=datetime.now(timezone.utc).isoformat(), **fields)
    write_json(OUT / "progress.json", value)
    print(json.dumps({"status": status, **fields}), flush=True)


def billing(label):
    now = datetime.now(timezone.utc)
    start = datetime(2026, 9, 22, 16, tzinfo=timezone.utc)
    # Discover current run's app IDs without dispatching any cloud compute.
    apps = json.loads(subprocess.check_output(["modal", "app", "list", "--json"], text=True))
    ids = set(read("progress.json")["app_ids"].values())
    for a in apps:
        if a["description"] in ("prc-online-8b-eta020-to-0p6b", "prc-eta020-0p6b-a100-remainder"):
            ids.add(a["app_id"])
    for path in fallback.OUT.glob("app_*.json"):
        ids.add(json.loads(path.read_text())["app_id"])
    rows = [asdict(r) for r in Workspace.from_context().billing.report(
        start=start, end=now.replace(minute=0, second=0, microsecond=0)+timedelta(hours=1), resolution="h")
        if r.object_id in ids]
    amount = sum((Decimal(str(r["cost"])) for r in rows), Decimal(0))
    value = {"checked_utc": now.isoformat(), "app_ids": sorted(ids),
             "reported_task_cost_usd": str(amount), "task_rows": rows, "billing_may_lag": True}
    (OUT / f"billing_{label}.json").write_text(json.dumps(value, indent=2, default=str)+"\n")
    return float(amount)


def authorize(stage, available, pilot_sha):
    plan = read("setup.json")
    write_json(OUT / f"approval_{stage}.json", {
        "stage": stage, "plan_sha256": digest(OUT / "setup.json"),
        "approval_reference": REFERENCE, "explicit_user_approval": True,
        "user_instruction": read("conditional_continuation_authorization.json")["user_instruction"],
        "authorization_evidence": "conditional_continuation_authorization.json",
        "authorized_spend_usd": plan["stages"][stage]["allowance_usd"],
        "confirmed_available_budget_usd": available,
        "pilot_result_sha256": pilot_sha,
        "recorded_utc": datetime.now(timezone.utc).isoformat()})


def timing_cost(result, rate):
    refs = [r for r in result["files"] if r["path"].endswith("/timing.json")]
    if not refs:
        raise ValueError("missing worker timing evidence")
    total = 0
    for ref in refs:
        path = OUT / "cache/results" / ref["path"]
        if digest(path) != ref["sha256"]:
            raise ValueError("timing evidence changed")
        total += (json.loads(path.read_text())["wall_seconds"]+92)*rate
    return total


def execute():
    auth = read("conditional_continuation_authorization.json")
    if (auth["approval_reference"] != REFERENCE or auth["explicit_user_approval"] is not True
            or auth["total_budget_usd"] != 35):
        raise ValueError("missing conditional authorization")
    # Exclusive marker: a second controller must never duplicate submission.
    with (OUT / "continuation_controller_attempt.json").open("x") as f:
        json.dump({"created_unix": time.time(), "approval_reference": REFERENCE}, f)
    state("waiting_for_pilot_cost_gate")
    deadline = time.monotonic()+3600
    while not ((OUT / "pilot_assessment.json").exists() and (OUT / "collected_pilot.json").exists()):
        if time.monotonic() > deadline:
            raise TimeoutError("pilot result unavailable; no continuation launched")
        time.sleep(10)
    plan, pilot = read("setup.json"), read("result_pilot.json")
    ref = next(r for r in pilot["files"] if r["path"].endswith("/timing.json"))
    timing_path = OUT / "cache/results" / ref["path"]
    if digest(timing_path) != ref["sha256"]:
        raise ValueError("pilot timing changed")
    assessment = primary.assess_pilot(plan, pilot, json.loads(timing_path.read_text()))
    reported = billing("before_continuation")
    spent = max(reported, .15 + timing_cost(pilot, .00134892))
    use_h200 = (assessment["fits_reviewed_remaining_allowance"]
                and spent+30.2+.05 <= 35)
    choice = {"pilot_assessment": assessment, "spent_upper_usd": spent,
              "reported_cost_usd": reported, "selection": "H200" if use_h200 else "A100-80GB",
              "conditional_approval_reference": REFERENCE, "budget_usd": 35}
    if use_h200:
        choice.update(estimated_total_usd=spent+assessment["remaining450_estimate_usd_with_margin"]+.05,
                      remaining_worker_seconds_with_margin=assessment["remaining_worker_seconds_with_15pct_margin"],
                      deadline_total_resource_envelope_usd=spent+30.2+.05)
        write_json(OUT / "continuation_decision.json", choice)
        state("remaining450_h200_launching", **choice)
        authorize("replay", 35-spent, digest(OUT / "result_pilot.json"))
        primary.run("replay", REFERENCE)
        upper_after_replay = spent + timing_cost(read("result_replay.json"), .00134892)
        paid = billing("before_scoring")
        if max(upper_after_replay, paid)+.05 > 35:
            raise ValueError("budget insufficient for scoring; stop")
        state("scoring_500", spent_upper_usd=max(upper_after_replay, paid))
        authorize("score", 35-max(upper_after_replay, paid), digest(OUT / "result_pilot.json"))
        primary.run("score", REFERENCE)
        final_upper = upper_after_replay + timing_cost(read("result_score.json"), 4*.0000131+8*.00000222)
        prepared = read("result_prepare.json")["prepared"]
    else:
        allowance = fallback.budget(spent)
        if not allowance["fits"]:
            raise ValueError("neither H200 nor original A100 plan fits remaining budget; stop")
        choice.update(allowance)
        write_json(OUT / "continuation_decision.json", choice)
        payload = {"plan": plan, "original": read("result_prepare.json")["prepared"],
                   "pilot_trace": next(r for r in pilot["files"] if r["path"].endswith("/trace.pt")),
                   "fallback_source_sha256": digest("eta020_a100_fallback.py"),
                   "approval_reference": REFERENCE, "budget": allowance}
        fallback.OUT.mkdir(parents=True, exist_ok=True)
        write_json(fallback.OUT / "setup.json", payload)
        (fallback.OUT / "execution_source.py").write_bytes(Path("eta020_a100_fallback.py").read_bytes())
        state("fallback_preparation", **choice)
        fallback.run("prepare", payload)
        # Check actual billing again before the GPU stage; the approved reserve
        # already includes this CPU preparation's bounded resource envelope.
        if billing("before_a100_replay") + 9*(allowance["worker_deadline_seconds"]+92)*fallback.RATE + .05 > 35:
            raise ValueError("updated billed cost would exceed budget; stop")
        state("remaining450_a100_launching", **choice)
        fallback.run("replay", payload)
        upper_after_replay = spent + .05 + timing_cost(json.loads((fallback.OUT / "result_replay.json").read_text()), fallback.RATE)
        paid = billing("before_scoring")
        if max(upper_after_replay, paid)+.05 > 35:
            raise ValueError("budget insufficient for scoring; stop")
        state("scoring_500", spent_upper_usd=max(upper_after_replay, paid))
        fallback.run("score", payload)
        final_upper = upper_after_replay + timing_cost(json.loads((fallback.OUT / "result_score.json").read_text()), 4*.0000131+8*.00000222)
        prepared = json.loads((fallback.OUT / "result_prepare.json").read_text())["prepared"]
    final_paid = billing("at_completion")
    state("complete", selection=choice["selection"], completed_records=500,
          reported_cost_usd=final_paid, resource_cost_upper_estimate_usd=final_upper,
          budget_remaining_using_upper_estimate_usd=35-max(final_paid, final_upper),
          result_root=prepared["root"], counts=read("native_comparison.json")["detector_0p6b"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    if parser.parse_args().execute:
        try:
            execute()
        except Exception as exc:
            state("stopped_needs_inspection", error=str(exc), automatic_retry=False)
            raise
    else:
        print(json.dumps({"paid_compute_launched": False, "fallback_worst_first_phase": fallback.budget(4.75)}))
