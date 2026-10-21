"""
Weekly MLOps flow (Prefect): monitor, then retrain when there is a reason to.

    python -m src.flows.mlops_flow            # run once now
    python -m src.flows.mlops_flow --serve    # keep running on a schedule (Mondays 06:00 Nairobi time)

Retraining runs when either is true:
  - new labelled records are waiting in data/new/ (for example FundiFix inspection results), or
  - the hold-out monitoring batch is red (performance or bias drift past its limit).
A retrained model only becomes "challenger". Promotion stays a human decision
(python -m src.models.retrain --promote <version> --approved-by "<name, role>").
"""
import argparse
from pathlib import Path

from prefect import flow, get_run_logger, task
from prefect.schedules import Cron

from src.models import retrain as rt
from src.monitoring import monitor

NEW_DATA_DIR = Path("data/new")


@task(name="monitor", retries=1)
def run_monitoring():
    return monitor.main()


@task(name="retrain-with-gates")
def run_retraining(new_file):
    return rt.retrain(str(new_file) if new_file else None)


@flow(name="fundifix-weekly-mlops")
def weekly_mlops():
    log = get_run_logger()
    summary = run_monitoring()
    holdout = next(b for b in summary["batches"] if b["name"] == "holdout")
    new_files = sorted(NEW_DATA_DIR.glob("*.csv")) if NEW_DATA_DIR.exists() else []
    for b in summary["batches"]:
        for a in b["alerts"]:
            log.warning(f"[{a['severity'].upper()}] {b['label']}: {a['message']}")
    if new_files or holdout["status"] == "red":
        reason = f"new data: {new_files[-1].name}" if new_files else "hold-out batch is red"
        log.info(f"Retraining ({reason})")
        result = run_retraining(new_files[-1] if new_files else None)
        if new_files:
            new_files[-1].rename(new_files[-1].with_suffix(".used.csv"))
        log.info(f"Gates passed: {result['passed']}. Registered version {result['registered_version']}.")
    else:
        log.info("No new labelled data and no red alert on the hold-out batch. No retraining.")
    return summary["run_at"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true")
    if ap.parse_args().serve:
        weekly_mlops.serve(name="weekly", schedule=Cron("0 6 * * 1", timezone="Africa/Nairobi"))
    else:
        weekly_mlops()
