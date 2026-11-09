"""Baseline 1: a fixed, Airflow-style workflow with predefined execution steps.

    extract → transform → train → evaluate

This is the shape of the Part 1 DAG and of the in-class `solution_baseline_prep` DAG.
Every run does the same steps with the same configuration. There is no validation and
no model choice. Like an Airflow task with `retries=2`, a failed step is re-run
*identically*, so transient errors recover but deterministic ones don't.

The configuration is fixed in code, as it would be in a DAG. Like Part 1, it knows the
schema: the time column is `hour` if it exists, and every other column is a feature.
"""
from __future__ import annotations

import time

import pandas as pd

from adaptive.training import check_fault, train_eval

AIRFLOW_RETRIES = 2
FIXED_SPEC = {
    "model_family": "random_forest",
    "hyperparameters": {"n_estimators": 300, "max_depth": 16},
    "drop_columns": [],
}


def run_fixed(*, dataset_path: str, target_column: str, fault_plan: dict | None = None, seed: int = 42,
              holdout_frac: float = 0.2, **_ignored) -> dict:
    steps: list[dict] = []
    t0 = time.time()
    df = pd.read_csv(dataset_path)  # extract
    steps.append({"node": "extract", "note": f"{len(df)} rows"})
    time_column = "hour" if "hour" in df.columns else None  # transform (fixed config)
    steps.append({"node": "transform", "note": f"time column={time_column}"})

    attempts, last_error, res = 0, None, None
    for attempt in range(AIRFLOW_RETRIES + 1):  # train (+ identical retries)
        attempts += 1
        try:
            check_fault(fault_plan, FIXED_SPEC["model_family"], attempt)
            res = train_eval(df, target_column, FIXED_SPEC, time_column=time_column, seed=seed,
                             holdout_frac=holdout_frac, on_holdout=True, return_predictions=True)
            steps.append({"node": "train", "note": f"attempt {attempt + 1}: ok", "status": "succeeded"})
            break
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            steps.append({"node": "train", "note": f"attempt {attempt + 1}: {last_error}", "status": "failed"})

    if res is None:
        return {"system": "fixed_airflow", "status": "crashed", "error": last_error, "steps": steps,
                "attempts": attempts, "duration_s": round(time.time() - t0, 2)}
    steps.append({"node": "evaluate", "note": f"holdout f1={res['metrics']['f1']:.3f}"})
    return {
        "system": "fixed_airflow",
        "status": "trained",
        "experiments": [{"model_family": "random_forest", "status": "succeeded", "val_f1": None}],
        "attempts": attempts,
        "holdout": {"metrics": res["metrics"], "majority_baseline": res["majority_metrics"],
                    "features_used": res["features_used"], "split": res["split"]},
        "flagged": [],
        "steps": steps,
        "duration_s": round(time.time() - t0, 2),
    }
