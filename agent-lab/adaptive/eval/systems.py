"""Run each of the three systems on one case and normalize the output.

Normalized result (what the checks read):
    system, status ("trained" | "rejected" | "crashed"), error, flagged (leakage columns
    the system itself identified), features_used, experiments [{model_family, status,
    val_f1, hyperparameters, feature_ops, agent}], attempts (training runs incl. retries),
    holdout {metrics, majority_baseline, split}, agent_path, decisions, duration_s
"""
from __future__ import annotations

import time
import traceback

import pandas as pd

# Imported up front so a broken harness fails loudly instead of being scored as a "crash".
from adaptive.baselines.fixed_pipeline import run_fixed
from adaptive.baselines.inclass import inclass_graph as inclass_g
from adaptive.baselines.inclass import inclass_runtime as inclass_rt
from adaptive.graph import run_adaptive
from adaptive.training import check_fault

SYSTEMS = ("fixed_airflow", "inclass_agent", "adaptive")
SYSTEM_LABELS = {
    "fixed_airflow": "Fixed Airflow workflow",
    "inclass_agent": "In-class agent loop",
    "adaptive": "Adaptive multi-agent (ours)",
}


def run_system(system: str, *, dataset_path: str, target_column: str, target_score: float, budget: int,
               fault_plan: dict | None) -> dict:
    t0 = time.time()
    try:
        if system == "fixed_airflow":
            out = _run_fixed(dataset_path, target_column, fault_plan)
        elif system == "inclass_agent":
            out = _run_inclass(dataset_path, target_column, target_score, budget, fault_plan)
        else:
            out = _run_adaptive(dataset_path, target_column, target_score, budget, fault_plan)
    except Exception as exc:  # noqa: BLE001 — a crash is a result, not a harness error
        out = {"status": "crashed", "error": f"{type(exc).__name__}: {exc}",
               "trace": traceback.format_exc(limit=3), "experiments": [], "attempts": 0}
    out["system"] = system
    out.setdefault("duration_s", round(time.time() - t0, 2))
    out.setdefault("flagged", [])
    out.setdefault("agent_path", [s.get("node") for s in out.get("steps") or []])
    return out


# ---------------------------------------------------------------------------
def _run_fixed(dataset_path, target_column, fault_plan):
    out = run_fixed(dataset_path=dataset_path, target_column=target_column, fault_plan=fault_plan)
    if out.get("holdout"):
        out["features_used"] = out["holdout"]["features_used"]
    return out


def _run_inclass(dataset_path, target_column, target_score, budget, fault_plan):
    """The unmodified in-class loop. Faults are injected by wrapping its model builder."""
    g, rt = inclass_g, inclass_rt
    original = rt._build_model
    calls = {"n": 0}

    def faulty_build(family, hp):
        n = calls["n"]
        calls["n"] += 1
        check_fault(fault_plan, family, n)
        return original(family, hp)

    rt._build_model = faulty_build
    try:
        res = g.run_solution_loop(dataset_path=dataset_path, target_column=target_column,
                                  target_score=target_score, max_cycles=budget)
    finally:
        rt._build_model = original

    steps = res.get("steps") or []
    experiments = [
        {"agent": "ml_engineer", "model_family": s.get("model_family"), "status": s.get("status"),
         "val_f1": s.get("f1"), "hyperparameters": (s.get("spec") or {}).get("hyperparameters"), "feature_ops": []}
        for s in steps if s.get("node") == "ml_engineer"
    ]
    df = pd.read_csv(dataset_path)
    best = res.get("best_spec") or {}
    drops = [c for c in best.get("drop_columns") or [] if c in df.columns]
    features = [c for c in df.columns if c != target_column and c not in drops]
    y = df[target_column]
    majority_acc = float((y == y.mode().iloc[0]).mean())  # stratified split ≈ same class balance
    profile = res.get("profile") or {}
    hold = res.get("holdout_metrics") or {}
    return {
        "status": "trained" if hold else "crashed",
        "experiments": experiments,
        "attempts": calls["n"] - (1 if hold else 0),  # minus the final holdout fit
        "flagged": list(profile.get("leakage_suspects") or []),
        "features_used": features,
        "holdout": {"metrics": {k: hold.get(k) for k in ("f1", "accuracy")}, "split": "random (stratified)",
                    "majority_baseline": {"accuracy": majority_acc, "f1": 0.0}} if hold else None,
        "steps": steps,
    }


def _run_adaptive(dataset_path, target_column, target_score, budget, fault_plan):
    rep = run_adaptive(dataset_path=dataset_path, target_column=target_column, target_score=target_score,
                       budget=budget, fault_plan=fault_plan, use_llm_supervisor=False)
    v = rep.get("validation") or {}
    hold = rep.get("holdout")
    status = "trained" if hold else ("rejected" if v.get("status") == "invalid" else "crashed")
    return {
        "status": status,
        "error": None if hold else rep.get("stop_reason"),
        "experiments": rep.get("experiments") or [],
        "attempts": len(rep.get("experiments") or []),
        "flagged": list(v.get("leakage_columns") or []),
        "features_used": (hold or {}).get("features_used") or [],
        "holdout": hold,
        "decisions": rep.get("decisions") or [],
        "steps": rep.get("steps") or [],
        "validation": v,
    }
