"""Part 3: adaptive multi-agent workflow (LangGraph).

A Supervisor agent decides, after every step, which agent runs next, based on the
intermediate results so far. Nothing after the first two steps is fixed in advance.

    START → data_validator → supervisor ⇄ {data_scientist, ml_engineer, tuner,
                                            feature_engineer, error_recovery}
                                 │            (every training agent → judge → supervisor)
                                 ├→ holdout_eval → report_writer → END
                                 └→ report_writer → END   (dataset rejected)

Agents and responsibilities
  data_validator    code   contract, binary target, leakage, time column (reused from Part 2)
  data_scientist    LLM*   profile + risks (reused from Part 2)
  supervisor        code   (+ LLM* tie-breaker) picks the next agent from the state
  ml_engineer       code   tries a new model family (explore)
  tuner             code   more capacity for the best family so far (exploit)
  feature_engineer  code   adds derived features, then re-trains the best spec
  error_recovery    code   retry → repair hyperparameters → give up on that family
  judge             code   scores each run, tracks best / improvement / plateau
  holdout_eval      code   opens the sealed holdout once
  report_writer     code   evaluation report incl. the decision log and agent path
  (* uses OpenRouter when a key is set and ADAPTIVE_USE_LLM=1, else a heuristic)
"""
from __future__ import annotations

import json
import operator
import os
import time
from pathlib import Path
from typing import Annotated, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from adaptive.training import FEATURE_BUNDLES, InjectedFault, check_fault, train_eval
from loop_runtime import (
    FAMILIES,
    _CAPACITY_GRID,
    _resolve_model,
    extract_json_object,
    llm_configured,
    openrouter_chat,
)
from solution.langgraph.graph import node_data_scientist, node_data_validator

REPO = Path(__file__).resolve().parents[1]
MAX_RETRIES_PER_SPEC = 2  # error_recovery: 1) plain retry  2) repaired hyperparameters
PLATEAU_DELTA = 0.01  # val F1 gain below this counts as "no progress"
FAR_BELOW_TARGET = 0.15  # gap that makes the supervisor try features before more models


def _append(a: list, b: list) -> list:
    return list(a or []) + list(b or [])


class AdaptiveState(TypedDict, total=False):
    dataset_path: str
    target_column: str
    target_score: float
    budget: int  # max training runs, retries included
    seed: int
    holdout_frac: float
    model: str
    fault_plan: dict  # evaluation only
    use_llm_supervisor: bool
    validation: dict
    profile: dict
    feature_ops: list
    feature_bundles_used: int
    blocked_families: list
    pending_spec: dict  # spec a worker wants judged (or retried)
    retries_for_pending: int
    attempts: int
    experiments: Annotated[list, _append]
    last_result: dict
    best: dict
    plateau_count: int
    next_agent: str
    decisions: Annotated[list, _append]
    holdout: dict
    report: dict
    stop_reason: str
    steps: Annotated[list, operator.add]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _df(state: AdaptiveState) -> pd.DataFrame:
    return pd.read_csv(state["dataset_path"])


def _signature(spec: dict, ops: list) -> str:
    return json.dumps(
        [spec.get("model_family"), spec.get("hyperparameters"), sorted(ops or [])], sort_keys=True, default=str
    )


def _tried_signatures(state: AdaptiveState) -> set[str]:
    return {e["signature"] for e in state.get("experiments") or [] if e.get("status") == "succeeded"}


def _families_tried(state: AdaptiveState) -> list[str]:
    return list(dict.fromkeys(e["model_family"] for e in state.get("experiments") or []))


def _base_spec(state: AdaptiveState, family: str, level: int) -> dict:
    grid = _CAPACITY_GRID[family]
    return {
        "model_family": family,
        "hyperparameters": dict(grid[min(level, len(grid) - 1)]),
        "capacity_level": min(level, len(grid) - 1),
        "drop_columns": list((state.get("validation") or {}).get("must_drop") or []),
    }


def _run(state: AdaptiveState, spec: dict, ops: list, agent: str, note: str) -> dict:
    """Train once (with fault injection) and return a state patch for the judge."""
    attempt = int(state.get("attempts") or 0)
    validation = state.get("validation") or {}
    record = {
        "attempt": attempt + 1,
        "agent": agent,
        "model_family": spec["model_family"],
        "hyperparameters": spec["hyperparameters"],
        "feature_ops": list(ops),
        "signature": _signature(spec, ops),
    }
    try:
        check_fault(state.get("fault_plan"), spec["model_family"], attempt)
        res = train_eval(
            _df(state),
            state["target_column"],
            spec,
            feature_ops=ops,
            time_column=validation.get("time_column"),
            seed=int(state.get("seed") or 42),
            holdout_frac=float(state.get("holdout_frac") or 0.2),
        )
        record.update(status="succeeded", val_f1=res["metrics"]["f1"], val_metrics=res["metrics"],
                      features_used=res["features_used"], split=res["split"])
        msg = f"{note} · {spec['model_family']} val f1={res['metrics']['f1']:.3f}"
    except Exception as exc:  # noqa: BLE001 — failures are data for the supervisor
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}", val_f1=None)
        msg = f"{note} · {spec['model_family']} FAILED: {exc}"
    return {
        "attempts": attempt + 1,
        "pending_spec": {**spec, "feature_ops": list(ops)},
        "last_result": record,
        "steps": [{"node": agent, "note": msg, "status": record["status"], "output": record}],
    }


# ---------------------------------------------------------------------------
# agents
# ---------------------------------------------------------------------------
def node_ml_engineer(state: AdaptiveState) -> dict:
    """Explore: train a model family that hasn't been tried (and isn't blocked)."""
    tried = set(_families_tried(state)) | set(state.get("blocked_families") or [])
    family = next((f for f in FAMILIES if f not in tried), None) or "random_forest"
    spec = _base_spec(state, family, 0)
    patch = _run(state, spec, state.get("feature_ops") or [], "ml_engineer", "explore new family")
    patch["retries_for_pending"] = 0
    return patch


def node_tuner(state: AdaptiveState) -> dict:
    """Exploit: next capacity level for the best family, skipping configs already run."""
    best = state.get("best") or {}
    family = best.get("model_family") or "random_forest"
    ops = list(state.get("feature_ops") or [])
    done = _tried_signatures(state)
    spec = None
    for level in range(int(best.get("capacity_level", 0)) + 1, len(_CAPACITY_GRID[family])):
        candidate = _base_spec(state, family, level)
        if _signature(candidate, ops) not in done:
            spec = candidate
            break
    spec = spec or _base_spec(state, family, len(_CAPACITY_GRID[family]) - 1)
    patch = _run(state, spec, ops, "tuner", f"tune {family} capacity level {spec['capacity_level']}")
    patch["retries_for_pending"] = 0
    return patch


def node_feature_engineer(state: AdaptiveState) -> dict:
    """Add the next feature bundle, then re-train the best spec so its effect is measured."""
    used = int(state.get("feature_bundles_used") or 0)
    ops = list(state.get("feature_ops") or []) + FEATURE_BUNDLES[used]
    best = state.get("best") or {}
    family = best.get("model_family") or "logistic_regression"
    spec = _base_spec(state, family, int(best.get("capacity_level", 0)))
    patch = _run(state, spec, ops, "feature_engineer", f"add features {FEATURE_BUNDLES[used]}")
    patch.update(feature_ops=ops, feature_bundles_used=used + 1, retries_for_pending=0)
    return patch


def node_error_recovery(state: AdaptiveState) -> dict:
    """Retry 1: same spec (transient errors). Retry 2: safe default hyperparameters."""
    pending = dict(state.get("pending_spec") or {})
    retries = int(state.get("retries_for_pending") or 0) + 1
    ops = pending.pop("feature_ops", state.get("feature_ops") or [])
    family = pending.get("model_family") or "random_forest"
    if retries == 1:
        spec, note = pending, "retry same spec"
    else:
        spec, note = _base_spec(state, family, 0), "retry with repaired (default) hyperparameters"
    patch = _run(state, spec, ops, "error_recovery", f"{note} [{retries}/{MAX_RETRIES_PER_SPEC}]")
    patch["retries_for_pending"] = retries
    return patch


def node_judge(state: AdaptiveState) -> dict:
    """Score the last run; keep the best; detect plateaus; block families that keep failing."""
    rec = dict(state.get("last_result") or {})
    best = dict(state.get("best") or {})
    target = float(state.get("target_score") or 0.8)
    patch: dict = {"experiments": [rec]}
    plateau = int(state.get("plateau_count") or 0)
    if rec.get("status") == "succeeded":
        prev = float(best.get("val_f1") or 0.0)
        gain = rec["val_f1"] - prev
        if rec["val_f1"] > prev:
            spec = dict(state.get("pending_spec") or {})
            best = {
                "val_f1": rec["val_f1"],
                "model_family": rec["model_family"],
                "capacity_level": spec.get("capacity_level", 0),
                "spec": {k: v for k, v in spec.items() if k != "feature_ops"},
                "feature_ops": spec.get("feature_ops") or [],
                "attempt": rec["attempt"],
            }
            patch["best"] = best
        plateau = plateau + 1 if gain < PLATEAU_DELTA else 0
        verdict = "target met" if rec["val_f1"] >= target else ("improved" if gain >= PLATEAU_DELTA else "no progress")
        note = f"{verdict}: val f1 {rec['val_f1']:.3f} (best {max(prev, rec['val_f1']):.3f}, target {target:.2f})"
    else:
        verdict = "failed"
        note = f"run failed: {rec.get('error')}"
        if int(state.get("retries_for_pending") or 0) >= MAX_RETRIES_PER_SPEC:
            blocked = list(dict.fromkeys(list(state.get("blocked_families") or []) + [rec["model_family"]]))
            patch["blocked_families"] = blocked
            note += f" · retries exhausted, blocking {rec['model_family']}"
    patch["plateau_count"] = plateau
    patch["steps"] = [{"node": "judge", "note": note, "verdict": verdict}]
    return patch


def _rule_decision(state: AdaptiveState) -> tuple[str, str, list[str]]:
    """Return (next agent, reason, allowed alternatives). Pure function of the state."""
    v = state.get("validation") or {}
    if v.get("status") == "invalid":
        return "report_writer", f"dataset rejected: {'; '.join(v.get('issues') or [])}", []
    if not state.get("profile"):
        return "data_scientist", "no profile yet", []
    last = state.get("last_result") or {}
    attempts = int(state.get("attempts") or 0)
    budget = int(state.get("budget") or 8)
    best = state.get("best") or {}
    target = float(state.get("target_score") or 0.8)
    if last.get("status") == "failed" and int(state.get("retries_for_pending") or 0) < MAX_RETRIES_PER_SPEC and attempts < budget:
        return "error_recovery", f"last run failed ({last.get('error')}); retry", []
    if attempts >= budget:
        if best:
            return "holdout_eval", f"budget of {budget} training runs used; finalize best", []
        return "report_writer", "budget used and no run succeeded", []
    if best and float(best["val_f1"]) >= target:
        return "holdout_eval", f"target met (val f1 {best['val_f1']:.3f} ≥ {target:.2f})", []

    blocked = set(state.get("blocked_families") or [])
    untried = [f for f in FAMILIES if f not in set(_families_tried(state)) | blocked]
    bundles_left = int(state.get("feature_bundles_used") or 0) < len(FEATURE_BUNDLES)
    plateau = int(state.get("plateau_count") or 0)
    tunable = bool(best) and best.get("model_family") not in blocked and int(best.get("capacity_level", 0)) < len(
        _CAPACITY_GRID[best["model_family"]]
    ) - 1
    gap = target - float(best.get("val_f1") or 0)
    n_ok = len({e["model_family"] for e in state.get("experiments") or [] if e.get("status") == "succeeded"})
    options = [a for a, ok in (("ml_engineer", bool(untried)), ("feature_engineer", bundles_left and bool(best)),
                               ("tuner", tunable)) if ok]

    if not best:
        return ("ml_engineer", "no successful model yet; start with a baseline family", options) if untried else (
            "report_writer", "every model family failed", [])
    if bundles_left and (plateau >= 1 or (gap > FAR_BELOW_TARGET and n_ok >= 2)):
        why = "progress stalled" if plateau >= 1 else f"still {gap:.2f} below target after {n_ok} working families"
        return "feature_engineer", f"{why}; features are the likely bottleneck", options
    if untried:
        return "ml_engineer", f"untried families: {untried}", options
    if tunable:
        return "tuner", f"all families tried; give {best['model_family']} more capacity", options
    if bundles_left:
        return "feature_engineer", "no tuning left; try new features", options
    return "holdout_eval", "search space exhausted; finalize best", []


def node_supervisor(state: AdaptiveState) -> dict:
    choice, reason, options = _rule_decision(state)
    source = "rules"
    model = _resolve_model(state)
    if state.get("use_llm_supervisor") and len(options) > 1 and llm_configured(model):
        system = (
            "You are the Supervisor of an ML research team. Choose the next agent. "
            f"Allowed: {options}. Return ONLY JSON: {{\"next_agent\": <one of allowed>, \"reason\": <short>}}."
        )
        summary = {
            "target": state.get("target_score"), "best": state.get("best"),
            "experiments": [{k: e.get(k) for k in ("agent", "model_family", "val_f1", "status", "feature_ops")}
                            for e in state.get("experiments") or []][-6:],
            "budget_left": int(state.get("budget") or 8) - int(state.get("attempts") or 0),
            "rule_suggestion": choice,
        }
        content, _err = openrouter_chat(system, json.dumps(summary, default=str), model=model, max_tokens=200, temperature=0.1)
        parsed = extract_json_object(content or "") or {}
        if parsed.get("next_agent") in options:  # the LLM can only pick an allowed agent
            choice, reason, source = parsed["next_agent"], str(parsed.get("reason") or reason), f"llm · {model}"
    decision = {
        "step": len(state.get("decisions") or []) + 1,
        "next_agent": choice,
        "reason": reason,
        "source": source,
        "attempts_used": int(state.get("attempts") or 0),
        "best_val_f1": (state.get("best") or {}).get("val_f1"),
    }
    return {
        "next_agent": choice,
        "decisions": [decision],
        "steps": [{"node": "supervisor", "note": f"→ {choice}: {reason}", "output": decision}],
    }


def node_holdout_eval(state: AdaptiveState) -> dict:
    best = state["best"]
    validation = state.get("validation") or {}
    res = train_eval(
        _df(state), state["target_column"], best["spec"],
        feature_ops=best.get("feature_ops"), time_column=validation.get("time_column"),
        seed=int(state.get("seed") or 42), holdout_frac=float(state.get("holdout_frac") or 0.2),
        on_holdout=True, return_predictions=True,
    )
    labels = sorted(set(res["y_true"]) | set(res["y_pred"]), key=str)
    cm = [[sum(1 for t, p in zip(res["y_true"], res["y_pred"]) if t == a and p == b) for b in labels] for a in labels]
    holdout = {
        "metrics": res["metrics"],
        "majority_baseline": res["majority_metrics"],
        "confusion_matrix": {"labels": [str(x) for x in labels], "rows_true_cols_pred": cm},
        "features_used": res["features_used"],
        "split": res["split"],
        "n_holdout": res["n_eval"],
    }
    m = res["metrics"]
    return {
        "holdout": holdout,
        "stop_reason": (state.get("decisions") or [{}])[-1].get("reason", "finalize"),
        "steps": [{"node": "holdout_eval", "note": f"holdout f1={m['f1']:.3f} acc={m['accuracy']:.3f}", "metrics": m}],
    }


def node_report_writer(state: AdaptiveState) -> dict:
    path = [s.get("node") for s in state.get("steps") or []] + ["report_writer"]
    import sklearn

    report = {
        "system": "adaptive",
        "sklearn_version": sklearn.__version__,
        "dataset": Path(state["dataset_path"]).name,
        "target": state.get("target_column"),
        "validation": state.get("validation"),
        "decisions": state.get("decisions") or [],
        "experiments": [{k: v for k, v in e.items() if k != "signature"} for e in state.get("experiments") or []],
        "best": state.get("best"),
        "holdout": state.get("holdout"),
        "agent_path": path,
        "stop_reason": state.get("stop_reason") or (state.get("decisions") or [{}])[-1].get("reason"),
    }
    out_dir = Path(os.getenv("LOOP_REPORT_DIR", str(REPO / "artifacts" / "loop_reports")))
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        f = out_dir / f"adaptive-report-{time.strftime('%Y%m%d-%H%M%S')}.json"
        f.write_text(json.dumps(report, indent=2, default=str))
        report["saved_to"] = str(f)
    except OSError as exc:
        report["saved_to"] = f"not saved: {exc}"
    note = (f"holdout f1={state['holdout']['metrics']['f1']:.3f}" if state.get("holdout") else f"no model: {report['stop_reason']}")
    return {"report": report, "steps": [{"node": "report_writer", "note": note}]}


# ---------------------------------------------------------------------------
# graph
# ---------------------------------------------------------------------------
WORKERS = ("ml_engineer", "tuner", "feature_engineer", "error_recovery")
ROUTES = ("data_scientist", *WORKERS, "holdout_eval", "report_writer")


def build_graph():
    g = StateGraph(AdaptiveState)
    g.add_node("data_validator", node_data_validator)
    g.add_node("supervisor", node_supervisor)
    g.add_node("data_scientist", node_data_scientist)
    g.add_node("ml_engineer", node_ml_engineer)
    g.add_node("tuner", node_tuner)
    g.add_node("feature_engineer", node_feature_engineer)
    g.add_node("error_recovery", node_error_recovery)
    g.add_node("judge", node_judge)
    g.add_node("holdout_eval", node_holdout_eval)
    g.add_node("report_writer", node_report_writer)

    g.add_edge(START, "data_validator")
    g.add_edge("data_validator", "supervisor")
    g.add_conditional_edges("supervisor", lambda s: s["next_agent"], {r: r for r in ROUTES})
    g.add_edge("data_scientist", "supervisor")
    for w in WORKERS:
        g.add_edge(w, "judge")
    g.add_edge("judge", "supervisor")
    g.add_edge("holdout_eval", "report_writer")
    g.add_edge("report_writer", END)
    return g.compile()


def run_adaptive(
    *,
    dataset_path: str,
    target_column: str,
    target_score: float = 0.8,
    budget: int = 8,
    seed: int = 42,
    holdout_frac: float = 0.2,
    fault_plan: dict | None = None,
    use_llm_supervisor: bool | None = None,
    model: str | None = None,
) -> dict:
    if use_llm_supervisor is None:
        use_llm_supervisor = os.getenv("ADAPTIVE_USE_LLM", "0") == "1"
    initial: AdaptiveState = {
        "dataset_path": str(Path(dataset_path).resolve()),
        "target_column": target_column,
        "target_score": target_score,
        "budget": budget,
        "seed": seed,
        "holdout_frac": holdout_frac,
        "fault_plan": fault_plan or {},
        "use_llm_supervisor": use_llm_supervisor,
        "steps": [],
        "experiments": [],
        "decisions": [],
        "feature_ops": [],
    }
    if model:
        initial["model"] = model
    app = build_graph()
    final = app.invoke(initial, config={"recursion_limit": 30 + budget * 8})
    return final["report"] | {"steps": final.get("steps") or []}


def mermaid() -> str:
    return build_graph().get_graph().draw_mermaid()
