"""Simple research loop — one self-contained LangGraph file.

Flow (modified for the Citi Bike hourly-demand dataset; see ../../CHANGES.md):
  START → data_validator ──(invalid)─────────────────────────────┐
              │ ok                                                  ▼
          data_scientist → hypothesis → ml_engineer → judge     report_writer → END
                             ↑______________________________|        ▲
                             |  (revise: memory → context → hypothesis)
                             └── accept/stop → holdout_eval ────────┘

New vs the in-class loop:
  * data_validator  (code agent) checks the dataset contract published by Airflow and
                    finds the time column and target-leakage columns.
  * ml_engineer     always drops the validator's leakage columns (a guardrail that holds
                    even if the LLM forgets) and splits chronologically when there is a
                    time column.
  * report_writer   (code agent) writes the holdout results (confusion matrix, majority-class
                    baseline, feature importances) and the experiment history to a report.

LLM / train helpers live in loop_runtime.py (Cursor SDK + OpenRouter + heuristics).
"""

from __future__ import annotations

import json
import operator
import os
import sys
from pathlib import Path
from typing import Annotated, TypedDict

import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score
from langgraph.graph import END, START, StateGraph

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
# Repo root locally; /app in the gateway container (loop_runtime.py is mounted there).
for _p in (REPO, Path("/app")):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from loop_runtime import (  # noqa: E402
    DEFAULT_MODEL,
    FAMILIES,
    _SPEC_SCHEMA_HINT,
    _build_pipeline,
    _classification_metrics,
    _coerce_experiment_spec,
    _fit_score,
    _heuristic_plan,
    _heuristic_profile,
    _heuristic_spec,
    _heuristic_verdict,
    _repair_heuristic,
    _resolve_model,
    _split,
    _validate_spec,
    extract_json_object,
    llm_configured,
    openrouter_chat,
)

# ---------------------------------------------------------------------------
# STATE
# ---------------------------------------------------------------------------
class SimpleLoopState(TypedDict, total=False):
    dataset_path: str
    target_column: str
    target_score: float
    max_cycles: int
    cycle: int
    seed: int
    holdout_frac: float
    model: str  # OpenRouter / Cursor model id from the UI / env
    profile: dict
    brief: dict
    hypothesis: dict  # conceptual plan from hypothesis_generator (not HPT numbers)
    experiment_spec: dict  # concrete HPT config from ml_engineer
    execution_result: dict
    verdict: dict
    memory: dict
    best_spec: dict
    best_val_f1: float
    validation: dict  # data_validator output: status, issues, time_column, must_drop
    holdout_details: dict  # holdout_eval output reused by report_writer (single holdout fit)
    report: dict  # report_writer output
    holdout_metrics: dict
    stop_reason: str
    steps: Annotated[list, operator.add]


# ---------------------------------------------------------------------------
# NODES
# ---------------------------------------------------------------------------
def node_data_scientist(state: SimpleLoopState) -> dict:
    """Profile CSV via OpenRouter; heuristic fallback if key/parse fails."""
    path = Path(state["dataset_path"])
    target = state["target_column"]
    df = pd.read_csv(path)
    if target not in df.columns:
        return {
            "profile": {},
            "steps": [{"node": "data_scientist", "note": f"target {target!r} missing", "status": "error"}],
        }

    profile = _heuristic_profile(df, target)
    used_llm = False
    note = "heuristic"
    model = _resolve_model(state)

    if llm_configured(model):
        missing = {c: float(df[c].isna().mean()) for c in df.columns}
        nunique = {c: int(df[c].nunique(dropna=True)) for c in df.columns}
        sample = df.head(6).astype(str).replace("nan", "").to_dict(orient="records")
        system = (
            "You are the Data Scientist only — not the hypothesizer and not the ML engineer. "
            "Your job is dataset understanding: profile the CSV and flag risk. "
            "Do NOT propose model families, hyperparameters, or experiment configs. "
            "Return ONLY one JSON object with keys: "
            "n_rows (int), n_columns (int), target (string), target_rate (number|null), "
            "numeric_features (string[]), categorical_features (string[]), "
            "leakage_suspects (string[]), recommended_drops (string[]), "
            "notes (string[], short bullets about data risks — leakage, imbalance, sparsity). "
            "Flag PassengerId/Name/Ticket/Cabin-style identifiers and near-unique text as leakage. "
            "Do not invent columns that are not present."
        )
        user = (
            f"dataset={path.name}\ntarget={target}\nn_rows={len(df)}\n"
            f"columns={list(df.columns)}\n"
            f"dtypes={{{', '.join(f'{c}:{df[c].dtype}' for c in df.columns)}}}\n"
            f"missing_rate={missing}\nnunique={nunique}\n"
            f"sample_rows={sample}\n"
        )
        content, err = openrouter_chat(system, user, model=model, max_tokens=900, temperature=0.15)
        if content:
            parsed = extract_json_object(content)
            if parsed and (
                "recommended_drops" in parsed
                or "leakage_suspects" in parsed
                or "numeric_features" in parsed
            ):
                profile = parsed
                profile.setdefault("n_rows", int(len(df)))
                profile.setdefault("n_columns", int(len(df.columns)))
                profile.setdefault("target", target)
                profile.setdefault("recommended_drops", profile.get("leakage_suspects") or [])
                used_llm = True
                note = f"llm profile · {model}"
            else:
                note = f"llm unparsable; heuristic ({err or 'ok'})"
        else:
            note = f"{err or 'no llm'}; heuristic"

    must_drop = list((state.get("validation") or {}).get("must_drop") or [])
    if must_drop:
        profile = dict(profile)
        for key in ("recommended_drops", "leakage_suspects"):
            current = [c for c in (profile.get(key) or []) if isinstance(c, str)]
            profile[key] = list(dict.fromkeys(current + must_drop))
        notes = profile.get("notes") if isinstance(profile.get("notes"), list) else []
        profile["notes"] = [*notes, f"data_validator: always drop {must_drop} (leakage / time key)"]

    return {
        "profile": profile,
        "steps": [
            {
                "node": "data_scientist",
                "note": note,
                "used_llm": used_llm,
                "model": model if used_llm else None,
                "output": profile,
            }
        ],
    }


def node_hypothesis(state: SimpleLoopState) -> dict:
    """Conceptual next experiment from DS profile + Judge feedback — no HPT numbers."""
    profile = state.get("profile") or {}
    brief = state.get("brief") or {}
    verdict = state.get("verdict") or {}
    history = list(brief.get("history") or (state.get("memory") or {}).get("history") or [])
    model = _resolve_model(state)

    plan = _heuristic_plan(profile, history, verdict)
    used_llm = False
    note = "heuristic plan"

    if llm_configured(model):
        system = (
            "You are the Hypothesis Generator only — not the Data Scientist and not the ML Engineer. "
            "Read the data-scientist profile and the latest judge verdict / lessons. "
            "Propose ONE conceptual research direction for the next trial. "
            "Do NOT invent concrete hyperparameter values, C/learning_rate/max_depth numbers, "
            "or a full ExperimentSpec — the ML Engineer owns HPT config. "
            "Return ONLY one JSON object with keys: "
            "direction (1-2 sentences: what should change conceptually), "
            "why (cite profile signals and/or judge observation), "
            "focus (one of: baseline_then_iterate|capacity_or_regularization|model_family_switch|feature_drops), "
            "respect_drops (string[] of columns to keep dropped — copy from profile when relevant), "
            "avoid_repeats (string[] of model families or ideas already tried). "
            "Stay conceptual: e.g. 'try a tree ensemble with less regularization' — never 'C=0.5'."
        )
        user = (
            f"profile={json.dumps(profile, default=str)}\n"
            f"latest_verdict={json.dumps(verdict, default=str)}\n"
            f"brief={json.dumps(brief, default=str)}\n"
            f"recent_history={json.dumps(history[-5:], default=str)}\n"
        )
        content, err = openrouter_chat(system, user, model=model, max_tokens=500, temperature=0.4)
        if content:
            parsed = extract_json_object(content)
            if parsed and parsed.get("direction"):
                plan = {
                    "direction": parsed.get("direction"),
                    "why": parsed.get("why") or plan.get("why"),
                    "focus": parsed.get("focus") or plan.get("focus"),
                    "respect_drops": parsed.get("respect_drops")
                    if isinstance(parsed.get("respect_drops"), list)
                    else plan.get("respect_drops"),
                    "avoid_repeats": parsed.get("avoid_repeats")
                    if isinstance(parsed.get("avoid_repeats"), list)
                    else plan.get("avoid_repeats"),
                }
                used_llm = True
                note = f"live conceptual hypothesis · {model}"
            else:
                note = f"llm unparsable; heuristic ({err or 'ok'})"
        else:
            note = f"{err or 'no llm'}; heuristic"

    return {
        "hypothesis": plan,
        "steps": [
            {
                "node": "hypothesis_generator",
                "note": note,
                "hypothesis": plan,
                "output": plan,
                "used_llm": used_llm,
                "model": model if used_llm else None,
            }
        ],
    }


def node_ml_engineer(state: SimpleLoopState) -> dict:
    """Turn conceptual hypothesis into concrete HPT ExperimentSpec, then train/val."""
    df = pd.read_csv(state["dataset_path"])
    target = state["target_column"]
    profile = state.get("profile") or {}
    plan = dict(state.get("hypothesis") or {})
    brief = state.get("brief") or {}
    history = list(brief.get("history") or (state.get("memory") or {}).get("history") or [])
    prev_spec = dict(state.get("experiment_spec") or state.get("best_spec") or {})
    seed = int(state.get("seed") or 42)
    holdout_frac = float(state.get("holdout_frac") or 0.2)
    model_id = _resolve_model(state)

    spec = _heuristic_spec(profile, history, plan)
    used_llm = False
    repaired = False
    note = "heuristic HPT"

    default_drops = (
        plan.get("respect_drops")
        or profile.get("recommended_drops")
        or profile.get("leakage_suspects")
        or []
    )
    next_eid = f"exp-{len(history) + 1}"
    raw_llm: str | None = None

    if llm_configured(model_id):
        system = (
            "You are the ML Engineer only — not the Hypothesis Generator and not the Judge. "
            "The hypothesis agent already decided the conceptual direction. "
            "Your job: translate that plan into a concrete hyperparameter / training config. "
            "Return ONLY one JSON object matching this schema exactly: "
            f"{_SPEC_SCHEMA_HINT} "
            "model_family MUST be one of logistic_regression|random_forest|hist_gradient_boosting. "
            "hyperparameters MUST be a JSON object with concrete numbers for that family "
            "(C/max_iter OR n_estimators/max_depth OR learning_rate/max_iter). "
            "drop_columns is a string array. rationale is one sentence. "
            "No markdown fences, no commentary."
        )
        user = (
            f"hypothesis_plan={json.dumps(plan, default=str)}\n"
            f"profile={json.dumps({k: profile.get(k) for k in ('recommended_drops', 'leakage_suspects', 'n_rows', 'target_rate')}, default=str)}\n"
            f"prior_spec={json.dumps(prev_spec, default=str)}\n"
            f"recent_history={json.dumps(history[-3:], default=str)}\n"
            f"allowed_families={list(FAMILIES)}\n"
            f"next_experiment_id={next_eid}\n"
        )
        content, err = openrouter_chat(
            system, user, model=model_id, max_tokens=700, temperature=0.2, json_object=True
        )
        raw_llm = content
        coerced = _coerce_experiment_spec(
            extract_json_object(content or ""),
            experiment_id=next_eid,
            default_drops=default_drops,
        )
        if coerced is None and content and llm_configured(model_id):
            # One strict repair pass — common with Cursor prose wrapping JSON.
            fix_system = (
                "Convert the text into ONE valid ExperimentSpec JSON object. "
                f"Schema example: {_SPEC_SCHEMA_HINT} "
                f"model_family must be one of {list(FAMILIES)}. "
                "Output JSON only."
            )
            fix_user = (
                f"raw_model_output={content[:3500]}\n"
                f"plan={json.dumps(plan, default=str)}\n"
                f"next_experiment_id={next_eid}\n"
            )
            content2, err2 = openrouter_chat(
                fix_system,
                fix_user,
                model=model_id,
                max_tokens=500,
                temperature=0.0,
                json_object=True,
            )
            raw_llm = content2 or content
            coerced = _coerce_experiment_spec(
                extract_json_object(content2 or ""),
                experiment_id=next_eid,
                default_drops=default_drops,
            )
            err = err2 or err
        if coerced:
            spec = coerced
            used_llm = True
            note = f"live HPT config · {model_id}"
        else:
            note = f"llm unparsable HPT; heuristic ({err or 'ok'})"

    err = _validate_spec(spec)
    if err and llm_configured(model_id):
        system = (
            "You are the ML Engineer repairing an invalid HPT ExperimentSpec. "
            f"Return ONLY JSON matching: {_SPEC_SCHEMA_HINT} "
            f"model_family must be one of {list(FAMILIES)}."
        )
        user = (
            f"error={err}\n"
            f"spec={json.dumps(spec, default=str)}\n"
            f"plan={json.dumps(plan, default=str)}\n"
            f"raw_model_output={(raw_llm or '')[:2000]}\n"
        )
        content, _llm_err = openrouter_chat(
            system, user, model=model_id, max_tokens=500, temperature=0.0, json_object=True
        )
        coerced = _coerce_experiment_spec(
            extract_json_object(content or ""),
            experiment_id=next_eid,
            default_drops=default_drops,
        )
        if coerced:
            spec = coerced
            used_llm = True
            repaired = True
            note = f"repaired HPT after: {err}"
            err = _validate_spec(spec)
    if err:
        spec = _repair_heuristic(spec, err)
        repaired = True
        note = f"heuristic repair after: {err}"
        err = _validate_spec(spec)

    validation = state.get("validation") or {}
    time_column = validation.get("time_column")
    must_drop = list(validation.get("must_drop") or [])
    if must_drop:
        before = list(spec.get("drop_columns") or [])
        spec = {**spec, "drop_columns": list(dict.fromkeys(before + must_drop))}
        added = [c for c in must_drop if c not in before]
        if added:
            note = f"{note} · guard dropped {added}"

    cycle = int(state.get("cycle") or 0) + 1
    if err:
        result = {"status": "failed", "error": err, "validation_metrics": {}}
        return {
            "execution_result": result,
            "experiment_spec": spec,
            "cycle": cycle,
            "steps": [
                {
                    "node": "ml_engineer",
                    "note": note,
                    "status": "failed",
                    "repaired": repaired,
                    "used_llm": used_llm,
                    "experiment_id": spec.get("experiment_id"),
                    "model_family": spec.get("model_family"),
                    "spec": spec,
                    "output": {"experiment_spec": spec, "error": err},
                }
            ],
        }

    try:
        result = _fit_score(df, target, spec, seed, holdout_frac, on_holdout=False, time_column=time_column)
    except Exception as exc:  # noqa: BLE001
        if not repaired:
            spec = _repair_heuristic(spec, str(exc))
            repaired = True
            note = f"heuristic repair after run error: {exc}"
            try:
                result = _fit_score(df, target, spec, seed, holdout_frac, on_holdout=False, time_column=time_column)
            except Exception as exc2:  # noqa: BLE001
                result = {"status": "failed", "error": str(exc2), "validation_metrics": {}}
        else:
            result = {"status": "failed", "error": str(exc), "validation_metrics": {}}

    f1 = float((result.get("validation_metrics") or {}).get("f1") or 0)
    patch: dict = {
        "execution_result": result,
        "experiment_spec": spec,
        "cycle": cycle,
        "steps": [
            {
                "node": "ml_engineer",
                "note": note if result.get("status") != "succeeded" else f"{note} · f1={f1:.3f}",
                "f1": f1 if result.get("status") == "succeeded" else None,
                "status": result.get("status"),
                "repaired": repaired,
                "used_llm": used_llm,
                "experiment_id": spec.get("experiment_id"),
                "model_family": spec.get("model_family"),
                "validation_metrics": result.get("validation_metrics") or {},
                "duration_seconds": result.get("duration_seconds"),
                "spec": spec,
                "output": {
                    "experiment_spec": spec,
                    "execution_result": {
                        "status": result.get("status"),
                        "validation_metrics": result.get("validation_metrics"),
                        "error": result.get("error"),
                    },
                },
            }
        ],
    }
    if result.get("status") == "succeeded" and f1 > float(state.get("best_val_f1") or -1):
        patch["best_val_f1"] = f1
        patch["best_spec"] = spec
    return patch


def node_judge(state: SimpleLoopState) -> dict:
    """Verdict via OpenRouter; heuristic fallback."""
    result = state.get("execution_result") or {}
    spec = state.get("experiment_spec") or {}
    plan = state.get("hypothesis") or {}
    brief = state.get("brief") or {}
    target_score = float(state.get("target_score") or 0.8)
    model = _resolve_model(state)

    verdict = _heuristic_verdict(result, target_score)
    used_llm = False
    note = "heuristic"

    if llm_configured(model):
        system = (
            "You are the Judge only — not the Hypothesis Generator and not the ML Engineer. "
            "Decide from VALIDATION evidence only — never invent holdout scores, "
            "never rewrite hyperparameters. "
            "Return ONLY one JSON object with keys: "
            "action (accept|revise|reject), observation (1-2 sentences with the val F1), "
            "next_action (short conceptual instruction for the hypothesis agent — no HPT numbers). "
            "accept only if validation F1 meets or exceeds target_score; "
            "revise when under target but recoverable; reject for hopeless/broken runs."
        )
        user = (
            f"target_score={target_score}\n"
            f"hypothesis_plan={json.dumps(plan, default=str)}\n"
            f"spec={json.dumps(spec, default=str)}\n"
            f"result={json.dumps(result, default=str)}\n"
            f"brief={json.dumps(brief, default=str)}\n"
        )
        content, err = openrouter_chat(system, user, model=model, max_tokens=500, temperature=0.1)
        if content:
            parsed = extract_json_object(content)
            if parsed and parsed.get("action") in {"accept", "revise", "reject"}:
                verdict = {
                    "action": parsed["action"],
                    "observation": parsed.get("observation") or verdict.get("observation"),
                    "next_action": parsed.get("next_action") or verdict.get("next_action"),
                }
                used_llm = True
                note = f"llm verdict · {model}"
            else:
                note = f"llm unparsable; heuristic ({err or 'ok'})"
        else:
            note = f"{err or 'no llm'}; heuristic"

    return {
        "verdict": verdict,
        "steps": [
            {
                "node": "judge",
                "note": note,
                "verdict": verdict,
                "used_llm": used_llm,
                "model": model if used_llm else None,
            }
        ],
    }


def node_memory_writer(state: SimpleLoopState) -> dict:
    """Append this cycle's lesson so the next hypothesis can see history."""
    mem = dict(state.get("memory") or {"history": [], "findings": [], "lessons": []})
    hist = list(mem.get("history") or [])
    hist.append(
        {
            "cycle": state.get("cycle"),
            "hypothesis": state.get("hypothesis"),
            "spec": state.get("experiment_spec"),
            "model_family": (state.get("experiment_spec") or {}).get("model_family"),
            "f1": (state.get("execution_result") or {}).get("validation_metrics", {}).get("f1"),
            "verdict": state.get("verdict"),
        }
    )
    mem["history"] = hist
    lessons = list(mem.get("lessons") or [])
    lessons.append(str((state.get("verdict") or {}).get("observation") or ""))
    mem["lessons"] = lessons
    return {"memory": mem, "steps": [{"node": "memory_writer", "note": "filed lesson"}]}


def node_context_builder(state: SimpleLoopState) -> dict:
    """Build a short brief for the next hypothesis from recent memory."""
    mem = state.get("memory") or {}
    brief = {
        "history": mem.get("history") or [],
        "lessons": (mem.get("lessons") or [])[-3:],
    }
    return {"brief": brief, "steps": [{"node": "context_builder", "note": "next brief"}]}


def node_holdout(state: SimpleLoopState) -> dict:
    """Score the best spec ONCE on the sealed holdout. This is the only holdout fit: the
    metrics, confusion matrix, baseline and importances all come from this one model."""
    df = pd.read_csv(state["dataset_path"])
    target = state["target_column"]
    spec = state.get("best_spec") or state.get("experiment_spec") or {}
    seed = int(state.get("seed") or 42)
    holdout_frac = float(state.get("holdout_frac") or 0.2)
    if _validate_spec(spec):
        spec = _repair_heuristic(dict(spec), "invalid best_spec")
    time_column = (state.get("validation") or {}).get("time_column")
    X_tr, y_tr, X_ho, y_ho, _, split_kind = _split(
        df, target, list(spec.get("drop_columns") or []), seed, holdout_frac,
        on_holdout=True, time_column=time_column,
    )
    pipe = _build_pipeline(X_tr, spec)
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_ho)
    labels = sorted(pd.unique(pd.concat([y_tr, y_ho])))
    majority = y_tr.mode().iloc[0]
    cm = pd.crosstab(pd.Categorical(y_ho, categories=labels), pd.Categorical(pred, categories=labels), dropna=False)
    metrics = {**_classification_metrics(y_ho, pred), "n_holdout": int(len(y_ho)), "split": split_kind}
    details = {
        "best_spec": spec,
        "split": split_kind,
        "features_used": list(X_tr.columns),
        "holdout_metrics": metrics,
        "majority_baseline": {"predicts": str(majority), **_classification_metrics(y_ho, [majority] * len(y_ho))},
        "confusion_matrix": {"labels": [str(v) for v in labels], "rows_true_cols_pred": cm.values.tolist()},
        "feature_importances": _feature_importances(pipe),
    }
    return {
        "holdout_metrics": metrics,
        "holdout_details": details,
        "stop_reason": "finalize",
        "steps": [{"node": "holdout_eval", "note": "sealed holdout", "metrics": metrics}],
    }


# ---------------------------------------------------------------------------
# NEW AGENTS (Citi Bike adaptation)
# ---------------------------------------------------------------------------
TIME_COLUMN_NAMES = ("hour", "timestamp", "datetime", "date", "time", "started_at")
LEAKAGE_AUC_THRESHOLD = 0.98  # one column alone separates the classes this well = leakage
MIN_ROWS = 200
MAX_MISSING_RATE = 0.2


def _load_contract(csv_path: Path) -> dict:
    """Optional sidecar `<name>.contract.json` written by the Airflow pipeline."""
    sidecar = csv_path.with_suffix(".contract.json")
    if sidecar.exists():
        try:
            return json.loads(sidecar.read_text())
        except (OSError, ValueError):
            return {}
    return {}


def _detect_time_column(df: pd.DataFrame, target: str, contract: dict) -> str | None:
    declared = contract.get("time_column")
    if declared in df.columns:
        return declared
    for col in df.columns:
        if col == target or col.lower() not in TIME_COLUMN_NAMES:
            continue
        parsed = pd.to_datetime(df[col], errors="coerce")
        if parsed.notna().mean() > 0.95:
            return col
    return None


def node_data_validator(state: SimpleLoopState) -> dict:
    """Code agent (no LLM): check the dataset before any modeling.

    Blocking issues route straight to report_writer; warnings are passed on.
    Its `must_drop` list is enforced by the ML Engineer on every experiment.
    """
    path = Path(state["dataset_path"])
    target = state["target_column"]
    df = pd.read_csv(path)
    contract = _load_contract(path)
    issues: list[str] = []
    warnings: list[str] = []

    if target not in df.columns:
        issues.append(f"target column {target!r} not in dataset")
    elif df[target].nunique(dropna=True) != 2:
        issues.append(
            f"target {target!r} has {df[target].nunique(dropna=True)} classes; this loop needs a binary target"
        )
    if len(df) < MIN_ROWS:
        issues.append(f"only {len(df)} rows (need at least {MIN_ROWS})")
    if contract.get("target") and contract["target"] != target:
        warnings.append(f"contract says target is {contract['target']!r}, run uses {target!r}")

    missing = df.isna().mean()
    for col, rate in missing[missing > MAX_MISSING_RATE].items():
        warnings.append(f"{col}: {rate:.0%} missing")
    if target in df.columns and df[target].isna().any():
        issues.append(f"target has {int(df[target].isna().sum())} missing values")

    time_column = _detect_time_column(df, target, contract)
    if time_column:
        dup = int(df[time_column].duplicated().sum())
        if dup:
            issues.append(f"{dup} duplicate {time_column!r} values (expected one row per period)")

    leakage = [c for c in contract.get("leakage_columns") or [] if c in df.columns]
    if target in df.columns and df[target].nunique(dropna=True) == 2 and not issues:
        # Label-dependent screening uses the TRAINING segment only (same split the ML
        # Engineer uses), so validation/holdout labels never influence which columns stay.
        X_tr, y_tr, *_ = _split(
            df, target, [], int(state.get("seed") or 42), float(state.get("holdout_frac") or 0.2),
            on_holdout=False, time_column=time_column,
        )
        y = pd.Series(pd.factorize(y_tr)[0], index=y_tr.index)
        for col in X_tr.columns:
            if col in leakage or y.nunique() < 2:
                continue
            if pd.api.types.is_numeric_dtype(X_tr[col]) and X_tr[col].nunique() > 1:
                # AUC of the raw column as a score: 1.0 means a single threshold on this
                # column reproduces the label, e.g. a label defined as `trips >= q`.
                ok = X_tr[col].notna()
                auc = roc_auc_score(y[ok], X_tr.loc[ok, col])
                auc = max(auc, 1 - auc)
                if auc >= LEAKAGE_AUC_THRESHOLD:
                    leakage.append(col)
                    warnings.append(f"{col}: alone separates the target (AUC={auc:.3f}), treated as leakage")

    must_drop = list(dict.fromkeys(leakage + ([time_column] if time_column else [])))
    target_rate = (
        float(pd.Series(pd.factorize(df[target])[0]).mean()) if target in df.columns else None
    )
    validation = {
        "status": "invalid" if issues else "ok",
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "target_rate": target_rate,
        "time_column": time_column,
        "split": "chronological" if time_column else "random (stratified)",
        "leakage_columns": leakage,
        "must_drop": must_drop,
        "contract_found": bool(contract),
        "issues": issues,
        "warnings": warnings,
    }
    note = (
        f"invalid: {issues[0]}"
        if issues
        else f"ok · split={validation['split']} · drop {must_drop or 'nothing'}"
    )
    return {
        "validation": validation,
        "steps": [
            {
                "node": "data_validator",
                "note": note,
                "status": "failed" if issues else "completed",
                "used_llm": False,
                "output": validation,
            }
        ],
    }


def route_after_validator(state: SimpleLoopState) -> str:
    return "report" if (state.get("validation") or {}).get("status") == "invalid" else "ok"


def _feature_importances(pipe, top: int = 8) -> dict:
    try:
        names = list(pipe.named_steps["pre"].get_feature_names_out())
        clf = pipe.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            values = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            values = abs(clf.coef_[0])
        else:
            return {}
    except Exception:  # noqa: BLE001
        return {}
    pairs = sorted(zip(names, map(float, values)), key=lambda kv: -kv[1])[:top]
    return {n.split("__", 1)[-1]: round(v, 4) for n, v in pairs}


def node_report_writer(state: SimpleLoopState) -> dict:
    """Code agent (no LLM): turn the run into an evaluation report a human can check.

    Formats holdout_eval's single holdout fit (no re-training) together with what the F1
    number alone hides: the confusion matrix, a majority-class baseline, feature
    importances, and every experiment the loop ran.
    """
    validation = state.get("validation") or {}
    memory = state.get("memory") or {}
    experiments = [
        {
            "cycle": h.get("cycle"),
            "model_family": h.get("model_family"),
            "hyperparameters": (h.get("spec") or {}).get("hyperparameters"),
            "val_f1": h.get("f1"),
            "verdict": (h.get("verdict") or {}).get("action"),
        }
        for h in memory.get("history") or []
        if isinstance(h, dict)
    ]
    last = state.get("execution_result") or {}
    if last and state.get("experiment_spec") and (
        not experiments or experiments[-1].get("cycle") != state.get("cycle")
    ):
        spec = state.get("experiment_spec") or {}
        experiments.append(
            {
                "cycle": state.get("cycle"),
                "model_family": spec.get("model_family"),
                "hyperparameters": spec.get("hyperparameters"),
                "val_f1": (last.get("validation_metrics") or {}).get("f1"),
                "verdict": (state.get("verdict") or {}).get("action"),
            }
        )

    report: dict = {
        "sklearn_version": sklearn.__version__,
        "dataset": Path(state["dataset_path"]).name,
        "target": state.get("target_column"),
        "validation": validation,
        "experiments": experiments,
    }

    details = state.get("holdout_details") or {}
    if details:
        report.update(details)
        note = (
            f"holdout f1={details['holdout_metrics']['f1']:.3f} "
            f"acc={details['holdout_metrics']['accuracy']:.3f} "
            f"(majority baseline acc={details['majority_baseline']['accuracy']:.3f})"
        )
    else:
        note = "no model trained: " + "; ".join(validation.get("issues") or ["no best spec"])

    out_dir = Path(os.getenv("LOOP_REPORT_DIR", str(REPO / "artifacts" / "loop_reports")))
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
        path = out_dir / f"report-{stamp}.json"
        path.write_text(json.dumps(report, indent=2, default=str))
        report["saved_to"] = str(path)
    except OSError as exc:
        report["saved_to"] = f"not saved: {exc}"

    return {
        "report": report,
        "stop_reason": state.get("stop_reason") or "invalid_dataset",
        "steps": [
            {
                "node": "report_writer",
                "note": note,
                "used_llm": False,
                "output": report,
                "metrics": report.get("holdout_metrics"),
            }
        ],
    }


def route_after_judge(state: SimpleLoopState) -> str:
    """Router = plain code (not an LLM)."""
    action = str((state.get("verdict") or {}).get("action") or "stop")
    cycle = int(state.get("cycle") or 0)
    max_cycles = int(state.get("max_cycles") or 3)
    if action == "accept":
        return "holdout"
    if action == "revise" and cycle < max_cycles:
        return "memory"
    return "holdout"


# ---------------------------------------------------------------------------
# GRAPH WIRING
# ---------------------------------------------------------------------------
def build_graph():
    g = StateGraph(SimpleLoopState)

    g.add_node("data_validator", node_data_validator)
    g.add_node("data_scientist", node_data_scientist)
    g.add_node("hypothesis_generator", node_hypothesis)
    g.add_node("ml_engineer", node_ml_engineer)
    g.add_node("judge", node_judge)
    g.add_node("memory_writer", node_memory_writer)
    g.add_node("context_builder", node_context_builder)
    g.add_node("holdout_eval", node_holdout)
    g.add_node("report_writer", node_report_writer)

    g.add_edge(START, "data_validator")
    g.add_conditional_edges(
        "data_validator",
        route_after_validator,
        {"ok": "data_scientist", "report": "report_writer"},
    )
    g.add_edge("data_scientist", "hypothesis_generator")
    g.add_edge("hypothesis_generator", "ml_engineer")
    g.add_edge("ml_engineer", "judge")
    g.add_conditional_edges(
        "judge",
        route_after_judge,
        {
            "memory": "memory_writer",
            "holdout": "holdout_eval",
        },
    )
    g.add_edge("memory_writer", "context_builder")
    g.add_edge("context_builder", "hypothesis_generator")  # loop back
    g.add_edge("holdout_eval", "report_writer")
    g.add_edge("report_writer", END)
    return g.compile()


REQUIRED_NODES = frozenset(
    {
        "data_validator",
        "report_writer",
        "data_scientist",
        "hypothesis_generator",
        "ml_engineer",
        "judge",
        "memory_writer",
        "context_builder",
        "holdout_eval",
    }
)


def assess_graph_ready(app) -> dict:
    """Return readiness dict from a compiled LangGraph app."""
    raw = app.get_graph().nodes
    present = {n for n in raw if n not in {START, END, "__start__", "__end__"}}
    missing = sorted(REQUIRED_NODES - present)
    return {
        "ready": not missing,
        "nodes": sorted(present),
        "missing_nodes": missing,
        "error": None if not missing else f"missing nodes: {', '.join(missing)}",
    }


def get_status() -> dict:
    """Import-safe status for the gateway / UI."""
    try:
        app = build_graph()
    except NotImplementedError as exc:
        return {
            "ready": False,
            "nodes": [],
            "missing_nodes": sorted(REQUIRED_NODES),
            "error": str(exc) or "build_graph not implemented",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ready": False,
            "nodes": [],
            "missing_nodes": sorted(REQUIRED_NODES),
            "error": str(exc),
        }
    return assess_graph_ready(app)


def _initial_state(
    *,
    dataset_path: str,
    target_column: str,
    target_score: float,
    max_cycles: int,
    seed: int,
    holdout_frac: float,
    model: str | None = None,
) -> SimpleLoopState:
    state: SimpleLoopState = {
        "dataset_path": str(Path(dataset_path).resolve()),
        "target_column": target_column,
        "target_score": target_score,
        "max_cycles": max_cycles,
        "cycle": 0,
        "seed": seed,
        "holdout_frac": holdout_frac,
        "best_val_f1": -1.0,
        "steps": [],
        "memory": {"history": [], "findings": [], "lessons": []},
        "brief": {},
    }
    chosen = (model or "").strip() or DEFAULT_MODEL
    state["model"] = chosen
    return state


def _finalize_payload(final: dict) -> dict:
    return {
        "dataset_path": final.get("dataset_path"),
        "target_column": final.get("target_column"),
        "cycles": final.get("cycle"),
        "best_val_f1": final.get("best_val_f1"),
        "best_spec": final.get("best_spec"),
        "holdout_metrics": final.get("holdout_metrics"),
        "validation": final.get("validation"),
        "report": final.get("report"),
        "verdict": final.get("verdict"),
        "stop_reason": final.get("stop_reason"),
        "profile": final.get("profile"),
        "memory": final.get("memory"),
        "model": final.get("model"),
        "steps": final.get("steps") or [],
    }


def run_solution_loop(
    *,
    dataset_path: str,
    target_column: str = "Survived",
    target_score: float = 0.8,
    max_cycles: int = 3,
    seed: int = 42,
    holdout_frac: float = 0.2,
    model: str | None = None,
) -> dict:
    """Run the whole graph to completion; return a summary dict."""
    app = build_graph()
    initial = _initial_state(
        dataset_path=dataset_path,
        target_column=target_column,
        target_score=target_score,
        max_cycles=max_cycles,
        seed=seed,
        holdout_frac=holdout_frac,
        model=model,
    )
    final = app.invoke(initial, config={"recursion_limit": 10 + max_cycles * 6})
    return _finalize_payload(final)


def stream_solution_loop(
    *,
    dataset_path: str,
    target_column: str = "Survived",
    target_score: float = 0.8,
    max_cycles: int = 3,
    seed: int = 42,
    holdout_frac: float = 0.2,
    model: str | None = None,
):
    """Yield UI events as each node starts/finishes (used by the gateway SSE)."""
    app = build_graph()
    initial = _initial_state(
        dataset_path=dataset_path,
        target_column=target_column,
        target_score=target_score,
        max_cycles=max_cycles,
        seed=seed,
        holdout_frac=holdout_frac,
        model=model,
    )
    yield {
        "type": "run_started",
        "dataset_path": str(Path(dataset_path).resolve()),
        "target_column": target_column,
        "model": initial.get("model"),
    }

    merged: dict = dict(initial)
    last_node: str | None = None
    try:
        for update in app.stream(
            initial, stream_mode="updates", config={"recursion_limit": 10 + max_cycles * 6}
        ):
            if not isinstance(update, dict):
                continue
            for node_name, patch in update.items():
                if last_node == "judge" and node_name in {"memory_writer", "holdout_eval"}:
                    action = str((merged.get("verdict") or {}).get("action") or "stop")
                    yield {
                        "type": "router",
                        "node": "router",
                        "status": "completed",
                        "action": action,
                        "route": "memory" if node_name == "memory_writer" else "holdout",
                    }
                    if node_name == "memory_writer":
                        yield {
                            "type": "node_completed",
                            "node": "research_memory",
                            "status": "completed",
                            "note": "history + lessons filed",
                        }

                yield {"type": "node_started", "node": node_name, "status": "running"}
                if isinstance(patch, dict):
                    steps = list(patch.get("steps") or [])
                    # Mirror Annotated[list, operator.add] — stream patches omit prior steps.
                    merged.update({k: v for k, v in patch.items() if k != "steps"})
                    if steps:
                        merged["steps"] = list(merged.get("steps") or []) + steps
                    note = steps[-1].get("note") if steps else None
                    yield {
                        "type": "node_completed",
                        "node": node_name,
                        "status": "completed",
                        "note": note,
                        "step": steps[-1] if steps else None,
                        "cycle": merged.get("cycle"),
                        "best_val_f1": merged.get("best_val_f1"),
                        "verdict": merged.get("verdict"),
                    }
                else:
                    yield {"type": "node_completed", "node": node_name, "status": "completed"}
                last_node = node_name
    except Exception as exc:  # noqa: BLE001
        yield {"type": "error", "message": str(exc)}
        return

    yield {"type": "run_completed", "result": _finalize_payload(merged)}


if __name__ == "__main__":
    ds = REPO / "datasets" / "Titanic-Dataset.csv"
    result = run_solution_loop(dataset_path=str(ds), target_column="Survived", max_cycles=3)
    print(json.dumps(result, indent=2, default=str))
