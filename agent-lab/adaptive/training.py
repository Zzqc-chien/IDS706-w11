"""Training helpers shared by the adaptive system and the fixed baseline.

* Feature operations: derived columns that the Feature Engineer agent can switch on.
  Each one uses only values from the same row, so it can't leak future information.
* Fault injection: lets the evaluation suite make training fail on purpose, to test
  how each system recovers.
* train_eval: apply feature ops → time-aware split → fit → metrics (+ predictions).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from loop_runtime import _build_pipeline, _classification_metrics, _split

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
FEATURE_BUNDLES: list[list[str]] = [
    ["cyclical_time", "rush_hour"],  # tried first: time-of-day shape
    ["weather_comfort"],  # second: non-linear weather effect
]


def _hour_of_day(df: pd.DataFrame, time_column: str | None) -> pd.Series | None:
    if "hour_of_day" in df.columns:
        return pd.to_numeric(df["hour_of_day"], errors="coerce")
    if time_column and time_column in df.columns:
        return pd.to_datetime(df[time_column], errors="coerce").dt.hour
    return None


def _day_of_week(df: pd.DataFrame, time_column: str | None) -> pd.Series | None:
    if "day_of_week" in df.columns:
        return pd.to_numeric(df["day_of_week"], errors="coerce")
    if time_column and time_column in df.columns:
        return pd.to_datetime(df[time_column], errors="coerce").dt.dayofweek
    return None


def apply_feature_ops(df: pd.DataFrame, ops: list[str], time_column: str | None) -> tuple[pd.DataFrame, list[str]]:
    """Return (df with new columns, names of columns actually added)."""
    out = df.copy()
    added: list[str] = []
    hod = _hour_of_day(out, time_column)
    dow = _day_of_week(out, time_column)
    for op in ops:
        if op == "cyclical_time" and hod is not None:
            out["hour_sin"] = np.sin(2 * np.pi * hod / 24)
            out["hour_cos"] = np.cos(2 * np.pi * hod / 24)
            added += ["hour_sin", "hour_cos"]
        elif op == "rush_hour" and hod is not None:
            weekday = (dow < 5) if dow is not None else True
            out["is_rush_hour"] = (weekday & (hod.between(7, 9) | hod.between(16, 19))).astype(int)
            added.append("is_rush_hour")
        elif op == "weather_comfort" and "temp_c" in out.columns:
            out["temp_comfort"] = -((pd.to_numeric(out["temp_c"], errors="coerce") - 22.0) ** 2)
            added.append("temp_comfort")
    return out, added


# ---------------------------------------------------------------------------
# Fault injection (evaluation only)
# ---------------------------------------------------------------------------
class InjectedFault(RuntimeError):
    pass


def check_fault(fault_plan: dict | None, family: str, attempt_index: int) -> None:
    """Raise like a real training failure would. attempt_index counts all training calls."""
    plan = fault_plan or {}
    if attempt_index < int(plan.get("transient_failures", 0)):
        raise InjectedFault(f"injected transient failure (attempt {attempt_index + 1}): worker lost")
    if family in (plan.get("broken_families") or []):
        raise InjectedFault(f"injected persistent failure: {family} crashes on this platform")


# ---------------------------------------------------------------------------
# Train + evaluate
# ---------------------------------------------------------------------------
def train_eval(
    df: pd.DataFrame,
    target: str,
    spec: dict,
    *,
    feature_ops: list[str] | None = None,
    time_column: str | None = None,
    seed: int = 42,
    holdout_frac: float = 0.2,
    on_holdout: bool = False,
    return_predictions: bool = False,
) -> dict:
    data, added = apply_feature_ops(df, list(feature_ops or []), time_column)
    drops = list(spec.get("drop_columns") or [])
    X_tr, y_tr, X_ev, y_ev, n_hold, split_kind = _split(
        data, target, drops, seed, holdout_frac, on_holdout=on_holdout, time_column=time_column
    )
    pipe = _build_pipeline(X_tr, spec)
    t0 = time.time()
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_ev)
    out = {
        "metrics": _classification_metrics(y_ev, pred),
        "features_used": list(X_tr.columns),
        "features_added": added,
        "split": split_kind,
        "n_train": int(len(X_tr)),
        "n_eval": int(len(X_ev)),
        "duration_s": round(time.time() - t0, 3),
    }
    if return_predictions:
        majority = y_tr.mode().iloc[0]
        out["y_true"] = list(y_ev)
        out["y_pred"] = list(pred)
        out["majority_metrics"] = _classification_metrics(y_ev, [majority] * len(y_ev))
        out["pipeline"] = pipe
    return out
