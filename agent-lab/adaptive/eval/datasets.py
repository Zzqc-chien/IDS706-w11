"""Synthetic datasets for the evaluation suite, shaped like the real Citi Bike hand-off.

Same columns as citibike_hourly_latest.csv, so the suite runs offline and
deterministically. Each case only changes the property it tests: leakage columns, a
broken target, harder noise, and so on.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def make_hourly(
    *,
    days: int = 150,
    seed: int = 0,
    noise: float = 0.25,
    leakage: bool = False,
    contract_leakage: list[str] | None = None,
    multiclass_target: bool = False,
) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    hours = pd.date_range("2026-04-01", periods=24 * days, freq="h")
    hod = hours.hour.to_numpy()
    dow = hours.dayofweek.to_numpy()
    weekend = dow >= 5
    doy = hours.dayofyear.to_numpy()
    temp = 14 + 10 * np.sin((doy - 100) / 365 * 2 * np.pi) + 4 * np.sin((hod - 9) / 24 * 2 * np.pi) + rng.normal(0, 2.5, len(hours))
    rain = np.where(rng.random(len(hours)) < 0.09, rng.gamma(1.3, 1.6, len(hours)), 0.0)
    commute = np.exp(-((hod - 8) ** 2) / 2.5) + 1.3 * np.exp(-((hod - 17.5) ** 2) / 3.5)
    leisure = np.exp(-((hod - 14) ** 2) / 14)
    shape = np.where(weekend, 0.15 + 1.1 * leisure, 0.1 + commute + 0.35 * leisure)
    comfort = np.exp(-((temp - 23) ** 2) / 120)
    demand = 300 * shape * comfort * np.where(rain > 0.1, 0.45, 1.0)
    trips = rng.poisson(np.maximum(demand * np.exp(rng.normal(0, noise, len(hours))), 0.1))

    df = pd.DataFrame(
        {
            "hour": hours.strftime("%Y-%m-%d %H:%M"),
            "temp_c": temp.round(1),
            "precip_mm": rain.round(1),
            "wind_kmh": rng.gamma(3, 4, len(hours)).round(1),
            "humidity_pct": rng.integers(35, 96, len(hours)),
            "hour_of_day": hod,
            "day_of_week": dow,
            "month": hours.month,
            "is_weekend": weekend.astype(int),
            "is_holiday": 0,
            "is_raining": (rain > 0.1).astype(int),
        }
    )
    train_part = trips[: int(len(trips) * 0.6)]  # thresholds from the training segment only, like Part 1
    q75 = np.quantile(train_part, 0.75)
    if multiclass_target:
        target = np.digitize(trips, np.quantile(train_part, [0.33, 0.66]))  # low / mid / high
    else:
        target = (trips >= q75).astype(int)
    df.insert(1, "high_demand", target)
    if leakage:
        df["trips"] = trips
        df["member_share"] = np.clip(0.75 - 0.0004 * trips + rng.normal(0, 0.05, len(hours)), 0, 1).round(3)
    contract = {"target": "high_demand", "time_column": "hour", "leakage_columns": contract_leakage or []}
    return df, contract


def write_dataset(df: pd.DataFrame, contract: dict | None, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.csv"
    df.to_csv(path, index=False)
    sidecar = path.with_suffix(".contract.json")
    if contract is not None:
        sidecar.write_text(json.dumps(contract, indent=2))
    elif sidecar.exists():
        sidecar.unlink()
    return path
