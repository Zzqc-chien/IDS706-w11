"""Publish the cleaned table for the agent workflow (Part 2).

The agent loop does binary classification, so the hand-off adds a label:
high_demand = 1 when an hour's trips are at or above the 75th percentile.

A sidecar `<name>.contract.json` tells the agents' data_validator which column is the
time key and which columns are target leakage. `trips`, `member_share`, `ebike_share` and
`avg_duration_min` are computed from the same rides the label counts, so a model may not
use them; they stay in the CSV on purpose, so the validator's guardrail has something to catch.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

DATASET_NAME = "citibike_hourly_latest"
TARGET = "high_demand"
TIME_COLUMN = "hour"
LEAKAGE_COLUMNS = ["trips", "member_share", "ebike_share", "avg_duration_min"]
HIGH_DEMAND_QUANTILE = 0.75


def build_agent_dataset(hourly: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    threshold = float(hourly["trips"].quantile(HIGH_DEMAND_QUANTILE))
    df = hourly.sort_values(TIME_COLUMN).copy()
    df[TARGET] = (df["trips"] >= threshold).astype(int)
    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN]).dt.strftime("%Y-%m-%d %H:%M")
    front = [TIME_COLUMN, TARGET]
    df = df[front + [c for c in df.columns if c not in front]]
    contract = {
        "dataset": f"{DATASET_NAME}.csv",
        "produced_by": "Airflow DAG citibike_weather_pipeline (task analysis.publish_for_agents)",
        "grain": "one row per hour, Jersey City Citi Bike system",
        "target": TARGET,
        "target_definition": f"trips >= {threshold:.0f} (the {HIGH_DEMAND_QUANTILE:.0%} quantile)",
        "positive_rate": round(float(df[TARGET].mean()), 4),
        "time_column": TIME_COLUMN,
        "leakage_columns": LEAKAGE_COLUMNS,
        "n_rows": int(len(df)),
        "period": f"{df[TIME_COLUMN].iloc[0]} to {df[TIME_COLUMN].iloc[-1]}",
    }
    return df, contract


def publish(hourly: pd.DataFrame, out_dir: str | Path) -> str:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df, contract = build_agent_dataset(hourly)
    csv_path = out_dir / f"{DATASET_NAME}.csv"
    tmp = csv_path.with_suffix(".csv.part")
    df.to_csv(tmp, index=False)
    tmp.replace(csv_path)
    (out_dir / f"{DATASET_NAME}.contract.json").write_text(json.dumps(contract, indent=2))
    return str(csv_path)
