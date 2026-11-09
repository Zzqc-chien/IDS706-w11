"""Publish the cleaned table for the agent workflow (Part 2).

The agent loop does binary classification, so the hand-off adds a label:
high_demand = 1 when an hour's trips reach a threshold. The threshold is the 75th
percentile of trips in the **training segment only**: the first 60% of hours, the part
the agents train on (the next 20% is validation, the last 20% the sealed holdout).
Validation and holdout hours are labelled with that same fixed threshold, so their
demand never influences the label definition.

A sidecar `<name>.contract.json` tells the agents' data_validator which column is the
time key and which columns are target leakage. `trips`, `member_share`, `ebike_share` and
`avg_duration_min` come from the same rides the label counts. They stay in the CSV on
purpose, so the validator's guardrail can be checked.
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
TRAIN_SEGMENT = 0.60  # matches the agents' split: 60% train / 20% validation / 20% holdout


def build_agent_dataset(hourly: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = hourly.sort_values(TIME_COLUMN).reset_index(drop=True).copy()
    n_train = int(len(df) * TRAIN_SEGMENT)
    threshold = float(df["trips"].iloc[:n_train].quantile(HIGH_DEMAND_QUANTILE))
    df[TARGET] = (df["trips"] >= threshold).astype(int)
    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN]).dt.strftime("%Y-%m-%d %H:%M")
    front = [TIME_COLUMN, TARGET]
    df = df[front + [c for c in df.columns if c not in front]]
    contract = {
        "dataset": f"{DATASET_NAME}.csv",
        "produced_by": "Airflow DAG citibike_weather_pipeline (task analysis.publish_for_agents)",
        "grain": "one row per hour, Jersey City Citi Bike system",
        "target": TARGET,
        "target_definition": f"trips >= {threshold:.0f}",
        "threshold_basis": f"{HIGH_DEMAND_QUANTILE:.0%} quantile of trips in the first {TRAIN_SEGMENT:.0%} of hours "
                           f"(up to {df[TIME_COLUMN].iloc[n_train - 1]}); later hours use the same fixed threshold",
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
