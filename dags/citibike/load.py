"""Load the merged table into the pipeline database, and read it back."""
from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine, inspect, text

from citibike import config

TABLE = "hourly_demand"


def get_engine(uri: str | None = None):
    return create_engine(uri or config.PIPELINE_DB_URI, future=True)


def upsert_hourly(df: pd.DataFrame, uri: str | None = None, table: str = TABLE) -> int:
    """Replace the rows for the hours in df (idempotent: re-running a month never duplicates it)."""
    engine = get_engine(uri)
    with engine.begin() as conn:
        if inspect(conn).has_table(table):
            conn.execute(
                text(f"DELETE FROM {table} WHERE hour >= :lo AND hour <= :hi"),
                {"lo": df["hour"].min().to_pydatetime(), "hi": df["hour"].max().to_pydatetime()},
            )
        df.to_sql(table, conn, if_exists="append", index=False, chunksize=1000)
        total = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one()
    engine.dispose()
    return int(total)


def read_hourly(uri: str | None = None, table: str = TABLE, months: list[str] | None = None) -> pd.DataFrame:
    """Read the cleaned table. With `months`, only those months (the run's own window);
    the table keeps older months from earlier runs, so without a filter the analysis
    would silently grow over time."""
    engine = get_engine(uri)
    with engine.connect() as conn:
        df = pd.read_sql(text(f"SELECT * FROM {table} ORDER BY hour"), conn, parse_dates=["hour"])
    engine.dispose()
    if months:
        df = df[df["hour"].dt.strftime("%Y-%m").isin(set(months))].reset_index(drop=True)
    return df


def save_metrics(metrics: dict, run_id: str, uri: str | None = None) -> None:
    """Append one row per run. Metrics are stored as JSON so adding a metric later never
    breaks the table schema (an earlier version used one column per metric)."""
    import json

    row = pd.DataFrame([{
        "run_id": run_id,
        "created_at": pd.Timestamp.now(tz="UTC").tz_localize(None),
        "metrics_json": json.dumps(metrics, default=str),
    }])
    engine = get_engine(uri)
    with engine.begin() as conn:
        row.to_sql("model_runs", conn, if_exists="append", index=False)
    engine.dispose()
