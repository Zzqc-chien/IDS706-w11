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


def read_hourly(uri: str | None = None, table: str = TABLE) -> pd.DataFrame:
    engine = get_engine(uri)
    with engine.connect() as conn:
        df = pd.read_sql(text(f"SELECT * FROM {table} ORDER BY hour"), conn, parse_dates=["hour"])
    engine.dispose()
    return df


def save_metrics(metrics: dict, run_id: str, uri: str | None = None) -> None:
    row = pd.DataFrame(
        [{"run_id": run_id, "created_at": pd.Timestamp.now(tz="UTC").tz_localize(None), **metrics}]
    )
    engine = get_engine(uri)
    with engine.begin() as conn:
        row.to_sql("model_metrics", conn, if_exists="append", index=False)
    engine.dispose()
