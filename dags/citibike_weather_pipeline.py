"""
Citi Bike (Jersey City) + weather pipeline.

    plan_months
        ├── trips   [TaskGroup]  extract_trips (one mapped task per month, parallel) → clean_trips
        └── weather [TaskGroup]  extract_weather (one mapped task per month, parallel) → clean_weather
    merge → load_to_postgres
        └── analysis [TaskGroup]  make_plots ∥ train_model ∥ publish_for_agents

The trips and weather branches don't depend on each other, so they run in parallel;
inside each branch the monthly downloads run in parallel too (dynamic task mapping).
Tasks pass file paths through XCom, never DataFrames.
"""
from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pendulum
from airflow.sdk import Param, dag, get_current_context, task, task_group

from citibike import config


def _window_dir(months: list[str]) -> Path:
    path = config.PROCESSED_DIR / f"{months[0]}_{months[-1]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


@dag(
    dag_id="citibike_weather_pipeline",
    description="Ingest Citi Bike trips + weather, clean, merge, load to Postgres, analyze",
    schedule="@monthly",  # new Citi Bike files are published monthly
    start_date=pendulum.datetime(2026, 1, 1, tz="America/New_York"),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=1)},
    params={
        "months": Param(
            [],
            type="array",
            description="Months to process as YYYY-MM. Empty = the N most recent published months.",
        ),
        "n_months": Param(config.DEFAULT_N_MONTHS, type="integer", minimum=1, maximum=24),
    },
    tags=["ids706", "citibike"],
)
def citibike_weather_pipeline():
    @task
    def plan_months() -> list[str]:
        from citibike.extract import months_for_run

        ctx = get_current_context()
        months = ctx["params"].get("months") or []
        if not months:
            run_date = ctx.get("logical_date") or ctx["dag_run"].run_after
            months = months_for_run(
                run_date.date(), int(ctx["params"]["n_months"]), config.PUBLISH_LAG_MONTHS
            )
        months = sorted(months)
        print(f"Processing months: {months}")
        return months

    # ---------- trips branch ----------
    @task(max_active_tis_per_dagrun=4)
    def extract_trips(month: str) -> str:
        from citibike.extract import download_trips

        return download_trips(month)

    @task
    def clean_trips(paths: list[str], months: list[str]) -> str:
        import pandas as pd

        from citibike.transform import clean_trips as clean, read_trip_zip, trips_to_hourly

        raw = pd.concat([read_trip_zip(p) for p in paths], ignore_index=True)
        trips, report = clean(raw, months)
        hourly = trips_to_hourly(trips, months)
        out = _window_dir(months)
        hourly.to_parquet(out / "trips_hourly.parquet", index=False)
        (out / "trips_quality.json").write_text(json.dumps(report, indent=2))
        print(f"Trip quality report: {report}")
        return str(out / "trips_hourly.parquet")

    @task_group(group_id="trips")
    def trips_group(months):
        return clean_trips(extract_trips.expand(month=months), months)

    # ---------- weather branch ----------
    @task(max_active_tis_per_dagrun=4)
    def extract_weather(month: str) -> str:
        from citibike.extract import download_weather

        return download_weather(month)

    @task
    def clean_weather(paths: list[str], months: list[str]) -> str:
        import pandas as pd

        from citibike.transform import clean_weather as clean, read_weather_json

        raw = pd.concat([read_weather_json(p) for p in paths], ignore_index=True)
        weather, report = clean(raw, months)
        out = _window_dir(months)
        weather.to_parquet(out / "weather_hourly.parquet", index=False)
        (out / "weather_quality.json").write_text(json.dumps(report, indent=2))
        print(f"Weather quality report: {report}")
        return str(out / "weather_hourly.parquet")

    @task_group(group_id="weather")
    def weather_group(months):
        return clean_weather(extract_weather.expand(month=months), months)

    # ---------- merge + load ----------
    @task
    def merge(trips_path: str, weather_path: str) -> str:
        import pandas as pd

        from citibike.transform import merge_hourly

        merged = merge_hourly(pd.read_parquet(trips_path), pd.read_parquet(weather_path))
        path = Path(trips_path).with_name("hourly_demand.parquet")
        merged.to_parquet(path, index=False)
        print(f"Merged {len(merged)} hourly rows, columns: {list(merged.columns)}")
        return str(path)

    @task
    def load_to_postgres(merged_path: str) -> int:
        import pandas as pd

        from citibike.load import upsert_hourly

        total = upsert_hourly(pd.read_parquet(merged_path))
        print(f"hourly_demand now holds {total} rows")
        return total

    # ---------- analysis (reads back from the database) ----------
    # Analysis reads back only this run's months, not everything the table has accumulated.
    @task
    def make_plots(_rows: int, months: list[str]) -> str:
        from citibike.analyze import make_plots as plot
        from citibike.load import read_hourly

        return plot(read_hourly(months=months), config.OUTPUT_DIR)

    @task
    def train_model(_rows: int, months: list[str]) -> dict:
        from citibike.analyze import train_and_evaluate
        from citibike.load import read_hourly, save_metrics

        metrics = train_and_evaluate(read_hourly(months=months), config.OUTPUT_DIR)
        save_metrics(metrics, run_id=get_current_context()["run_id"])
        print(json.dumps(metrics, indent=2))
        return metrics

    @task
    def publish_for_agents(_rows: int, months: list[str]) -> str:
        """Hand-off to Part 2: CSV + contract in the agent lab's datasets folder."""
        from citibike.load import read_hourly
        from citibike.publish import publish

        path = publish(read_hourly(months=months), config.AGENT_DATASET_DIR)
        print(f"Published agent dataset: {path}")
        return path

    @task_group(group_id="analysis")
    def analysis_group(rows, months):
        make_plots(rows, months)
        train_model(rows, months)
        publish_for_agents(rows, months)

    months = plan_months()
    merged = merge(trips_group(months), weather_group(months))
    analysis_group(load_to_postgres(merged), months)


citibike_weather_pipeline()
