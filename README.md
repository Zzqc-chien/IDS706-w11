# Citi Bike demand pipeline: Airflow + agent-based ML

An end-to-end pipeline that ingests **Citi Bike trip records** and **hourly weather**, cleans and merges them in **Apache Airflow 3.4**, loads the result into **PostgreSQL**, then analyzes it: exploratory plots plus a model that forecasts hourly bike demand. Part 2 trains and evaluates a model on the same cleaned data with an agent-based workflow.

## Quick start

Requirements: Docker Desktop (4 GB+ memory for Docker) and `make`.

```bash
make up          # build the image and start Airflow + two Postgres databases
                 # UI: http://localhost:8080   user: airflow   password: airflow
make trigger     # unpause the DAG and trigger a run (or click ▶ in the UI)
make test        # unit tests, run inside the container
make down        # stop (data is kept);  make clean  deletes everything
```

The cleaned data lives in the `pipeline-db` database, which is reachable from your machine:

```bash
psql -h localhost -p 5433 -U pipeline citibike      # password: pipeline
citibike=> SELECT hour, trips, temp_c, precip_mm FROM hourly_demand LIMIT 5;
```

Outputs (plots and metrics) are written to `data/output/`.

**Offline demo:** `make sample-data` writes synthetic raw files with the real formats to `data/raw/`. The extract tasks treat files that already exist as a cache, so the DAG then runs without downloading anything.

**Dev container:** opening the folder in VS Code → "Reopen in Container" uses `.devcontainer/devcontainer.json`. It starts the same stack and attaches to the `dev` service, with the repo mounted at `/workspace`.

## Data sources

| Dataset | Source | Grain | Join key |
|---|---|---|---|
| Citi Bike trips, Jersey City / Hoboken | `s3.amazonaws.com/tripdata/JC-YYYYMM-citibike-tripdata.csv.zip` (public monthly files) | one row per ride | ride start time → hour |
| Hourly weather (temperature, precipitation, wind, humidity) | Open-Meteo historical API (free, no key), at Jersey City's coordinates | one row per hour | hour |

The Jersey City system uses the same schema as NYC Citi Bike, but its monthly files are a few MB instead of ~1 GB, which keeps the pipeline fast on a laptop.

## Pipeline structure

```
plan_months
   ├── trips   (TaskGroup): extract_trips ×N months (parallel, mapped) → clean_trips
   └── weather (TaskGroup): extract_weather ×N months (parallel, mapped) → clean_weather
merge → load_to_postgres
   └── analysis (TaskGroup): make_plots ∥ train_model
```

- **Scheduling.** `schedule="@monthly"`, because Citi Bike publishes one file per month. Each scheduled run processes the 6 most recent *published* months, ending 2 months before the run date because of the publishing delay; a run on 2026-11-07 processes 2026-04 … 2026-09. `catchup=False` and `max_active_runs=1`. A manual run can set the `months` parameter (for example `["2026-08","2026-09"]`) in the trigger form. Tasks retry twice, one minute apart, which covers transient download failures.
- **Parallel processing.**
  1. The trips and weather branches have no dependency on each other, so they run at the same time.
  2. Inside each branch, the download for each month is a separate mapped task (`.expand(month=months)`); up to 4 per branch run at once.
  3. `make_plots` and `train_model` both depend only on the load, so they run in parallel too.
  
  The LocalExecutor runs up to 8 tasks at once (`AIRFLOW__CORE__PARALLELISM`).
- **Data passing.** Tasks pass file paths (Parquet) through XCom, never DataFrames.
- **Idempotent and cached.**
  - Raw files are cached in `data/raw/` and written atomically, so a re-run never downloads a file twice or sees a half-downloaded one.
  - The load deletes the hours it is about to insert, so re-running a month replaces it instead of duplicating it.

## Transformations

**Trips** (`clean_trips` → `trips_to_hourly`). Each step's row count is logged as a data-quality report (`data/processed/<window>/trips_quality.json`).

1. Drop rows missing `ride_id`, `started_at` or `ended_at`.
2. Drop duplicate `ride_id`s.
3. Parse timestamps (the files mix formats with and without milliseconds) and drop unparseable ones.
4. Keep rides of 1–240 minutes. Shorter ones are false starts or re-docks; longer ones are lost or unreturned bikes.
5. Drop rides that start outside the requested months.
6. Aggregate to one row per hour: `trips`, `member_share`, `ebike_share`, `avg_duration_min`. Hours with no rides are kept with `trips = 0`, so quiet hours aren't silently missing.

**Weather** (`clean_weather`):

1. Coerce values to numbers.
2. Set physically impossible values to missing.
3. Drop duplicate hours (these can appear at the daylight-saving switch).
4. Reindex to the full hourly range and fill gaps of up to 3 hours by interpolation; hours that still have gaps are dropped.

**Merge** (`merge_hourly`): a one-to-one join on the hour (validated), plus calendar features: `hour_of_day`, `day_of_week`, `is_weekend`, `is_holiday` (US federal holidays) and `is_raining` (precipitation > 0.1 mm). The result is the `hourly_demand` table.

## Analysis (Part 1)

Both analysis tasks **read the data back from Postgres**.

- `eda.png`: average trips by hour (weekday vs weekend), trips vs temperature (dry vs raining), daily trips vs rainfall.
- `train_model`: a `RandomForestRegressor` predicts trips per hour from time and weather features. It is compared with a seasonal baseline: the average for the same day-of-week and hour in the training data.
  - The split is **chronological** (last 20% of hours as the test set), so the model is never evaluated on hours earlier than the ones it trained on.
  - Metrics: MAE, RMSE and R², written to `data/output/metrics.json` and appended to the `model_metrics` table with the run id. Plot: `predictions.png`.

### Results

> Replace this with the numbers from your own run (`data/output/metrics.json`).

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Seasonal baseline (day × hour average) | _ | _ | _ |
| Random forest (time + weather) | _ | _ | _ |

Key findings: _fill in after the run. For example: the weekday commute peaks (8am and 6pm) vs the weekend midday peak, how much rain lowers demand, and whether the weather features beat the baseline._

## Part 2: agent-based ML workflow

_To be added: the in-class agent template adapted to train and evaluate a model on `hourly_demand`._

## Screenshots (`docs/screenshots/`)

- [ ] DAG run succeeded (Grid view, all tasks green)
- [ ] DAG Graph view (shows the TaskGroups and the parallel branches)
- [ ] Agent graph (Part 2)
- [ ] Model evaluation results (Part 1 `metrics.json` / `predictions.png`, and Part 2)

## Repository layout

```
dags/citibike_weather_pipeline.py   the DAG (wiring only)
dags/citibike/                      pipeline logic: config, extract, transform, load, analyze
scripts/make_sample_data.py         synthetic raw files for tests and offline runs
tests/                              pytest unit tests
Dockerfile, docker-compose.yaml     Airflow 3.4 (LocalExecutor) + Postgres ×2
.devcontainer/                      VS Code dev container
Makefile                            up / trigger / test / down / clean
```
