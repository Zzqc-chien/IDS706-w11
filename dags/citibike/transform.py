"""Cleaning and merging. Pure pandas functions so they are easy to unit-test."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from citibike.extract import month_bounds

TRIP_COLUMNS = ["ride_id", "rideable_type", "started_at", "ended_at", "member_casual"]
MIN_DURATION_MIN = 1.0  # shorter = false starts / re-docks
MAX_DURATION_MIN = 240.0  # longer = lost or unreturned bikes


def hourly_index(months: list[str]) -> pd.DatetimeIndex:
    first, _ = month_bounds(min(months))
    _, last = month_bounds(max(months))
    return pd.date_range(
        pd.Timestamp(first), pd.Timestamp(last) + pd.Timedelta(hours=23), freq="h", name="hour"
    )


def read_trip_zip(path: str | Path) -> pd.DataFrame:
    """Read every CSV inside a Citi Bike zip (some months are split into several parts)."""
    frames = []
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if name.lower().endswith(".csv") and not name.startswith("__MACOSX"):
                with zf.open(name) as fh:
                    frames.append(
                        pd.read_csv(fh, usecols=lambda c: c in TRIP_COLUMNS, dtype=str)
                    )
    if not frames:
        raise ValueError(f"No CSV found in {path}")
    return pd.concat(frames, ignore_index=True)


def clean_trips(raw: pd.DataFrame, months: list[str]) -> tuple[pd.DataFrame, dict]:
    """Clean trip records. Returns (clean trips, data-quality report)."""
    report = {"rows_in": int(len(raw))}
    df = raw.copy()

    missing = df[["ride_id", "started_at", "ended_at"]].isna().any(axis=1)
    report["dropped_missing_required"] = int(missing.sum())
    df = df[~missing]

    dup = df.duplicated(subset="ride_id", keep="first")
    report["dropped_duplicate_ride_id"] = int(dup.sum())
    df = df[~dup]

    df["started_at"] = pd.to_datetime(df["started_at"], format="mixed", errors="coerce")
    df["ended_at"] = pd.to_datetime(df["ended_at"], format="mixed", errors="coerce")
    bad_ts = df["started_at"].isna() | df["ended_at"].isna()
    report["dropped_bad_timestamp"] = int(bad_ts.sum())
    df = df[~bad_ts]

    df["duration_min"] = (df["ended_at"] - df["started_at"]).dt.total_seconds() / 60
    bad_dur = ~df["duration_min"].between(MIN_DURATION_MIN, MAX_DURATION_MIN)
    report["dropped_duration_out_of_range"] = int(bad_dur.sum())
    df = df[~bad_dur]

    idx = hourly_index(months)
    outside = (df["started_at"] < idx[0]) | (df["started_at"] >= idx[-1] + pd.Timedelta(hours=1))
    report["dropped_outside_window"] = int(outside.sum())
    df = df[~outside]

    df["member_casual"] = df["member_casual"].str.strip().str.lower()
    df["rideable_type"] = df["rideable_type"].str.strip().str.lower()
    report["rows_out"] = int(len(df))
    return df.reset_index(drop=True), report


def trips_to_hourly(trips: pd.DataFrame, months: list[str]) -> pd.DataFrame:
    """Aggregate trips to one row per hour. Hours with no trips get trips = 0."""
    t = trips.assign(
        hour=trips["started_at"].dt.floor("h"),
        is_member=(trips["member_casual"] == "member").astype(float),
        is_ebike=(trips["rideable_type"] == "electric_bike").astype(float),
    )
    hourly = t.groupby("hour").agg(
        trips=("ride_id", "size"),
        member_share=("is_member", "mean"),
        ebike_share=("is_ebike", "mean"),
        avg_duration_min=("duration_min", "mean"),
    )
    hourly = hourly.reindex(hourly_index(months))
    hourly["trips"] = hourly["trips"].fillna(0).astype(int)
    hourly[["member_share", "ebike_share", "avg_duration_min"]] = hourly[
        ["member_share", "ebike_share", "avg_duration_min"]
    ].fillna(0.0)
    return hourly.reset_index()


def read_weather_json(path: str | Path) -> pd.DataFrame:
    payload = json.loads(Path(path).read_text())
    return pd.DataFrame(payload["hourly"])


def clean_weather(raw: pd.DataFrame, months: list[str]) -> tuple[pd.DataFrame, dict]:
    """Clean hourly weather: types, duplicates (DST), gaps, range checks."""
    report = {"rows_in": int(len(raw))}
    df = raw.rename(
        columns={
            "time": "hour",
            "temperature_2m": "temp_c",
            "precipitation": "precip_mm",
            "wind_speed_10m": "wind_kmh",
            "relative_humidity_2m": "humidity_pct",
        }
    )
    df["hour"] = pd.to_datetime(df["hour"], errors="coerce")
    df = df.dropna(subset=["hour"])
    value_cols = ["temp_c", "precip_mm", "wind_kmh", "humidity_pct"]
    for col in value_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Physically impossible values -> missing, then treated like any other gap.
    limits = {"temp_c": (-40, 50), "precip_mm": (0, 200), "wind_kmh": (0, 250), "humidity_pct": (0, 100)}
    out_of_range = 0
    for col, (lo, hi) in limits.items():
        bad = ~df[col].between(lo, hi) & df[col].notna()
        out_of_range += int(bad.sum())
        df.loc[bad, col] = float("nan")
    report["values_out_of_range"] = out_of_range

    dup = df.duplicated(subset="hour", keep="first")
    report["dropped_duplicate_hour"] = int(dup.sum())
    df = df[~dup].set_index("hour").sort_index()

    df = df.reindex(hourly_index(months))
    report["missing_values_before_fill"] = int(df[value_cols].isna().sum().sum())
    df[value_cols] = df[value_cols].interpolate(limit=3, limit_direction="both")
    still_missing = df[value_cols].isna().any(axis=1)
    report["dropped_unfillable_hours"] = int(still_missing.sum())
    df = df[~still_missing]
    report["rows_out"] = int(len(df))
    return df.reset_index(), report


def merge_hourly(trips_hourly: pd.DataFrame, weather_hourly: pd.DataFrame) -> pd.DataFrame:
    """Join trips and weather on the hour and add calendar features."""
    df = trips_hourly.merge(weather_hourly, on="hour", how="inner", validate="one_to_one")
    holidays = USFederalHolidayCalendar().holidays(df["hour"].min(), df["hour"].max())
    day = df["hour"].dt.normalize()
    df["hour_of_day"] = df["hour"].dt.hour
    df["day_of_week"] = df["hour"].dt.day_of_week
    df["month"] = df["hour"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_holiday"] = day.isin(holidays).astype(int)
    df["is_raining"] = (df["precip_mm"] > 0.1).astype(int)
    return df.sort_values("hour").reset_index(drop=True)
