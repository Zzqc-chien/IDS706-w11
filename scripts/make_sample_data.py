"""Generate small synthetic Citi Bike + Open-Meteo raw files in the real formats.

Used by the tests, and for an offline demo run: files placed in data/raw/ are picked up
by the extract tasks as a cache, so no download happens.

    python scripts/make_sample_data.py --months 2026-04 2026-05 --out data/raw
"""
from __future__ import annotations

import argparse
import io
import json
import uuid
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def _weather(hours: pd.DatetimeIndex, rng) -> pd.DataFrame:
    doy = hours.day_of_year.to_numpy()
    hod = hours.hour.to_numpy()
    temp = 12 + 12 * np.sin((doy - 110) / 365 * 2 * np.pi) + 4 * np.sin((hod - 9) / 24 * 2 * np.pi)
    temp += rng.normal(0, 1.5, len(hours))
    rain = np.where(rng.random(len(hours)) < 0.08, rng.gamma(1.2, 1.5, len(hours)), 0.0)
    return pd.DataFrame(
        {
            "time": hours.strftime("%Y-%m-%dT%H:%M"),
            "temperature_2m": temp.round(1),
            "precipitation": rain.round(1),
            "wind_speed_10m": rng.gamma(3, 4, len(hours)).round(1),
            "relative_humidity_2m": rng.integers(35, 95, len(hours)),
        }
    )


def _expected_trips(w: pd.DataFrame) -> np.ndarray:
    t = pd.to_datetime(w["time"])
    hod, weekend = t.dt.hour.to_numpy(), (t.dt.day_of_week >= 5).to_numpy()
    commute = np.exp(-((hod - 8) ** 2) / 3) + np.exp(-((hod - 18) ** 2) / 4)
    leisure = np.exp(-((hod - 14) ** 2) / 12)
    shape = np.where(weekend, 0.2 + 1.2 * leisure, 0.15 + 1.4 * commute + 0.4 * leisure)
    temp_effect = np.clip((w["temperature_2m"].to_numpy() + 5) / 25, 0.1, 1.3)
    rain_effect = np.where(w["precipitation"].to_numpy() > 0.1, 0.45, 1.0)
    return 40 * shape * temp_effect * rain_effect


def make_month(month: str, out: Path, seed: int = 0) -> None:
    rng = np.random.default_rng(seed + int(month.replace("-", "")))
    start = pd.Timestamp(f"{month}-01")
    hours = pd.date_range(start, start + pd.offsets.MonthEnd(0) + pd.Timedelta(hours=23), freq="h")
    weather = _weather(hours, rng)

    counts = rng.poisson(_expected_trips(weather))
    starts = np.repeat(hours.to_numpy(), counts) + pd.to_timedelta(
        rng.integers(0, 3600, counts.sum()), unit="s"
    ).to_numpy()
    n = len(starts)
    duration = pd.to_timedelta(rng.gamma(2.0, 6.0, n) * 60, unit="s").to_numpy()
    trips = pd.DataFrame(
        {
            "ride_id": [uuid.UUID(int=int(x)).hex[-16:].upper() for x in rng.integers(0, 2**63, n)],
            "rideable_type": rng.choice(["classic_bike", "electric_bike"], n, p=[0.55, 0.45]),
            "started_at": pd.Series(starts).dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3],
            "ended_at": pd.Series(starts + duration).dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3],
            "start_station_name": "Grove St PATH",
            "start_station_id": "JC005",
            "end_station_name": "Hoboken Terminal",
            "end_station_id": "HB101",
            "start_lat": 40.7196, "start_lng": -74.0431, "end_lat": 40.7359, "end_lng": -74.0303,
            "member_casual": rng.choice(["member", "casual"], n, p=[0.7, 0.3]),
        }
    )
    # Realistic dirt: duplicates, missing timestamps, 0-second and multi-day rides.
    dirty = trips.sample(frac=0.01, random_state=1)
    trips.loc[trips.sample(frac=0.005, random_state=2).index, "ended_at"] = None
    trips.loc[trips.sample(frac=0.005, random_state=3).index, "ended_at"] = trips["started_at"]
    trips = pd.concat([trips, dirty], ignore_index=True)

    (out / "trips").mkdir(parents=True, exist_ok=True)
    (out / "weather").mkdir(parents=True, exist_ok=True)
    ym = month.replace("-", "")
    buf = io.StringIO()
    trips.to_csv(buf, index=False)
    with zipfile.ZipFile(out / "trips" / f"JC-{ym}-citibike-tripdata.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"JC-{ym}-citibike-tripdata.csv", buf.getvalue())

    payload = {
        "latitude": 40.72, "longitude": -74.04, "timezone": "America/New_York",
        "hourly_units": {"time": "iso8601", "temperature_2m": "°C"},
        "hourly": {c: weather[c].tolist() for c in weather.columns},
    }
    (out / "weather" / f"weather-{month}.json").write_text(json.dumps(payload))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="+", required=True)
    ap.add_argument("--out", default="data/raw")
    args = ap.parse_args()
    for m in args.months:
        make_month(m, Path(args.out))
        print(f"wrote sample data for {m} -> {args.out}")
