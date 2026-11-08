"""Ingestion: download raw files once and cache them under data/raw/."""
from __future__ import annotations

import calendar
import json
import logging
from datetime import date
from pathlib import Path

import requests

from citibike import config

log = logging.getLogger(__name__)


def months_for_run(run_date: date, n_months: int, lag: int) -> list[str]:
    """Return n_months consecutive 'YYYY-MM' strings ending `lag` months before run_date."""
    end_index = run_date.year * 12 + (run_date.month - 1) - lag
    return [
        f"{i // 12:04d}-{i % 12 + 1:02d}" for i in range(end_index - n_months + 1, end_index + 1)
    ]


def month_bounds(month: str) -> tuple[date, date]:
    year, mon = map(int, month.split("-"))
    return date(year, mon, 1), date(year, mon, calendar.monthrange(year, mon)[1])


def download_trips(month: str, raw_dir: Path | None = None) -> str:
    """Download one month of Citi Bike trips (zip). Skips the download if it is cached."""
    raw_dir = Path(raw_dir or config.RAW_DIR) / "trips"
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / f"JC-{month.replace('-', '')}-citibike-tripdata.zip"
    if target.exists() and target.stat().st_size > 0:
        log.info("trips %s: using cached %s", month, target)
        return str(target)

    ym = month.replace("-", "")
    errors = []
    for pattern in config.TRIP_URL_PATTERNS:
        url = pattern.format(ym=ym)
        resp = requests.get(url, timeout=config.HTTP_TIMEOUT)
        if resp.status_code == 200:
            tmp = target.with_suffix(".part")
            tmp.write_bytes(resp.content)
            tmp.replace(target)  # atomic: a half-downloaded file is never visible
            log.info("trips %s: downloaded %s (%.1f MB)", month, url, len(resp.content) / 1e6)
            return str(target)
        errors.append(f"{url} -> HTTP {resp.status_code}")
    raise FileNotFoundError(
        f"No Citi Bike file for {month} (not published yet?). Tried: {'; '.join(errors)}"
    )


def download_weather(month: str, raw_dir: Path | None = None) -> str:
    """Download hourly weather for one month from Open-Meteo. Skips it if cached."""
    raw_dir = Path(raw_dir or config.RAW_DIR) / "weather"
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / f"weather-{month}.json"
    if target.exists() and target.stat().st_size > 0:
        log.info("weather %s: using cached %s", month, target)
        return str(target)

    start, end = month_bounds(month)
    params = {
        "latitude": config.LATITUDE,
        "longitude": config.LONGITUDE,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": ",".join(config.WEATHER_VARS),
        "timezone": config.TIMEZONE,
    }
    resp = requests.get(config.WEATHER_URL, params=params, timeout=config.HTTP_TIMEOUT)
    resp.raise_for_status()
    payload = resp.json()
    if "hourly" not in payload:
        raise ValueError(f"Unexpected weather response for {month}: {str(payload)[:200]}")
    tmp = target.with_suffix(".part")
    tmp.write_text(json.dumps(payload))
    tmp.replace(target)
    log.info("weather %s: downloaded %d hours", month, len(payload["hourly"]["time"]))
    return str(target)
