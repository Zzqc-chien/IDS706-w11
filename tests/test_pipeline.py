from datetime import date

import pandas as pd
import pytest

from conftest import MONTHS
from citibike import analyze, extract, transform


def test_months_for_run_respects_lag_and_year_boundary():
    assert extract.months_for_run(date(2026, 11, 7), 6, 2) == [
        "2026-04", "2026-05", "2026-06", "2026-07", "2026-08", "2026-09"
    ]
    assert extract.months_for_run(date(2026, 2, 1), 3, 2) == ["2025-10", "2025-11", "2025-12"]


def test_extract_uses_cache_without_network(raw_dir, monkeypatch):
    def no_network(*a, **k):
        raise AssertionError("should not download a cached file")

    monkeypatch.setattr(extract.requests, "get", no_network)
    assert extract.download_trips("2026-04", raw_dir).endswith("JC-202604-citibike-tripdata.zip")
    assert extract.download_weather("2026-04", raw_dir).endswith("weather-2026-04.json")


def test_extract_reports_unpublished_month(tmp_path, monkeypatch):
    class Resp:
        status_code = 404

    monkeypatch.setattr(extract.requests, "get", lambda *a, **k: Resp())
    with pytest.raises(FileNotFoundError, match="not published"):
        extract.download_trips("2099-01", tmp_path)


@pytest.fixture(scope="module")
def cleaned(raw_dir):
    raw = pd.concat(
        [transform.read_trip_zip(raw_dir / "trips" / f"JC-{m.replace('-', '')}-citibike-tripdata.zip") for m in MONTHS]
    )
    return raw, *transform.clean_trips(raw, MONTHS)


def test_clean_trips_removes_dirty_rows(cleaned):
    raw, trips, report = cleaned
    assert trips["ride_id"].is_unique
    assert trips["duration_min"].between(1, 240).all()
    assert report["dropped_duplicate_ride_id"] > 0
    assert report["dropped_missing_required"] > 0
    assert report["dropped_duration_out_of_range"] > 0
    dropped = sum(v for k, v in report.items() if k.startswith("dropped_"))
    assert report["rows_in"] - dropped == report["rows_out"] == len(trips)


def test_hourly_aggregation_keeps_every_hour_and_every_trip(cleaned):
    _, trips, _ = cleaned
    hourly = transform.trips_to_hourly(trips, MONTHS)
    assert len(hourly) == (30 + 31) * 24  # April + May, including zero-trip hours
    assert hourly["hour"].is_unique
    assert hourly["trips"].sum() == len(trips)


def test_clean_weather_fills_gaps_and_drops_dst_duplicates(raw_dir):
    raw = transform.read_weather_json(raw_dir / "weather" / "weather-2026-04.json")
    raw = pd.concat([raw, raw.iloc[[5]]])  # duplicate hour
    raw.loc[raw.index[10:12], "temperature_2m"] = None  # short gap
    raw.loc[raw.index[20], "relative_humidity_2m"] = 300  # impossible value
    weather, report = transform.clean_weather(raw, ["2026-04"])
    assert len(weather) == 30 * 24
    assert weather.notna().all().all()
    assert report["dropped_duplicate_hour"] == 1
    assert report["values_out_of_range"] == 1


def test_merge_and_model(raw_dir, cleaned, tmp_path):
    _, trips, _ = cleaned
    weather = pd.concat([transform.read_weather_json(raw_dir / "weather" / f"weather-{m}.json") for m in MONTHS])
    w, _ = transform.clean_weather(weather, MONTHS)
    merged = transform.merge_hourly(transform.trips_to_hourly(trips, MONTHS), w)
    assert len(merged) == (30 + 31) * 24
    assert merged.loc[merged["hour"] == "2026-05-25 12:00", "is_holiday"].item() == 1  # Memorial Day

    metrics = analyze.train_and_evaluate(merged, tmp_path)
    assert metrics["rf_r2"] > metrics["baseline_r2"]  # weather features add signal
    assert (tmp_path / "metrics.json").exists() and (tmp_path / "predictions.png").exists()
    assert analyze.make_plots(merged, tmp_path).endswith("eda.png")


def test_publish_for_agents_writes_csv_and_contract(raw_dir, cleaned, tmp_path):
    import json

    from citibike.publish import LEAKAGE_COLUMNS, publish

    _, trips, _ = cleaned
    weather = pd.concat([transform.read_weather_json(raw_dir / "weather" / f"weather-{m}.json") for m in MONTHS])
    w, _ = transform.clean_weather(weather, MONTHS)
    merged = transform.merge_hourly(transform.trips_to_hourly(trips, MONTHS), w)

    path = publish(merged, tmp_path)
    df = pd.read_csv(path)
    contract = json.loads((tmp_path / "citibike_hourly_latest.contract.json").read_text())
    assert list(df.columns[:2]) == ["hour", "high_demand"]
    assert set(df["high_demand"]) == {0, 1}
    assert 0.2 < df["high_demand"].mean() < 0.3
    assert contract["time_column"] == "hour" and contract["leakage_columns"] == LEAKAGE_COLUMNS
    assert contract["n_rows"] == len(df) == len(merged)


def test_fill_short_gaps_only_fills_up_to_three_hours():
    s = pd.Series([1.0, None, None, None, 5.0, None, None, None, None, None, None, 12.0])
    out = transform.fill_short_gaps(s)
    assert out.iloc[1:4].notna().all()  # 3-hour gap filled
    assert out.iloc[5:11].isna().all()  # 6-hour gap left missing (pandas' limit=3 would fill it)


def test_non_consecutive_months_do_not_include_the_gap():
    idx = transform.hourly_index(["2026-04", "2026-06"])
    assert len(idx) == (30 + 30) * 24
    assert not (idx.month == 5).any()


def test_publish_threshold_uses_training_segment_only(raw_dir, cleaned):
    from citibike.publish import TRAIN_SEGMENT, build_agent_dataset

    _, trips, _ = cleaned
    hourly = transform.trips_to_hourly(trips, MONTHS)
    # make the later (validation/holdout) period much busier: the threshold must not move
    late = hourly.index >= int(len(hourly) * TRAIN_SEGMENT)
    busier = hourly.copy()
    busier.loc[late, "trips"] *= 10
    t1 = build_agent_dataset(hourly)[1]["target_definition"]
    t2 = build_agent_dataset(busier)[1]["target_definition"]
    assert t1 == t2


def test_model_report_includes_time_only_ablation(raw_dir, cleaned, tmp_path):
    _, trips, _ = cleaned
    weather = pd.concat([transform.read_weather_json(raw_dir / "weather" / f"weather-{m}.json") for m in MONTHS])
    w, _ = transform.clean_weather(weather, MONTHS)
    merged = transform.merge_hourly(transform.trips_to_hourly(trips, MONTHS), w)
    m = analyze.train_and_evaluate(merged, tmp_path)
    assert {"baseline_r2", "rf_time_r2", "rf_r2", "sklearn_version"} <= set(m)
