"""Tests for the agents added for the Citi Bike workflow (data_validator, report_writer, guard)."""
import json

import numpy as np
import pandas as pd
import pytest

from solution.langgraph import graph as g


@pytest.fixture()
def hourly_csv(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("CURSOR_API_KEY", raising=False)
    monkeypatch.setenv("LOOP_REPORT_DIR", str(tmp_path / "reports"))
    rng = np.random.default_rng(0)
    hours = pd.date_range("2026-04-01", periods=24 * 60, freq="h")
    hod = hours.hour.to_numpy()
    temp = rng.normal(18, 6, len(hours))
    trips = rng.poisson(5 + 30 * np.exp(-((hod - 17) ** 2) / 6) * (temp > 12))
    df = pd.DataFrame(
        {
            "hour": hours.strftime("%Y-%m-%d %H:%M"),
            "trips": trips,
            "member_share": rng.random(len(hours)),
            "temp_c": temp.round(1),
            "hour_of_day": hod,
            "day_of_week": hours.dayofweek,
        }
    )
    df.insert(1, "high_demand", (df["trips"] >= df["trips"].quantile(0.75)).astype(int))
    path = tmp_path / "hourly.csv"
    df.to_csv(path, index=False)
    path.with_suffix(".contract.json").write_text(
        json.dumps({"target": "high_demand", "time_column": "hour", "leakage_columns": ["member_share"]})
    )
    return path


def _state(path, target="high_demand"):
    return g._initial_state(
        dataset_path=str(path), target_column=target, target_score=0.8,
        max_cycles=3, seed=42, holdout_frac=0.2,
    )


def test_validator_reads_contract_and_flags_correlated_leakage(hourly_csv):
    out = g.node_data_validator(_state(hourly_csv))["validation"]
    assert out["status"] == "ok"
    assert out["time_column"] == "hour" and out["split"] == "chronological"
    # member_share from the contract; trips caught by the correlation check
    assert {"member_share", "trips", "hour"} <= set(out["must_drop"])


def test_validator_rejects_non_binary_target(hourly_csv):
    out = g.node_data_validator(_state(hourly_csv, target="trips"))["validation"]
    assert out["status"] == "invalid"
    assert g.route_after_validator({"validation": out}) == "report"


def test_ml_engineer_guard_drops_leakage_even_if_spec_keeps_it(hourly_csv, monkeypatch):
    state = {**_state(hourly_csv), **g.node_data_validator(_state(hourly_csv))}
    monkeypatch.setattr(
        g, "_heuristic_spec",
        lambda *a, **k: {"experiment_id": "exp-1", "model_family": "random_forest",
                         "hyperparameters": {"n_estimators": 50, "max_depth": 5}, "drop_columns": []},
    )
    out = g.node_ml_engineer(state)
    assert {"trips", "member_share", "hour"} <= set(out["experiment_spec"]["drop_columns"])
    assert out["execution_result"]["split"] == "chronological"
    assert out["execution_result"]["validation_metrics"]["f1"] < 0.999  # no leakage


def test_full_loop_ends_with_report(hourly_csv):
    result = g.run_solution_loop(dataset_path=str(hourly_csv), target_column="high_demand", max_cycles=3)
    nodes = [s["node"] for s in result["steps"]]
    assert nodes[0] == "data_validator" and nodes[-1] == "report_writer"
    report = result["report"]
    assert report["split"] == "chronological"
    assert "trips" not in report["features_used"] and "hour" not in report["features_used"]
    assert sum(map(sum, report["confusion_matrix"]["rows_true_cols_pred"])) == result["holdout_metrics"]["n_holdout"]
    assert report["holdout_metrics"]["f1"] == pytest.approx(result["holdout_metrics"]["f1"])
    assert report["saved_to"].endswith(".json")


def test_invalid_dataset_skips_modeling(hourly_csv):
    result = g.run_solution_loop(dataset_path=str(hourly_csv), target_column="trips", max_cycles=2)
    assert [s["node"] for s in result["steps"]] == ["data_validator", "report_writer"]
    assert result["report"]["validation"]["status"] == "invalid"


def test_report_matches_the_single_holdout_fit(hourly_csv):
    result = g.run_solution_loop(dataset_path=str(hourly_csv), target_column="high_demand", max_cycles=2)
    report, hold = result["report"], result["holdout_metrics"]
    cm = report["confusion_matrix"]["rows_true_cols_pred"]
    tp, fp, fn = cm[1][1], cm[0][1], cm[1][0]
    assert hold["f1"] == pytest.approx(2 * tp / (2 * tp + fp + fn))
    assert report["sklearn_version"]


def test_validator_screens_on_training_segment_only(hourly_csv, tmp_path):
    df = pd.read_csv(hourly_csv)
    n_train = int(len(df) * 0.6)
    df["future_leak"] = 0.0
    df.loc[n_train:, "future_leak"] = df.loc[n_train:, "high_demand"]  # perfect only after the train segment
    path = tmp_path / "late_leak.csv"
    df.to_csv(path, index=False)
    out = g.node_data_validator(_state(path))["validation"]
    assert "future_leak" not in out["leakage_columns"]  # holdout labels must not drive the screen
