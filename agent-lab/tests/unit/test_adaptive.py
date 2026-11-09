"""Part 3: the adaptive multi-agent system and its evaluation harness."""
import pytest

from adaptive.eval.cases import build_cases, score_case
from adaptive.eval.datasets import make_hourly, write_dataset
from adaptive.eval.systems import run_system
from adaptive.graph import _rule_decision, run_adaptive


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("CURSOR_API_KEY", raising=False)
    monkeypatch.setenv("LOOP_REPORT_DIR", str(tmp_path / "reports"))


@pytest.fixture()
def leaky_csv(tmp_path):
    df, contract = make_hourly(days=90, leakage=True)
    return write_dataset(df, contract, tmp_path, "leaky")


def test_supervisor_routes_and_excludes_leakage(leaky_csv):
    r = run_adaptive(dataset_path=str(leaky_csv), target_column="high_demand", budget=6)
    assert r["agent_path"][:2] == ["data_validator", "supervisor"]
    assert r["agent_path"][-2:] == ["holdout_eval", "report_writer"]
    assert "trips" not in r["holdout"]["features_used"]
    assert r["holdout"]["split"] == "chronological"
    assert len(r["experiments"]) <= 6


def test_broken_family_is_retried_then_blocked(leaky_csv):
    r = run_adaptive(dataset_path=str(leaky_csv), target_column="high_demand",
                     fault_plan={"broken_families": ["random_forest"]})
    rf = [e for e in r["experiments"] if e["model_family"] == "random_forest"]
    assert [e["status"] for e in rf] == ["failed"] * 3  # first try + 2 recoveries, then blocked
    assert any(d["next_agent"] == "error_recovery" for d in r["decisions"])
    assert r["holdout"] and r["best"]["model_family"] != "random_forest"


def test_invalid_target_skips_training(tmp_path):
    df, contract = make_hourly(days=30, multiclass_target=True)
    path = write_dataset(df, contract, tmp_path, "multi")
    r = run_adaptive(dataset_path=str(path), target_column="high_demand")
    assert r["experiments"] == [] and r["holdout"] is None
    assert [d["next_agent"] for d in r["decisions"]] == ["report_writer"]


def test_unreachable_target_uses_varied_strategies_without_repeats(leaky_csv):
    r = run_adaptive(dataset_path=str(leaky_csv), target_column="high_demand", target_score=0.99, budget=8)
    agents = {e["agent"] for e in r["experiments"]}
    assert {"ml_engineer", "tuner", "feature_engineer"} <= agents
    sigs = [(e["model_family"], str(e["hyperparameters"]), tuple(e["feature_ops"])) for e in r["experiments"]]
    assert len(sigs) == len(set(sigs)) and len(sigs) == 8


def test_rule_decision_finalizes_when_target_met():
    state = {"validation": {"status": "ok"}, "profile": {"x": 1}, "attempts": 1, "budget": 8,
             "best": {"val_f1": 0.9, "model_family": "random_forest", "capacity_level": 0},
             "target_score": 0.8, "experiments": [{"model_family": "random_forest", "status": "succeeded"}]}
    assert _rule_decision(state)[0] == "holdout_eval"


@pytest.mark.parametrize("system", ["fixed_airflow", "inclass_agent", "adaptive"])
def test_harness_runs_every_system(system, tmp_path):
    case = next(c for c in build_cases() if c.id == "C2")
    path = case.dataset_builder(tmp_path)
    r = run_system(system, dataset_path=str(path), **case.run_params())
    scored = score_case(case, r)
    assert r["status"] in {"trained", "rejected", "crashed"} and 0 <= scored["score"] <= 1
    if system == "adaptive":
        assert scored["passed"]
    if system == "fixed_airflow":  # uses every column, so the label leaks through `trips`
        assert not scored["passed"]
