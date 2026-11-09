"""Evaluation suite: each case has an input, the expected behaviour, scoring criteria and a
pass/fail threshold.

Checks come in three levels:
  intermediate   quality of what happens on the way (leakage handling, split, recovery)
  final          quality of the delivered outcome (holdout F1, plausibility, decision)
  coordination   how well the agents worked together (budget, early stop, no repeated
                 experiments, variety of strategies, wasted retries)

A case passes when every `required` check scores 1.0 AND the mean score of all its
checks is at least `pass_threshold`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from adaptive.eval.datasets import make_hourly, write_dataset

REPO = Path(__file__).resolve().parents[2]


@dataclass
class Check:
    name: str
    level: str  # intermediate | final | coordination
    criterion: str  # human-readable scoring rule
    fn: Callable[[dict, "Case"], tuple[float, str]]
    required: bool = False


@dataclass
class Case:
    id: str
    title: str
    input: dict  # dataset spec + run parameters (shown in the report)
    expected: str
    checks: list[Check]
    pass_threshold: float
    dataset_builder: Callable[[Path], Path] | None = None
    note: str = ""
    params: dict = field(default_factory=dict)

    def run_params(self) -> dict:
        p = {"target_column": "high_demand", "target_score": 0.8, "budget": 8, "fault_plan": None}
        p.update(self.params)
        return p


# ---------------------------------------------------------------------------
# check factories — each returns (score in [0, 1], detail)
# ---------------------------------------------------------------------------
def _f1(r: dict) -> float | None:
    return ((r.get("holdout") or {}).get("metrics") or {}).get("f1")


def trained(level="final", required=True) -> Check:
    return Check("delivers a trained model", level, "status == trained", lambda r, c: (
        float(r["status"] == "trained"), r["status"] + (f" ({r.get('error')})" if r.get("error") else "")), required)


def f1_at_least(t: float, required=False) -> Check:
    def fn(r, c):
        f = _f1(r)
        return (0.0, "no model") if f is None else (float(f >= t), f"holdout F1 {f:.3f} (need ≥ {t})")
    return Check(f"holdout F1 ≥ {t}", "final", f"F1 on the sealed holdout ≥ {t}", fn, required)


def beats_majority(required=False) -> Check:
    def fn(r, c):
        h = r.get("holdout") or {}
        if not h:
            return 0.0, "no model"
        acc, base = h["metrics"]["accuracy"], h["majority_baseline"]["accuracy"]
        return float(acc > base and (h["metrics"]["f1"] or 0) > 0), f"acc {acc:.3f} vs majority {base:.3f}"
    return Check("beats majority-class baseline", "final", "accuracy > always-majority and F1 > 0", fn, required)


def plausible(max_f1=0.99, required=True) -> Check:
    def fn(r, c):
        f = _f1(r)
        return (1.0, "no model") if f is None else (float(f < max_f1), f"F1 {f:.3f} (≥ {max_f1} means the label leaked)")
    return Check("score is plausible (no leak)", "final", f"holdout F1 < {max_f1}", fn, required)


def leak_excluded(cols: list[str], required=True) -> Check:
    def fn(r, c):
        if r["status"] == "rejected":
            return 1.0, "dataset rejected, nothing trained"
        if r["status"] != "trained":
            return 0.0, f"no model ({r['status']})"
        used = [col for col in cols if col in (r.get("features_used") or [])]
        return 1 - len(used) / len(cols), ("used: " + ", ".join(used)) if used else "none used"
    return Check(f"leakage columns excluded {cols}", "intermediate", "fraction of leak columns NOT used as features", fn, required)


def leak_flagged(cols: list[str]) -> Check:
    def fn(r, c):
        hit = [col for col in cols if col in (r.get("flagged") or [])]
        return len(hit) / len(cols), f"flagged: {hit or 'none'}"
    return Check(f"system flags leakage {cols}", "intermediate", "fraction of leak columns the system itself flagged", fn)


def time_aware_split() -> Check:
    def fn(r, c):
        split = (r.get("holdout") or {}).get("split")
        return (0.0, "no model") if not split else (float(split == "chronological"), split)
    return Check("time-aware split", "intermediate", "holdout = latest hours (chronological)", fn)


def rejects(required=True) -> Check:
    return Check("rejects the invalid dataset", "final", "status == rejected (not trained, not crashed)",
                 lambda r, c: (float(r["status"] == "rejected"), r["status"]), required)


def no_wasted_training() -> Check:
    return Check("no training on invalid data", "coordination", "0 training runs",
                 lambda r, c: (float((r.get("attempts") or 0) == 0), f"{r.get('attempts') or 0} runs"))


def within_budget(required=False) -> Check:
    def fn(r, c):
        n, b = r.get("attempts") or 0, c.run_params()["budget"]
        return float(n <= b), f"{n} training runs (budget {b})"
    return Check("stays within budget", "coordination", "training runs ≤ budget", fn, required)


def stops_early(max_runs: int, required=False) -> Check:
    def fn(r, c):
        ok = [e for e in r.get("experiments") or []]
        return float(r["status"] == "trained" and len(ok) <= max_runs), f"{len(ok)} runs (target met → stop by {max_runs})"
    return Check(f"stops once target met (≤ {max_runs} runs)", "coordination", f"training runs ≤ {max_runs}", fn, required)


def no_duplicate_experiments() -> Check:
    def fn(r, c):
        sigs = [json.dumps([e.get("model_family"), e.get("hyperparameters"), sorted(e.get("feature_ops") or [])],
                           sort_keys=True, default=str) for e in r.get("experiments") or [] if e.get("status") == "succeeded"]
        dup = len(sigs) - len(set(sigs))
        return (1.0 if not sigs else 1 - dup / len(sigs)), f"{dup} repeated of {len(sigs)}"
    return Check("no repeated experiments", "coordination", "1 − repeated / successful runs", fn)


def strategy_variety(k: int = 2) -> Check:
    def fn(r, c):
        kinds = set()
        for e in r.get("experiments") or []:
            agent = e.get("agent")
            if agent in ("ml_engineer", "tuner", "feature_engineer"):
                kinds.add(agent)
        families = {e.get("model_family") for e in r.get("experiments") or []}
        if not kinds and len(families) > 1:
            kinds = {"ml_engineer"}
        return min(len(kinds), k) / k, f"strategies used: {sorted(kinds) or ['fixed']}"
    return Check(f"uses ≥ {k} kinds of fix (models/tuning/features)", "coordination", "distinct strategy agents / k", fn)


def recovers(required=True) -> Check:
    def fn(r, c):
        failed = [e for e in r.get("experiments") or [] if e.get("status") == "failed"]
        return float(r["status"] == "trained"), f"{len(failed)} failed runs, final status {r['status']}"
    return Check("recovers and delivers a model", "intermediate", "trained despite injected failures", fn, required)


def limited_retry_waste(max_failed: int) -> Check:
    def fn(r, c):
        failed = [e for e in r.get("experiments") or [] if e.get("status") == "failed"]
        # systems without per-run records (the fixed DAG) report failed attempts in `attempts`
        n = len(failed) if failed or r["status"] == "trained" else int(r.get("attempts") or 0)
        return float(n <= max_failed), f"{n} failed runs (≤ {max_failed} allowed)"
    return Check(f"wastes ≤ {max_failed} runs on failures", "coordination", "failed runs ≤ limit", fn)


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------
def _synthetic(name: str, **kw):
    contract = kw.pop("_contract", "default")

    def build(out_dir: Path) -> Path:
        df, default_contract = make_hourly(**kw)
        return write_dataset(df, None if contract is None else default_contract if contract == "default" else contract,
                             out_dir, name)
    return build


def build_cases() -> list[Case]:
    cases = [
        Case("C1", "Clean hourly data",
             {"data": "synthetic hourly, 150 days, no leakage", "target_score": 0.8, "budget": 8},
             "Train a model that beats the majority baseline with holdout F1 ≥ 0.75, on a time-aware split, within budget.",
             [trained(), f1_at_least(0.75, required=True), beats_majority(required=True), time_aware_split(), within_budget()],
             0.8, _synthetic("c1_clean")),
        Case("C2", "Strong leakage, no contract",
             {"data": "synthetic + `trips` (defines the label) + `member_share`; contract lists nothing"},
             "Detect that `trips` leaks the label, exclude it, and still forecast well (F1 ≥ 0.75, but not a too-good-to-be-true ≥ 0.99).",
             [trained(), leak_excluded(["trips"]), leak_flagged(["trips"]), plausible(), f1_at_least(0.75)],
             0.8, _synthetic("c2_leak", leakage=True)),
        Case("C3", "Leakage declared in the data contract",
             {"data": "synthetic + `trips`, `member_share`; contract lists both"},
             "Honor the producer's contract: exclude both declared columns.",
             [trained(), leak_excluded(["trips", "member_share"]), leak_flagged(["trips", "member_share"]), plausible(), f1_at_least(0.75)],
             0.8, _synthetic("c3_contract", leakage=True, _contract={"target": "high_demand", "time_column": "hour",
                                                                   "leakage_columns": ["trips", "member_share"]})),
        Case("C4", "Weak leakage, no contract (limitation probe)",
             {"data": "synthetic + `trips`, `member_share`; no contract file"},
             "Exclude `member_share` too. It is derived from the same rides but only weakly separates the classes. "
             "Expected to be hard for every system; included to show the limits of automatic detection.",
             [leak_excluded(["member_share"])],
             1.0, _synthetic("c4_weak_leak", leakage=True, _contract=None),
             note="Known limitation: needs a contract (C3) or domain knowledge."),
        Case("C5", "Invalid target (3 classes)",
             {"data": "synthetic, `high_demand` replaced by low/mid/high"},
             "Reject before any training and say why (this workflow is binary classification). No crash, no wasted runs.",
             [rejects(), no_wasted_training()],
             1.0, _synthetic("c5_multiclass", multiclass_target=True)),
        Case("C6", "Transient training failure",
             {"data": "clean synthetic", "fault_plan": {"transient_failures": 1}},
             "The first training run crashes (worker lost). Retry, and deliver a model of normal quality.",
             [recovers(), f1_at_least(0.75), limited_retry_waste(1)],
             0.8, _synthetic("c6_transient"), params={"fault_plan": {"transient_failures": 1}}),
        Case("C7", "Persistent failure of one model family",
             {"data": "clean synthetic", "fault_plan": {"broken_families": ["random_forest"]}},
             "Random forest always crashes. Stop retrying it after a bounded number of tries, route to another family, and still deliver.",
             [recovers(), f1_at_least(0.75), limited_retry_waste(3)],
             0.8, _synthetic("c7_broken_rf"), params={"fault_plan": {"broken_families": ["random_forest"]}}),
        Case("C8", "Easy target: stop early",
             {"data": "clean synthetic", "target_score": 0.6, "budget": 8},
             "The target is easy; stop as soon as it's met (≤ 2 training runs) instead of spending the budget.",
             [trained(), stops_early(2, required=True), f1_at_least(0.6)],
             1.0, _synthetic("c8_easy"), params={"target_score": 0.6}),
        Case("C9", "Unreachable target: search well",
             {"data": "noisier synthetic", "target_score": 0.97, "budget": 8},
             "The target can't be reached. Use the budget on varied, non-repeated ideas (models, tuning, features), stop at the budget, and still deliver the best model.",
             [trained(), within_budget(required=True), no_duplicate_experiments(), strategy_variety(3), f1_at_least(0.7)],
             0.75, _synthetic("c9_hard", noise=0.45), params={"target_score": 0.97}),
    ]
    real = REPO / "datasets" / "citibike_hourly_latest.csv"
    if real.exists():
        leak = json.loads(real.with_suffix(".contract.json").read_text()).get("leakage_columns", []) \
            if real.with_suffix(".contract.json").exists() else ["trips"]
        cases.append(Case(
            "C10", "Real Citi Bike data (from Part 1)",
            {"data": "datasets/citibike_hourly_latest.csv (published by Airflow)", "target_score": 0.8, "budget": 8},
            "Exclude the contract's leakage columns, split by time, and forecast busy hours with F1 ≥ 0.75, beating the majority baseline.",
            [trained(), leak_excluded(leak), plausible(), time_aware_split(), f1_at_least(0.75, required=True), beats_majority()],
            0.8, lambda out_dir: real))
    titanic = REPO / "datasets" / "Titanic-Dataset.csv"
    if titanic.exists():
        cases.append(Case(
            "C11", "Generality: Titanic (no time column)",
            {"data": "datasets/Titanic-Dataset.csv, target Survived", "target_score": 0.8, "budget": 6},
            "Work on a non-time-series dataset: random split, F1 ≥ 0.65, beats majority, within budget.",
            [trained(), f1_at_least(0.65), beats_majority(), within_budget()],
            0.75, lambda out_dir: titanic, params={"target_column": "Survived", "budget": 6}))
    return cases


def score_case(case: Case, result: dict) -> dict:
    rows = []
    for chk in case.checks:
        try:
            s, detail = chk.fn(result, case)
        except Exception as exc:  # noqa: BLE001
            s, detail = 0.0, f"check error: {exc}"
        rows.append({"name": chk.name, "level": chk.level, "criterion": chk.criterion, "required": chk.required,
                     "score": round(float(s), 3), "detail": detail})
    mean = sum(r["score"] for r in rows) / len(rows)
    req_ok = all(r["score"] == 1.0 for r in rows if r["required"])
    return {"checks": rows, "score": round(mean, 3), "passed": bool(req_ok and mean >= case.pass_threshold)}
