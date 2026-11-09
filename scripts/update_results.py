"""Collect the results of the latest runs and write them into the READMEs.

    make results        (= python3 scripts/update_results.py)

Run it after Part 1 (Airflow), Part 2 (/solution-loop) and Part 3 (make part3-eval).
It only uses the standard library, so it runs with any Python 3.

1. Copies the evidence into docs/results/ (and the two Part 1 plots into docs/screenshots/).
2. Rewrites the blocks between <!-- results:NAME --> and <!-- /results:NAME --> in
   README.md and agent-lab/adaptive/README.md. Everything outside the blocks is untouched.

A part whose output files don't exist yet is skipped; its block keeps its previous text.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "agent-lab"
RESULTS = ROOT / "docs" / "results"
SHOTS = ROOT / "docs" / "screenshots"


def _f(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def _replace_block(path: Path, name: str, body: str) -> bool:
    text = path.read_text()
    pattern = re.compile(rf"(<!-- results:{name} -->\n).*?(\n<!-- /results:{name} -->)", re.S)
    if not pattern.search(text):
        print(f"  ! no results:{name} block in {path.relative_to(ROOT)}")
        return False
    path.write_text(pattern.sub(lambda m: m.group(1) + body.strip() + m.group(2), text))
    return True


# ---------------------------------------------------------------------------
def part1() -> str | None:
    src = ROOT / "data" / "output" / "metrics.json"
    if not src.exists():
        return None
    m = json.loads(src.read_text())
    shutil.copy(src, RESULTS / "part1-metrics.json")
    for name, dest in (("eda.png", "03-eda.png"), ("predictions.png", "04-model-predictions.png")):
        if (src.parent / name).exists():
            shutil.copy(src.parent / name, SHOTS / dest)
    imp = list((m.get("feature_importances") or {}).items())[:4]
    rows = [
        ("Seasonal baseline: mean of the same day-of-week × hour", "baseline"),
        ("Random forest, time features only", "rf_time"),
        ("Random forest, time + weather features", "rf"),
    ]
    table = "\n".join(
        f"| {label} | {_f(m.get(f'{k}_mae'), 1)} | {_f(m.get(f'{k}_rmse'), 1)} | {_f(m.get(f'{k}_r2'))} |"
        for label, k in rows if m.get(f"{k}_r2") is not None
    )
    weather = ""
    if m.get("rf_time_r2") is not None:
        weather = (
            f"\nThe two random forests have the same settings and differ only in their inputs. Adding weather changes "
            f"R² from {_f(m['rf_time_r2'])} to {_f(m['rf_r2'])} and MAE from {_f(m['rf_time_mae'], 1)} "
            f"to {_f(m['rf_mae'], 1)} trips/hour; that difference is what weather contributes."
        )
    return f"""Train: {m['train_period']} ({m['n_train']:,} hours) · test: {m['test_period']} ({m['n_test']:,} hours) · scikit-learn {m.get('sklearn_version', '?')}

| Model | MAE (trips/hour) | RMSE | R² |
|---|---|---|---|
{table}
{weather}

Top feature importances (time + weather forest): {", ".join(f"`{k}` {v:.2f}" for k, v in imp)}.

![Exploratory plots](docs/screenshots/03-eda.png)
![Predictions on the test period](docs/screenshots/04-model-predictions.png)

Raw numbers: [`docs/results/part1-metrics.json`](docs/results/part1-metrics.json)"""


# ---------------------------------------------------------------------------
def part2() -> str | None:
    reports = sorted((LAB / "artifacts" / "loop_reports").glob("report-*.json"))
    if not reports:
        return None
    r = json.loads(reports[-1].read_text())
    if not r.get("holdout_metrics"):
        return None
    shutil.copy(reports[-1], RESULTS / "part2-agent-report.json")
    contract_path = LAB / "datasets" / "citibike_hourly_latest.contract.json"
    contract = json.loads(contract_path.read_text()) if contract_path.exists() else {}
    exp_rows = "\n".join(
        f"| {e.get('cycle')} | {e.get('model_family')} `{json.dumps(e.get('hyperparameters'))}` | "
        f"{_f(e.get('val_f1'))} | {e.get('verdict') or '—'} |"
        for e in r.get("experiments") or []
    )
    h, b = r["holdout_metrics"], r.get("majority_baseline") or {}
    cm = (r.get("confusion_matrix") or {}).get("rows_true_cols_pred") or [[0, 0], [0, 0]]
    fam = (r.get("best_spec") or {}).get("model_family")
    feats = ", ".join(f"`{c}`" for c in r.get("features_used") or [])
    label = contract.get("target_definition", "")
    basis = contract.get("threshold_basis", "")
    return f"""Dataset: `{r.get('dataset')}`, label `high_demand` = {label}; threshold basis: {basis}. Split: {r.get('split')}; the holdout is the last {h.get('n_holdout', '?')} hours. scikit-learn {r.get('sklearn_version', '?')}. Agents ran their heuristic fallbacks (no LLM key).

| Cycle | Model (hyperparameters) | Validation F1 | Judge |
|---|---|---|---|
{exp_rows}

Holdout result for the selected model ({fam}):

| | F1 | Accuracy | Precision | Recall |
|---|---|---|---|---|
| {fam} | {_f(h.get('f1'))} | {_f(h.get('accuracy'))} | {_f(h.get('precision'))} | {_f(h.get('recall'))} |
| Always predict "not busy" | {_f(b.get('f1'))} | {_f(b.get('accuracy'))} | — | {_f(b.get('recall'))} |

Confusion matrix (rows = actual, columns = predicted): normal → {cm[0][0]} normal / {cm[0][1]} busy; busy → {cm[1][0]} normal / {cm[1][1]} busy.

Features used: {feats}. The four leakage columns from the contract and the time key were dropped.

Full report: [`docs/results/part2-agent-report.json`](docs/results/part2-agent-report.json)"""


# ---------------------------------------------------------------------------
def part3() -> tuple[str, str] | None:
    out = LAB / "adaptive" / "eval" / "out"
    if not (out / "summary.md").exists():
        return None
    summary = (out / "summary.md").read_text()
    payload = json.loads((out / "results.json").read_text())
    shutil.copy(out / "summary.md", RESULTS / "part3-summary.md")
    shutil.copy(out / "results.json", RESULTS / "part3-results.json")
    by = {(x["case"], x["system"]): x for x in payload["results"]}
    sys_ids = list(payload["systems"])
    passed = {s: sum(by[(c["id"], s)]["passed"] for c in payload["cases"]) for s in sys_ids}
    n = len(payload["cases"])
    c10 = ""
    if any(c["id"] == "C10" for c in payload["cases"]):
        f1 = {s: ((by[("C10", s)]["result"].get("holdout") or {}).get("metrics") or {}).get("f1") for s in sys_ids}
        used = {s: [c for c in ("trips", "member_share", "ebike_share", "avg_duration_min")
                    if c in (by[("C10", s)]["result"].get("features_used") or [])] for s in sys_ids}
        c10 = ("On the real Citi Bike data (C10): "
               + "; ".join(f"{payload['systems'][s]} F1 {_f(f1[s])}"
                           + (f" (trained on {', '.join(used[s])})" if used[s] else " (no leakage columns)")
                           for s in sys_ids) + ".")
    head = (f"Cases passed (of {n}): " + ", ".join(f"{payload['systems'][s]} **{passed[s]}**" for s in sys_ids) + ". " + c10
            + "\n\nFull tables: [`docs/results/part3-summary.md`](docs/results/part3-summary.md) · dashboard: "
              "[`agent-lab/adaptive/eval/out/dashboard.html`](agent-lab/adaptive/eval/out/dashboard.html)")
    detail = summary + ("\n" + c10 if c10 else "")
    return head, detail


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    SHOTS.mkdir(parents=True, exist_ok=True)
    readme, adaptive_readme = ROOT / "README.md", LAB / "adaptive" / "README.md"
    for name, fn in (("part1", part1), ("part2", part2)):
        body = fn()
        if body is None:
            print(f"- {name}: no output found yet, skipped")
        elif _replace_block(readme, name, body):
            print(f"- {name}: README updated")
    p3 = part3()
    if p3 is None:
        print("- part3: no adaptive/eval/out/summary.md yet, skipped")
    else:
        _replace_block(readme, "part3", p3[0])
        _replace_block(adaptive_readme, "part3", p3[1])
        print("- part3: README + adaptive/README updated")
    print(f"Evidence copied to {RESULTS.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
