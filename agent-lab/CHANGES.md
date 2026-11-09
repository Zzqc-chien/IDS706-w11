# Changes to the in-class agent workflow (Part 2)

This folder is the in-class template ([Kedar-V/automl-research-lab](https://github.com/Kedar-V/automl-research-lab), branch `refactor/lab-solution-layout`), adapted to train and evaluate a model on the cleaned Citi Bike data from Part 1. Only the **solution loop** (`/solution-loop`) was changed; the student lab and the advanced stack are untouched.

## The task

Part 1's Airflow task `analysis.publish_for_agents` reads the cleaned `hourly_demand` table from Postgres and writes two files into `datasets/`:

- `citibike_hourly_latest.csv`: one row per hour.
  - Target: `high_demand`, which is 1 when that hour's trips are at or above the 75th percentile (about 25% of hours).
  - Features: weather (`temp_c`, `precip_mm`, `wind_kmh`, `humidity_pct`, `is_raining`) and calendar (`hour_of_day`, `day_of_week`, `month`, `is_weekend`, `is_holiday`).
- `citibike_hourly_latest.contract.json`: a contract naming the time column (`hour`) and the **leakage columns** (`trips`, `member_share`, `ebike_share`, `avg_duration_min`). These are computed from the same rides the label counts, so a model that used them would score perfectly but be useless for forecasting. They stay in the CSV on purpose, so the guardrail below has something to catch.

The loop's job: predict whether an hour will be busy from weather and calendar information alone. Metric: F1 on the positive class (the template's metric; it suits a 25%-positive target). Accuracy, precision and recall are reported too.

## Agent changes

```
START → Data Validator ──invalid──────────────────────────────────────┐
            │ ok                                                         ▼
        Data Scientist → Hypothesis → ML Engineer → Judge → Router   Report Writer → END
                            ▲                                 │  │        ▲
                            └── Context ← Memory ← Writer ←───┘  └→ Holdout ┘
```

| Agent | Change | Why |
|---|---|---|
| **Data Validator** (new, code) | Runs first. Reads the contract, detects the time column, and flags leakage: contract columns, plus any numeric column that **on its own** separates the classes (AUC ≥ 0.98). It also checks for a binary target, enough rows, missing values and duplicate time keys. Blocking problems route straight to the Report Writer, so no model is trained. | Catching leakage needs no LLM. Code is deterministic and can't be talked out of it. The AUC rule also catches leaks the contract misses: on the in-class fraud dataset it flags `fraud_probability` (AUC = 1.0). |
| **Data Scientist** (modified) | The validator's must-drop columns are merged into its profile, so the Hypothesis agent sees them. | Keeps the LLM agents informed. |
| **ML Engineer** (modified) | (1) **Guardrail:** after the LLM or heuristic writes the spec, the validator's must-drop columns are always added to `drop_columns`. (2) **Chronological split** whenever there is a time column: the last 20% of hours is the sealed holdout, and the last 25% of the rest is validation. | (1) An LLM can forget to drop a column; the guard makes that impossible. (2) Neighbouring hours are highly correlated, so the template's random split would let the model train on hours right next to the ones it is tested on. |
| **Hypothesis / ML Engineer heuristics** (modified, `loop_runtime.py`) | New fallback strategy: a linear baseline, then each untried model family once, then more capacity for the best family so far. | The original heuristic kept re-tuning logistic regression after every "below target" verdict and never reached a tree model. |
| **Report Writer** (new, code) | Runs last. Re-fits the best spec on the holdout split and writes `artifacts/loop_reports/report-<time>.json` with: the holdout metrics, a **majority-class baseline**, the **confusion matrix**, feature importances, the features actually used, and every experiment's family, hyperparameters, validation F1 and verdict. | One F1 number hides a lot: a model that always predicts "not busy" already gets around 75% accuracy. The baseline and confusion matrix make that visible. |

The UI's live agent graph (`frontend/src/graph/simpleLoopConfig.ts`) shows the two new nodes and the `ok` / `invalid` routes. The gateway's `/solution-loop/info` lists them. `infra/docker-compose.yml` mounts `artifacts/loop_reports/` so reports are saved on your machine.

## Files changed

- `solution/langgraph/graph.py`: new nodes `node_data_validator` and `node_report_writer`, the validator router, the guardrail, and the time-split wiring
- `loop_runtime.py`: `_split` (chronological or random), `_build_pipeline`, precision/recall metrics, new heuristic strategy
- `frontend/src/graph/simpleLoopConfig.ts`, `frontend/src/components/LiveSimpleLoopGraph.tsx`: the graph
- `advanced/services/gateway/app/main.py`: node list in `/solution-loop/info`
- `infra/docker-compose.yml`: `LOOP_REPORT_DIR` and its volume
- `tests/unit/test_citibike_agents.py`: new tests (validator, guardrail, full loop, invalid-dataset route)

The test suite gives the same result as the unmodified template (the advanced-stack contract tests need `mlflow`/`pyarrow`, so they fail the same way in both), plus the 5 new tests passing. The Titanic dataset still runs: no time column, so it uses the original random split.
