# Changes to the in-class agent workflow (Part 2)

This folder is the in-class template ([Kedar-V/automl-research-lab](https://github.com/Kedar-V/automl-research-lab), branch `refactor/lab-solution-layout`). Only the **solution loop** (`/solution-loop`) was changed; the student lab and the advanced stack are untouched. Part 3 lives in `adaptive/`.

## Input from Part 1

`datasets/citibike_hourly_latest.csv`: one row per hour, target `high_demand`, weather and calendar features.

`citibike_hourly_latest.contract.json` gives:
- **the time column:** `hour`
- **the label threshold and how it was set:** 75th percentile of trips in the first 60% of hours, the training segment
- **the leakage columns:** `trips`, `member_share`, `ebike_share`, `avg_duration_min`

## Agent changes

```
START → Data Validator ──invalid──────────────────────────────────────┐
            │ ok                                                         ▼
        Data Scientist → Hypothesis → ML Engineer → Judge → Router   Report Writer → END
                            ▲                                 │  │        ▲
                            └── Context ← Memory ← Writer ←───┘  └→ Holdout ┘
```

| Agent | Change | Reason |
|---|---|---|
| **Data Validator** (new, code) | Runs first. Reads the contract and detects the time column. Checks for a binary target, ≥ 200 rows, missing values and duplicate time keys. Flags as leakage the contract's columns plus any numeric column that alone separates the classes (AUC ≥ 0.98), **computed on the training segment only**. Blocking problems route straight to the Report Writer. | Deterministic checks for problems an LLM profile can miss. On the in-class fraud dataset the AUC rule also flags `fraud_probability`. |
| **Data Scientist** (modified) | The validator's must-drop columns are merged into its profile. | The Hypothesis agent sees them. |
| **ML Engineer** (modified) | (1) Always adds the validator's must-drop columns to `drop_columns`, whatever the LLM or heuristic proposed. (2) Chronological split when there is a time column: last 20% holdout, the 20% before it validation. | (1) Leakage removal doesn't depend on the LLM. (2) A random split puts neighbouring, highly correlated hours on both sides. |
| **Fallback heuristics** (modified, `loop_runtime.py`) | Linear baseline → each untried family once → more capacity for the best family. | The original kept re-tuning logistic regression after every "below target". |
| **Holdout eval** (modified) | The only holdout fit. It produces the metrics, majority-class baseline, confusion matrix and feature importances together. | The UI and the report describe the same model. |
| **Report Writer** (new, code) | Formats the holdout results and the experiment history into `artifacts/loop_reports/report-<time>.json`, with the scikit-learn version. | A record of the run that can be checked beyond the screenshots. |

Other runtime changes in `loop_runtime.py`:
- numeric features are standardized, so logistic regression converges
- gradient boosting has a fixed `random_state`
- precision and recall are reported

## Files changed

- `solution/langgraph/graph.py`: validator, report writer, routing, guardrail, time split, single holdout fit
- `loop_runtime.py`: `_split`, `_build_pipeline` (scaling), seeds, metrics, heuristics
- `frontend/src/graph/simpleLoopConfig.ts`, `frontend/src/components/LiveSimpleLoopGraph.tsx`: the live graph
- `advanced/services/gateway/app/main.py`: node list in `/solution-loop/info`
- `infra/docker-compose.yml`: `LOOP_REPORT_DIR` and its volume
- `tests/unit/test_citibike_agents.py`: validator, guardrail, full loop, invalid-dataset route

Run the tests with `make test-agents` from the repository root. The template's own suite gives the same result as before these changes. Its advanced-stack contract tests need `mlflow`/`pyarrow` and fail without them, both before and after the changes.
