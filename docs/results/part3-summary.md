| | Fixed Airflow workflow | In-class agent loop | Adaptive multi-agent (ours) |
|---|---|---|---|
| **Cases passed** | **5/11** | **1/11** | **10/11** |
| Mean score | 0.60 | 0.55 | 0.91 |
| Intermediate checks | 0.30 | 0.20 | 0.90 |
| Final checks | 0.78 | 0.70 | 1.00 |
| Coordination checks | 0.78 | 0.77 | 1.00 |
| Crashes | 2 | 1 | 0 |
| Training runs (all cases) | 16 | 51 | 40 |
| Wall time (s) | 9.9 | 14.4 | 13.2 |

| Case | Fixed Airflow workflow | In-class agent loop | Adaptive multi-agent (ours) |
|---|---|---|---|
| C1 Clean hourly data | ✅ 1.00 · F1 0.89 | ❌ 0.40 · F1 0.55 | ✅ 1.00 · F1 0.89 |
| C2 Strong leakage, no contract | ❌ 0.40 · F1 1.00 | ❌ 0.60 · F1 0.99 | ✅ 1.00 · F1 0.89 |
| C3 Leakage declared in the data contract | ❌ 0.40 · F1 1.00 | ❌ 0.60 · F1 0.99 | ✅ 1.00 · F1 0.89 |
| C4 Weak leakage, no contract (limitation probe) | ❌ 0.00 · F1 1.00 | ❌ 0.00 · F1 0.99 | ❌ 0.00 · F1 0.89 |
| C5 Invalid target (3 classes) | ❌ 0.00 · crashed | ❌ 0.50 · crashed | ✅ 1.00 · rejected |
| C6 Transient training failure | ✅ 1.00 · F1 0.89 | ❌ 0.67 · F1 0.55 | ✅ 1.00 · F1 0.89 |
| C7 Persistent failure of one model family | ❌ 0.33 · crashed | ❌ 0.67 · F1 0.55 | ✅ 1.00 · F1 0.89 |
| C8 Easy target: stop early | ✅ 1.00 · F1 0.89 | ❌ 0.33 · F1 0.55 | ✅ 1.00 · F1 0.72 |
| C9 Unreachable target: search well | ✅ 0.80 · F1 0.79 | ❌ 0.59 · F1 0.51 | ✅ 1.00 · F1 0.79 |
| C10 Real Citi Bike data (from Part 1) | ❌ 0.67 · F1 1.00 | ❌ 0.67 · F1 0.97 | ✅ 1.00 · F1 0.86 |
| C11 Generality: Titanic (no time column) | ✅ 1.00 · F1 0.73 | ✅ 1.00 · F1 0.76 | ✅ 1.00 · F1 0.73 |
