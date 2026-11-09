"""Run the adaptive multi-agent workflow on one dataset.

    python -m adaptive.run_cli --dataset datasets/citibike_hourly_latest.csv --target high_demand
    ADAPTIVE_USE_LLM=1 OPENROUTER_API_KEY=... python -m adaptive.run_cli ...   # LLM supervisor tie-breaks
"""
from __future__ import annotations

import argparse
import json
import warnings

from adaptive.graph import run_adaptive


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--target", default="high_demand")
    p.add_argument("--target-score", type=float, default=0.8)
    p.add_argument("--budget", type=int, default=8, help="max training runs, retries included")
    p.add_argument("--fault", choices=["none", "transient", "broken_rf"], default="none",
                   help="inject a training failure to watch the recovery path")
    a = p.parse_args()
    warnings.filterwarnings("ignore")
    fault = {"none": None, "transient": {"transient_failures": 1},
             "broken_rf": {"broken_families": ["random_forest"]}}[a.fault]
    r = run_adaptive(dataset_path=a.dataset, target_column=a.target, target_score=a.target_score,
                     budget=a.budget, fault_plan=fault)
    print("\nSupervisor decisions:")
    for d in r["decisions"]:
        print(f"  {d['step']:>2}. → {d['next_agent']:<17} {d['reason']}  [{d['source']}]")
    print("\nTraining runs:")
    for e in r["experiments"]:
        f1 = "FAILED" if e["status"] == "failed" else f"val F1 {e['val_f1']:.3f}"
        print(f"  {e['attempt']}. {e['agent']:<17} {e['model_family']:<23} {f1}  features+{e['feature_ops']}")
    h = r.get("holdout")
    if h:
        m, b = h["metrics"], h["majority_baseline"]
        print(f"\nHoldout ({h['split']}, n={h['n_holdout']}): F1 {m['f1']:.3f} · accuracy {m['accuracy']:.3f} "
              f"· precision {m['precision']:.3f} · recall {m['recall']:.3f} (majority acc {b['accuracy']:.3f})")
        print("Confusion matrix (rows true, cols pred):", h["confusion_matrix"]["rows_true_cols_pred"])
    else:
        print("\nNo model:", r.get("stop_reason"))
    print("\nFull report:", r.get("saved_to"))


if __name__ == "__main__":
    main()
