"""Run the evaluation suite on all three systems and write the results + dashboard.

    python -m adaptive.eval.run                 # all cases, all systems
    python -m adaptive.eval.run --cases C2 C7   # a subset

Outputs (adaptive/eval/out/):
    results.json     every normalized result and check score
    summary.md       comparison tables (pasted into adaptive/README.md)
    dashboard.html   self-contained dashboard (open it in a browser)
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

from adaptive.eval.cases import build_cases, score_case
from adaptive.eval.dashboard import write_dashboard
from adaptive.eval.systems import SYSTEM_LABELS, SYSTEMS, run_system

OUT = Path(__file__).resolve().parent / "out"


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    out = Path(args.out)
    os.environ.setdefault("LOOP_REPORT_DIR", str(out / "reports"))
    for key in ("OPENROUTER_API_KEY", "CURSOR_API_KEY"):  # deterministic: heuristics only
        os.environ.pop(key, None)
    warnings.filterwarnings("ignore")

    cases = [c for c in build_cases() if not args.cases or c.id in args.cases]
    results = []
    for case in cases:
        path = case.dataset_builder(out / "data")
        params = case.run_params()
        print(f"\n{case.id} {case.title}")
        for system in SYSTEMS:
            r = run_system(system, dataset_path=str(path), **params)
            scored = score_case(case, r)
            results.append({"case": case.id, "system": system, "result": _slim(r), **scored})
            mark = "PASS" if scored["passed"] else "fail"
            print(f"  {SYSTEM_LABELS[system]:<30} {mark}  score={scored['score']:.2f}  "
                  f"status={r['status']}  f1={_fmt(((r.get('holdout') or {}).get('metrics') or {}).get('f1'))}  "
                  f"runs={r.get('attempts')}  {r.get('duration_s')}s")

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M"),
        "systems": SYSTEM_LABELS,
        "cases": [{"id": c.id, "title": c.title, "input": c.input, "expected": c.expected, "note": c.note,
                   "pass_threshold": c.pass_threshold,
                   "checks": [{"name": k.name, "level": k.level, "criterion": k.criterion, "required": k.required}
                              for k in c.checks]} for c in cases],
        "results": results,
    }
    from adaptive.graph import mermaid

    payload["adaptive_graph_mermaid"] = mermaid()
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(payload, indent=2, default=str))
    (out / "summary.md").write_text(summary_markdown(payload))
    write_dashboard(payload, out / "dashboard.html")
    print(f"\nwrote {out / 'results.json'}, {out / 'summary.md'}, {out / 'dashboard.html'}")
    print("\n" + summary_markdown(payload))
    return payload


def _fmt(x):
    return "—" if x is None else f"{x:.3f}"


def _slim(r: dict) -> dict:
    keep = ("system", "status", "error", "flagged", "features_used", "experiments", "attempts", "holdout",
            "decisions", "agent_path", "duration_s")
    out = {k: r.get(k) for k in keep}
    if out.get("holdout"):
        out["holdout"] = {k: v for k, v in out["holdout"].items() if k != "pipeline"}
    return out


def aggregate(payload: dict) -> dict:
    agg = {}
    for s in payload["systems"]:
        rows = [r for r in payload["results"] if r["system"] == s]
        by_level = {}
        for lvl in ("intermediate", "final", "coordination"):
            scores = [c["score"] for r in rows for c in r["checks"] if c["level"] == lvl]
            by_level[lvl] = round(sum(scores) / len(scores), 3) if scores else None
        f1s = [((r["result"].get("holdout") or {}).get("metrics") or {}).get("f1") for r in rows]
        f1s = [f for f in f1s if f is not None]
        agg[s] = {
            "passed": sum(r["passed"] for r in rows), "total": len(rows),
            "mean_score": round(sum(r["score"] for r in rows) / len(rows), 3) if rows else 0,
            "by_level": by_level,
            "crashes": sum(r["result"]["status"] == "crashed" for r in rows),
            "training_runs": sum(r["result"].get("attempts") or 0 for r in rows),
            "time_s": round(sum(r["result"].get("duration_s") or 0 for r in rows), 1),
        }
    return agg


def summary_markdown(payload: dict) -> str:
    sys_ids = list(payload["systems"])
    agg = aggregate(payload)
    lines = ["| | " + " | ".join(payload["systems"][s] for s in sys_ids) + " |",
             "|---|" + "---|" * len(sys_ids)]
    lines.append("| **Cases passed** | " + " | ".join(f"**{agg[s]['passed']}/{agg[s]['total']}**" for s in sys_ids) + " |")
    lines.append("| Mean score | " + " | ".join(f"{agg[s]['mean_score']:.2f}" for s in sys_ids) + " |")
    for lvl in ("intermediate", "final", "coordination"):
        lines.append(f"| {lvl.capitalize()} checks | " + " | ".join(
            "—" if agg[s]["by_level"][lvl] is None else f"{agg[s]['by_level'][lvl]:.2f}" for s in sys_ids) + " |")
    lines.append("| Crashes | " + " | ".join(str(agg[s]["crashes"]) for s in sys_ids) + " |")
    lines.append("| Training runs (all cases) | " + " | ".join(str(agg[s]["training_runs"]) for s in sys_ids) + " |")
    lines.append("| Wall time (s) | " + " | ".join(str(agg[s]["time_s"]) for s in sys_ids) + " |")
    lines += ["", "| Case | " + " | ".join(payload["systems"][s] for s in sys_ids) + " |", "|---|" + "---|" * len(sys_ids)]
    for c in payload["cases"]:
        cells = []
        for s in sys_ids:
            r = next(x for x in payload["results"] if x["case"] == c["id"] and x["system"] == s)
            f1 = ((r["result"].get("holdout") or {}).get("metrics") or {}).get("f1")
            cells.append(f"{'✅' if r['passed'] else '❌'} {r['score']:.2f}" + (f" · F1 {f1:.2f}" if f1 is not None else f" · {r['result']['status']}"))
        lines.append(f"| {c['id']} {c['title']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
