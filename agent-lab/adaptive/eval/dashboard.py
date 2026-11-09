"""Write a self-contained HTML dashboard for an evaluation run (graphs via Mermaid CDN)."""
from __future__ import annotations

import html
import json
from collections import Counter
from pathlib import Path

CSS = """
:root{--ink:#1d2a2a;--mist:#5d6b6b;--line:#d9e1df;--bg:#f6f8f7;--ok:#1f7a4d;--bad:#b3261e;--accent:#2f6f62}
*{box-sizing:border-box}body{margin:0;font:14px/1.5 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;color:var(--ink);background:var(--bg)}
main{max-width:1180px;margin:0 auto;padding:32px 24px 64px}h1{font-size:26px;margin:0 0 4px}h2{font-size:18px;margin:36px 0 12px}
.sub{color:var(--mist);margin:0 0 24px}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}
.card{background:#fff;border:1px solid var(--line);border-radius:10px;padding:16px}.card h3{margin:0 0 6px;font-size:15px}
.big{font-size:30px;font-weight:650}.muted{color:var(--mist);font-size:12px}.ours{border-color:var(--accent);box-shadow:0 0 0 2px #2f6f6222}
.bar{height:8px;background:#e9eeec;border-radius:4px;overflow:hidden;margin:3px 0 8px}.bar>span{display:block;height:100%;background:var(--accent)}
table{width:100%;border-collapse:collapse;background:#fff;border:1px solid var(--line);border-radius:10px;overflow:hidden}
th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{background:#eef3f1;font-weight:600;font-size:13px}
.pass{color:var(--ok);font-weight:600}.fail{color:var(--bad);font-weight:600}details{background:#fff;border:1px solid var(--line);border-radius:10px;margin:10px 0;padding:0 14px}
summary{cursor:pointer;padding:12px 0;font-weight:600}.kv{display:grid;grid-template-columns:140px 1fr;gap:4px 12px;margin:6px 0 12px}
.kv div:nth-child(odd){color:var(--mist)}code{background:#eef3f1;padding:1px 4px;border-radius:4px;font-size:12px}
.pill{display:inline-block;padding:0 6px;border-radius:9px;font-size:11px;background:#eef3f1;color:var(--mist);margin-left:4px}
.mermaid{background:#fff;border:1px solid var(--line);border-radius:10px;padding:12px;overflow:auto}
ol.decisions{margin:6px 0 12px;padding-left:22px}ol.decisions li{margin:2px 0}
"""


def _e(x) -> str:
    return html.escape(str(x))


def _path_mermaid(path: list[str]) -> str:
    """Executed agent path as a graph; edge labels = how many times that hop happened."""
    path = [p for p in path if p]
    edges = Counter(zip(["START", *path], [*path, "END"]))
    lines = ["flowchart LR", "  START((START))", "  END((END))"]
    lines += [f"  {a} -->|{n}x| {b}" for (a, b), n in edges.items()]
    return "\n".join(lines)


def write_dashboard(payload: dict, path: Path) -> None:
    from adaptive.eval.run import aggregate

    systems = payload["systems"]
    agg = aggregate(payload)
    res = {(r["case"], r["system"]): r for r in payload["results"]}

    cards = []
    for s, label in systems.items():
        a = agg[s]
        bars = "".join(
            f"<div class='muted'>{lvl} {'—' if v is None else f'{v:.2f}'}</div><div class='bar'><span style='width:{(v or 0) * 100:.0f}%'></span></div>"
            for lvl, v in a["by_level"].items())
        cards.append(f"<div class='card{' ours' if s == 'adaptive' else ''}'><h3>{_e(label)}</h3>"
                     f"<div class='big'>{a['passed']}/{a['total']}</div><div class='muted'>cases passed · mean score {a['mean_score']:.2f}</div>"
                     f"<div style='margin-top:10px'>{bars}</div><div class='muted'>{a['crashes']} crashes · {a['training_runs']} training runs · {a['time_s']} s</div></div>")

    head = "".join(f"<th>{_e(l)}</th>" for l in systems.values())
    rows = []
    for c in payload["cases"]:
        cells = []
        for s in systems:
            r = res[(c["id"], s)]
            f1 = ((r["result"].get("holdout") or {}).get("metrics") or {}).get("f1")
            extra = f"F1 {f1:.3f}" if f1 is not None else r["result"]["status"]
            cells.append(f"<td><span class='{'pass' if r['passed'] else 'fail'}'>{'PASS' if r['passed'] else 'FAIL'}</span> "
                         f"{r['score']:.2f}<br><span class='muted'>{_e(extra)}</span></td>")
        rows.append(f"<tr><td><b>{_e(c['id'])}</b> {_e(c['title'])}</td>{''.join(cells)}</tr>")

    details = []
    for c in payload["cases"]:
        check_rows = []
        for i, chk in enumerate(c["checks"]):
            cells = []
            for s in systems:
                row = res[(c["id"], s)]["checks"][i]
                cls = "pass" if row["score"] == 1 else "fail" if row["score"] == 0 else ""
                cells.append(f"<td><span class='{cls}'>{row['score']:.2f}</span><br><span class='muted'>{_e(row['detail'])}</span></td>")
            req = "<span class='pill'>required</span>" if chk["required"] else ""
            check_rows.append(f"<tr><td>{_e(chk['name'])}{req}<br><span class='muted'>{_e(chk['level'])} · {_e(chk['criterion'])}</span></td>{''.join(cells)}</tr>")
        ad = res[(c["id"], "adaptive")]["result"]
        decisions = "".join(f"<li><code>{_e(d['next_agent'])}</code> — {_e(d['reason'])}</li>" for d in ad.get("decisions") or [])
        mer = _path_mermaid(ad.get("agent_path") or [])
        details.append(
            f"<details><summary>{_e(c['id'])} · {_e(c['title'])}</summary>"
            f"<div class='kv'><div>Input</div><div><code>{_e(json.dumps(c['input']))}</code></div>"
            f"<div>Expected</div><div>{_e(c['expected'])}</div><div>Pass rule</div><div>all required checks = 1 and mean ≥ {c['pass_threshold']}</div>"
            + (f"<div>Note</div><div>{_e(c['note'])}</div>" if c.get("note") else "") +
            f"</div><table><tr><th>Check</th>{head}</tr>{''.join(check_rows)}</table>"
            f"<p><b>Adaptive supervisor decisions</b></p><ol class='decisions'>{decisions or '<li>—</li>'}</ol>"
            f"<p><b>Adaptive executed path</b> <span class='muted'>(edge label = times taken)</span></p><pre class='mermaid'>{_e(mer)}</pre></details>")

    doc = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Part 3 · agent workflow evaluation</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>{CSS}</style></head><body><main>
<h1>Adaptive multi-agent workflow: evaluation</h1>
<p class="sub">Generated {_e(payload['generated_at'])} · {len(payload['cases'])} cases × {len(systems)} systems · deterministic (heuristic agents, no LLM)</p>
<div class="cards">{''.join(cards)}</div>
<h2>Results by case</h2><table><tr><th>Case</th>{head}</tr>{''.join(rows)}</table>
<h2>Adaptive workflow graph</h2><p class="muted">Static graph compiled by LangGraph. The Supervisor chooses the outgoing edge at run time.</p>
<pre class="mermaid">{_e(payload.get('adaptive_graph_mermaid', ''))}</pre>
<h2>Case details</h2>{''.join(details)}
</main><script type="module">import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
mermaid.initialize({{startOnLoad:false,theme:"neutral"}});
document.querySelectorAll("details").forEach(d=>d.addEventListener("toggle",()=>{{if(d.open)mermaid.run({{nodes:d.querySelectorAll(".mermaid:not([data-processed])")}});}}));
mermaid.run({{nodes:document.querySelectorAll("main > .mermaid")}});</script></body></html>"""
    path.write_text(doc)
