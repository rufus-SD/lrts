"""Self-contained HTML report generator."""

from __future__ import annotations

from datetime import datetime, timezone
from html import escape

from lrts.engines.report import ReportSummary

_CSS = """\
:root{--bg:#0f172a;--card:#1e293b;--border:#334155;--text:#e2e8f0;--dim:#94a3b8;
--green:#22c55e;--yellow:#eab308;--red:#ef4444;--blue:#3b82f6}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
background:var(--bg);color:var(--text);padding:2rem;max-width:960px;margin:0 auto;line-height:1.6}
header{text-align:center;margin-bottom:2rem;padding:1.5rem;border-bottom:1px solid var(--border)}
header h1{font-size:2rem;letter-spacing:.5em;font-weight:300}
header p{color:var(--dim);font-size:.9rem}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:1.5rem;margin-bottom:1.5rem}
.card h2{font-size:1rem;text-transform:uppercase;letter-spacing:.05em;color:var(--dim);margin-bottom:1rem}
.drift-bar{height:8px;background:var(--border);border-radius:4px;overflow:hidden;margin:.5rem 0}
.drift-fill{height:100%;border-radius:4px}
.stats{display:flex;gap:2rem;flex-wrap:wrap;margin-top:1rem}
.stat-value{font-size:1.5rem;font-weight:700}
.stat-label{color:var(--dim);font-size:.75rem;text-transform:uppercase;letter-spacing:.05em}
.verdict-text{font-weight:700;margin-top:.25rem}
table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--border);
border-radius:12px;overflow:hidden;margin-bottom:1.5rem}
th{background:var(--bg);padding:.75rem 1rem;text-align:left;font-weight:600;font-size:.75rem;
text-transform:uppercase;letter-spacing:.05em;color:var(--dim)}
td{padding:.75rem 1rem;border-top:1px solid var(--border);font-size:.875rem;vertical-align:top}
tr:hover{background:rgba(255,255,255,.02)}
.badge{display:inline-block;padding:.15rem .5rem;border-radius:4px;font-size:.7rem;
font-weight:700;text-transform:uppercase}
.badge-pass{background:rgba(34,197,94,.15);color:var(--green)}
.badge-drift{background:rgba(239,68,68,.15);color:var(--red)}
.badge-error{background:rgba(234,179,8,.15);color:var(--yellow)}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-left:.3rem;vertical-align:middle}
.sep{border:0;border-top:1px solid var(--border);margin:2rem 0}
.reason{color:var(--dim);font-size:.8rem}
details{margin-top:.4rem}
details summary{cursor:pointer;color:var(--blue);font-size:.75rem}
details pre{white-space:pre-wrap;word-break:break-word;font-size:.8rem;
background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:.5rem;margin-top:.3rem;max-height:200px;overflow:auto}
footer{text-align:center;margin-top:2rem;padding:1rem;color:var(--dim);font-size:.8rem;
border-top:1px solid var(--border)}
@media(max-width:600px){.stats{flex-direction:column;gap:.75rem}td,th{padding:.5rem}}
"""


def _drift_color(score: float) -> str:
    if score < 0.15:
        return "var(--green)"
    if score < 0.3:
        return "var(--yellow)"
    return "var(--red)"


def _score_html(score: float | None) -> str:
    if score is None:
        return '<span style="color:var(--dim)">—</span>'
    pct = int(score * 100)
    if score >= 0.85:
        color = "var(--green)"
    elif score >= 0.6:
        color = "var(--yellow)"
    else:
        color = "var(--red)"
    return (
        f'<span style="color:{color}">{pct}%</span>'
        f'<span class="dot" style="background:{color}"></span>'
    )


def _verdict_badge(verdict: str) -> str:
    if verdict == "pass":
        return '<span class="badge badge-pass">PASS</span>'
    if verdict == "fail":
        return '<span class="badge badge-drift">DRIFT</span>'
    if verdict == "error":
        return '<span class="badge badge-error">ERROR</span>'
    return f'<span class="badge">{escape(verdict)}</span>'


def _verdict_text(score: float) -> str:
    if score < 0.15:
        return '<span class="verdict-text" style="color:var(--green)">LOW — behavior is stable</span>'
    if score < 0.3:
        return '<span class="verdict-text" style="color:var(--yellow)">MODERATE — some changes detected</span>'
    return '<span class="verdict-text" style="color:var(--red)">HIGH — significant behavioral change</span>'


def _extract_judge_reason(detail: dict) -> str:
    for ev in detail.get("evaluations", []):
        if ev.get("evaluator") == "judge":
            return ev.get("detail", {}).get("reason", "")
    return ""


def _report_section(report: ReportSummary) -> str:
    drift_pct = int(report.drift_score * 100)
    color = _drift_color(report.drift_score)

    rows = []
    for i, item in enumerate(report.items, 1):
        reason_text = escape(_extract_judge_reason(item.get("detail", {})))

        v2_full = escape(item.get("output") or item.get("output_preview") or "")
        v1_full = escape(item.get("baseline_output") or item.get("baseline_preview") or "")

        parts = []
        if reason_text:
            parts.append(f"<div>{reason_text}</div>")
        if v2_full or v1_full:
            parts.append(
                f"<details><summary>Show full outputs</summary>"
                f"<div style='margin-top:.3rem'><strong style='color:var(--blue)'>v2:</strong>"
                f"<pre>{v2_full}</pre></div>"
                f"<div style='margin-top:.3rem'><strong style='color:var(--dim)'>v1:</strong>"
                f"<pre>{v1_full}</pre></div>"
                f"</details>"
            )

        reason_html = "\n".join(parts) if parts else '<span style="color:var(--dim)">—</span>'

        rows.append(
            f"<tr>"
            f"<td style='text-align:right'>{i}</td>"
            f"<td>{escape(item['input'])}</td>"
            f"<td style='text-align:center'>{_score_html(item.get('similarity'))}</td>"
            f"<td style='text-align:center'>{_verdict_badge(item['verdict'])}</td>"
            f"<td class='reason'>{reason_html}</td>"
            f"</tr>"
        )

    return f"""
<div class="card">
  <h2>Regression Summary</h2>
  <div>
    <strong>Drift</strong>
    <div class="drift-bar"><div class="drift-fill" style="width:{drift_pct}%;background:{color}"></div></div>
    <span style="color:{color};font-weight:700">{drift_pct}%</span>
    &nbsp;&nbsp;{_verdict_text(report.drift_score)}
  </div>
  <div class="stats">
    <div><div class="stat-value">{report.total}</div><div class="stat-label">Total</div></div>
    <div><div class="stat-value" style="color:var(--green)">{report.passed}</div><div class="stat-label">Identical</div></div>
    <div><div class="stat-value" style="color:var(--red)">{report.failed}</div><div class="stat-label">Drifted</div></div>
    <div><div class="stat-value" style="color:var(--yellow)">{report.errors}</div><div class="stat-label">Errors</div></div>
  </div>
  <div style="margin-top:1rem;color:var(--dim);font-size:.8rem">Run: {escape(report.run_id)}</div>
</div>
<table>
<thead><tr><th>#</th><th>Question</th><th>Similarity</th><th>Status</th><th>What changed</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
"""


def render_html(reports: list[ReportSummary]) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    sections = []
    for idx, report in enumerate(reports):
        if len(reports) > 1:
            sections.append(f'<h3 style="color:var(--dim);margin-bottom:.5rem">Test {idx + 1} of {len(reports)}</h3>')
        sections.append(_report_section(report))
        if idx < len(reports) - 1:
            sections.append("<hr class='sep'>")

    body = "\n".join(sections)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LRTS Report</title>
<style>{_CSS}</style>
</head>
<body>
<header>
  <h1>L R T S</h1>
  <p>LLM Regression Testing System</p>
</header>
{body}
<footer>Generated by LRTS v0.1.0 &middot; {now}</footer>
</body>
</html>
"""
