"""HTML report generator — renders structured experiment data into a self-contained HTML report."""
from __future__ import annotations

import html
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


def _esc(text) -> str:
    """HTML-escape text safely."""
    return html.escape(str(text)) if text else ""


def _build_kpi_cards(data: dict) -> str:
    best_ll = data.get("best_log_loss", 0)
    best_acc = data.get("best_accuracy", 0)
    total_exp = data.get("total_experiments", 0)
    cal_err = data.get("calibration_error", "N/A")
    edges = data.get("edges_found", 0)
    iteration = data.get("iteration", "?")

    return f"""
    <div class="kpis">
      <div class="kpi kpi-accent"><div class="value">#{_esc(iteration)}</div><div class="label">Iteration</div></div>
      <div class="kpi kpi-green"><div class="value">{best_ll:.4f}</div><div class="label">Best Log Loss</div>
        <div class="delta up">{((0.693 - best_ll) / 0.693 * 100):.1f}% better than random</div></div>
      <div class="kpi kpi-blue"><div class="value">{best_acc:.1f}%</div><div class="label">Best Accuracy</div>
        <div class="delta up">+{best_acc - 55.3:.1f}pp vs baseline</div></div>
      <div class="kpi kpi-purple"><div class="value">{total_exp}</div><div class="label">Experiments</div></div>
      <div class="kpi kpi-pink"><div class="value">{edges}</div><div class="label">Market Edges</div></div>
    </div>"""


def _build_convergence_chart(runs: list[dict]) -> str:
    """Build SVG convergence chart from run history.

    Each run dict: {"iteration": int, "best_log_loss": float, "experiments": int}
    """
    if not runs:
        return '<div style="color:#666;text-align:center;padding:40px;">No convergence data available yet.</div>'

    n = len(runs)
    # Chart dimensions
    left, right = 80, 860
    top, bottom = 30, 270
    chart_w = right - left
    chart_h = bottom - top

    # Y range
    y_min_val = 0.605
    y_max_val = 0.700

    def val_to_y(v):
        return top + (y_max_val - v) / (y_max_val - y_min_val) * chart_h

    def iter_to_x(i):
        if n == 1:
            return (left + right) / 2
        return left + i * chart_w / (n - 1)

    # Grid lines
    grid_vals = [0.700, 0.680, 0.660, 0.640, 0.620, 0.605]
    grid_lines = ""
    for gv in grid_vals:
        gy = val_to_y(gv)
        grid_lines += f'<line x1="{left}" y1="{gy:.0f}" x2="{right}" y2="{gy:.0f}" stroke="#1e2235" stroke-width="0.5"/>\n'
        grid_lines += f'<text x="{left - 8}" y="{gy + 4:.0f}" fill="#6b7194" font-size="10" text-anchor="end" font-family="Inter,sans-serif">{gv:.3f}</text>\n'

    # Random baseline
    rand_y = val_to_y(0.693)
    grid_lines += f'<line x1="{left}" y1="{rand_y:.0f}" x2="{right}" y2="{rand_y:.0f}" stroke="#ef444440" stroke-width="1" stroke-dasharray="6,4"/>\n'
    grid_lines += f'<text x="{right + 4}" y="{rand_y + 4:.0f}" fill="#ef4444" font-size="9" font-family="Inter,sans-serif" opacity="0.6">Random</text>\n'

    # Points and line
    points = []
    for i, run in enumerate(runs):
        x = iter_to_x(i)
        y = val_to_y(run["best_log_loss"])
        points.append((x, y, run))

    # Path
    path_d = " ".join(f"{'M' if i == 0 else 'L'}{x:.0f},{y:.0f}" for i, (x, y, _) in enumerate(points))

    # Area fill
    area_d = path_d + f" L{points[-1][0]:.0f},{bottom} L{points[0][0]:.0f},{bottom} Z"

    # Gradient colors
    colors = ["#6366f1", "#7c3aed", "#8b5cf6", "#a855f7", "#c084fc", "#d946ef", "#ec4899", "#f43f5e", "#f97316", "#22c55e"]

    circles = ""
    labels = ""
    for i, (x, y, run) in enumerate(points):
        c = colors[i % len(colors)]
        is_best = (run["best_log_loss"] == min(r["best_log_loss"] for r in runs))
        fill_col = "#22c55e" if is_best else c
        label_col = "#22c55e" if is_best else "#a5b4fc"
        fw = "800" if is_best else "700"

        circles += f'<circle cx="{x:.0f}" cy="{y:.0f}" r="6" fill="{fill_col}" stroke="#0a0b10" stroke-width="3"/>\n'
        labels += f'<text x="{x:.0f}" y="{y - 12:.0f}" fill="{label_col}" font-size="10" text-anchor="middle" font-family="Inter,sans-serif" font-weight="{fw}">{run["best_log_loss"]:.4f}</text>\n'
        labels += f'<text x="{x:.0f}" y="{bottom + 18:.0f}" fill="#6b7194" font-size="11" text-anchor="middle" font-family="Inter,sans-serif" font-weight="600">Iter {run["iteration"]}</text>\n'
        labels += f'<text x="{x:.0f}" y="{bottom + 32:.0f}" fill="#4a5068" font-size="9" text-anchor="middle" font-family="Inter,sans-serif">{run["experiments"]} exps</text>\n'

    return f"""
    <svg viewBox="0 0 900 320" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;">
      <defs>
        <linearGradient id="lg" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="#6366f1"/><stop offset="100%" stop-color="#ec4899"/></linearGradient>
        <linearGradient id="ag" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#6366f1" stop-opacity="0.15"/><stop offset="100%" stop-color="#6366f1" stop-opacity="0"/></linearGradient>
      </defs>
      {grid_lines}
      <path d="{area_d}" fill="url(#ag)"/>
      <path d="{path_d}" fill="none" stroke="url(#lg)" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
      {circles}
      {labels}
    </svg>"""


def _build_experiment_table(experiments: list[dict]) -> str:
    if not experiments:
        return '<div style="color:#666;text-align:center;padding:20px;">No experiments found.</div>'

    rows = ""
    for i, exp in enumerate(experiments, 1):
        ll = exp.get("log_loss", 0)
        acc = exp.get("accuracy", 0)
        brier = exp.get("brier_score", 0)
        name = _esc(exp.get("name", ""))
        method = _esc(exp.get("method", ""))

        # Rank badge
        if i == 1:
            rank = f'<span class="rank-badge rank-1">{i}</span>'
        elif i <= 3:
            rank = f'<span class="rank-badge rank-2">{i}</span>'
        else:
            rank = f'<span class="rank-badge rank-n">{i}</span>'

        # Performance bar (relative to range 0.62-0.68)
        bar_pct = max(0, min(100, (0.68 - ll) / (0.68 - 0.62) * 100))
        bar_cls = "bar" if bar_pct > 40 else ("bar bar-ok" if bar_pct > 15 else "bar bar-bad")

        # Color for log loss
        if ll <= 0.622:
            ll_style = 'color:#22c55e;font-weight:700'
        elif ll <= 0.630:
            ll_style = 'color:#4ade80;font-weight:600'
        elif ll > 0.650:
            ll_style = 'color:#ef4444'
        else:
            ll_style = ''

        top_cls = ' class="top-model"' if i <= 3 else ''
        rows += f"""<tr{top_cls}><td>{rank}</td><td>{name}</td><td>{method}</td>
          <td style="{ll_style}">{ll:.4f}</td><td>{acc:.1f}%</td><td>{brier:.4f}</td>
          <td><span class="{bar_cls}" style="width:{bar_pct:.0f}%"></span></td></tr>\n"""

    return f"""
    <table class="mini-table">
      <thead><tr><th>#</th><th>Experiment</th><th>Method</th><th>Log Loss</th><th>Accuracy</th><th>Brier</th><th>Performance</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>"""


def _build_edges_table(edges: list[dict]) -> str:
    if not edges:
        return '<div style="color:#666;text-align:center;padding:20px;">No betting edges computed. Run with Polymarket data to see edges.</div>'

    rows = ""
    for e in edges:
        conf_cls = "tag-discovery" if e.get("confidence") == "HIGH" else "tag-insight"
        edge_val = e.get("edge", 0)
        edge_color = "#22c55e" if edge_val > 0 else "#ef4444"
        rows += f"""<tr>
          <td>{_esc(e.get('home_team',''))} vs {_esc(e.get('away_team',''))}</td>
          <td>{e.get('model_prob', 0):.1%}</td>
          <td>{e.get('market_prob', 0):.1%}</td>
          <td style="color:{edge_color};font-weight:700">{edge_val:+.1%}</td>
          <td><span class="finding-tag {conf_cls}">{_esc(e.get('confidence',''))}</span></td>
          <td>{_esc(e.get('bet_direction',''))}</td>
          <td>${e.get('volume', 0):,.0f}</td></tr>\n"""

    return f"""
    <table class="mini-table">
      <thead><tr><th>Game</th><th>Model</th><th>Market</th><th>Edge</th><th>Confidence</th><th>Direction</th><th>Volume</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>"""


def _build_next_steps(steps: list[dict]) -> str:
    if not steps:
        return ""

    cards = ""
    for s in steps:
        impact = s.get("impact", "MEDIUM")
        signal_cls = "signal-high" if impact == "HIGH" else "signal-med"
        cards += f"""
        <div class="method-card">
          <h3>{_esc(s.get('title', ''))}</h3>
          <div class="method-score"><span class="signal {signal_cls}"><span class="signal-dot"></span> {_esc(impact)} IMPACT</span></div>
          <p>{_esc(s.get('description', ''))}</p>
        </div>"""
    return f'<div class="methods-grid">{cards}</div>'


def render_report(data: dict) -> str:
    """Render a structured data dict into a self-contained HTML report.

    Expected data keys:
        iteration, timestamp, run_id,
        best_log_loss, best_accuracy, total_experiments, calibration_error, edges_found,
        executive_summary, best_model (dict),
        runs (list of iteration dicts for convergence),
        experiments (list sorted by log_loss),
        edges (list of edge dicts),
        next_steps (list of step dicts with title/description/impact),
        analysis_text (free-form LLM analysis)
    """
    timestamp = data.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M"))
    iteration = data.get("iteration", "?")
    run_id = data.get("run_id", "")

    kpi_html = _build_kpi_cards(data)
    chart_html = _build_convergence_chart(data.get("runs", []))
    table_html = _build_experiment_table(data.get("experiments", []))
    edges_html = _build_edges_table(data.get("edges", []))
    next_steps_html = _build_next_steps(data.get("next_steps", []))

    # Best model info
    bm = data.get("best_model", {})
    best_model_html = f"""
    <div class="best-model-card">
      <div class="bm-header">
        <div class="bm-icon">&#x1F3C6;</div>
        <div><div class="bm-name">{_esc(bm.get('name', 'N/A'))}</div>
          <div class="bm-method">{_esc(bm.get('method', ''))}</div></div>
      </div>
      <div class="bm-metrics">
        <div class="bm-metric"><div class="bm-val">{bm.get('log_loss', 0):.4f}</div><div class="bm-label">Log Loss</div></div>
        <div class="bm-metric"><div class="bm-val">{bm.get('accuracy', 0):.1f}%</div><div class="bm-label">Accuracy</div></div>
        <div class="bm-metric"><div class="bm-val">{bm.get('brier_score', 0):.4f}</div><div class="bm-label">Brier Score</div></div>
      </div>
    </div>"""

    exec_summary = _esc(data.get("executive_summary", ""))
    analysis = data.get("analysis_text", "")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Predicto Report — Iteration {_esc(str(iteration))}</title>
<style>
  :root {{ --bg:#0a0b10; --card:#12141e; --card-border:#1e2235; --text:#d1d5e0; --text-dim:#6b7194; --text-bright:#f0f2f8; --accent:#6366f1; --green:#22c55e; --red:#ef4444; --amber:#f59e0b; --blue:#3b82f6; --pink:#ec4899; --purple:#a855f7; }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif; background:var(--bg); color:var(--text); line-height:1.6; }}
  .page {{ max-width:1200px; margin:0 auto; padding:40px 32px 60px; }}
  .header {{ text-align:center; margin-bottom:36px; }}
  .header h1 {{ font-size:2rem; background:linear-gradient(135deg,#6366f1,#a78bfa,#ec4899); -webkit-background-clip:text; -webkit-text-fill-color:transparent; font-weight:800; margin-bottom:4px; }}
  .header .sub {{ color:var(--text-dim); font-size:0.9rem; }}

  .kpis {{ display:grid; grid-template-columns:repeat(5,1fr); gap:14px; margin-bottom:40px; }}
  .kpi {{ background:var(--card); border:1px solid var(--card-border); border-radius:14px; padding:18px; text-align:center; position:relative; overflow:hidden; }}
  .kpi::before {{ content:''; position:absolute; top:0; left:0; right:0; height:3px; border-radius:14px 14px 0 0; }}
  .kpi.kpi-accent::before {{ background:var(--accent); }}
  .kpi.kpi-green::before {{ background:var(--green); }}
  .kpi.kpi-blue::before {{ background:var(--blue); }}
  .kpi.kpi-purple::before {{ background:var(--purple); }}
  .kpi.kpi-pink::before {{ background:var(--pink); }}
  .kpi .value {{ font-size:1.8rem; font-weight:800; color:var(--text-bright); }}
  .kpi .label {{ font-size:0.7rem; color:var(--text-dim); text-transform:uppercase; letter-spacing:1px; font-weight:600; }}
  .kpi .delta {{ font-size:0.7rem; margin-top:4px; font-weight:600; }}
  .kpi .delta.up {{ color:var(--green); }}

  .section {{ margin-bottom:40px; }}
  .section-title {{ font-size:1.25rem; font-weight:700; color:var(--text-bright); margin-bottom:4px; }}
  .section-sub {{ font-size:0.82rem; color:var(--text-dim); margin-bottom:16px; }}
  .card {{ background:var(--card); border:1px solid var(--card-border); border-radius:14px; padding:24px; }}
  .summary-text {{ font-size:0.9rem; color:var(--text); line-height:1.8; white-space:pre-wrap; }}

  .best-model-card {{ background:linear-gradient(135deg,rgba(99,102,241,0.08),rgba(168,85,247,0.08)); border:1px solid #2d2960; border-radius:14px; padding:24px; }}
  .bm-header {{ display:flex; align-items:center; gap:14px; margin-bottom:16px; }}
  .bm-icon {{ font-size:2rem; }}
  .bm-name {{ font-size:1.1rem; font-weight:700; color:var(--text-bright); }}
  .bm-method {{ font-size:0.8rem; color:var(--accent); }}
  .bm-metrics {{ display:flex; gap:32px; }}
  .bm-metric {{ text-align:center; }}
  .bm-val {{ font-size:1.5rem; font-weight:800; color:var(--green); }}
  .bm-label {{ font-size:0.7rem; color:var(--text-dim); text-transform:uppercase; letter-spacing:0.5px; }}

  .mini-table {{ width:100%; border-collapse:collapse; font-size:0.8rem; }}
  .mini-table th {{ text-align:left; color:var(--text-dim); font-weight:700; padding:8px 10px; border-bottom:1px solid var(--card-border); font-size:0.65rem; text-transform:uppercase; letter-spacing:1px; }}
  .mini-table td {{ padding:6px 10px; border-bottom:1px solid #151825; font-variant-numeric:tabular-nums; }}
  .mini-table tr.top-model td {{ background:rgba(99,102,241,0.06); }}
  .rank-badge {{ display:inline-flex; align-items:center; justify-content:center; width:22px; height:22px; border-radius:6px; font-weight:800; font-size:0.7rem; }}
  .rank-1 {{ background:#2d1a4e; color:#c084fc; }}
  .rank-2 {{ background:#1a2e3a; color:#60a5fa; }}
  .rank-n {{ background:#1a1a22; color:#6b7194; }}
  .bar {{ height:6px; border-radius:3px; background:linear-gradient(90deg,#15803d,#22c55e); display:inline-block; vertical-align:middle; }}
  .bar-bad {{ background:linear-gradient(90deg,#7f1d1d,#ef4444); }}
  .bar-ok {{ background:linear-gradient(90deg,#713f12,#f59e0b); }}
  .finding-tag {{ display:inline-block; font-size:0.7rem; padding:2px 8px; border-radius:6px; font-weight:600; }}
  .tag-discovery {{ background:rgba(99,102,241,0.15); color:#818cf8; }}
  .tag-insight {{ background:rgba(245,158,11,0.15); color:#fbbf24; }}
  .tag-confirmed {{ background:rgba(34,197,94,0.15); color:#4ade80; }}

  .methods-grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); gap:14px; }}
  .method-card {{ background:var(--card); border:1px solid var(--card-border); border-radius:14px; padding:20px; }}
  .method-card h3 {{ font-size:0.95rem; color:var(--text-bright); margin-bottom:4px; }}
  .method-card .method-score {{ font-size:0.75rem; margin-bottom:10px; }}
  .method-card p {{ font-size:0.8rem; color:var(--text-dim); line-height:1.5; }}
  .signal {{ display:inline-flex; align-items:center; gap:5px; font-size:0.75rem; font-weight:600; }}
  .signal-dot {{ width:8px; height:8px; border-radius:50%; }}
  .signal-high .signal-dot {{ background:var(--green); }}
  .signal-high {{ color:var(--green); }}
  .signal-med .signal-dot {{ background:var(--amber); }}
  .signal-med {{ color:var(--amber); }}

  .analysis {{ background:var(--card); border:1px solid var(--card-border); border-radius:14px; padding:24px; font-size:0.85rem; color:#a0a6be; line-height:1.8; white-space:pre-wrap; }}
  .footer {{ text-align:center; color:#333; font-size:0.75rem; margin-top:40px; padding-top:20px; border-top:1px solid #151825; }}

  @media (max-width:800px) {{ .kpis {{ grid-template-columns:repeat(2,1fr); }} .methods-grid {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<div class="page">
  <div class="header">
    <h1>Predicto Analysis Report</h1>
    <div class="sub">Iteration {_esc(str(iteration))} &middot; {_esc(timestamp)} &middot; Run {_esc(run_id)}</div>
  </div>

  {kpi_html}

  <div class="section">
    <div class="section-title">Executive Summary</div>
    <div class="card"><div class="summary-text">{_esc(exec_summary) if exec_summary else 'No summary provided.'}</div></div>
  </div>

  <div class="section">
    <div class="section-title">Best Model</div>
    {best_model_html}
  </div>

  <div class="section">
    <div class="section-title">Model Convergence</div>
    <div class="section-sub">Best log loss achieved at each pipeline iteration</div>
    <div class="card">{chart_html}</div>
  </div>

  <div class="section">
    <div class="section-title">All Experiments — Ranked</div>
    <div class="section-sub">Every experiment across all iterations, sorted by log loss</div>
    <div class="card" style="overflow-x:auto;">{table_html}</div>
  </div>

  <div class="section">
    <div class="section-title">Betting Edge Analysis</div>
    <div class="section-sub">Model predictions vs Polymarket odds — where our model disagrees with the market</div>
    <div class="card" style="overflow-x:auto;">{edges_html}</div>
  </div>

  <div class="section">
    <div class="section-title">Recommended Next Steps</div>
    <div class="section-sub">Promising directions to improve model accuracy</div>
    {next_steps_html}
  </div>

  {"<div class='section'><div class='section-title'>Detailed Analysis</div><div class='analysis'>" + _esc(analysis) + "</div></div>" if analysis else ""}

  <div class="footer">Predicto Multi-Agent AI System &middot; Built with Claude Code + Anthropic SDK</div>
</div>
</body>
</html>"""
