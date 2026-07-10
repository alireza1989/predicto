#!/usr/bin/env python3
"""Predicto Web Dashboard — view reports, track progress, trigger new iterations."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request, send_file

BASE_DIR = Path(__file__).parent.parent
REPORTS_DIR = BASE_DIR / "data" / "reports"
AGENTS_DIR = BASE_DIR / "agents"
PROMPT_OVERRIDES_PATH = BASE_DIR / "data" / "prompt_overrides.json"

app = Flask(__name__)

# ── Pipeline state ────────────────────────────────────────────────────────────
_pipeline_status = {"running": False, "started": None, "log": "", "finished": None, "error": None}
_pipeline_lock = threading.Lock()

# ── Agent metadata ────────────────────────────────────────────────────────────
AGENTS = [
    {
        "id": "data_agent",
        "file": "data_agent.py",
        "name": "Data Agent",
        "icon": "🌐",
        "model": "claude-sonnet",
        "desc": "Fetches NBA game results, upcoming schedules, advanced team stats, and live Polymarket betting odds.",
        "color": "#6366f1",
    },
    {
        "id": "feature_agent",
        "file": "feature_agent.py",
        "name": "Feature Agent",
        "icon": "⚙️",
        "model": "claude-sonnet",
        "desc": "Engineers 69 features: Elo ratings, rolling stats, momentum, head-to-head records, SOS, and advanced NBA analytics.",
        "color": "#3b82f6",
    },
    {
        "id": "meta_scientist",
        "file": "meta_scientist.py",
        "name": "Meta-Scientist",
        "icon": "🧠",
        "model": "claude-opus",
        "desc": "Designs and runs ML experiments. Reads persistent history, picks what to try next, updates knowledge after each run.",
        "color": "#a855f7",
    },
    {
        "id": "eval_agent",
        "file": "eval_agent.py",
        "name": "Eval Agent",
        "icon": "📐",
        "model": "claude-sonnet",
        "desc": "Compares all experiments side-by-side, validates calibration, checks for overfitting, and promotes the best model.",
        "color": "#22c55e",
    },
    {
        "id": "report_agent",
        "file": "report_agent.py",
        "name": "Report Agent",
        "icon": "📊",
        "model": "claude-sonnet",
        "desc": "Generates structured HTML reports with convergence charts, betting edge analysis vs Polymarket, and next-step recommendations.",
        "color": "#ec4899",
    },
]


# ── Data helpers ──────────────────────────────────────────────────────────────

def _get_reports() -> list[dict]:
    if not REPORTS_DIR.exists():
        return []
    reports = []
    for f in sorted(REPORTS_DIR.glob("predicto_report_*.html"), key=lambda p: p.stat().st_mtime, reverse=True):
        name = f.stem
        iteration = "?"
        for p in name.split("_"):
            if p.startswith("iter"):
                iteration = p[4:]
                break
        reports.append({
            "filename": f.name,
            "iteration": iteration,
            "modified": datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
            "size_kb": round(f.stat().st_size / 1024, 1),
        })
    return reports


def _get_stats() -> dict:
    """Pull experiment stats from SQLite for the dashboard."""
    import sqlite3
    db_path = BASE_DIR / "data" / "predicto.db"
    if not db_path.exists():
        return {}
    try:
        conn = sqlite3.connect(str(db_path))

        # All completed experiments
        cursor = conn.execute(
            "SELECT name, method, metrics_json, created_at FROM experiment_log "
            "WHERE status='completed' ORDER BY created_at ASC"
        )
        rows = cursor.fetchall()

        experiments = []
        for row in rows:
            m = json.loads(row[2] or "{}")
            if m.get("log_loss"):
                raw_acc = m.get("accuracy", 0)
                acc_pct = raw_acc * 100 if raw_acc < 1 else raw_acc
                experiments.append({
                    "name": row[0],
                    "method": row[1],
                    "log_loss": m["log_loss"],
                    "accuracy": acc_pct,
                    "brier_score": m.get("brier_score", 0),
                    "created_at": row[3],
                })

        experiments_sorted = sorted(experiments, key=lambda x: x["log_loss"])

        # Run history for convergence
        cursor = conn.execute(
            "SELECT run_id, started_at, finished_at, status, summary FROM run_log ORDER BY started_at ASC"
        )
        run_rows = cursor.fetchall()
        conn.close()

        sorted_runs = [(r[0], r[1], r[2]) for r in run_rows if r[3] == "completed"]
        timed = sorted(experiments, key=lambda x: x["created_at"])

        runs_data = []
        cumulative_best = float("inf")
        for i, (rid, started, finished) in enumerate(sorted_runs, 1):
            run_end = finished or datetime.now().isoformat()
            run_exps = [e for e in timed if e["created_at"] >= started and e["created_at"] <= run_end]
            for e in run_exps:
                if e["log_loss"] < cumulative_best:
                    cumulative_best = e["log_loss"]
            total_so_far = len([e for e in timed if e["created_at"] <= run_end])
            best_in_run = min((e["log_loss"] for e in run_exps), default=None)
            runs_data.append({
                "iteration": i,
                "best_log_loss": round(cumulative_best, 4) if cumulative_best < float("inf") else 0.693,
                "best_in_run": round(best_in_run, 4) if best_in_run else None,
                "experiments": total_so_far,
            })

        best = experiments_sorted[0] if experiments_sorted else None
        method_counts = {}
        for e in experiments:
            m = e["method"].split("(")[0]
            method_counts[m] = method_counts.get(m, 0) + 1

        return {
            "best_log_loss": round(best["log_loss"], 4) if best else None,
            "best_accuracy": round(best["accuracy"], 1) if best else None,
            "best_method": best["method"] if best else None,
            "best_name": best["name"] if best else None,
            "total_experiments": len(experiments),
            "total_runs": len(sorted_runs),
            "random_baseline": 0.693,
            "improvement_pct": round((0.693 - best["log_loss"]) / 0.693 * 100, 1) if best else 0,
            "experiments_top10": experiments_sorted[:10],
            "runs_data": runs_data,
            "method_counts": method_counts,
        }
    except Exception as e:
        return {"error": str(e)}


def _get_run_history() -> list[dict]:
    import sqlite3
    db_path = BASE_DIR / "data" / "predicto.db"
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute(
        "SELECT run_id, started_at, finished_at, status, summary FROM run_log ORDER BY started_at DESC LIMIT 20"
    )
    runs = [{"run_id": r[0], "started": r[1], "finished": r[2], "status": r[3], "summary": r[4] or ""}
            for r in cursor.fetchall()]
    conn.close()
    return runs


def _read_agent_prompt(agent_file: str) -> str:
    """Extract SYSTEM_PROMPT string from an agent Python file."""
    # Check overrides first
    overrides = _load_prompt_overrides()
    agent_id = agent_file.replace(".py", "")
    if agent_id in overrides:
        return overrides[agent_id]

    path = AGENTS_DIR / agent_file
    if not path.exists():
        return ""
    content = path.read_text()
    # Match SYSTEM_PROMPT = """\...""" (with optional backslash after opening quotes)
    match = re.search(r'SYSTEM_PROMPT\s*=\s*"""\\?\n?(.*?)"""', content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def _save_agent_prompt(agent_id: str, new_prompt: str) -> bool:
    """Save a prompt override to the overrides JSON file."""
    overrides = _load_prompt_overrides()
    overrides[agent_id] = new_prompt
    PROMPT_OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROMPT_OVERRIDES_PATH.write_text(json.dumps(overrides, indent=2))
    return True


def _load_prompt_overrides() -> dict:
    if PROMPT_OVERRIDES_PATH.exists():
        try:
            return json.loads(PROMPT_OVERRIDES_PATH.read_text())
        except Exception:
            return {}
    return {}


def _reset_agent_prompt(agent_id: str) -> bool:
    """Remove override, reverting to default prompt from file."""
    overrides = _load_prompt_overrides()
    if agent_id in overrides:
        del overrides[agent_id]
        PROMPT_OVERRIDES_PATH.write_text(json.dumps(overrides, indent=2))
    return True


# ── Pipeline runner ───────────────────────────────────────────────────────────

def _run_pipeline_thread(fetch_data: bool = False):
    global _pipeline_status
    try:
        cmd = [sys.executable, str(BASE_DIR / "main.py")]
        if not fetch_data:
            cmd.append("--skip-data")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(BASE_DIR),
        )
        output_lines = []
        for line in proc.stdout:
            output_lines.append(line)
            with _pipeline_lock:
                _pipeline_status["log"] = "".join(output_lines[-150:])
        proc.wait()
        with _pipeline_lock:
            _pipeline_status["running"] = False
            _pipeline_status["finished"] = datetime.now().isoformat()
            if proc.returncode != 0:
                _pipeline_status["error"] = f"Pipeline exited with code {proc.returncode}"
    except Exception as e:
        with _pipeline_lock:
            _pipeline_status["running"] = False
            _pipeline_status["finished"] = datetime.now().isoformat()
            _pipeline_status["error"] = str(e)


# ── SVG convergence mini-chart ────────────────────────────────────────────────

def _build_mini_chart(runs_data: list[dict]) -> str:
    if not runs_data:
        return '<div style="color:#4a5068;text-align:center;padding:40px;font-size:0.85rem;">Run your first iteration to see convergence.</div>'

    n = len(runs_data)
    left, right, top, bottom = 60, 740, 20, 160
    chart_w, chart_h = right - left, bottom - top
    y_min_val, y_max_val = 0.605, 0.700

    def vy(v):
        return top + (y_max_val - v) / (y_max_val - y_min_val) * chart_h

    def vx(i):
        return left if n == 1 else left + i * chart_w / (n - 1)

    # Grid
    grid = ""
    for gv in [0.700, 0.680, 0.660, 0.640, 0.620]:
        gy = vy(gv)
        grid += f'<line x1="{left}" y1="{gy:.0f}" x2="{right}" y2="{gy:.0f}" stroke="#1e2235" stroke-width="1"/>'
        grid += f'<text x="{left-6}" y="{gy+4:.0f}" fill="#4a5068" font-size="9" text-anchor="end" font-family="Inter,sans-serif">{gv:.3f}</text>'

    rand_y = vy(0.693)
    grid += f'<line x1="{left}" y1="{rand_y:.0f}" x2="{right}" y2="{rand_y:.0f}" stroke="#ef444450" stroke-width="1" stroke-dasharray="5,4"/>'
    grid += f'<text x="{right+4}" y="{rand_y+4:.0f}" fill="#ef4444" font-size="8" font-family="Inter,sans-serif" opacity="0.7">Random</text>'

    points = [(vx(i), vy(r["best_log_loss"]), r) for i, r in enumerate(runs_data)]
    path_d = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y, _) in enumerate(points))
    area_d = path_d + f" L{points[-1][0]:.1f},{bottom} L{points[0][0]:.1f},{bottom} Z"

    circles, labels = "", ""
    colors = ["#6366f1", "#7c3aed", "#8b5cf6", "#a855f7", "#c084fc", "#ec4899", "#f43f5e"]
    global_best_ll = min(r["best_log_loss"] for r in runs_data)
    for i, (x, y, r) in enumerate(points):
        is_best = r["best_log_loss"] == global_best_ll
        col = "#22c55e" if is_best else colors[i % len(colors)]
        circles += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{col}" stroke="#0a0b10" stroke-width="2"/>'
        labels += f'<text x="{x:.1f}" y="{bottom+14:.0f}" fill="#6b7194" font-size="9" text-anchor="middle" font-family="Inter,sans-serif">#{r["iteration"]}</text>'
        label_color = "#22c55e" if is_best else "#a5b4fc"
        labels += f'<text x="{x:.1f}" y="{y-8:.0f}" fill="{label_color}" font-size="8" text-anchor="middle" font-family="Inter,sans-serif" font-weight="700">{r["best_log_loss"]:.4f}</text>'

    return f'''<svg viewBox="0 0 800 195" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;">
      <defs>
        <linearGradient id="lg" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="#6366f1"/><stop offset="100%" stop-color="#ec4899"/></linearGradient>
        <linearGradient id="ag" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#6366f1" stop-opacity="0.12"/><stop offset="100%" stop-color="#6366f1" stop-opacity="0"/></linearGradient>
      </defs>
      {grid}
      <path d="{area_d}" fill="url(#ag)"/>
      <path d="{path_d}" fill="none" stroke="url(#lg)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
      {circles}{labels}
    </svg>'''


# ── HTML templates ─────────────────────────────────────────────────────────────

COMMON_CSS = """
:root { --bg:#0a0b10; --card:#12141e; --card2:#0e1018; --card-border:#1e2235; --text:#d1d5e0; --text-dim:#6b7194; --text-bright:#f0f2f8; --accent:#6366f1; --green:#22c55e; --red:#ef4444; --amber:#f59e0b; --blue:#3b82f6; --pink:#ec4899; --purple:#a855f7; }
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif; background:var(--bg); color:var(--text); min-height:100vh; }
a { text-decoration:none; color:inherit; }

.topbar { background:var(--card); border-bottom:1px solid var(--card-border); padding:0 32px; display:flex; align-items:center; justify-content:space-between; height:56px; position:sticky; top:0; z-index:100; }
.topbar-left { display:flex; align-items:center; gap:32px; }
.topbar h1 { font-size:1.1rem; font-weight:800; white-space:nowrap; }
.topbar h1 .gradient { background:linear-gradient(135deg,#6366f1,#a78bfa,#ec4899); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.topbar h1 .aka { font-size:0.7rem; color:var(--text-dim); font-weight:500; -webkit-text-fill-color:var(--text-dim); margin-left:6px; }
.nav { display:flex; gap:4px; }
.nav a { padding:6px 14px; border-radius:8px; font-size:0.82rem; font-weight:600; color:var(--text-dim); transition:all 0.15s; }
.nav a:hover { color:var(--text-bright); background:rgba(255,255,255,0.05); }
.nav a.active { color:var(--text-bright); background:rgba(99,102,241,0.12); }

.page { max-width:1280px; margin:0 auto; padding:28px 32px 60px; }

.btn { padding:9px 20px; border:none; border-radius:9px; font-size:0.82rem; font-weight:700; cursor:pointer; transition:all 0.2s; display:inline-flex; align-items:center; gap:7px; }
.btn-primary { background:var(--accent); color:white; }
.btn-primary:hover { background:#4f46e5; transform:translateY(-1px); box-shadow:0 4px 16px rgba(99,102,241,0.35); }
.btn-primary:disabled { opacity:0.45; cursor:not-allowed; transform:none; box-shadow:none; }
.btn-secondary { background:var(--card); color:var(--text); border:1px solid var(--card-border); }
.btn-secondary:hover { border-color:var(--accent); color:var(--text-bright); }
.btn-danger { background:rgba(239,68,68,0.1); color:var(--red); border:1px solid rgba(239,68,68,0.2); }
.btn-danger:hover { background:rgba(239,68,68,0.2); }
.btn-success { background:rgba(34,197,94,0.12); color:var(--green); border:1px solid rgba(34,197,94,0.25); }
.btn-success:hover { background:rgba(34,197,94,0.2); }
.btn-sm { padding:5px 12px; font-size:0.75rem; border-radius:7px; }

.section-label { font-size:0.65rem; text-transform:uppercase; letter-spacing:1.5px; font-weight:700; color:var(--text-dim); margin-bottom:12px; }
.section-title { font-size:1.05rem; font-weight:700; color:var(--text-bright); margin-bottom:4px; }

.card { background:var(--card); border:1px solid var(--card-border); border-radius:14px; padding:20px; }

.status-banner { background:var(--card); border:1px solid var(--accent); border-radius:12px; padding:16px 20px; margin-bottom:24px; display:none; }
.status-banner.active { display:block; }
.status-header { display:flex; align-items:center; gap:10px; margin-bottom:10px; }
.spinner { width:16px; height:16px; border:2px solid var(--card-border); border-top-color:var(--accent); border-radius:50%; animation:spin 0.8s linear infinite; flex-shrink:0; }
@keyframes spin { to { transform:rotate(360deg); } }
.log-area { background:#060709; border-radius:8px; padding:12px; font-family:'JetBrains Mono','Fira Code',monospace; font-size:0.7rem; color:#5a6280; max-height:180px; overflow-y:auto; white-space:pre-wrap; word-break:break-all; line-height:1.6; }

.badge { display:inline-block; padding:2px 8px; border-radius:6px; font-size:0.68rem; font-weight:700; }
.badge-completed { background:rgba(34,197,94,0.1); color:var(--green); }
.badge-running { background:rgba(99,102,241,0.1); color:var(--accent); }
.badge-failed { background:rgba(239,68,68,0.1); color:var(--red); }
"""

TOPBAR_HTML = """
<div class="topbar">
  <div class="topbar-left">
    <h1>
      <span class="gradient">Predicto Dashboard</span>
      <span class="aka">(AKA Cheap Schmidty)</span>
    </h1>
    <nav class="nav">
      <a href="/" class="{{ 'active' if page == 'dashboard' else '' }}">Dashboard</a>
      <a href="/agents" class="{{ 'active' if page == 'agents' else '' }}">Agents & Prompts</a>
    </nav>
  </div>
  <div style="font-size:0.72rem;color:var(--text-dim);">{{ stat_summary }}</div>
</div>
"""

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Predicto Dashboard (AKA Cheap Schmidty)</title>
<style>""" + COMMON_CSS + """
/* Dashboard-specific */
.kpi-grid { display:grid; grid-template-columns:repeat(5,1fr); gap:12px; margin-bottom:28px; }
.kpi { background:var(--card); border:1px solid var(--card-border); border-radius:13px; padding:16px; position:relative; overflow:hidden; }
.kpi::before { content:''; position:absolute; top:0; left:0; right:0; height:3px; border-radius:13px 13px 0 0; }
.kpi.c-indigo::before { background:var(--accent); }
.kpi.c-green::before { background:var(--green); }
.kpi.c-blue::before { background:var(--blue); }
.kpi.c-purple::before { background:var(--purple); }
.kpi.c-pink::before { background:var(--pink); }
.kpi .kv { font-size:1.75rem; font-weight:800; color:var(--text-bright); line-height:1; margin-bottom:4px; }
.kpi .kl { font-size:0.65rem; color:var(--text-dim); text-transform:uppercase; letter-spacing:1px; font-weight:700; margin-bottom:6px; }
.kpi .kd { font-size:0.7rem; font-weight:600; }
.kpi .kd.up { color:var(--green); }
.kpi .kd.dim { color:var(--text-dim); }

.two-col { display:grid; grid-template-columns:2fr 1fr; gap:16px; margin-bottom:28px; }
.chart-card { background:var(--card); border:1px solid var(--card-border); border-radius:14px; padding:20px; }

.exp-table { width:100%; border-collapse:collapse; font-size:0.78rem; }
.exp-table th { text-align:left; color:var(--text-dim); font-weight:700; padding:7px 10px; border-bottom:1px solid var(--card-border); font-size:0.62rem; text-transform:uppercase; letter-spacing:1px; }
.exp-table td { padding:6px 10px; border-bottom:1px solid #0d0f18; font-variant-numeric:tabular-nums; }
.exp-table tr:first-child td { color:var(--text-bright); }
.rank-1 { color:#c084fc; font-weight:800; }
.rank-other { color:var(--text-dim); font-weight:600; }

.method-chip { display:inline-block; padding:2px 8px; border-radius:6px; font-size:0.68rem; font-weight:600; background:rgba(99,102,241,0.1); color:#818cf8; }
.method-chip.gb { background:rgba(59,130,246,0.1); color:#60a5fa; }
.method-chip.lgbm { background:rgba(245,158,11,0.1); color:#fbbf24; }
.method-chip.xgb { background:rgba(236,72,153,0.1); color:#f472b6; }
.method-chip.nn { background:rgba(34,197,94,0.1); color:#4ade80; }
.method-chip.ens { background:rgba(168,85,247,0.1); color:#c084fc; }

.perf-bar-wrap { background:#0d0f18; border-radius:4px; height:5px; width:80px; }
.perf-bar { height:5px; border-radius:4px; background:linear-gradient(90deg,#15803d,#22c55e); }
.perf-bar.mid { background:linear-gradient(90deg,#92400e,#f59e0b); }
.perf-bar.low { background:linear-gradient(90deg,#7f1d1d,#ef4444); }

.method-breakdown { display:flex; flex-direction:column; gap:8px; }
.mb-row { display:flex; align-items:center; gap:10px; }
.mb-label { font-size:0.75rem; color:var(--text-dim); width:120px; flex-shrink:0; }
.mb-bar-wrap { flex:1; background:#0d0f18; border-radius:4px; height:6px; }
.mb-bar { height:6px; border-radius:4px; }
.mb-count { font-size:0.72rem; color:var(--text-dim); width:24px; text-align:right; }

.reports-grid { display:grid; gap:10px; margin-bottom:28px; }
.report-row { background:var(--card); border:1px solid var(--card-border); border-radius:12px; padding:14px 18px; display:flex; align-items:center; justify-content:space-between; transition:all 0.15s; }
.report-row:hover { border-color:var(--accent); transform:translateX(3px); box-shadow:0 2px 12px rgba(99,102,241,0.1); }
.r-iter { background:linear-gradient(135deg,rgba(99,102,241,0.15),rgba(168,85,247,0.15)); color:#a5b4fc; font-weight:800; font-size:0.95rem; padding:7px 13px; border-radius:9px; min-width:46px; text-align:center; }
.r-name { font-weight:600; color:var(--text-bright); font-size:0.87rem; }
.r-date { font-size:0.72rem; color:var(--text-dim); margin-top:2px; }
.btn-view { background:rgba(99,102,241,0.1); color:#818cf8; border:1px solid rgba(99,102,241,0.2); padding:5px 14px; border-radius:7px; font-size:0.77rem; font-weight:600; cursor:pointer; transition:all 0.15s; text-decoration:none; }
.btn-view:hover { background:rgba(99,102,241,0.2); color:#a5b4fc; }

.runs-table { width:100%; border-collapse:collapse; font-size:0.77rem; }
.runs-table th { text-align:left; color:var(--text-dim); font-weight:700; padding:7px 10px; border-bottom:1px solid var(--card-border); font-size:0.62rem; text-transform:uppercase; letter-spacing:1px; }
.runs-table td { padding:7px 10px; border-bottom:1px solid #0d0f18; }

.empty { text-align:center; padding:48px 20px; color:var(--text-dim); }
.empty .ei { font-size:2.5rem; margin-bottom:12px; opacity:0.25; }

@media(max-width:900px) { .kpi-grid { grid-template-columns:repeat(2,1fr); } .two-col { grid-template-columns:1fr; } }
</style>
</head>
<body>

""" + TOPBAR_HTML + """

<div class="page">

  <!-- Status Banner -->
  <div class="status-banner" id="statusBanner">
    <div class="status-header">
      <div class="spinner" id="statusSpinner"></div>
      <span style="font-weight:700;color:var(--text-bright);" id="statusText">Running pipeline...</span>
    </div>
    <div class="log-area" id="logArea"></div>
  </div>

  <!-- Actions -->
  <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:28px;">
    <button class="btn btn-primary" id="runBtn" onclick="startPipeline(false)">
      <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><polygon points="4,2 14,8 4,14"/></svg>
      Run New Iteration
    </button>
    <button class="btn" id="runDataBtn" onclick="startPipeline(true)" style="background:linear-gradient(135deg,#0e7490,#0891b2);color:#fff;border:none;padding:9px 18px;border-radius:8px;font-size:13px;font-weight:600;cursor:pointer;display:flex;align-items:center;gap:6px;">
      <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M8 1a5 5 0 0 1 5 5v1h1a2 2 0 0 1 0 4h-1v1a5 5 0 0 1-10 0v-1H2a2 2 0 0 1 0-4h1V6a5 5 0 0 1 5-5zm0 2a3 3 0 0 0-3 3v1h6V6a3 3 0 0 0-3-3z"/></svg>
      Fetch Data &amp; Run
    </button>
    <button class="btn btn-secondary" onclick="location.reload()">
      <svg width="13" height="13" viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M1 7a6 6 0 1 1 1.5 3.9"/><path d="M1 11V7h4"/></svg>
      Refresh
    </button>
    <a href="/agents" class="btn btn-secondary">
      <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="8" cy="6" r="3"/><path d="M2 14c0-3.3 2.7-6 6-6s6 2.7 6 6"/></svg>
      Edit Agent Prompts
    </a>
  </div>

  {% if stats and stats.total_experiments %}

  <!-- KPI Cards -->
  <div class="section-label">Model Performance</div>
  <div class="kpi-grid">
    <div class="kpi c-green">
      <div class="kl">Best Log Loss</div>
      <div class="kv">{{ stats.best_log_loss }}</div>
      <div class="kd up">↓ {{ stats.improvement_pct }}% vs random</div>
    </div>
    <div class="kpi c-blue">
      <div class="kl">Best Accuracy</div>
      <div class="kv">{{ stats.best_accuracy }}%</div>
      <div class="kd up">+{{ "%.1f"|format(stats.best_accuracy - 55.0) }}pp vs baseline</div>
    </div>
    <div class="kpi c-indigo">
      <div class="kl">Pipeline Runs</div>
      <div class="kv">{{ stats.total_runs }}</div>
      <div class="kd dim">iterations completed</div>
    </div>
    <div class="kpi c-purple">
      <div class="kl">Experiments</div>
      <div class="kv">{{ stats.total_experiments }}</div>
      <div class="kd dim">models evaluated</div>
    </div>
    <div class="kpi c-pink">
      <div class="kl">Random Baseline</div>
      <div class="kv">0.693</div>
      <div class="kd dim">log loss ceiling</div>
    </div>
  </div>

  <!-- Convergence Chart + Method Breakdown -->
  <div class="two-col">
    <div class="chart-card">
      <div class="section-label" style="margin-bottom:8px;">Convergence — Best Log Loss per Iteration</div>
      {{ chart_svg | safe }}
    </div>
    <div class="chart-card">
      <div class="section-label" style="margin-bottom:14px;">Experiments by Method</div>
      <div class="method-breakdown">
        {% set max_count = stats.method_counts.values()|list|max if stats.method_counts else 1 %}
        {% set method_colors = {'logistic_regression':'#6366f1','gradient_boosting':'#3b82f6','lightgbm':'#f59e0b','xgboost':'#ec4899','neural_network':'#22c55e','ensemble':'#a855f7'} %}
        {% for method, count in stats.method_counts.items()|sort(attribute='1', reverse=True) %}
        <div class="mb-row">
          <div class="mb-label">{{ method.replace('_',' ').title()[:18] }}</div>
          <div class="mb-bar-wrap">
            <div class="mb-bar" style="width:{{ (count / max_count * 100)|int }}%;background:{{ method_colors.get(method.split('(')[0], '#6366f1') }};"></div>
          </div>
          <div class="mb-count">{{ count }}</div>
        </div>
        {% endfor %}
      </div>
      {% if stats.best_name %}
      <div style="margin-top:20px;padding-top:16px;border-top:1px solid var(--card-border);">
        <div class="section-label" style="margin-bottom:8px;">🏆 Best Model</div>
        <div style="font-size:0.82rem;color:var(--text-bright);font-weight:600;margin-bottom:4px;">{{ stats.best_name[:45] }}</div>
        <div style="font-size:0.72rem;color:var(--text-dim);">{{ stats.best_method }}</div>
      </div>
      {% endif %}
    </div>
  </div>

  <!-- Top Experiments Table -->
  <div class="section-label">Top Experiments — Ranked by Log Loss</div>
  <div class="card" style="overflow-x:auto;margin-bottom:28px;">
    <table class="exp-table">
      <thead><tr><th>#</th><th>Name</th><th>Method</th><th>Log Loss</th><th>Accuracy</th><th>Performance</th></tr></thead>
      <tbody>
        {% for exp in stats.experiments_top10 %}
        <tr>
          <td><span class="{{ 'rank-1' if loop.index == 1 else 'rank-other' }}">{{ loop.index }}</span></td>
          <td style="max-width:280px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{{ exp.name }}</td>
          <td>
            {% set m = exp.method.split('(')[0] %}
            {% set chip_cls = 'gb' if 'gradient' in m else ('lgbm' if 'lgbm' in m or 'light' in m else ('xgb' if 'xgb' in m else ('nn' if 'neural' in m else ('ens' if 'ensemble' in m else '')))) %}
            <span class="method-chip {{ chip_cls }}">{{ exp.method[:28] }}</span>
          </td>
          <td style="font-weight:{% if loop.index == 1 %}700{% else %}400{% endif %};color:{% if exp.log_loss < 0.619 %}#22c55e{% elif exp.log_loss < 0.625 %}#4ade80{% else %}var(--text){% endif %};">{{ "%.4f"|format(exp.log_loss) }}</td>
          <td>{{ "%.1f"|format(exp.accuracy) }}%</td>
          <td>
            {% set bar = [(0.680 - exp.log_loss) / (0.680 - 0.615) * 100, 0]|max %}
            {% set bar = [bar, 100]|min %}
            <div class="perf-bar-wrap"><div class="perf-bar {{ 'mid' if bar < 40 else ('' if bar >= 60 else 'mid') }}" style="width:{{ bar|int }}%"></div></div>
          </td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>

  {% else %}
  <div class="empty">
    <div class="ei">🧪</div>
    <p>No experiments yet. Click <strong>Run New Iteration</strong> to start.</p>
  </div>
  {% endif %}

  <!-- Reports -->
  <div class="section-label">HTML Reports</div>
  {% if reports %}
  <div class="reports-grid">
    {% for r in reports %}
    <div class="report-row" onclick="window.open('/report/{{ r.filename }}','_blank')">
      <div style="display:flex;align-items:center;gap:14px;">
        <div class="r-iter">#{{ r.iteration }}</div>
        <div>
          <div class="r-name">Iteration {{ r.iteration }} Report</div>
          <div class="r-date">{{ r.modified }} &middot; {{ r.size_kb }} KB</div>
        </div>
      </div>
      <a class="btn-view" href="/report/{{ r.filename }}" target="_blank" onclick="event.stopPropagation()">View →</a>
    </div>
    {% endfor %}
  </div>
  {% else %}
  <div style="color:var(--text-dim);font-size:0.85rem;margin-bottom:28px;padding:20px;text-align:center;">
    No reports generated yet.
  </div>
  {% endif %}

  <!-- Run History -->
  {% if runs %}
  <div class="section-label">Pipeline Run History</div>
  <div class="card" style="overflow:hidden;padding:0;">
    <table class="runs-table">
      <thead><tr><th>Run ID</th><th>Started</th><th>Finished</th><th>Status</th><th>Summary</th></tr></thead>
      <tbody>
        {% for run in runs %}
        <tr>
          <td style="font-family:monospace;color:var(--accent);font-size:0.72rem;">{{ run.run_id }}</td>
          <td style="color:var(--text-dim);">{{ run.started[:16] if run.started else '-' }}</td>
          <td style="color:var(--text-dim);">{{ run.finished[:16] if run.finished else '-' }}</td>
          <td><span class="badge badge-{{ run.status }}">{{ run.status }}</span></td>
          <td style="color:var(--text-dim);font-size:0.72rem;max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{{ run.summary[:70] }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
  {% endif %}

</div>

<script>
let polling = null;

function startPipeline(fetchData) {
  const btn = document.getElementById('runBtn');
  const btnData = document.getElementById('runDataBtn');
  if (btn) { btn.disabled = true; btn.innerHTML = '<div class="spinner"></div> Running...'; }
  if (btnData) btnData.disabled = true;
  document.getElementById('statusBanner').classList.add('active');
  document.getElementById('logArea').textContent = (fetchData ? 'Fetching data + running pipeline...' : 'Starting pipeline...') + '\\n';

  fetch('/api/run', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({fetch_data: !!fetchData}) })
    .then(r => r.json())
    .then(data => {
      if (data.error) { alert(data.error); resetBtn(); return; }
      polling = setInterval(pollStatus, 2000);
    });
}

function resetBtn() {
  const btn = document.getElementById('runBtn');
  const btnData = document.getElementById('runDataBtn');
  if (btn) { btn.disabled = false; btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><polygon points="4,2 14,8 4,14"/></svg> Run New Iteration'; }
  if (btnData) btnData.disabled = false;
}

function pollStatus() {
  fetch('/api/status').then(r => r.json()).then(data => {
    const la = document.getElementById('logArea');
    la.textContent = data.log || '';
    la.scrollTop = la.scrollHeight;
    if (!data.running) {
      clearInterval(polling);
      resetBtn();
      document.getElementById('statusSpinner').style.display = 'none';
      const st = document.getElementById('statusText');
      st.textContent = data.error ? 'Pipeline failed: ' + data.error : 'Done! Refreshing...';
      st.style.color = data.error ? 'var(--red)' : 'var(--green)';
      if (!data.error) setTimeout(() => location.reload(), 1800);
    }
  });
}

fetch('/api/status').then(r => r.json()).then(data => {
  if (data.running) {
    document.getElementById('runBtn').disabled = true;
    document.getElementById('runBtn').innerHTML = '<div class="spinner"></div> Running...';
    document.getElementById('statusBanner').classList.add('active');
    document.getElementById('logArea').textContent = data.log || '';
    polling = setInterval(pollStatus, 2000);
  }
});
</script>
</body>
</html>"""


AGENTS_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Agents & Prompts — Predicto</title>
<style>""" + COMMON_CSS + """
/* Agents page */
.flow-wrap { background:var(--card); border:1px solid var(--card-border); border-radius:14px; padding:28px 20px; margin-bottom:32px; overflow-x:auto; }
.flow { display:flex; align-items:center; gap:0; min-width:700px; }
.flow-node { flex:1; display:flex; flex-direction:column; align-items:center; gap:0; position:relative; }
.flow-box { background:var(--card2); border:1px solid var(--card-border); border-radius:13px; padding:16px 14px; text-align:center; cursor:pointer; transition:all 0.2s; width:100%; max-width:148px; }
.flow-box:hover { transform:translateY(-3px); box-shadow:0 6px 20px rgba(0,0,0,0.4); }
.flow-box.active { transform:translateY(-3px); }
.flow-icon { font-size:1.6rem; margin-bottom:6px; }
.flow-name { font-size:0.78rem; font-weight:700; color:var(--text-bright); margin-bottom:3px; }
.flow-model { font-size:0.62rem; padding:2px 7px; border-radius:5px; font-weight:600; display:inline-block; }
.flow-model.sonnet { background:rgba(99,102,241,0.12); color:#818cf8; }
.flow-model.opus { background:rgba(168,85,247,0.12); color:#c084fc; }
.flow-desc { font-size:0.65rem; color:var(--text-dim); margin-top:5px; line-height:1.4; }
.flow-arrow { display:flex; align-items:center; padding:0 4px; flex-shrink:0; }
.flow-arrow svg { color:var(--text-dim); }

.loop-indicator { text-align:center; margin-top:18px; }
.loop-badge { display:inline-flex; align-items:center; gap:8px; background:rgba(168,85,247,0.08); border:1px solid rgba(168,85,247,0.2); border-radius:20px; padding:6px 16px; font-size:0.72rem; color:#c084fc; font-weight:600; }

.agents-list { display:flex; flex-direction:column; gap:14px; }
.agent-card { background:var(--card); border:1px solid var(--card-border); border-radius:14px; overflow:hidden; transition:border-color 0.2s; }
.agent-card.open { border-color:var(--active-color, var(--accent)); }
.agent-header { padding:16px 20px; display:flex; align-items:center; gap:14px; cursor:pointer; user-select:none; }
.agent-header:hover { background:rgba(255,255,255,0.02); }
.agent-icon-wrap { width:42px; height:42px; border-radius:11px; display:flex; align-items:center; justify-content:center; font-size:1.3rem; flex-shrink:0; }
.agent-info { flex:1; }
.agent-title { font-size:0.9rem; font-weight:700; color:var(--text-bright); display:flex; align-items:center; gap:8px; }
.agent-subtitle { font-size:0.73rem; color:var(--text-dim); margin-top:2px; }
.agent-chevron { color:var(--text-dim); transition:transform 0.2s; flex-shrink:0; }
.agent-card.open .agent-chevron { transform:rotate(180deg); }

.agent-body { display:none; padding:0 20px 20px; border-top:1px solid var(--card-border); }
.agent-card.open .agent-body { display:block; }
.prompt-label { font-size:0.65rem; text-transform:uppercase; letter-spacing:1px; font-weight:700; color:var(--text-dim); margin-top:16px; margin-bottom:8px; display:flex; justify-content:space-between; align-items:center; }
.prompt-area { width:100%; background:#060709; border:1px solid var(--card-border); border-radius:9px; padding:14px; font-family:'JetBrains Mono','Fira Code',monospace; font-size:0.73rem; color:#c0c8e0; line-height:1.7; resize:vertical; min-height:200px; outline:none; transition:border-color 0.15s; }
.prompt-area:focus { border-color:var(--accent); }
.prompt-actions { display:flex; gap:8px; margin-top:10px; align-items:center; }
.save-status { font-size:0.72rem; color:var(--text-dim); margin-left:4px; }
.save-status.ok { color:var(--green); }
.save-status.err { color:var(--red); }
.override-badge { display:inline-block; padding:2px 8px; border-radius:6px; font-size:0.65rem; font-weight:700; background:rgba(245,158,11,0.1); color:var(--amber); margin-left:6px; }
</style>
</head>
<body>

""" + TOPBAR_HTML + """

<div class="page">

  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:24px;">
    <div>
      <div class="section-title">Agentic Pipeline</div>
      <div style="color:var(--text-dim);font-size:0.82rem;margin-top:4px;">5 specialized agents run in sequence each iteration. Click any agent below to view and edit its system prompt.</div>
    </div>
    <button class="btn btn-primary" id="runBtn" onclick="startPipeline(false)">
      <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><polygon points="4,2 14,8 4,14"/></svg>
      Run New Iteration
    </button>
  </div>

  <!-- Status Banner -->
  <div class="status-banner" id="statusBanner">
    <div class="status-header">
      <div class="spinner" id="statusSpinner"></div>
      <span style="font-weight:700;color:var(--text-bright);" id="statusText">Running pipeline...</span>
    </div>
    <div class="log-area" id="logArea"></div>
  </div>

  <!-- Flow Diagram -->
  <div class="flow-wrap">
    <div class="section-label" style="margin-bottom:20px;text-align:center;">Agent Execution Flow</div>
    <div class="flow">
      {% for agent in agents %}
      <div class="flow-node">
        <div class="flow-box" id="flow-{{ agent.id }}" style="border-color:{{ agent.color }}22;"
             onclick="openAgent('{{ agent.id }}')"
             onmouseover="this.style.borderColor='{{ agent.color }}80'"
             onmouseout="this.style.borderColor='{{ agent.color }}22'">
          <div class="flow-icon">{{ agent.icon }}</div>
          <div class="flow-name">{{ agent.name }}</div>
          <div class="flow-model {{ 'opus' if 'opus' in agent.model else 'sonnet' }}">{{ agent.model }}</div>
          <div class="flow-desc">{{ agent.desc[:70] }}</div>
        </div>
        {% if not loop.last %}
        <!-- step label below box -->
        <div style="font-size:0.6rem;color:var(--text-dim);margin-top:6px;">Step {{ loop.index }}</div>
        {% else %}
        <div style="font-size:0.6rem;color:var(--text-dim);margin-top:6px;">Step {{ loop.index }}</div>
        {% endif %}
      </div>
      {% if not loop.last %}
      <div class="flow-arrow">
        <svg width="28" height="14" viewBox="0 0 28 14" fill="none">
          <line x1="0" y1="7" x2="20" y2="7" stroke="currentColor" stroke-width="1.5"/>
          <polyline points="14,2 22,7 14,12" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"/>
        </svg>
      </div>
      {% endif %}
      {% endfor %}
    </div>

    <!-- Shared data layer -->
    <div style="display:flex;justify-content:center;margin-top:20px;padding-top:16px;border-top:1px solid var(--card-border);">
      <div style="display:flex;align-items:center;gap:8px;background:#0d0f18;border:1px solid #1e2235;border-radius:8px;padding:8px 20px;">
        <span style="font-size:0.75rem;color:var(--text-dim);font-weight:600;">Shared Data Layer:</span>
        <span style="font-size:0.7rem;color:#4a5068;">SQLite DB</span>
        <span style="color:var(--card-border);">·</span>
        <span style="font-size:0.7rem;color:#4a5068;">Parquet Files</span>
        <span style="color:var(--card-border);">·</span>
        <span style="font-size:0.7rem;color:#4a5068;">Scientist History MD</span>
        <span style="color:var(--card-border);">·</span>
        <span style="font-size:0.7rem;color:#4a5068;">Prompt Overrides JSON</span>
      </div>
    </div>

    <!-- Loop indicator -->
    <div class="loop-indicator" style="margin-top:16px;">
      <div class="loop-badge">
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.5">
          <path d="M1 7a6 6 0 1 1 1.5 3.9"/><path d="M1 11V7h4"/>
        </svg>
        Self-improving loop — Meta-Scientist reads history &amp; improves each iteration
      </div>
    </div>
  </div>

  <!-- Agent Prompt Editors -->
  <div class="section-label" style="margin-bottom:12px;">Agent System Prompts — Click to Expand &amp; Edit</div>
  <div class="agents-list">
    {% for agent in agents %}
    <div class="agent-card" id="card-{{ agent.id }}" style="--active-color:{{ agent.color }};">
      <div class="agent-header" onclick="toggleAgent('{{ agent.id }}')">
        <div class="agent-icon-wrap" style="background:{{ agent.color }}18;">
          <span>{{ agent.icon }}</span>
        </div>
        <div class="agent-info">
          <div class="agent-title">
            {{ agent.name }}
            <span class="flow-model {{ 'opus' if 'opus' in agent.model else 'sonnet' }}">{{ agent.model }}</span>
            {% if agent.id in overrides %}
            <span class="override-badge">PROMPT OVERRIDDEN</span>
            {% endif %}
          </div>
          <div class="agent-subtitle">{{ agent.desc }}</div>
        </div>
        <svg class="agent-chevron" width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2">
          <polyline points="4,6 8,10 12,6"/>
        </svg>
      </div>
      <div class="agent-body">
        <div class="prompt-label">
          <span>System Prompt {{ '⚠️ Override active — will be used instead of default' if agent.id in overrides else '(default from code)' }}</span>
          <span style="color:var(--text-dim);font-size:0.62rem;">{{ agent.prompt_chars }} chars</span>
        </div>
        <textarea class="prompt-area" id="prompt-{{ agent.id }}" rows="12">{{ agent.prompt }}</textarea>
        <div class="prompt-actions">
          <button class="btn btn-success btn-sm" onclick="savePrompt('{{ agent.id }}')">
            <svg width="11" height="11" viewBox="0 0 12 12" fill="currentColor"><path d="M1 6l4 4 6-7"/></svg>
            Save Override
          </button>
          {% if agent.id in overrides %}
          <button class="btn btn-danger btn-sm" onclick="resetPrompt('{{ agent.id }}')">Reset to Default</button>
          {% endif %}
          <span class="save-status" id="status-{{ agent.id }}"></span>
        </div>
        <div style="font-size:0.68rem;color:var(--text-dim);margin-top:8px;line-height:1.5;">
          💡 Saving stores an override in <code style="color:#818cf8;">data/prompt_overrides.json</code>.
          The next pipeline run will inject this prompt. Reset to revert to the hardcoded default.
        </div>
      </div>
    </div>
    {% endfor %}
  </div>

</div>

<script>
function toggleAgent(id) {
  const card = document.getElementById('card-' + id);
  card.classList.toggle('open');
}

function openAgent(id) {
  const card = document.getElementById('card-' + id);
  if (!card.classList.contains('open')) card.classList.add('open');
  card.scrollIntoView({ behavior:'smooth', block:'nearest' });
}

function savePrompt(agentId) {
  const prompt = document.getElementById('prompt-' + agentId).value;
  const statusEl = document.getElementById('status-' + agentId);
  statusEl.textContent = 'Saving...';
  statusEl.className = 'save-status';

  fetch('/api/prompts/' + agentId, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt })
  }).then(r => r.json()).then(data => {
    if (data.ok) {
      statusEl.textContent = '✓ Saved — will apply on next run';
      statusEl.className = 'save-status ok';
      setTimeout(() => location.reload(), 1200);
    } else {
      statusEl.textContent = '✗ Error: ' + (data.error || 'Unknown');
      statusEl.className = 'save-status err';
    }
  }).catch(e => {
    statusEl.textContent = '✗ ' + e.message;
    statusEl.className = 'save-status err';
  });
}

function resetPrompt(agentId) {
  if (!confirm('Reset "' + agentId + '" to default prompt?')) return;
  fetch('/api/prompts/' + agentId + '/reset', { method: 'POST' })
    .then(r => r.json())
    .then(data => { if (data.ok) location.reload(); });
}

let polling = null;
function startPipeline(fetchData) {
  const btn = document.getElementById('runBtn');
  const btnData = document.getElementById('runDataBtn');
  if (btn) { btn.disabled = true; btn.innerHTML = '<div class="spinner"></div> Running...'; }
  if (btnData) btnData.disabled = true;
  document.getElementById('statusBanner').classList.add('active');
  document.getElementById('logArea').textContent = (fetchData ? 'Fetching data + running pipeline...' : 'Starting pipeline...') + '\\n';
  fetch('/api/run', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({fetch_data: !!fetchData}) }).then(r => r.json()).then(data => {
    if (data.error) { alert(data.error); if (btn) { btn.disabled = false; btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><polygon points="4,2 14,8 4,14"/></svg> Run New Iteration'; } if (btnData) btnData.disabled = false; return; }
    polling = setInterval(pollStatus, 2000);
  });
}
function pollStatus() {
  fetch('/api/status').then(r => r.json()).then(data => {
    const la = document.getElementById('logArea');
    la.textContent = data.log || '';
    la.scrollTop = la.scrollHeight;
    if (!data.running) {
      clearInterval(polling);
      const rb = document.getElementById('runBtn');
      const rbd = document.getElementById('runDataBtn');
      if (rb) { rb.disabled = false; rb.innerHTML = '<svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><polygon points="4,2 14,8 4,14"/></svg> Run New Iteration'; }
      if (rbd) rbd.disabled = false;
      document.getElementById('statusSpinner').style.display = 'none';
      const st = document.getElementById('statusText');
      st.textContent = data.error ? 'Failed: ' + data.error : 'Done!';
      st.style.color = data.error ? 'var(--red)' : 'var(--green)';
    }
  });
}
fetch('/api/status').then(r => r.json()).then(data => {
  if (data.running) {
    document.getElementById('runBtn').disabled = true;
    document.getElementById('runBtn').innerHTML = '<div class="spinner"></div> Running...';
    document.getElementById('statusBanner').classList.add('active');
    document.getElementById('logArea').textContent = data.log || '';
    polling = setInterval(pollStatus, 2000);
  }
});
</script>
</body>
</html>"""


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def dashboard():
    stats = _get_stats()
    reports = _get_reports()
    runs = _get_run_history()
    chart_svg = _build_mini_chart(stats.get("runs_data", []))
    total_exp = stats.get("total_experiments", 0)
    total_runs = stats.get("total_runs", 0)
    stat_summary = f"{total_exp} experiments · {total_runs} runs"
    return render_template_string(
        DASHBOARD_HTML,
        page="dashboard",
        stats=stats,
        reports=reports,
        runs=runs,
        chart_svg=chart_svg,
        stat_summary=stat_summary,
    )


@app.route("/agents")
def agents_page():
    overrides = _load_prompt_overrides()
    agents_data = []
    for a in AGENTS:
        prompt = _read_agent_prompt(a["file"])
        agents_data.append({**a, "prompt": prompt, "prompt_chars": len(prompt)})

    stats = _get_stats()
    total_exp = stats.get("total_experiments", 0)
    total_runs = stats.get("total_runs", 0)
    stat_summary = f"{total_exp} experiments · {total_runs} runs"

    return render_template_string(
        AGENTS_HTML,
        page="agents",
        agents=agents_data,
        overrides=overrides,
        stat_summary=stat_summary,
    )


@app.route("/report/<filename>")
def view_report(filename):
    path = REPORTS_DIR / filename
    if not path.exists() or not path.name.startswith("predicto_report_"):
        return "Report not found", 404
    return send_file(path)


@app.route("/api/run", methods=["POST"])
def start_run():
    global _pipeline_status
    data = request.get_json(silent=True) or {}
    fetch_data = bool(data.get("fetch_data", False))
    with _pipeline_lock:
        if _pipeline_status["running"]:
            return jsonify({"error": "Pipeline is already running"})
        _pipeline_status = {"running": True, "started": datetime.now().isoformat(),
                            "log": "", "finished": None, "error": None,
                            "fetch_data": fetch_data}
    thread = threading.Thread(target=_run_pipeline_thread, args=(fetch_data,), daemon=True)
    thread.start()
    return jsonify({"status": "started", "fetch_data": fetch_data})


@app.route("/api/status")
def pipeline_status():
    with _pipeline_lock:
        return jsonify(_pipeline_status)


@app.route("/api/reports")
def list_reports():
    return jsonify(_get_reports())


@app.route("/api/stats")
def api_stats():
    return jsonify(_get_stats())


@app.route("/api/prompts/<agent_id>", methods=["POST"])
def save_prompt(agent_id: str):
    valid_ids = {a["id"] for a in AGENTS}
    if agent_id not in valid_ids:
        return jsonify({"ok": False, "error": "Unknown agent"}), 400
    data = request.get_json()
    if not data or "prompt" not in data:
        return jsonify({"ok": False, "error": "Missing prompt"}), 400
    try:
        _save_agent_prompt(agent_id, data["prompt"])
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/prompts/<agent_id>/reset", methods=["POST"])
def reset_prompt(agent_id: str):
    valid_ids = {a["id"] for a in AGENTS}
    if agent_id not in valid_ids:
        return jsonify({"ok": False, "error": "Unknown agent"}), 400
    _reset_agent_prompt(agent_id)
    return jsonify({"ok": True})


@app.route("/api/prompts")
def get_prompts():
    overrides = _load_prompt_overrides()
    return jsonify({
        a["id"]: {
            "prompt": _read_agent_prompt(a["file"]),
            "overridden": a["id"] in overrides,
        }
        for a in AGENTS
    })


if __name__ == "__main__":
    print("\n" + "=" * 54)
    print("  Predicto Dashboard (AKA Cheap Schmidty)")
    print("  http://localhost:5001")
    print("=" * 54 + "\n")
    app.run(debug=True, port=5001)
