#!/usr/bin/env python3
"""Predicto Web Dashboard — view reports, track progress, trigger new iterations."""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template_string, send_file

BASE_DIR = Path(__file__).parent.parent
REPORTS_DIR = BASE_DIR / "data" / "reports"

app = Flask(__name__)

# Track running pipeline
_pipeline_status = {"running": False, "started": None, "log": "", "finished": None, "error": None}
_pipeline_lock = threading.Lock()


def _get_reports() -> list[dict]:
    """Get all HTML reports sorted by modification time (newest first)."""
    if not REPORTS_DIR.exists():
        return []
    reports = []
    for f in sorted(REPORTS_DIR.glob("predicto_report_*.html"), key=lambda p: p.stat().st_mtime, reverse=True):
        # Parse iteration from filename: predicto_report_iter7_20260413_123456.html
        name = f.stem
        parts = name.split("_")
        iteration = "?"
        for p in parts:
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


def _get_run_history() -> list[dict]:
    """Get pipeline run history from SQLite."""
    import sqlite3
    db_path = BASE_DIR / "data" / "predicto.db"
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute(
        "SELECT run_id, started_at, finished_at, status, summary FROM run_log ORDER BY started_at DESC LIMIT 20"
    )
    runs = []
    for row in cursor.fetchall():
        runs.append({
            "run_id": row[0],
            "started": row[1],
            "finished": row[2],
            "status": row[3],
            "summary": row[4] or "",
        })
    conn.close()
    return runs


def _run_pipeline_thread():
    """Run the pipeline in a background thread."""
    global _pipeline_status
    try:
        proc = subprocess.Popen(
            [sys.executable, str(BASE_DIR / "main.py"), "--skip-data"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(BASE_DIR),
        )
        output_lines = []
        for line in proc.stdout:
            output_lines.append(line)
            with _pipeline_lock:
                _pipeline_status["log"] = "".join(output_lines[-100:])  # keep last 100 lines

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


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Predicto Dashboard</title>
<style>
  :root { --bg:#0a0b10; --card:#12141e; --card-border:#1e2235; --text:#d1d5e0; --text-dim:#6b7194; --text-bright:#f0f2f8; --accent:#6366f1; --green:#22c55e; --red:#ef4444; --amber:#f59e0b; }
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif; background:var(--bg); color:var(--text); min-height:100vh; }

  .topbar { background:var(--card); border-bottom:1px solid var(--card-border); padding:16px 32px; display:flex; align-items:center; justify-content:space-between; }
  .topbar h1 { font-size:1.3rem; background:linear-gradient(135deg,#6366f1,#a78bfa,#ec4899); -webkit-background-clip:text; -webkit-text-fill-color:transparent; font-weight:800; }
  .topbar .subtitle { font-size:0.75rem; color:var(--text-dim); }

  .page { max-width:1200px; margin:0 auto; padding:32px; }

  .actions { display:flex; gap:12px; align-items:center; margin-bottom:32px; }
  .btn { padding:10px 24px; border:none; border-radius:10px; font-size:0.85rem; font-weight:700; cursor:pointer; transition:all 0.2s; display:inline-flex; align-items:center; gap:8px; }
  .btn-primary { background:var(--accent); color:white; }
  .btn-primary:hover { background:#4f46e5; transform:translateY(-1px); box-shadow:0 4px 16px rgba(99,102,241,0.3); }
  .btn-primary:disabled { opacity:0.5; cursor:not-allowed; transform:none; box-shadow:none; }
  .btn-secondary { background:var(--card); color:var(--text); border:1px solid var(--card-border); }

  .status-banner { background:var(--card); border:1px solid var(--accent); border-radius:12px; padding:16px 20px; margin-bottom:24px; display:none; }
  .status-banner.active { display:block; }
  .status-banner .status-header { display:flex; align-items:center; gap:10px; margin-bottom:8px; }
  .status-banner .spinner { width:18px; height:18px; border:2px solid var(--card-border); border-top-color:var(--accent); border-radius:50%; animation:spin 0.8s linear infinite; }
  @keyframes spin { to { transform:rotate(360deg); } }
  .status-banner .log-area { background:#080910; border-radius:8px; padding:12px; font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:var(--text-dim); max-height:200px; overflow-y:auto; white-space:pre-wrap; word-break:break-all; }

  .section-title { font-size:1.1rem; font-weight:700; color:var(--text-bright); margin-bottom:16px; }

  .reports-grid { display:grid; gap:12px; }
  .report-row { background:var(--card); border:1px solid var(--card-border); border-radius:12px; padding:16px 20px; display:flex; align-items:center; justify-content:space-between; transition:all 0.2s; cursor:pointer; }
  .report-row:hover { border-color:var(--accent); transform:translateX(4px); box-shadow:0 2px 12px rgba(99,102,241,0.1); }
  .report-info { display:flex; align-items:center; gap:16px; }
  .report-iter { background:linear-gradient(135deg,rgba(99,102,241,0.15),rgba(168,85,247,0.15)); color:#a5b4fc; font-weight:800; font-size:1rem; padding:8px 14px; border-radius:10px; min-width:50px; text-align:center; }
  .report-meta { }
  .report-meta .name { font-weight:600; color:var(--text-bright); font-size:0.9rem; }
  .report-meta .date { font-size:0.75rem; color:var(--text-dim); margin-top:2px; }
  .report-actions { display:flex; gap:8px; }
  .btn-view { background:rgba(99,102,241,0.1); color:#818cf8; border:1px solid rgba(99,102,241,0.2); padding:6px 16px; border-radius:8px; font-size:0.8rem; font-weight:600; cursor:pointer; transition:all 0.2s; text-decoration:none; }
  .btn-view:hover { background:rgba(99,102,241,0.2); }

  .empty-state { text-align:center; padding:60px 20px; color:var(--text-dim); }
  .empty-state .icon { font-size:3rem; margin-bottom:16px; opacity:0.3; }
  .empty-state p { font-size:0.9rem; }

  .runs-table { width:100%; border-collapse:collapse; font-size:0.8rem; }
  .runs-table th { text-align:left; color:var(--text-dim); font-weight:600; padding:8px 12px; border-bottom:1px solid var(--card-border); font-size:0.7rem; text-transform:uppercase; letter-spacing:0.5px; }
  .runs-table td { padding:8px 12px; border-bottom:1px solid #151825; }
  .status-badge { display:inline-block; padding:2px 8px; border-radius:6px; font-size:0.7rem; font-weight:600; }
  .status-completed { background:rgba(34,197,94,0.1); color:var(--green); }
  .status-running { background:rgba(99,102,241,0.1); color:var(--accent); }
  .status-failed { background:rgba(239,68,68,0.1); color:var(--red); }
</style>
</head>
<body>
<div class="topbar">
  <div>
    <h1>Predicto Dashboard</h1>
    <div class="subtitle">Multi-Agent NBA Prediction System</div>
  </div>
  <div style="font-size:0.75rem;color:var(--text-dim);">{{ report_count }} reports &middot; {{ run_count }} runs</div>
</div>

<div class="page">
  <!-- Pipeline Status Banner -->
  <div class="status-banner" id="statusBanner">
    <div class="status-header">
      <div class="spinner" id="statusSpinner"></div>
      <span style="font-weight:700;color:var(--text-bright);" id="statusText">Running new iteration...</span>
    </div>
    <div class="log-area" id="logArea"></div>
  </div>

  <!-- Actions -->
  <div class="actions">
    <button class="btn btn-primary" id="runBtn" onclick="startPipeline()">
      <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><polygon points="4,2 14,8 4,14"/></svg>
      Run New Iteration
    </button>
    <button class="btn btn-secondary" onclick="location.reload()">
      <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M1 7a6 6 0 1 1 1.5 3.9"/><path d="M1 11V7h4"/></svg>
      Refresh
    </button>
  </div>

  <!-- Reports List -->
  <div class="section-title">Reports</div>
  {% if reports %}
  <div class="reports-grid">
    {% for r in reports %}
    <div class="report-row" onclick="window.open('/report/{{ r.filename }}', '_blank')">
      <div class="report-info">
        <div class="report-iter">#{{ r.iteration }}</div>
        <div class="report-meta">
          <div class="name">Iteration {{ r.iteration }} Report</div>
          <div class="date">{{ r.modified }} &middot; {{ r.size_kb }} KB</div>
        </div>
      </div>
      <div class="report-actions">
        <a class="btn-view" href="/report/{{ r.filename }}" target="_blank" onclick="event.stopPropagation()">View Report</a>
      </div>
    </div>
    {% endfor %}
  </div>
  {% else %}
  <div class="empty-state">
    <div class="icon">📊</div>
    <p>No reports yet. Run your first iteration to generate a report.</p>
  </div>
  {% endif %}

  <!-- Run History -->
  {% if runs %}
  <div class="section-title" style="margin-top:40px;">Pipeline Run History</div>
  <div style="background:var(--card);border:1px solid var(--card-border);border-radius:12px;overflow:hidden;">
    <table class="runs-table">
      <thead><tr><th>Run ID</th><th>Started</th><th>Status</th><th>Summary</th></tr></thead>
      <tbody>
        {% for run in runs %}
        <tr>
          <td style="font-family:monospace;color:var(--accent);">{{ run.run_id }}</td>
          <td>{{ run.started }}</td>
          <td><span class="status-badge status-{{ run.status }}">{{ run.status }}</span></td>
          <td style="color:var(--text-dim);font-size:0.75rem;">{{ run.summary[:80] }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
  {% endif %}
</div>

<script>
let polling = null;

function startPipeline() {
  const btn = document.getElementById('runBtn');
  btn.disabled = true;
  btn.innerHTML = '<div class="spinner" style="width:14px;height:14px;border:2px solid rgba(255,255,255,0.3);border-top-color:white;border-radius:50%;animation:spin 0.8s linear infinite;"></div> Running...';

  const banner = document.getElementById('statusBanner');
  banner.classList.add('active');
  document.getElementById('logArea').textContent = 'Starting pipeline...\\n';

  fetch('/api/run', { method: 'POST' })
    .then(r => r.json())
    .then(data => {
      if (data.error) {
        alert(data.error);
        btn.disabled = false;
        btn.innerHTML = 'Run New Iteration';
        return;
      }
      polling = setInterval(pollStatus, 2000);
    });
}

function pollStatus() {
  fetch('/api/status')
    .then(r => r.json())
    .then(data => {
      const logArea = document.getElementById('logArea');
      logArea.textContent = data.log || '';
      logArea.scrollTop = logArea.scrollHeight;

      if (!data.running) {
        clearInterval(polling);
        const btn = document.getElementById('runBtn');
        btn.disabled = false;
        btn.innerHTML = '<svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><polygon points="4,2 14,8 4,14"/></svg> Run New Iteration';

        document.getElementById('statusSpinner').style.display = 'none';
        document.getElementById('statusText').textContent = data.error
          ? 'Pipeline failed: ' + data.error
          : 'Pipeline completed! Refreshing...';
        document.getElementById('statusText').style.color = data.error ? 'var(--red)' : 'var(--green)';

        if (!data.error) {
          setTimeout(() => location.reload(), 2000);
        }
      }
    });
}

// Check if pipeline is already running on page load
fetch('/api/status').then(r => r.json()).then(data => {
  if (data.running) {
    document.getElementById('runBtn').disabled = true;
    document.getElementById('runBtn').innerHTML = '<div class="spinner" style="width:14px;height:14px;border:2px solid rgba(255,255,255,0.3);border-top-color:white;border-radius:50%;animation:spin 0.8s linear infinite;"></div> Running...';
    document.getElementById('statusBanner').classList.add('active');
    document.getElementById('logArea').textContent = data.log || '';
    polling = setInterval(pollStatus, 2000);
  }
});
</script>
</body>
</html>"""


@app.route("/")
def dashboard():
    reports = _get_reports()
    runs = _get_run_history()
    return render_template_string(DASHBOARD_HTML, reports=reports, runs=runs,
                                  report_count=len(reports), run_count=len(runs))


@app.route("/report/<filename>")
def view_report(filename):
    path = REPORTS_DIR / filename
    if not path.exists() or not path.name.startswith("predicto_report_"):
        return "Report not found", 404
    return send_file(path)


@app.route("/api/run", methods=["POST"])
def start_run():
    global _pipeline_status
    with _pipeline_lock:
        if _pipeline_status["running"]:
            return jsonify({"error": "Pipeline is already running"})
        _pipeline_status = {"running": True, "started": datetime.now().isoformat(), "log": "", "finished": None, "error": None}

    thread = threading.Thread(target=_run_pipeline_thread, daemon=True)
    thread.start()
    return jsonify({"status": "started"})


@app.route("/api/status")
def pipeline_status():
    with _pipeline_lock:
        return jsonify(_pipeline_status)


@app.route("/api/reports")
def list_reports():
    return jsonify(_get_reports())


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  Predicto Dashboard")
    print("  http://localhost:5000")
    print("=" * 50 + "\n")
    app.run(debug=True, port=5000)
