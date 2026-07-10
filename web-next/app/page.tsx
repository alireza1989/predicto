import LineChart from "@/components/LineChart";
import { dbConfigured, q, parseMetrics, type Experiment } from "@/lib/db";

export const revalidate = 300;

function SetupNotice() {
  return (
    <div className="notice">
      <strong>Database not connected.</strong> Set <code>DATABASE_URL</code> on
      this Vercel project (Settings → Environment Variables) to the Neon
      connection string, then redeploy.
    </div>
  );
}

export default async function Overview() {
  if (!dbConfigured) return (
    <>
      <h1>Overview</h1>
      <p className="sub">Autonomous multi-agent NBA prediction platform.</p>
      <SetupNotice />
    </>
  );

  const [experiments, runs, champion, perf, findings] = await Promise.all([
    q<Experiment>(
      `SELECT experiment_id, name, method, metrics_json, conclusion, status, created_at
       FROM experiment_log WHERE status = 'completed' ORDER BY created_at ASC`
    ),
    q<{ run_id: string; started_at: string; status: string; summary: string }>(
      `SELECT run_id, started_at, status, summary FROM run_log ORDER BY started_at DESC LIMIT 6`
    ),
    q<{ model_desc: string; log_loss: number; critic_verdict: string; promoted_at: string }>(
      `SELECT model_desc, log_loss, critic_verdict, promoted_at FROM promotions
       WHERE retired_at IS NULL ORDER BY promoted_at DESC LIMIT 1`
    ),
    q<{ n_preds: string; n_trades: string; pnl: string; avg_clv: string }>(
      `SELECT
         (SELECT COUNT(*) FROM predictions) AS n_preds,
         (SELECT COUNT(*) FROM paper_trades) AS n_trades,
         (SELECT COALESCE(SUM(pnl), 0) FROM paper_trades WHERE status = 'settled') AS pnl,
         (SELECT AVG(clv) FROM paper_trades WHERE clv IS NOT NULL) AS avg_clv`
    ),
    q<{ claim: string; confidence: string; source_agent: string; created_at: string }>(
      `SELECT claim, confidence, source_agent, created_at FROM findings
       WHERE status = 'active' ORDER BY created_at DESC LIMIT 6`
    ),
  ]);

  const withLoss = experiments
    .map((e) => ({ ...e, m: parseMetrics(e.metrics_json) }))
    .filter((e) => typeof e.m.log_loss === "number");

  const bestLoss = withLoss.length
    ? Math.min(...withLoss.map((e) => e.m.log_loss as number))
    : null;

  // Convergence: best-so-far log loss over experiment sequence
  let best = Infinity;
  const convergence = withLoss.map((e, i) => {
    best = Math.min(best, e.m.log_loss as number);
    return { x: `#${i + 1} ${String(e.created_at).slice(0, 10)}`, y: best };
  });

  const p = perf[0] ?? { n_preds: "0", n_trades: "0", pnl: "0", avg_clv: null };
  const champ = champion[0];

  return (
    <>
      <h1>Overview</h1>
      <p className="sub">
        Six-agent pipeline: data → features → meta-scientist → eval → critic →
        report. Metrics update after every run.
      </p>

      <div className="kpis">
        <div className="kpi">
          <div className="label">Best log loss</div>
          <div className="value">{bestLoss ? bestLoss.toFixed(4) : "—"}</div>
          <div className="hint">random = 0.6931</div>
        </div>
        <div className="kpi">
          <div className="label">Experiments</div>
          <div className="value">{withLoss.length}</div>
          <div className="hint">across {runs.length}+ runs</div>
        </div>
        <div className="kpi">
          <div className="label">Predictions logged</div>
          <div className="value">{p.n_preds}</div>
          <div className="hint">CLV ledger</div>
        </div>
        <div className="kpi">
          <div className="label">Paper trades</div>
          <div className="value">{p.n_trades}</div>
          <div className="hint">¼-Kelly sizing</div>
        </div>
        <div className="kpi">
          <div className="label">Paper PnL</div>
          <div
            className={`value ${Number(p.pnl) > 0 ? "good" : Number(p.pnl) < 0 ? "bad" : ""}`}
          >
            ${Number(p.pnl).toFixed(0)}
          </div>
          <div className="hint">settled trades</div>
        </div>
        <div className="kpi">
          <div className="label">Avg CLV</div>
          <div
            className={`value ${Number(p.avg_clv) > 0 ? "good" : ""}`}
          >
            {p.avg_clv != null ? `${(Number(p.avg_clv) * 100).toFixed(2)}%` : "—"}
          </div>
          <div className="hint">vs closing line</div>
        </div>
      </div>

      {champ && (
        <>
          <h2>Champion model</h2>
          <div className="card">
            <span className="badge champion">CHAMPION</span>{" "}
            <strong>{champ.model_desc}</strong> — log loss{" "}
            {Number(champ.log_loss).toFixed(4)}
            <div className="sub" style={{ margin: "6px 0 0" }}>
              Critic: {champ.critic_verdict} · promoted{" "}
              {String(champ.promoted_at).slice(0, 10)}
            </div>
          </div>
        </>
      )}

      <h2>Convergence — best log loss over experiments</h2>
      <div className="card">
        <LineChart
          points={convergence}
          yLabel="Best log loss (walk-forward CV)"
          baseline={0.6931}
          baselineLabel="random (0.693)"
        />
      </div>

      {findings.length > 0 && (
        <>
          <h2>Scientist findings — settled questions</h2>
          <div className="card">
            <table>
              <thead>
                <tr>
                  <th>Claim</th>
                  <th>Confidence</th>
                  <th>Agent</th>
                  <th>Date</th>
                </tr>
              </thead>
              <tbody>
                {findings.map((f, i) => (
                  <tr key={i}>
                    <td style={{ whiteSpace: "normal" }}>{f.claim}</td>
                    <td>
                      <span className="badge">{f.confidence}</span>
                    </td>
                    <td>{f.source_agent}</td>
                    <td>{String(f.created_at).slice(0, 10)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      <h2>Recent runs</h2>
      <div className="card">
        {runs.length === 0 ? (
          <div className="empty">No pipeline runs recorded yet.</div>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Run</th>
                <th>Started</th>
                <th>Status</th>
                <th>Summary</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((r) => (
                <tr key={r.run_id}>
                  <td>{r.run_id}</td>
                  <td>{String(r.started_at).slice(0, 16).replace("T", " ")}</td>
                  <td>
                    <span className="badge">{r.status}</span>
                  </td>
                  <td style={{ whiteSpace: "normal" }}>{r.summary ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </>
  );
}
