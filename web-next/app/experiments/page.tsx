import { dbConfigured, q, parseMetrics, type Experiment } from "@/lib/db";

export const revalidate = 300;

export default async function Experiments() {
  const experiments = dbConfigured
    ? await q<Experiment>(
        `SELECT experiment_id, name, method, metrics_json, conclusion, status, created_at
         FROM experiment_log WHERE status = 'completed' ORDER BY created_at DESC LIMIT 200`
      )
    : [];

  const rows = experiments
    .map((e) => ({ ...e, m: parseMetrics(e.metrics_json) }))
    .filter((e) => typeof e.m.log_loss === "number")
    .sort((a, b) => (a.m.log_loss as number) - (b.m.log_loss as number));

  const bestId = rows[0]?.experiment_id;

  return (
    <>
      <h1>Experiments</h1>
      <p className="sub">
        Every experiment the Meta-Scientist has run, ranked by walk-forward CV
        log loss. Champion status requires the Critic&apos;s audit, not rank
        alone.
      </p>
      <div className="card">
        {rows.length === 0 ? (
          <div className="empty">No experiments recorded yet.</div>
        ) : (
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>Name</th>
                <th>Method</th>
                <th className="num">Log loss</th>
                <th className="num">Brier</th>
                <th className="num">Accuracy</th>
                <th>Date</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((e, i) => (
                <tr key={e.experiment_id}>
                  <td>{i + 1}</td>
                  <td>
                    {e.name}{" "}
                    {e.experiment_id === bestId && (
                      <span className="badge champion">BEST</span>
                    )}
                  </td>
                  <td>{e.method}</td>
                  <td className="num">
                    {(e.m.log_loss as number).toFixed(4)}
                  </td>
                  <td className="num">
                    {e.m.brier_score != null
                      ? Number(e.m.brier_score).toFixed(4)
                      : "—"}
                  </td>
                  <td className="num">
                    {e.m.accuracy != null
                      ? `${(Number(e.m.accuracy) * 100).toFixed(1)}%`
                      : "—"}
                  </td>
                  <td>{String(e.created_at).slice(0, 10)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </>
  );
}
