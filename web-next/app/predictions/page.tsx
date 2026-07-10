import { dbConfigured, q, type Prediction } from "@/lib/db";

export const revalidate = 300;

export default async function Predictions() {
  const preds = dbConfigured
    ? await q<Prediction & { home_win: number | null }>(
        `SELECT p.prediction_id, p.game_date, p.home_team, p.away_team,
                p.p_home, p.market_prob_at_pred, p.edge, p.model_desc,
                p.created_at, o.home_win
         FROM predictions p
         LEFT JOIN outcomes o
           ON o.game_date = p.game_date
          AND o.home_team = p.home_team
          AND o.away_team = p.away_team
         ORDER BY p.created_at DESC LIMIT 200`
      )
    : [];

  return (
    <>
      <h1>Predictions</h1>
      <p className="sub">
        Every prediction is recorded the moment it is made, with the market
        price at that moment — this ledger is what makes closing-line-value
        analysis possible. Result fills in after the game settles.
      </p>
      <div className="card">
        {preds.length === 0 ? (
          <div className="empty">
            No predictions logged yet — they appear after the next pipeline run
            during NBA season (markets must be live).
          </div>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Game</th>
                <th>Date</th>
                <th className="num">Model P(home)</th>
                <th className="num">Market</th>
                <th className="num">Edge</th>
                <th>Model</th>
                <th>Result</th>
              </tr>
            </thead>
            <tbody>
              {preds.map((p) => {
                const edge = p.edge != null ? Number(p.edge) : null;
                const correct =
                  p.home_win != null
                    ? (Number(p.p_home) >= 0.5) === (Number(p.home_win) === 1)
                    : null;
                return (
                  <tr key={p.prediction_id}>
                    <td>
                      {p.home_team} vs {p.away_team}
                    </td>
                    <td>{String(p.game_date).slice(0, 10)}</td>
                    <td className="num">{Number(p.p_home).toFixed(3)}</td>
                    <td className="num">
                      {p.market_prob_at_pred != null
                        ? Number(p.market_prob_at_pred).toFixed(3)
                        : "—"}
                    </td>
                    <td
                      className={`num ${edge != null && edge > 0 ? "pos" : edge != null && edge < 0 ? "neg" : ""}`}
                    >
                      {edge != null
                        ? `${edge > 0 ? "+" : ""}${(edge * 100).toFixed(1)}%`
                        : "—"}
                    </td>
                    <td>{p.model_desc ?? "—"}</td>
                    <td>
                      {p.home_win == null ? (
                        <span className="badge">open</span>
                      ) : correct ? (
                        <span className="badge" style={{ color: "var(--good-text)", borderColor: "var(--good-text)" }}>
                          ✓ correct
                        </span>
                      ) : (
                        <span className="badge" style={{ color: "var(--critical)", borderColor: "var(--critical)" }}>
                          ✗ wrong
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </>
  );
}
