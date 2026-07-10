import { dbConfigured, q, type PaperTrade } from "@/lib/db";

export const revalidate = 300;

export default async function Performance() {
  const [trades, agg] = dbConfigured
    ? await Promise.all([
        q<PaperTrade>(
          `SELECT t.trade_id, t.side, t.model_prob, t.odds_taken, t.stake,
                  t.clv, t.pnl, t.status, t.placed_at,
                  p.home_team, p.away_team, p.game_date
           FROM paper_trades t
           LEFT JOIN predictions p ON p.prediction_id = t.prediction_id
           ORDER BY t.placed_at DESC LIMIT 100`
        ),
        q<{
          settled: string;
          staked: string;
          pnl: string;
          avg_clv: string;
          pos_clv: string;
        }>(
          `SELECT COUNT(*) AS settled,
                  COALESCE(SUM(stake), 0) AS staked,
                  COALESCE(SUM(pnl), 0) AS pnl,
                  AVG(clv) AS avg_clv,
                  AVG(CASE WHEN clv > 0 THEN 1.0 ELSE 0.0 END) AS pos_clv
           FROM paper_trades WHERE status = 'settled'`
        ),
      ])
    : [[], []];

  const a = agg[0] ?? {
    settled: "0",
    staked: "0",
    pnl: "0",
    avg_clv: null,
    pos_clv: null,
  };
  const roi = Number(a.staked) > 0 ? Number(a.pnl) / Number(a.staked) : null;

  return (
    <>
      <h1>Performance</h1>
      <p className="sub">
        Paper trading only — no real bets. Sustained positive closing-line
        value (CLV) is the gold-standard evidence of genuine edge; PnL alone is
        noisy.
      </p>

      <div className="kpis">
        <div className="kpi">
          <div className="label">Settled trades</div>
          <div className="value">{a.settled}</div>
        </div>
        <div className="kpi">
          <div className="label">Total staked</div>
          <div className="value">${Number(a.staked).toFixed(0)}</div>
        </div>
        <div className="kpi">
          <div className="label">ROI</div>
          <div className={`value ${roi != null && roi > 0 ? "good" : roi != null && roi < 0 ? "bad" : ""}`}>
            {roi != null ? `${(roi * 100).toFixed(1)}%` : "—"}
          </div>
        </div>
        <div className="kpi">
          <div className="label">Avg CLV</div>
          <div className={`value ${Number(a.avg_clv) > 0 ? "good" : ""}`}>
            {a.avg_clv != null ? `${(Number(a.avg_clv) * 100).toFixed(2)}%` : "—"}
          </div>
          <div className="hint">&gt; +1% sustained = real edge</div>
        </div>
        <div className="kpi">
          <div className="label">Positive-CLV rate</div>
          <div className="value">
            {a.pos_clv != null ? `${(Number(a.pos_clv) * 100).toFixed(0)}%` : "—"}
          </div>
        </div>
      </div>

      <h2>Trades</h2>
      <div className="card">
        {trades.length === 0 ? (
          <div className="empty">
            No paper trades yet — trades open automatically when the model
            finds a ≥5% edge on a liquid market.
          </div>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Game</th>
                <th>Side</th>
                <th className="num">Model P</th>
                <th className="num">Price taken</th>
                <th className="num">Stake</th>
                <th className="num">CLV</th>
                <th className="num">PnL</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {trades.map((t) => (
                <tr key={t.trade_id}>
                  <td>
                    {t.home_team ?? "?"} vs {t.away_team ?? "?"}{" "}
                    <span className="badge">{String(t.game_date ?? "").slice(5, 10)}</span>
                  </td>
                  <td>{t.side}</td>
                  <td className="num">{Number(t.model_prob).toFixed(3)}</td>
                  <td className="num">{Number(t.odds_taken).toFixed(3)}</td>
                  <td className="num">${Number(t.stake).toFixed(0)}</td>
                  <td className={`num ${Number(t.clv) > 0 ? "pos" : Number(t.clv) < 0 ? "neg" : ""}`}>
                    {t.clv != null ? `${(Number(t.clv) * 100).toFixed(2)}%` : "—"}
                  </td>
                  <td className={`num ${Number(t.pnl) > 0 ? "pos" : Number(t.pnl) < 0 ? "neg" : ""}`}>
                    {t.pnl != null ? `$${Number(t.pnl).toFixed(2)}` : "—"}
                  </td>
                  <td>
                    <span className="badge">{t.status}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </>
  );
}
