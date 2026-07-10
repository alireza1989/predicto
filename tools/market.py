"""Market intelligence: odds snapshots, prediction ledger, paper trading, CLV.

Closing-line value (CLV) is the gold-standard evidence of real predictive
edge: if the odds we acted on consistently beat the closing odds, the model
is ahead of the market. Everything here is paper trading — no real bets.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import pandas as pd

from tools import storage
from tools import polymarket as pm

logger = logging.getLogger(__name__)

DEFAULT_BANKROLL = 1000.0
KELLY_MULTIPLIER = 0.25  # quarter Kelly until CLV proves the edge
MIN_EDGE = 0.05
MIN_LIQUIDITY = 100.0
MAX_STAKE_FRACTION = 0.05  # never risk >5% of bankroll on one game


def kelly_fraction_binary(p: float, price: float) -> float:
    """Full-Kelly bankroll fraction for buying a binary share at `price`
    (pays $1 if it resolves YES) when our true probability is `p`.

    b = (1-price)/price  (net odds); f* = (b*p - (1-p)) / b = p - (1-p)*price/(1-price)
    """
    if price <= 0 or price >= 1:
        return 0.0
    f = p - (1 - p) * price / (1 - price)
    return max(0.0, f)


def snapshot_polymarket_odds(conn=None) -> dict:
    """Capture current Polymarket NBA moneyline odds into odds_snapshots.

    Orientation (which team is home) comes from the latest upcoming-games
    schedule; unmatched markets are stored with team_a as listed and
    source marked 'polymarket:unoriented'.
    """
    own_conn = conn is None
    if own_conn:
        conn = storage.init_db()

    client = pm.PolymarketClient()
    markets = pm.fetch_nba_game_markets(client)
    if markets.empty:
        logger.info("No active Polymarket NBA game markets")
        if own_conn:
            conn.close()
        return {"snapshots": 0, "markets": 0}

    upcoming = storage.load_latest_parquet("data/raw", "upcoming_games")
    schedule = {}
    if not upcoming.empty:
        for _, g in upcoming.iterrows():
            home = pm._normalize_team(str(g["HOME_TEAM"]))
            away = pm._normalize_team(str(g["AWAY_TEAM"]))
            schedule[frozenset((home, away))] = (home, away, str(g["GAME_DATE"])[:10])

    count = 0
    for _, mkt in markets.iterrows():
        ta = pm._normalize_team(str(mkt["team_a"]))
        tb = pm._normalize_team(str(mkt["team_b"]))
        key = frozenset((ta, tb))
        game_time = str(mkt.get("game_time", ""))[:10]
        if key in schedule:
            home, away, game_date = schedule[key]
            home_prob = float(mkt["team_a_price"]) if home == ta else float(mkt["team_b_price"])
            source = "polymarket"
        else:
            home, away, game_date = ta, tb, game_time or datetime.now().strftime("%Y-%m-%d")
            home_prob = float(mkt["team_a_price"])
            source = "polymarket:unoriented"
        storage.log_odds_snapshot(
            conn, game_date=game_date, home_team=home, away_team=away,
            source=source, home_prob=home_prob,
            volume=float(mkt.get("volume", 0) or 0),
            liquidity=float(mkt.get("liquidity", 0) or 0),
        )
        count += 1

    if own_conn:
        conn.close()
    logger.info("Captured %d odds snapshots", count)
    return {"snapshots": count, "markets": len(markets)}


def record_predictions_and_trades(
    conn,
    edges_df: pd.DataFrame,
    *,
    run_id: Optional[str] = None,
    experiment_id: Optional[str] = None,
    model_desc: Optional[str] = None,
    bankroll: float = DEFAULT_BANKROLL,
) -> dict:
    """Write every matched prediction to the ledger; open ¼-Kelly paper
    trades on edges above MIN_EDGE with adequate liquidity."""
    n_preds, n_trades = 0, 0
    for _, row in edges_df.iterrows():
        game_date = str(row.get("GAME_DATE", "") or row.get("game_time", ""))[:10]
        pred_id = storage.log_prediction(
            conn,
            game_date=game_date,
            home_team=str(row["HOME_TEAM"]),
            away_team=str(row["AWAY_TEAM"]),
            p_home=float(row["model_prob"]),
            run_id=run_id,
            experiment_id=experiment_id,
            market_prob=float(row["market_prob"]),
            model_desc=model_desc,
        )
        n_preds += 1

        edge = float(row["edge"])
        liquidity = float(row.get("market_liquidity", 0) or 0)
        if abs(edge) < MIN_EDGE or liquidity < MIN_LIQUIDITY:
            continue

        if edge > 0:  # model likes HOME more than market → buy home at market price
            side, p, price = "HOME", float(row["model_prob"]), float(row["market_prob"])
        else:  # model likes AWAY → buy away share at (1 - home market price)
            side, p, price = "AWAY", 1 - float(row["model_prob"]), 1 - float(row["market_prob"])

        f = kelly_fraction_binary(p, price) * KELLY_MULTIPLIER
        f = min(f, MAX_STAKE_FRACTION)
        if f <= 0:
            continue
        storage.log_paper_trade(
            conn, prediction_id=pred_id, side=side, model_prob=p,
            odds_taken=price, kelly_fraction=f, stake=round(f * bankroll, 2),
        )
        n_trades += 1

    logger.info("Ledger: %d predictions, %d paper trades", n_preds, n_trades)
    return {"predictions": n_preds, "paper_trades": n_trades}


def _closing_prob(conn, game_date: str, home_team: str, away_team: str) -> Optional[float]:
    """Last oriented snapshot captured for the game = our closing line."""
    row = conn.execute(
        """SELECT home_prob FROM odds_snapshots
           WHERE game_date = ? AND home_team = ? AND away_team = ?
             AND source = 'polymarket'
           ORDER BY captured_at DESC LIMIT 1""",
        (game_date, home_team, away_team),
    ).fetchone()
    return float(row[0]) if row and row[0] is not None else None


def settle_open_items(conn, results_df: pd.DataFrame) -> dict:
    """Settle predictions and paper trades against final game results.

    results_df: matchup-level results with GAME_DATE, HOME_TEAM, AWAY_TEAM,
    HOME_WIN (and optionally HOME_PTS/AWAY_PTS) — the data agent's standard
    output. Records outcomes, fills closing_prob, computes CLV and PnL.
    """
    if results_df.empty:
        return {"outcomes": 0, "trades_settled": 0}

    results = results_df.copy()
    results["GAME_DATE"] = results["GAME_DATE"].astype(str).str[:10]
    results["HOME_TEAM"] = results["HOME_TEAM"].map(lambda t: pm._normalize_team(str(t)))
    results["AWAY_TEAM"] = results["AWAY_TEAM"].map(lambda t: pm._normalize_team(str(t)))
    result_map = {
        (r["GAME_DATE"], r["HOME_TEAM"], r["AWAY_TEAM"]): r
        for _, r in results.iterrows()
    }

    # 1. Record outcomes for any game we predicted that now has a result
    pred_games = conn.execute(
        """SELECT DISTINCT p.game_date, p.home_team, p.away_team
           FROM predictions p
           LEFT JOIN outcomes o ON o.game_date = p.game_date
             AND o.home_team = p.home_team AND o.away_team = p.away_team
           WHERE o.game_date IS NULL""",
    ).fetchall()

    n_outcomes = 0
    for game_date, home, away in pred_games:
        key = (str(game_date)[:10], home, away)
        if key not in result_map:
            continue
        r = result_map[key]
        closing = _closing_prob(conn, *key)
        storage.record_outcome(
            conn, game_date=key[0], home_team=home, away_team=away,
            home_win=int(r["HOME_WIN"]),
            home_pts=int(r["HOME_PTS"]) if "HOME_PTS" in r and pd.notna(r.get("HOME_PTS")) else None,
            away_pts=int(r["AWAY_PTS"]) if "AWAY_PTS" in r and pd.notna(r.get("AWAY_PTS")) else None,
            closing_prob=closing,
        )
        n_outcomes += 1

    # 2. Settle open paper trades whose game now has an outcome
    open_trades = conn.execute(
        """SELECT t.trade_id, t.side, t.odds_taken, t.stake,
                  p.game_date, p.home_team, p.away_team
           FROM paper_trades t JOIN predictions p
             ON p.prediction_id = t.prediction_id
           WHERE t.status = 'open'""",
    ).fetchall()

    n_settled = 0
    for trade_id, side, odds_taken, stake, game_date, home, away in open_trades:
        out = conn.execute(
            """SELECT home_win, closing_prob FROM outcomes
               WHERE game_date = ? AND home_team = ? AND away_team = ?""",
            (str(game_date)[:10], home, away),
        ).fetchone()
        if out is None or out[0] is None:
            continue
        home_win, closing_home = int(out[0]), out[1]
        won = (home_win == 1) if side == "HOME" else (home_win == 0)
        # Bought shares at odds_taken paying $1: win pays stake*(1-q)/q, loss = -stake
        pnl = stake * (1 - odds_taken) / odds_taken if won else -stake
        closing_side = None
        if closing_home is not None:
            closing_side = closing_home if side == "HOME" else 1 - float(closing_home)
        clv = (closing_side - odds_taken) if closing_side is not None else None
        conn.execute(
            """UPDATE paper_trades
               SET status = 'settled', pnl = ?, clv = ?, closing_prob = ?, settled_at = ?
               WHERE trade_id = ?""",
            (round(pnl, 2), round(clv, 4) if clv is not None else None,
             closing_side, datetime.now().isoformat(), trade_id),
        )
        n_settled += 1
    conn.commit()

    logger.info("Settled %d outcomes, %d paper trades", n_outcomes, n_settled)
    return {"outcomes": n_outcomes, "trades_settled": n_settled}


def performance_summary(conn) -> dict:
    """Aggregate live performance: CLV, paper-trading PnL, prediction accuracy."""
    def one(sql, params=()):
        row = conn.execute(sql, params).fetchone()
        return row if row else (None,)

    n_preds = one("SELECT COUNT(*) FROM predictions")[0]
    n_open = one("SELECT COUNT(*) FROM paper_trades WHERE status = 'open'")[0]
    settled = conn.execute(
        "SELECT pnl, clv, stake FROM paper_trades WHERE status = 'settled'"
    ).fetchall()
    pnl = sum(r[0] for r in settled if r[0] is not None)
    staked = sum(r[2] for r in settled if r[2] is not None)
    clvs = [r[1] for r in settled if r[1] is not None]

    acc = conn.execute(
        """SELECT p.p_home, o.home_win FROM predictions p
           JOIN outcomes o ON o.game_date = p.game_date
             AND o.home_team = p.home_team AND o.away_team = p.away_team
           WHERE o.home_win IS NOT NULL""",
    ).fetchall()
    n_scored = len(acc)
    n_correct = sum(1 for p, w in acc if (p >= 0.5) == (int(w) == 1))

    return {
        "predictions_total": n_preds,
        "predictions_scored": n_scored,
        "live_accuracy": round(n_correct / n_scored, 4) if n_scored else None,
        "trades_settled": len(settled),
        "trades_open": n_open,
        "total_staked": round(staked, 2),
        "total_pnl": round(pnl, 2),
        "roi": round(pnl / staked, 4) if staked else None,
        "avg_clv": round(sum(clvs) / len(clvs), 4) if clvs else None,
        "positive_clv_rate": round(sum(1 for c in clvs if c > 0) / len(clvs), 4) if clvs else None,
    }
