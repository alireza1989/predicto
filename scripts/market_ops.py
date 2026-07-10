#!/usr/bin/env python3
"""Deterministic market operations — no LLM calls, safe to run on a cron.

Usage:
    python scripts/market_ops.py snapshot   # capture current Polymarket odds
    python scripts/market_ops.py settle     # settle predictions/trades vs results
    python scripts/market_ops.py summary    # print live performance summary
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv(override=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

from tools import storage, market  # noqa: E402


def cmd_snapshot() -> dict:
    return market.snapshot_polymarket_odds()


def cmd_settle() -> dict:
    """Settle against the freshest game results.

    Tries a live fetch of the current season first (cheap: one API call);
    falls back to the latest locally saved matchups parquet.
    """
    results = None
    try:
        from tools import nba
        import yaml

        config_path = Path(__file__).parent.parent / "config.yaml"
        seasons = ["2025-26"]
        if config_path.exists():
            with open(config_path) as f:
                seasons = yaml.safe_load(f).get("seasons", seasons)
        games = nba.fetch_season_games(seasons[-1])
        if games is not None and not games.empty:
            results = nba.games_to_matchups(games)
    except Exception as e:
        logging.warning("Live results fetch failed (%s); using cached matchups", e)

    if results is None or results.empty:
        results = storage.load_latest_parquet("data/raw", "nba_matchups")
    if results.empty:
        return {"error": "no results data available"}

    conn = storage.init_db()
    out = market.settle_open_items(conn, results)
    conn.close()
    return out


def cmd_summary() -> dict:
    conn = storage.init_db()
    out = market.performance_summary(conn)
    conn.close()
    return out


def main():
    parser = argparse.ArgumentParser(description="Predicto market operations")
    parser.add_argument("command", choices=["snapshot", "settle", "summary"])
    args = parser.parse_args()
    result = {"snapshot": cmd_snapshot, "settle": cmd_settle, "summary": cmd_summary}[args.command]()
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
