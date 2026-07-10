"""Data Collection Agent: fetches NBA and Polymarket data."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from agents.base import Agent, Tool
from tools import nba, polymarket, storage

logger = logging.getLogger(__name__)


def _build_tools(config: dict) -> list[Tool]:
    """Build the tool set for the Data Agent."""
    poly_client = polymarket.PolymarketClient(
        request_delay=config.get("polymarket", {}).get("request_delay", 0.5)
    )

    def fetch_nba_games(seasons: list[str]) -> str:
        """Fetch NBA game results for specified seasons."""
        delay = config.get("nba", {}).get("request_delay", 1.0)
        games_df = nba.fetch_multi_season_games(seasons, delay=delay)
        if games_df.empty:
            return json.dumps({"status": "error", "message": "No games fetched"})

        matchups = nba.games_to_matchups(games_df)
        path = storage.save_parquet(matchups, "data/raw/nba_matchups.parquet")
        return json.dumps({
            "status": "success",
            "total_games": len(matchups),
            "seasons": seasons,
            "date_range": [str(matchups["GAME_DATE"].min()), str(matchups["GAME_DATE"].max())],
            "saved_to": str(path),
        })

    def fetch_upcoming_nba_games(horizon_days: int = 14) -> str:
        """Fetch upcoming NBA games from schedule."""
        upcoming = nba.fetch_upcoming_games(horizon_days)
        if upcoming.empty:
            return json.dumps({"status": "no_upcoming_games", "horizon_days": horizon_days})

        path = storage.save_parquet(upcoming, "data/raw/upcoming_games.parquet")
        return json.dumps({
            "status": "success",
            "games_found": len(upcoming),
            "dates": upcoming["date"].unique().tolist()[:10],
            "saved_to": str(path),
        })

    def fetch_team_stats(season: str) -> str:
        """Fetch advanced team stats for a season."""
        delay = config.get("nba", {}).get("request_delay", 1.0)
        stats = nba.fetch_team_advanced_stats(season, delay=delay)
        if stats.empty:
            return json.dumps({"status": "error", "message": "No stats fetched"})

        path = storage.save_parquet(stats, f"data/raw/team_stats_{season}.parquet")
        return json.dumps({
            "status": "success",
            "teams": len(stats),
            "columns": stats.columns.tolist(),
            "saved_to": str(path),
        })

    def search_polymarket_nba() -> str:
        """Search for NBA-related markets on Polymarket."""
        markets_df = polymarket.find_nba_markets(poly_client)
        if markets_df.empty:
            return json.dumps({"status": "no_markets_found"})

        path = storage.save_parquet(markets_df, "data/raw/polymarket_nba.parquet")
        return json.dumps({
            "status": "success",
            "markets_found": len(markets_df),
            "sample_questions": markets_df["question"].head(5).tolist(),
            "total_volume": float(markets_df["volume"].sum()),
            "saved_to": str(path),
        })

    def enrich_market_orderbooks() -> str:
        """Add orderbook data (spread, depth) to saved Polymarket markets."""
        markets_df = storage.load_latest_parquet("data/raw", "polymarket_nba")
        if markets_df.empty:
            return json.dumps({"status": "error", "message": "No markets data found"})

        enriched = []
        for _, row in markets_df.iterrows():
            market_dict = row.to_dict()
            enriched_market = polymarket.enrich_market_with_orderbook(poly_client, market_dict)
            enriched.append(enriched_market)

        enriched_df = pd.DataFrame(enriched)
        path = storage.save_parquet(enriched_df, "data/raw/polymarket_nba_enriched.parquet")
        return json.dumps({
            "status": "success",
            "markets_enriched": len(enriched_df),
            "saved_to": str(path),
        })

    def fetch_polymarket_game_markets() -> str:
        """Fetch NBA game-level moneyline markets from Polymarket (team vs team with odds)."""
        markets_df = polymarket.fetch_nba_game_markets(poly_client)
        if markets_df.empty:
            return json.dumps({"status": "no_game_markets", "message": "No active NBA game markets found on Polymarket"})

        path = storage.save_parquet(markets_df, "data/raw/polymarket_game_markets.parquet")
        return json.dumps({
            "status": "success",
            "games_found": len(markets_df),
            "games": [
                {"matchup": row["event_title"], "prices": [row["team_a_price"], row["team_b_price"]], "volume": row["volume"]}
                for _, row in markets_df.head(15).iterrows()
            ],
            "total_volume": float(markets_df["volume"].sum()),
            "saved_to": str(path),
        })

    def fetch_player_logs(seasons: list[str]) -> str:
        """Fetch per-player per-game stats for the specified seasons.

        Returns game score, points, rebounds, assists, plus/minus and minutes
        for every player in every game. Used by the Feature Agent to compute
        roster-strength features.
        """
        delay = config.get("nba", {}).get("request_delay", 1.5)
        logs_df = nba.fetch_player_game_logs(seasons, delay=delay)
        if logs_df.empty:
            return json.dumps({"status": "error", "message": "No player logs fetched"})

        path = storage.save_parquet(logs_df, "data/raw/player_game_logs.parquet")
        # Summary per season
        season_counts = logs_df.groupby("SEASON").size().to_dict()
        players_count = logs_df["PLAYER_ID"].nunique()
        return json.dumps({
            "status": "success",
            "total_rows": len(logs_df),
            "unique_players": players_count,
            "seasons": season_counts,
            "columns": [c for c in logs_df.columns if c not in ["SEASON_ID", "VIDEO_AVAILABLE"]],
            "sample": logs_df[["PLAYER_NAME", "TEAM_ABBREVIATION", "GAME_DATE", "MIN", "PTS", "REB", "AST", "PLUS_MINUS"]].head(5).to_dict(orient="records"),
            "saved_to": str(path),
        }, default=str)

    def get_all_nba_teams() -> str:
        """Get list of all NBA teams with IDs."""
        teams = nba.get_all_teams()
        return teams.to_json(orient="records")

    return [
        Tool(
            name="fetch_nba_games",
            description="Fetch historical NBA game results for specified seasons. Returns matchup data with scores and outcomes.",
            input_schema={
                "type": "object",
                "properties": {
                    "seasons": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of seasons like ['2023-24', '2024-25']"
                    }
                },
                "required": ["seasons"]
            },
            func=fetch_nba_games,
        ),
        Tool(
            name="fetch_upcoming_nba_games",
            description="Fetch upcoming scheduled NBA games within a horizon.",
            input_schema={
                "type": "object",
                "properties": {
                    "horizon_days": {
                        "type": "integer",
                        "description": "Number of days ahead to look (default 14)"
                    }
                },
            },
            func=fetch_upcoming_nba_games,
        ),
        Tool(
            name="fetch_team_stats",
            description="Fetch advanced team stats (OFF_RATING, DEF_RATING, PACE, NET_RATING) for a season.",
            input_schema={
                "type": "object",
                "properties": {
                    "season": {"type": "string", "description": "Season like '2024-25'"}
                },
                "required": ["season"]
            },
            func=fetch_team_stats,
        ),
        Tool(
            name="search_polymarket_nba",
            description="Search Polymarket for active NBA/basketball markets. Returns market details with prices.",
            input_schema={"type": "object", "properties": {}},
            func=search_polymarket_nba,
        ),
        Tool(
            name="enrich_market_orderbooks",
            description="Add orderbook depth and spread data to previously fetched Polymarket markets.",
            input_schema={"type": "object", "properties": {}},
            func=enrich_market_orderbooks,
        ),
        Tool(
            name="fetch_polymarket_game_markets",
            description="Fetch active NBA game-level moneyline markets from Polymarket. Returns team matchups with win probabilities (odds) and volume.",
            input_schema={"type": "object", "properties": {}},
            func=fetch_polymarket_game_markets,
        ),
        Tool(
            name="fetch_player_logs",
            description="Fetch per-player per-game stats (PTS, REB, AST, STL, BLK, TOV, PLUS_MINUS, MIN) for specified seasons. IMPORTANT: call this after fetch_nba_games so the Feature Agent can compute player-strength features.",
            input_schema={
                "type": "object",
                "properties": {
                    "seasons": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Seasons to fetch, e.g. ['2022-23', '2023-24', '2024-25']"
                    }
                },
                "required": ["seasons"]
            },
            func=fetch_player_logs,
        ),
        Tool(
            name="get_all_nba_teams",
            description="Get list of all 30 NBA teams with IDs and metadata.",
            input_schema={"type": "object", "properties": {}},
            func=get_all_nba_teams,
        ),
    ]


SYSTEM_PROMPT = """\
You are the Data Collection Agent for Predicto, an NBA prediction system.

Your job is to collect and store raw data from two sources:
1. **NBA game data** via nba_api (historical results, team stats, upcoming schedule)
2. **Polymarket** prediction market data (NBA markets, prices, orderbooks)

## Your workflow:
1. Fetch historical NBA game results for the requested seasons (fetch_nba_games)
2. Fetch player game logs for the SAME seasons (fetch_player_logs) — critical for roster-strength features
3. Fetch upcoming NBA games (fetch_upcoming_nba_games)
4. Fetch advanced team stats for the current/recent season (fetch_team_stats)
5. Fetch Polymarket GAME-LEVEL markets (fetch_polymarket_game_markets) — actual moneyline odds
6. Optionally search broader Polymarket NBA markets and enrich with orderbooks

## Important rules:
- Always fetch data in the order above (games first, then markets)
- Report what you fetched: number of games, date ranges, number of markets found
- If a data source fails, report the error but continue with other sources
- NBA API has rate limits — the tools handle delays automatically
- Polymarket may have few or no NBA markets depending on the season

After collecting all data, provide a summary of what was fetched and any issues encountered.
"""


def create_data_agent(config: dict) -> Agent:
    """Create and return the Data Collection Agent."""
    return Agent(
        name="data_agent",
        system_prompt=SYSTEM_PROMPT,
        tools=_build_tools(config),
        model=config.get("models", {}).get("agent_model", "claude-sonnet-4-20250514"),
        max_tokens=config.get("models", {}).get("max_tokens", 4096),
    )
