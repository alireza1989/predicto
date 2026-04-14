"""NBA data fetching via nba_api and NBA CDN."""
from __future__ import annotations

import time
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from nba_api.stats.endpoints import leaguegamelog, leaguedashteamstats
from nba_api.stats.static import teams as nba_teams

logger = logging.getLogger(__name__)

NBA_CDN_SCHEDULE_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
CDN_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Referer": "https://www.nba.com/",
}
STATS_HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://stats.nba.com/",
    "Connection": "keep-alive",
}


def get_all_teams() -> pd.DataFrame:
    """Get all 30 NBA teams with IDs and metadata."""
    teams_list = nba_teams.get_teams()
    return pd.DataFrame(teams_list)


def fetch_season_games(
    season: str,
    season_type: str = "Regular Season",
    delay: float = 1.0,
) -> pd.DataFrame:
    """Fetch all team game logs for a season.

    Args:
        season: Season string like '2024-25'
        season_type: 'Regular Season' or 'Playoffs'
        delay: seconds to wait after request (rate limiting)

    Returns:
        DataFrame with one row per team per game (each game appears twice).
    """
    logger.info(f"Fetching game logs for {season} ({season_type})")
    log = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star=season_type,
        player_or_team_abbreviation="T",
        direction="ASC",
        sorter="DATE",
        headers=STATS_HEADERS,
        timeout=30,
    )
    time.sleep(delay)
    df = log.get_data_frames()[0]
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    logger.info(f"  Got {len(df)} team-game rows for {season}")
    return df


def fetch_multi_season_games(
    seasons: list[str],
    season_type: str = "Regular Season",
    delay: float = 1.0,
) -> pd.DataFrame:
    """Fetch game logs for multiple seasons."""
    all_dfs = []
    for season in seasons:
        try:
            df = fetch_season_games(season, season_type, delay)
            df["SEASON"] = season
            all_dfs.append(df)
        except Exception as e:
            logger.error(f"Failed to fetch {season}: {e}")
    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)


def fetch_team_advanced_stats(
    season: str,
    season_type: str = "Regular Season",
    delay: float = 1.0,
) -> pd.DataFrame:
    """Fetch advanced team stats (OFF_RATING, DEF_RATING, NET_RATING, PACE, etc.)."""
    logger.info(f"Fetching advanced team stats for {season}")
    stats = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        season_type_all_star=season_type,
        measure_type_detailed_defense="Advanced",
        per_mode_detailed="PerGame",
        headers=STATS_HEADERS,
        timeout=30,
    )
    time.sleep(delay)
    df = stats.get_data_frames()[0]
    logger.info(f"  Got stats for {len(df)} teams")
    return df


def fetch_upcoming_games(horizon_days: int = 14) -> pd.DataFrame:
    """Fetch upcoming NBA games from CDN schedule.

    Returns DataFrame with columns: game_id, date, home_team, away_team,
    home_team_id, away_team_id.
    """
    logger.info(f"Fetching NBA schedule (next {horizon_days} days)")
    resp = requests.get(NBA_CDN_SCHEDULE_URL, headers=CDN_HEADERS, timeout=30)
    resp.raise_for_status()
    schedule = resp.json()

    today = datetime.now().date()
    cutoff = today + timedelta(days=horizon_days)

    games = []
    for game_date_info in schedule.get("leagueSchedule", {}).get("gameDates", []):
        date_str = game_date_info.get("gameDate", "")
        # Parse date — format varies, try common ones
        try:
            game_date = datetime.strptime(date_str[:10], "%m/%d/%Y").date()
        except ValueError:
            try:
                game_date = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
            except ValueError:
                continue

        if game_date < today or game_date > cutoff:
            continue

        for game in game_date_info.get("games", []):
            home = game.get("homeTeam", {})
            away = game.get("awayTeam", {})
            games.append({
                "game_id": game.get("gameId", ""),
                "date": game_date.isoformat(),
                "home_team": home.get("teamTricode", ""),
                "away_team": away.get("teamTricode", ""),
                "home_team_id": home.get("teamId"),
                "away_team_id": away.get("teamId"),
                "home_team_name": home.get("teamName", ""),
                "away_team_name": away.get("teamName", ""),
                "status": game.get("gameStatusText", ""),
            })

    df = pd.DataFrame(games)
    logger.info(f"  Found {len(df)} upcoming games")
    return df


def games_to_matchups(games_df: pd.DataFrame) -> pd.DataFrame:
    """Convert team-level game logs (2 rows per game) to matchup rows (1 row per game).

    Returns DataFrame with: GAME_ID, GAME_DATE, HOME_TEAM, AWAY_TEAM,
    HOME_PTS, AWAY_PTS, HOME_WIN.
    """
    if games_df.empty:
        return pd.DataFrame()

    # Each game has two rows — identify home/away from MATCHUP column
    # Home games show "LAL vs. BOS", away shows "BOS @ LAL"
    home = games_df[games_df["MATCHUP"].str.contains(" vs. ")].copy()
    away = games_df[games_df["MATCHUP"].str.contains(" @ ")].copy()

    home = home.rename(columns={
        "TEAM_ABBREVIATION": "HOME_TEAM",
        "TEAM_ID": "HOME_TEAM_ID",
        "PTS": "HOME_PTS",
        "WL": "HOME_WL",
    })
    away = away.rename(columns={
        "TEAM_ABBREVIATION": "AWAY_TEAM",
        "TEAM_ID": "AWAY_TEAM_ID",
        "PTS": "AWAY_PTS",
        "WL": "AWAY_WL",
    })

    home_cols = ["GAME_ID", "GAME_DATE", "HOME_TEAM", "HOME_TEAM_ID", "HOME_PTS", "HOME_WL"]
    away_cols = ["GAME_ID", "AWAY_TEAM", "AWAY_TEAM_ID", "AWAY_PTS"]

    # Handle SEASON column if present
    if "SEASON" in home.columns:
        home_cols.append("SEASON")

    merged = home[home_cols].merge(away[away_cols], on="GAME_ID", how="inner")
    merged["HOME_WIN"] = (merged["HOME_WL"] == "W").astype(int)
    merged = merged.sort_values("GAME_DATE").reset_index(drop=True)
    return merged
