"""Feature engineering for NBA game prediction."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default Elo parameters
ELO_K = 20
ELO_HOME_ADVANTAGE = 100
ELO_INITIAL = 1500
ELO_SEASON_REVERT = 0.75  # revert 25% toward mean between seasons


def compute_elo_ratings(matchups: pd.DataFrame) -> pd.DataFrame:
    """Compute running Elo ratings from historical matchups.

    Args:
        matchups: DataFrame with GAME_DATE, HOME_TEAM, AWAY_TEAM, HOME_WIN, SEASON columns.

    Returns:
        DataFrame with Elo ratings before each game for both teams.
    """
    matchups = matchups.sort_values("GAME_DATE").reset_index(drop=True)
    ratings = {}  # team -> current elo
    records = []
    prev_season = None

    for _, row in matchups.iterrows():
        home = row["HOME_TEAM"]
        away = row["AWAY_TEAM"]
        season = row.get("SEASON", "")

        # Season reversion
        if season != prev_season and prev_season is not None:
            for team in ratings:
                ratings[team] = ELO_INITIAL + ELO_SEASON_REVERT * (ratings[team] - ELO_INITIAL)
            prev_season = season
        elif prev_season is None:
            prev_season = season

        home_elo = ratings.get(home, ELO_INITIAL)
        away_elo = ratings.get(away, ELO_INITIAL)

        # Expected scores (with home advantage)
        exp_home = 1.0 / (1.0 + 10 ** ((away_elo - home_elo - ELO_HOME_ADVANTAGE) / 400))

        records.append({
            "GAME_ID": row["GAME_ID"],
            "GAME_DATE": row["GAME_DATE"],
            "HOME_TEAM": home,
            "AWAY_TEAM": away,
            "home_elo_pre": home_elo,
            "away_elo_pre": away_elo,
            "home_elo_expected": exp_home,
            "HOME_WIN": row["HOME_WIN"],
        })

        # Update ratings
        actual = row["HOME_WIN"]
        ratings[home] = home_elo + ELO_K * (actual - exp_home)
        ratings[away] = away_elo + ELO_K * ((1 - actual) - (1 - exp_home))

    return pd.DataFrame(records)


def compute_rolling_stats(
    matchups: pd.DataFrame,
    windows: list[int] = [5, 10],
) -> pd.DataFrame:
    """Compute rolling team performance stats.

    For each game, computes rolling win%, points scored/allowed for both home and away teams.
    All stats are computed BEFORE the game (no leakage).
    """
    matchups = matchups.sort_values("GAME_DATE").reset_index(drop=True)

    # Build per-team game history
    team_games = {}  # team -> list of (date, pts_scored, pts_allowed, win)

    records = []

    for _, row in matchups.iterrows():
        home = row["HOME_TEAM"]
        away = row["AWAY_TEAM"]

        features = {"GAME_ID": row["GAME_ID"]}

        for team, prefix, pts_for, pts_against in [
            (home, "home", "HOME_PTS", "AWAY_PTS"),
            (away, "away", "AWAY_PTS", "HOME_PTS"),
        ]:
            history = team_games.get(team, [])

            for w in windows:
                recent = history[-w:] if len(history) >= w else history
                if recent:
                    features[f"{prefix}_win_pct_{w}"] = np.mean([g[3] for g in recent])
                    features[f"{prefix}_pts_scored_{w}"] = np.mean([g[1] for g in recent])
                    features[f"{prefix}_pts_allowed_{w}"] = np.mean([g[2] for g in recent])
                    features[f"{prefix}_net_pts_{w}"] = np.mean([g[1] - g[2] for g in recent])
                else:
                    features[f"{prefix}_win_pct_{w}"] = 0.5
                    features[f"{prefix}_pts_scored_{w}"] = 110.0  # league avg approx
                    features[f"{prefix}_pts_allowed_{w}"] = 110.0
                    features[f"{prefix}_net_pts_{w}"] = 0.0

        records.append(features)

        # Update histories AFTER computing features (no leakage)
        home_pts = row.get("HOME_PTS", 0) or 0
        away_pts = row.get("AWAY_PTS", 0) or 0
        team_games.setdefault(home, []).append(
            (row["GAME_DATE"], home_pts, away_pts, row["HOME_WIN"])
        )
        team_games.setdefault(away, []).append(
            (row["GAME_DATE"], away_pts, home_pts, 1 - row["HOME_WIN"])
        )

    return pd.DataFrame(records)


def compute_rest_days(matchups: pd.DataFrame) -> pd.DataFrame:
    """Compute rest days for each team before each game.

    Returns DataFrame with GAME_ID, home_rest_days, away_rest_days,
    home_is_b2b (back-to-back), away_is_b2b.
    """
    matchups = matchups.sort_values("GAME_DATE").reset_index(drop=True)
    last_game = {}  # team -> last game date

    records = []
    for _, row in matchups.iterrows():
        home = row["HOME_TEAM"]
        away = row["AWAY_TEAM"]
        game_date = pd.to_datetime(row["GAME_DATE"])

        home_rest = (game_date - last_game[home]).days if home in last_game else 3
        away_rest = (game_date - last_game[away]).days if away in last_game else 3

        records.append({
            "GAME_ID": row["GAME_ID"],
            "home_rest_days": min(home_rest, 7),  # cap at 7
            "away_rest_days": min(away_rest, 7),
            "home_is_b2b": int(home_rest <= 1),
            "away_is_b2b": int(away_rest <= 1),
            "rest_advantage": home_rest - away_rest,
        })

        last_game[home] = game_date
        last_game[away] = game_date

    return pd.DataFrame(records)


def compute_player_strength_features(
    matchups: pd.DataFrame,
    player_logs: pd.DataFrame,
    window: int = 10,
) -> pd.DataFrame:
    """Compute roster-strength features from historical player game logs.

    For each game, looks at the last `window` games each team's players
    appeared in (BEFORE this game — no leakage) and computes:
      - Roster strength score  (weighted sum of player efficiency)
      - Star player form       (top scorer's rolling avg)
      - Roster depth           (# of players with meaningful minutes)
      - Star dependency        (% of team pts from top player)

    All features computed with only past data. Missing player logs → 0 (neutral).

    Args:
        matchups: Game matchup DataFrame (needs GAME_ID, GAME_DATE, HOME_TEAM, AWAY_TEAM)
        player_logs: Player game log DataFrame from fetch_player_game_logs()
        window: Rolling window in games
    """
    if player_logs.empty:
        logger.warning("No player logs provided — skipping player strength features")
        return pd.DataFrame({"GAME_ID": matchups["GAME_ID"]})

    matchups = matchups.sort_values("GAME_DATE").reset_index(drop=True)

    # Ensure date types match
    player_logs = player_logs.copy()
    player_logs["GAME_DATE"] = pd.to_datetime(player_logs["GAME_DATE"])

    # Game Score (John Hollinger) — compact single-number player quality metric
    # GmSc = PTS + 0.4*FGM - 0.7*FGA - 0.4*(FTA-FTM) + 0.7*OREB + 0.3*DREB
    #        + STL + 0.7*AST + 0.7*BLK - 0.4*PF - TOV
    def game_score(row):
        return (
            row.get("PTS", 0) + 0.4 * row.get("FGM", 0)
            - 0.7 * row.get("FGA", 0)
            - 0.4 * (row.get("FTA", 0) - row.get("FTM", 0))
            + 0.7 * row.get("OREB", 0) + 0.3 * row.get("DREB", 0)
            + row.get("STL", 0) + 0.7 * row.get("AST", 0)
            + 0.7 * row.get("BLK", 0) - 0.4 * row.get("PF", 0)
            - row.get("TOV", 0)
        )

    player_logs["GAME_SCORE"] = player_logs.apply(game_score, axis=1)

    # Build lookup: (team_abbrev, date) → player stats for that game
    # Sort logs by date so we can efficiently query "all logs before date X for team T"
    player_logs_sorted = player_logs.sort_values("GAME_DATE").reset_index(drop=True)

    # Pre-build per-team history as a dict of lists (chronological)
    # team_history[abbrev] = list of (game_date, player_id, pts, game_score, min_float, plus_minus)
    team_history: dict = {}
    for _, row in player_logs_sorted.iterrows():
        team = row["TEAM_ABBREVIATION"]
        if team not in team_history:
            team_history[team] = []
        team_history[team].append({
            "date": row["GAME_DATE"],
            "player_id": row["PLAYER_ID"],
            "pts": float(row.get("PTS", 0) or 0),
            "game_score": float(row.get("GAME_SCORE", 0) or 0),
            "min": float(row.get("MIN_FLOAT", 0) or 0),
            "plus_minus": float(row.get("PLUS_MINUS", 0) or 0),
        })

    def team_features_before(team: str, before_date, prefix: str) -> dict:
        """Compute roster strength features for a team from games before a date."""
        feats = {
            f"{prefix}_roster_gmscore": 0.0,
            f"{prefix}_star_gmscore": 0.0,
            f"{prefix}_roster_depth": 0.0,
            f"{prefix}_star_dependency": 0.0,
            f"{prefix}_roster_plus_minus": 0.0,
        }
        if team not in team_history:
            return feats

        # Get all entries before this game date
        past = [e for e in team_history[team] if e["date"] < before_date]
        if not past:
            return feats

        # Get the last `window` unique game dates
        unique_dates = sorted(set(e["date"] for e in past), reverse=True)[:window]
        recent = [e for e in past if e["date"] in set(unique_dates)]

        # Per-player aggregation over recent window
        from collections import defaultdict
        player_gms = defaultdict(list)
        player_pm = defaultdict(list)
        player_pts = defaultdict(list)
        player_min = defaultdict(list)
        for e in recent:
            player_gms[e["player_id"]].append(e["game_score"])
            player_pm[e["player_id"]].append(e["plus_minus"])
            player_pts[e["player_id"]].append(e["pts"])
            player_min[e["player_id"]].append(e["min"])

        # Only count players who averaged meaningful minutes (≥10 avg)
        active_players = {
            pid: {
                "avg_gmscore": float(np.mean(player_gms[pid])),
                "avg_plus_minus": float(np.mean(player_pm[pid])),
                "avg_pts": float(np.mean(player_pts[pid])),
                "avg_min": float(np.mean(player_min[pid])),
            }
            for pid in player_gms
            if np.mean(player_min[pid]) >= 10
        }

        if not active_players:
            return feats

        gm_scores = [p["avg_gmscore"] for p in active_players.values()]
        pts_scores = [p["avg_pts"] for p in active_players.values()]
        pm_scores  = [p["avg_plus_minus"] for p in active_players.values()]

        total_gmscore = sum(gm_scores)
        star_gmscore  = max(gm_scores) if gm_scores else 0
        star_pts      = max(pts_scores) if pts_scores else 0
        total_pts     = sum(pts_scores) if pts_scores else 1
        depth         = len(active_players)  # # of players with ≥10 min
        avg_pm        = float(np.mean(pm_scores)) if pm_scores else 0

        feats[f"{prefix}_roster_gmscore"]    = round(total_gmscore, 3)
        feats[f"{prefix}_star_gmscore"]      = round(star_gmscore, 3)
        feats[f"{prefix}_roster_depth"]      = float(depth)
        feats[f"{prefix}_star_dependency"]   = round(star_pts / max(total_pts, 1), 3)
        feats[f"{prefix}_roster_plus_minus"] = round(avg_pm, 3)
        return feats

    records = []
    for _, row in matchups.iterrows():
        game_date = pd.to_datetime(row["GAME_DATE"])
        home = row["HOME_TEAM"]
        away = row["AWAY_TEAM"]

        home_feats = team_features_before(home, game_date, "home")
        away_feats = team_features_before(away, game_date, "away")

        combined = {"GAME_ID": row["GAME_ID"]}
        combined.update(home_feats)
        combined.update(away_feats)

        # Differentials — most predictive for model
        combined["player_gmscore_diff"]    = round(home_feats["home_roster_gmscore"] - away_feats["away_roster_gmscore"], 3)
        combined["player_star_diff"]       = round(home_feats["home_star_gmscore"] - away_feats["away_star_gmscore"], 3)
        combined["player_depth_diff"]      = home_feats["home_roster_depth"] - away_feats["away_roster_depth"]
        combined["player_plus_minus_diff"] = round(home_feats["home_roster_plus_minus"] - away_feats["away_roster_plus_minus"], 3)

        records.append(combined)

    return pd.DataFrame(records)


def compute_momentum_features(matchups: pd.DataFrame) -> pd.DataFrame:
    """Compute win streak, momentum, and trend features.

    For each game, computes current win/loss streak length, momentum (weighted recent form),
    and home/away specific performance. All computed BEFORE the game (no leakage).
    """
    matchups = matchups.sort_values("GAME_DATE").reset_index(drop=True)

    # Per-team trackers
    team_streak = {}      # team -> current streak (positive=wins, negative=losses)
    team_home_record = {}  # team -> list of home wins/losses (1/0)
    team_away_record = {}  # team -> list of away wins/losses (1/0)

    records = []

    for _, row in matchups.iterrows():
        home = row["HOME_TEAM"]
        away = row["AWAY_TEAM"]

        # Current streaks BEFORE this game
        h_streak = team_streak.get(home, 0)
        a_streak = team_streak.get(away, 0)

        # Home/away specific win rates (last 15 games in that context)
        h_home_games = team_home_record.get(home, [])
        a_away_games = team_away_record.get(away, [])
        h_home_wpct = np.mean(h_home_games[-15:]) if h_home_games else 0.55  # league avg home
        a_away_wpct = np.mean(a_away_games[-15:]) if a_away_games else 0.45  # league avg away

        records.append({
            "GAME_ID": row["GAME_ID"],
            "home_streak": h_streak,
            "away_streak": a_streak,
            "streak_diff": h_streak - a_streak,
            "home_home_wpct": h_home_wpct,
            "away_away_wpct": a_away_wpct,
            "home_away_split": h_home_wpct - a_away_wpct,
        })

        # Update streaks AFTER (no leakage)
        home_won = row["HOME_WIN"]
        if home_won:
            team_streak[home] = max(h_streak, 0) + 1
            team_streak[away] = min(a_streak, 0) - 1
        else:
            team_streak[home] = min(h_streak, 0) - 1
            team_streak[away] = max(a_streak, 0) + 1

        # Update home/away records
        team_home_record.setdefault(home, []).append(home_won)
        team_away_record.setdefault(away, []).append(1 - home_won)

    return pd.DataFrame(records)


def compute_h2h_features(matchups: pd.DataFrame) -> pd.DataFrame:
    """Compute head-to-head matchup history features.

    Tracks historical record between each pair of teams.
    All computed BEFORE the game (no leakage).
    """
    matchups = matchups.sort_values("GAME_DATE").reset_index(drop=True)

    # Track H2H results: (team_a, team_b) -> list of wins for team_a
    h2h_record = {}

    records = []

    for _, row in matchups.iterrows():
        home = row["HOME_TEAM"]
        away = row["AWAY_TEAM"]

        # Canonical key (alphabetical) so we always look up the same pair
        key = tuple(sorted([home, away]))
        history = h2h_record.get(key, [])

        if history:
            # Calculate home team's H2H win rate
            if key[0] == home:
                h2h_home_wins = sum(history)
            else:
                h2h_home_wins = len(history) - sum(history)
            h2h_wpct = h2h_home_wins / len(history)
            h2h_games = len(history)
        else:
            h2h_wpct = 0.5  # no history = neutral
            h2h_games = 0

        records.append({
            "GAME_ID": row["GAME_ID"],
            "h2h_home_wpct": h2h_wpct,
            "h2h_games": min(h2h_games, 20),  # cap for feature scaling
        })

        # Update H2H AFTER (no leakage)
        if key[0] == home:
            h2h_record.setdefault(key, []).append(row["HOME_WIN"])
        else:
            h2h_record.setdefault(key, []).append(1 - row["HOME_WIN"])

    return pd.DataFrame(records)


def compute_strength_of_schedule(matchups: pd.DataFrame, elo_df: pd.DataFrame) -> pd.DataFrame:
    """Compute strength of schedule using opponent Elo ratings.

    For each team, calculates the average Elo of their last N opponents.
    All computed BEFORE the game (no leakage).
    """
    matchups = matchups.sort_values("GAME_DATE").reset_index(drop=True)

    # Build a lookup of Elo ratings by GAME_ID
    elo_lookup = {}
    for _, row in elo_df.iterrows():
        elo_lookup[row["GAME_ID"]] = {
            row["HOME_TEAM"]: row["home_elo_pre"],
            row["AWAY_TEAM"]: row["away_elo_pre"],
        }

    # Track recent opponent Elos
    team_opp_elos = {}  # team -> list of opponent Elo ratings

    records = []

    for _, row in matchups.iterrows():
        home = row["HOME_TEAM"]
        away = row["AWAY_TEAM"]
        gid = row["GAME_ID"]

        # SOS for each team (avg opponent Elo over last 10 games)
        h_opp_elos = team_opp_elos.get(home, [])
        a_opp_elos = team_opp_elos.get(away, [])

        h_sos = np.mean(h_opp_elos[-10:]) if h_opp_elos else ELO_INITIAL
        a_sos = np.mean(a_opp_elos[-10:]) if a_opp_elos else ELO_INITIAL

        records.append({
            "GAME_ID": gid,
            "home_sos": h_sos,
            "away_sos": a_sos,
            "sos_diff": h_sos - a_sos,
        })

        # Update opponent Elo lists AFTER (no leakage)
        game_elos = elo_lookup.get(gid, {})
        if game_elos:
            team_opp_elos.setdefault(home, []).append(game_elos.get(away, ELO_INITIAL))
            team_opp_elos.setdefault(away, []).append(game_elos.get(home, ELO_INITIAL))

    return pd.DataFrame(records)


_FULL_NAME_TO_ABBREV = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP", "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS",
    "LA Clippers": "LAC",
}


def compute_advanced_stats_features(matchups: pd.DataFrame) -> pd.DataFrame:
    """Merge team-level advanced stats (OFF_RATING, DEF_RATING, PACE, etc.) into game features.

    Loads the most recent team_stats parquet and joins on team abbreviation.
    These are season-level aggregate stats, providing a different signal than rolling game stats.
    """
    from tools.storage import load_latest_parquet

    team_stats = load_latest_parquet("data/raw", "team_stats")
    if team_stats.empty:
        logger.warning("No team stats found, skipping advanced features")
        return pd.DataFrame({"GAME_ID": matchups["GAME_ID"]})

    # Find the team identifier column and map to abbreviations
    abbrev_col = None
    for col in ["TEAM_ABBREVIATION", "TEAM_ABB", "TEAM"]:
        if col in team_stats.columns:
            abbrev_col = col
            break

    if abbrev_col is None and "TEAM_NAME" in team_stats.columns:
        # Map full names to abbreviations
        team_stats = team_stats.copy()
        team_stats["_ABBREV"] = team_stats["TEAM_NAME"].map(_FULL_NAME_TO_ABBREV)
        abbrev_col = "_ABBREV"
        unmapped = team_stats[team_stats["_ABBREV"].isna()]["TEAM_NAME"].tolist()
        if unmapped:
            logger.warning(f"Could not map team names to abbreviations: {unmapped}")

    if abbrev_col is None:
        logger.warning(f"No team identifier column found in team_stats. Columns: {team_stats.columns.tolist()}")
        return pd.DataFrame({"GAME_ID": matchups["GAME_ID"]})

    # Select key advanced stats
    stat_cols = []
    desired = ["OFF_RATING", "DEF_RATING", "NET_RATING", "PACE", "EFG_PCT", "TS_PCT",
               "OREB_PCT", "DREB_PCT", "TM_TOV_PCT", "AST_RATIO", "PIE"]
    for col in desired:
        if col in team_stats.columns:
            stat_cols.append(col)

    if not stat_cols:
        logger.warning("No advanced stat columns found in team_stats")
        return pd.DataFrame({"GAME_ID": matchups["GAME_ID"]})

    # Build lookup: team_abbrev -> stats dict
    team_lookup = {}
    for _, row in team_stats.iterrows():
        team = row[abbrev_col]
        team_lookup[team] = {col: row[col] for col in stat_cols}

    records = []
    for _, row in matchups.iterrows():
        home = row["HOME_TEAM"]
        away = row["AWAY_TEAM"]
        feats = {"GAME_ID": row["GAME_ID"]}

        h_stats = team_lookup.get(home, {})
        a_stats = team_lookup.get(away, {})

        # Create differential features (home - away) for each stat
        for col in stat_cols:
            h_val = h_stats.get(col, 0) or 0
            a_val = a_stats.get(col, 0) or 0
            feats[f"home_{col.lower()}"] = float(h_val)
            feats[f"away_{col.lower()}"] = float(a_val)
            feats[f"diff_{col.lower()}"] = float(h_val) - float(a_val)

        records.append(feats)

    return pd.DataFrame(records)


def build_feature_matrix(
    elo_df: pd.DataFrame,
    rolling_df: pd.DataFrame,
    rest_df: pd.DataFrame,
    momentum_df: pd.DataFrame = None,
    h2h_df: pd.DataFrame = None,
    sos_df: pd.DataFrame = None,
    advanced_df: pd.DataFrame = None,
    player_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """Combine all features into a single feature matrix.

    Returns DataFrame ready for modeling with all features + HOME_WIN target.
    """
    # Start with elo
    features = elo_df[["GAME_ID", "GAME_DATE", "HOME_TEAM", "AWAY_TEAM",
                        "home_elo_pre", "away_elo_pre", "home_elo_expected", "HOME_WIN"]].copy()

    # Add derived elo features
    features["elo_diff"] = features["home_elo_pre"] - features["away_elo_pre"]

    # Merge rolling stats
    features = features.merge(rolling_df, on="GAME_ID", how="left")

    # Merge rest days
    features = features.merge(rest_df, on="GAME_ID", how="left")

    # Merge new feature groups if provided
    if momentum_df is not None and not momentum_df.empty:
        features = features.merge(momentum_df, on="GAME_ID", how="left")
        logger.info(f"Added momentum features: {[c for c in momentum_df.columns if c != 'GAME_ID']}")

    if h2h_df is not None and not h2h_df.empty:
        features = features.merge(h2h_df, on="GAME_ID", how="left")
        logger.info(f"Added H2H features: {[c for c in h2h_df.columns if c != 'GAME_ID']}")

    if sos_df is not None and not sos_df.empty:
        features = features.merge(sos_df, on="GAME_ID", how="left")
        logger.info(f"Added SOS features: {[c for c in sos_df.columns if c != 'GAME_ID']}")

    if advanced_df is not None and not advanced_df.empty:
        adv_cols = [c for c in advanced_df.columns if c != "GAME_ID"]
        if adv_cols:
            features = features.merge(advanced_df, on="GAME_ID", how="left")
            logger.info(f"Added advanced stats features: {len(adv_cols)} columns")

    if player_df is not None and not player_df.empty:
        player_cols = [c for c in player_df.columns if c != "GAME_ID"]
        if player_cols:
            features = features.merge(player_df, on="GAME_ID", how="left")
            logger.info(f"Added player strength features: {len(player_cols)} columns")

    # Fill any NaN with defaults
    features = features.fillna(0)

    return features


def get_feature_columns() -> list[str]:
    """Return the list of feature columns used for modeling (excluding target and IDs).

    Dynamically detects which features exist in the latest feature matrix.
    Falls back to base features if no matrix exists yet.
    """
    base_features = [
        # Elo (4)
        "home_elo_pre", "away_elo_pre", "elo_diff", "home_elo_expected",
        # Rolling 5-game (8)
        "home_win_pct_5", "home_pts_scored_5", "home_pts_allowed_5", "home_net_pts_5",
        "away_win_pct_5", "away_pts_scored_5", "away_pts_allowed_5", "away_net_pts_5",
        # Rolling 10-game (8)
        "home_win_pct_10", "home_pts_scored_10", "home_pts_allowed_10", "home_net_pts_10",
        "away_win_pct_10", "away_pts_scored_10", "away_pts_allowed_10", "away_net_pts_10",
        # Rest (5)
        "home_rest_days", "away_rest_days", "home_is_b2b", "away_is_b2b", "rest_advantage",
        # Momentum & streaks (6)
        "home_streak", "away_streak", "streak_diff",
        "home_home_wpct", "away_away_wpct", "home_away_split",
        # Head-to-head (2)
        "h2h_home_wpct", "h2h_games",
        # Strength of schedule (3)
        "home_sos", "away_sos", "sos_diff",
        # Advanced stats diffs (11 key diffs)
        "diff_off_rating", "diff_def_rating", "diff_net_rating", "diff_pace",
        "diff_efg_pct", "diff_ts_pct", "diff_oreb_pct", "diff_dreb_pct",
        "diff_tm_tov_pct", "diff_ast_ratio", "diff_pie",
    ]

    # Try to detect available features from the latest matrix
    try:
        from tools.storage import load_latest_parquet
        fm = load_latest_parquet("data/features", "feature_matrix")
        if not fm.empty:
            # Return only features that actually exist in the matrix
            available = [f for f in base_features if f in fm.columns]
            # Also include any advanced stat columns we didn't list explicitly
            for col in fm.columns:
                if col.startswith(("home_", "away_", "diff_")) and col not in available and col not in [
                    "HOME_TEAM", "AWAY_TEAM", "HOME_WIN", "GAME_ID", "GAME_DATE"
                ]:
                    available.append(col)
            return available
    except Exception:
        pass

    return base_features
