"""ESPN NBA injury report source (public JSON endpoint, no API key).

Injuries are the single biggest signal missing from the model (per 10 runs of
scientist history). There is no free historical archive, so this source
ACCRUES its own: every fetch snapshots the full league injury report into the
injury_snapshots table. Once enough history exists, injury features can enter
training; until then, current injuries annotate and adjust upcoming
predictions at inference time.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta

import pandas as pd
import requests

from sources.base import DataSource

logger = logging.getLogger(__name__)

ESPN_INJURIES_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
)

# Statuses that mean the player is expected to miss the game
OUT_STATUSES = {"out", "doubtful", "suspension"}
QUESTIONABLE_STATUSES = {"questionable", "day-to-day"}


class ESPNInjuriesSource(DataSource):
    name = "espn_injuries"
    kind = "injuries"
    freshness_sla = timedelta(hours=12)

    def fetch(self) -> pd.DataFrame:
        try:
            resp = requests.get(ESPN_INJURIES_URL, timeout=20)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as e:
            logger.warning("ESPN injuries fetch failed: %s", e)
            return pd.DataFrame()

        rows = []
        for team_block in payload.get("injuries", []):
            team = team_block.get("displayName", "")
            for inj in team_block.get("injuries", []):
                athlete = inj.get("athlete", {})
                rows.append({
                    "team": team,
                    "player": athlete.get("displayName", ""),
                    "position": (athlete.get("position") or {}).get("abbreviation", ""),
                    "status": str(inj.get("status", "")).lower(),
                    "detail": ((inj.get("details") or {}).get("type", "")
                               or (inj.get("type") or {}).get("description", "")),
                    "date": inj.get("date", ""),
                })
        df = pd.DataFrame(rows)
        logger.info("ESPN injuries: %d players across %d teams",
                    len(df), df["team"].nunique() if not df.empty else 0)
        return df

    def schema(self) -> dict:
        return {
            "team": "Full team name",
            "player": "Player display name",
            "position": "Position abbreviation",
            "status": "out | doubtful | questionable | day-to-day",
            "detail": "Injury description",
            "date": "Report date",
        }


def compute_impact_scores(injuries: pd.DataFrame,
                          player_logs: pd.DataFrame) -> pd.DataFrame:
    """Score each injury by the player's recent production.

    impact = avg game score over the player's last 10 games (Hollinger-lite:
    PTS + 0.7*REB + 0.7*AST + STL + BLK - TOV), scaled by miss probability
    (OUT/DOUBTFUL = 1.0, QUESTIONABLE = 0.5). Unknown players score 0.
    """
    if injuries.empty:
        return injuries

    inj = injuries.copy()
    per_player = {}
    if player_logs is not None and not player_logs.empty:
        logs = player_logs.copy()
        name_col = next((c for c in ("PLAYER_NAME", "player", "PLAYER") if c in logs.columns), None)
        date_col = next((c for c in ("GAME_DATE", "date") if c in logs.columns), None)
        if name_col and date_col:
            logs = logs.sort_values(date_col)
            for player, grp in logs.groupby(name_col):
                recent = grp.tail(10)
                gs = (
                    recent.get("PTS", pd.Series(dtype=float)).fillna(0)
                    + 0.7 * recent.get("REB", pd.Series(dtype=float)).fillna(0)
                    + 0.7 * recent.get("AST", pd.Series(dtype=float)).fillna(0)
                    + recent.get("STL", pd.Series(dtype=float)).fillna(0)
                    + recent.get("BLK", pd.Series(dtype=float)).fillna(0)
                    - recent.get("TOV", pd.Series(dtype=float)).fillna(0)
                )
                per_player[str(player).lower()] = float(gs.mean()) if len(gs) else 0.0

    def miss_prob(status: str) -> float:
        if status in OUT_STATUSES:
            return 1.0
        if status in QUESTIONABLE_STATUSES:
            return 0.5
        return 0.0

    inj["impact_score"] = [
        round(per_player.get(str(p).lower(), 0.0) * miss_prob(s), 2)
        for p, s in zip(inj["player"], inj["status"])
    ]
    return inj


def snapshot_injuries(conn, injuries: pd.DataFrame) -> int:
    """Accrue the current injury report into the historical archive."""
    if injuries.empty:
        return 0
    now = datetime.now().isoformat()
    for _, r in injuries.iterrows():
        conn.execute(
            """INSERT INTO injury_snapshots
               (snapshot_id, team, player, status, detail, impact_score, captured_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (str(uuid.uuid4())[:12], r["team"], r["player"], r.get("status", ""),
             r.get("detail", ""), float(r.get("impact_score", 0) or 0), now),
        )
    conn.commit()
    return len(injuries)


def team_injury_impact(injuries: pd.DataFrame) -> dict:
    """Aggregate impact per team: sum of impact scores of likely-out players."""
    if injuries.empty or "impact_score" not in injuries.columns:
        return {}
    agg = injuries.groupby("team")["impact_score"].sum()
    return {team: round(float(v), 2) for team, v in agg.items()}
