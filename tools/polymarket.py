"""Polymarket API client for fetching NBA market data (read-only)."""
from __future__ import annotations

import json
import time
import logging
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"


class PolymarketClient:
    """Read-only client for Polymarket Gamma + CLOB APIs."""

    def __init__(self, request_delay: float = 0.5):
        self.session = requests.Session()
        self.delay = request_delay

    def _get(self, base: str, path: str, params: Optional[dict] = None) -> dict | list:
        url = f"{base}{path}"
        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        time.sleep(self.delay)
        return resp.json()

    # -- Gamma API (discovery) --

    def search_events(self, query: str, limit: int = 100) -> list[dict]:
        """Search for events by keyword."""
        return self._get(GAMMA_BASE, "/events", params={
            "active": True,
            "closed": False,
            "limit": limit,
            "title": query,
        })

    def get_sports_metadata(self) -> list[dict]:
        """Get sports metadata (tags, series IDs, logos)."""
        return self._get(GAMMA_BASE, "/sports")

    def get_active_events(self, tag_id: Optional[int] = None, limit: int = 100) -> list[dict]:
        """Get active events, optionally filtered by tag."""
        params = {"active": True, "closed": False, "limit": limit}
        if tag_id:
            params["tag_id"] = tag_id
        return self._get(GAMMA_BASE, "/events", params=params)

    def get_market(self, market_id: str) -> dict:
        """Get a single market by ID."""
        return self._get(GAMMA_BASE, f"/markets/{market_id}")

    def get_markets(self, limit: int = 100, **filters) -> list[dict]:
        """Get markets with optional filters."""
        params = {"limit": limit, **filters}
        return self._get(GAMMA_BASE, "/markets", params=params)

    # -- CLOB API (orderbook/pricing) --

    def get_orderbook(self, token_id: str) -> dict:
        """Get full orderbook for a token."""
        return self._get(CLOB_BASE, "/book", params={"token_id": token_id})

    def get_midpoint(self, token_id: str) -> dict:
        """Get midpoint price for a token."""
        return self._get(CLOB_BASE, "/midpoint", params={"token_id": token_id})

    def get_spread(self, token_id: str) -> dict:
        """Get bid-ask spread for a token."""
        return self._get(CLOB_BASE, "/spread", params={"token_id": token_id})

    def get_price(self, token_id: str, side: str = "BUY") -> dict:
        """Get current price for a token."""
        return self._get(CLOB_BASE, "/price", params={
            "token_id": token_id, "side": side,
        })

    def get_price_history(self, token_id: str, interval: str = "1d") -> list[dict]:
        """Get historical prices for a token."""
        return self._get(CLOB_BASE, "/prices-history", params={
            "market": token_id, "interval": interval,
        })

    def get_last_trade_price(self, token_id: str) -> dict:
        """Get last trade price for a token."""
        return self._get(CLOB_BASE, "/last-trade-price", params={"token_id": token_id})


def find_nba_markets(client: PolymarketClient) -> pd.DataFrame:
    """Find active NBA/basketball markets on Polymarket.

    Returns DataFrame with market details including token IDs and prices.
    """
    logger.info("Searching for NBA markets on Polymarket")

    # Strategy: search with NBA keywords
    nba_keywords = ["NBA", "basketball", "Lakers", "Celtics", "Warriors",
                    "Nuggets", "Bucks", "76ers", "Knicks", "Heat"]

    all_markets = []
    seen_ids = set()

    for keyword in nba_keywords:
        try:
            events = client.search_events(keyword)
            for event in events:
                for market in event.get("markets", []):
                    mid = market.get("id", "")
                    if mid not in seen_ids:
                        seen_ids.add(mid)
                        all_markets.append(_parse_market(market, event))
        except Exception as e:
            logger.warning(f"Search for '{keyword}' failed: {e}")

    # Also try broad active events and filter
    try:
        events = client.get_active_events(limit=100)
        for event in events:
            title = (event.get("title", "") + " " + event.get("description", "")).lower()
            if any(k.lower() in title for k in ["nba", "basketball"]):
                for market in event.get("markets", []):
                    mid = market.get("id", "")
                    if mid not in seen_ids:
                        seen_ids.add(mid)
                        all_markets.append(_parse_market(market, event))
    except Exception as e:
        logger.warning(f"Broad search failed: {e}")

    df = pd.DataFrame(all_markets) if all_markets else pd.DataFrame()
    logger.info(f"  Found {len(df)} NBA-related markets")
    return df


def _parse_market(market: dict, event: dict) -> dict:
    """Parse a Gamma API market object into a flat dict."""
    # Parse JSON string fields
    outcomes = _safe_json_parse(market.get("outcomes", "[]"))
    outcome_prices = _safe_json_parse(market.get("outcomePrices", "[]"))
    clob_token_ids = _safe_json_parse(market.get("clobTokenIds", "[]"))

    return {
        "market_id": market.get("id", ""),
        "event_id": event.get("id", ""),
        "question": market.get("question", ""),
        "event_title": event.get("title", ""),
        "outcomes": outcomes,
        "outcome_prices": [float(p) for p in outcome_prices] if outcome_prices else [],
        "clob_token_ids": clob_token_ids,
        "volume": market.get("volumeNum", 0),
        "liquidity": market.get("liquidityNum", 0),
        "best_bid": market.get("bestBid"),
        "best_ask": market.get("bestAsk"),
        "last_trade_price": market.get("lastTradePrice"),
        "active": market.get("active", False),
        "closed": market.get("closed", False),
        "end_date": market.get("endDate", ""),
        "created_at": market.get("createdAt", ""),
    }


def enrich_market_with_orderbook(client: PolymarketClient, market: dict) -> dict:
    """Add orderbook depth and spread data to a market dict."""
    enriched = {**market}
    token_ids = market.get("clob_token_ids", [])

    for i, tid in enumerate(token_ids[:2]):  # max 2 outcomes (yes/no)
        prefix = f"outcome_{i}"
        try:
            book = client.get_orderbook(tid)
            bids = book.get("bids", [])
            asks = book.get("asks", [])

            enriched[f"{prefix}_best_bid"] = float(bids[0]["price"]) if bids else None
            enriched[f"{prefix}_best_ask"] = float(asks[0]["price"]) if asks else None
            enriched[f"{prefix}_bid_depth"] = sum(float(b["size"]) for b in bids[:5])
            enriched[f"{prefix}_ask_depth"] = sum(float(a["size"]) for a in asks[:5])
            enriched[f"{prefix}_spread"] = (
                enriched[f"{prefix}_best_ask"] - enriched[f"{prefix}_best_bid"]
                if enriched[f"{prefix}_best_bid"] and enriched[f"{prefix}_best_ask"]
                else None
            )
        except Exception as e:
            logger.warning(f"Failed to get orderbook for token {tid}: {e}")

    return enriched


def fetch_nba_game_markets(client: PolymarketClient) -> pd.DataFrame:
    """Fetch active NBA game-level moneyline markets from Polymarket.

    Uses tag_id=745 (NBA) and filters for "vs." game events.
    Returns DataFrame with one row per game: teams, market prices, game time.
    """
    logger.info("Fetching NBA game markets from Polymarket")

    events = client.get_active_events(tag_id=745, limit=100)
    game_events = [e for e in events if "vs." in e.get("title", "")]

    rows = []
    for event in game_events:
        title = event.get("title", "")
        markets = event.get("markets", [])

        # Find the moneyline market (question matches event title or has no spread/O/U)
        moneyline = None
        for m in markets:
            q = m.get("question", "")
            # Moneyline is the one that just says "Team A vs. Team B" (no spread, O/U, player)
            if q == title or (
                "spread" not in q.lower()
                and "o/u" not in q.lower()
                and "moneyline" not in q.lower()
                and "points" not in q.lower()
                and "rebounds" not in q.lower()
                and "assists" not in q.lower()
                and ":" not in q
            ):
                moneyline = m
                break

        if not moneyline:
            continue

        outcomes = _safe_json_parse(moneyline.get("outcomes", "[]"))
        prices = _safe_json_parse(moneyline.get("outcomePrices", "[]"))
        token_ids = _safe_json_parse(moneyline.get("clobTokenIds", "[]"))

        if len(outcomes) != 2 or len(prices) != 2:
            continue

        try:
            price_0 = float(prices[0])
            price_1 = float(prices[1])
        except (ValueError, TypeError):
            continue

        # Skip closed/settled markets (both prices at 0 or 1)
        if price_0 in (0, 1) and price_1 in (0, 1):
            continue

        game_time = moneyline.get("gameStartTime", event.get("endDate", ""))

        rows.append({
            "event_title": title,
            "market_id": moneyline.get("id", ""),
            "team_a": outcomes[0],
            "team_b": outcomes[1],
            "team_a_price": price_0,
            "team_b_price": price_1,
            "team_a_token": token_ids[0] if len(token_ids) > 0 else "",
            "team_b_token": token_ids[1] if len(token_ids) > 1 else "",
            "game_time": game_time,
            "volume": moneyline.get("volumeNum", 0),
            "liquidity": moneyline.get("liquidityNum", 0),
        })

    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    logger.info(f"  Found {len(df)} NBA game markets with moneyline prices")
    return df


# ── Team name normalization for matching ──────────────────────────────

# Map common short names (Polymarket uses these) to full team names (nba_api uses these)
_TEAM_NAME_MAP = {
    "hawks": "Atlanta Hawks", "celtics": "Boston Celtics", "nets": "Brooklyn Nets",
    "hornets": "Charlotte Hornets", "bulls": "Chicago Bulls", "cavaliers": "Cleveland Cavaliers",
    "mavericks": "Dallas Mavericks", "nuggets": "Denver Nuggets", "pistons": "Detroit Pistons",
    "warriors": "Golden State Warriors", "rockets": "Houston Rockets", "pacers": "Indiana Pacers",
    "clippers": "LA Clippers", "lakers": "Los Angeles Lakers",
    "grizzlies": "Memphis Grizzlies", "heat": "Miami Heat", "bucks": "Milwaukee Bucks",
    "timberwolves": "Minnesota Timberwolves", "pelicans": "New Orleans Pelicans",
    "knicks": "New York Knicks", "thunder": "Oklahoma City Thunder", "magic": "Orlando Magic",
    "76ers": "Philadelphia 76ers", "suns": "Phoenix Suns", "trail blazers": "Portland Trail Blazers",
    "blazers": "Portland Trail Blazers", "kings": "Sacramento Kings", "spurs": "San Antonio Spurs",
    "raptors": "Toronto Raptors", "jazz": "Utah Jazz", "wizards": "Washington Wizards",
}

# NBA abbreviation to full name mapping (feature matrix uses abbreviations)
_ABBREV_MAP = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "LA Clippers", "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies", "MIA": "Miami Heat", "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves", "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder", "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns", "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "WAS": "Washington Wizards",
}

# Also build reverse: full name → short name
_FULL_TO_SHORT = {v.lower(): k for k, v in _TEAM_NAME_MAP.items()}


def _normalize_team(name: str) -> str:
    """Normalize a team name to its full NBA name.

    Handles: abbreviations (LAL, GSW), short names (Lakers, Celtics), full names.
    """
    stripped = name.strip()
    upper = stripped.upper()

    # Check abbreviation first (LAL, GSW, OKC, etc.)
    if upper in _ABBREV_MAP:
        return _ABBREV_MAP[upper]

    lower = stripped.lower()
    # Direct match on short name (celtics, lakers, etc.)
    if lower in _TEAM_NAME_MAP:
        return _TEAM_NAME_MAP[lower]
    # Check if any short name is contained in the input
    for short, full in _TEAM_NAME_MAP.items():
        if short in lower:
            return full
    # Check if it's already a full name
    for full in _TEAM_NAME_MAP.values():
        if lower == full.lower():
            return full
    return name  # Return as-is if no match


def match_markets_to_predictions(
    predictions_df: pd.DataFrame,
    markets_df: pd.DataFrame,
) -> pd.DataFrame:
    """Match Polymarket game markets to model predictions by team names.

    Args:
        predictions_df: Model predictions with HOME_TEAM, AWAY_TEAM, model_prob columns
        markets_df: Polymarket game markets from fetch_nba_game_markets()

    Returns:
        Merged DataFrame with model_prob, market_prob (for home team), and edge.
    """
    if predictions_df.empty or markets_df.empty:
        return pd.DataFrame()

    # Normalize team names on both sides to full NBA names
    markets = markets_df.copy()
    markets["team_a_norm"] = markets["team_a"].apply(_normalize_team)
    markets["team_b_norm"] = markets["team_b"].apply(_normalize_team)

    matched = []
    for _, pred in predictions_df.iterrows():
        home = _normalize_team(str(pred.get("HOME_TEAM", "")))
        away = _normalize_team(str(pred.get("AWAY_TEAM", "")))
        model_prob = pred.get("model_prob", None)

        if not home or not away or model_prob is None:
            continue

        # Find matching market (home=team_a or home=team_b)
        for _, mkt in markets.iterrows():
            ta = mkt["team_a_norm"]
            tb = mkt["team_b_norm"]

            if home == ta and away == tb:
                # team_a is home, team_a_price = home win probability
                market_home_prob = mkt["team_a_price"]
            elif home == tb and away == ta:
                # team_b is home, team_b_price = home win probability
                market_home_prob = mkt["team_b_price"]
            else:
                continue

            edge = model_prob - market_home_prob
            matched.append({
                "HOME_TEAM": home,
                "AWAY_TEAM": away,
                "GAME_DATE": pred.get("GAME_DATE", ""),
                "model_prob": round(model_prob, 4),
                "market_prob": round(market_home_prob, 4),
                "edge": round(edge, 4),
                "abs_edge": round(abs(edge), 4),
                "market_volume": mkt["volume"],
                "market_liquidity": mkt["liquidity"],
                "market_id": mkt["market_id"],
                "game_time": mkt["game_time"],
                "bet_direction": "HOME" if edge > 0 else "AWAY",
                "confidence": (
                    "HIGH" if abs(edge) > 0.10 else
                    "MEDIUM" if abs(edge) > 0.05 else
                    "LOW"
                ),
            })
            break  # Found match, move to next prediction

    result = pd.DataFrame(matched) if matched else pd.DataFrame()
    if not result.empty:
        result = result.sort_values("abs_edge", ascending=False).reset_index(drop=True)
    logger.info(f"  Matched {len(result)} games between predictions and markets")
    return result


def _safe_json_parse(val) -> list:
    """Safely parse a JSON string or return as-is if already parsed."""
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return []
    return []
