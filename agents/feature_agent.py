"""Feature Engineering Agent: computes features from raw data with leakage validation."""
from __future__ import annotations

import json
import logging

import pandas as pd
import numpy as np

from agents.base import Agent, Tool
from tools import features, storage

logger = logging.getLogger(__name__)


def _build_tools(config: dict) -> list[Tool]:
    """Build the tool set for the Feature Agent."""

    def load_raw_matchups() -> str:
        """Load the most recent raw matchup data."""
        df = storage.load_latest_parquet("data/raw", "nba_matchups")
        if df.empty:
            return json.dumps({"status": "error", "message": "No matchup data found"})
        return json.dumps({
            "status": "success",
            "rows": len(df),
            "columns": df.columns.tolist(),
            "date_range": [str(df["GAME_DATE"].min()), str(df["GAME_DATE"].max())],
            "teams": sorted(df["HOME_TEAM"].unique().tolist()),
            "sample": df.head(3).to_dict(orient="records"),
        }, default=str)

    def compute_all_features() -> str:
        """Compute all features from raw matchups: Elo, rolling stats, rest, momentum, H2H, SOS, advanced stats."""
        matchups = storage.load_latest_parquet("data/raw", "nba_matchups")
        if matchups.empty:
            return json.dumps({"status": "error", "message": "No matchup data found"})

        # Compute core features
        elo_df = features.compute_elo_ratings(matchups)
        rolling_df = features.compute_rolling_stats(matchups)
        rest_df = features.compute_rest_days(matchups)

        # Compute new feature groups
        momentum_df = features.compute_momentum_features(matchups)
        h2h_df = features.compute_h2h_features(matchups)
        sos_df = features.compute_strength_of_schedule(matchups, elo_df)
        advanced_df = features.compute_advanced_stats_features(matchups)

        # Player strength features (requires player game logs)
        player_logs = storage.load_latest_parquet("data/raw", "player_game_logs")
        if not player_logs.empty:
            player_df = features.compute_player_strength_features(matchups, player_logs)
            logger.info(f"Player strength features computed: {[c for c in player_df.columns if c != 'GAME_ID']}")
        else:
            player_df = None
            logger.warning("No player game logs found — skipping player features. Run data agent with fetch_player_logs first.")

        # Combine all
        feature_matrix = features.build_feature_matrix(
            elo_df, rolling_df, rest_df,
            momentum_df=momentum_df,
            h2h_df=h2h_df,
            sos_df=sos_df,
            advanced_df=advanced_df,
            player_df=player_df,
        )

        # Save
        path = storage.save_parquet(feature_matrix, "data/features/feature_matrix.parquet")

        # Summary stats
        feature_cols = features.get_feature_columns()
        available = [c for c in feature_cols if c in feature_matrix.columns]
        missing = [c for c in feature_cols if c not in feature_matrix.columns]

        stats = {}
        for col in available:
            stats[col] = {
                "mean": round(float(feature_matrix[col].mean()), 4),
                "std": round(float(feature_matrix[col].std()), 4),
                "null_pct": round(float(feature_matrix[col].isna().mean() * 100), 2),
            }

        return json.dumps({
            "status": "success",
            "total_games": len(feature_matrix),
            "features_computed": len(available),
            "missing_features": missing,
            "home_win_rate": round(float(feature_matrix["HOME_WIN"].mean()), 4),
            "feature_stats": stats,
            "saved_to": str(path),
        }, default=str)

    def validate_features() -> str:
        """Run leakage and quality checks on the feature matrix."""
        fm = storage.load_latest_parquet("data/features", "feature_matrix")
        if fm.empty:
            return json.dumps({"status": "error", "message": "No feature matrix found"})

        issues = []
        feature_cols = features.get_feature_columns()
        available = [c for c in feature_cols if c in fm.columns]

        # Check 1: No NaN in critical columns
        for col in available:
            null_pct = fm[col].isna().mean() * 100
            if null_pct > 5:
                issues.append(f"High null rate in {col}: {null_pct:.1f}%")

        # Check 2: Target distribution is reasonable
        home_win_rate = fm["HOME_WIN"].mean()
        if home_win_rate < 0.45 or home_win_rate > 0.65:
            issues.append(f"Unusual home win rate: {home_win_rate:.4f}")

        # Check 3: Elo expected probability vs actual outcome correlation
        if "home_elo_expected" in fm.columns:
            corr = fm["home_elo_expected"].corr(fm["HOME_WIN"])
            if corr < 0.05:
                issues.append(f"Very low Elo-outcome correlation: {corr:.4f}")

        # Check 4: Feature ranges are sensible
        if "home_elo_pre" in fm.columns:
            elo_min, elo_max = fm["home_elo_pre"].min(), fm["home_elo_pre"].max()
            if elo_min < 1000 or elo_max > 2000:
                issues.append(f"Elo range suspicious: [{elo_min:.0f}, {elo_max:.0f}]")

        # Check 5: No future leakage — rolling stats should not be perfectly correlated with target
        for col in available:
            if "win_pct" in col:
                corr = abs(fm[col].corr(fm["HOME_WIN"]))
                if corr > 0.8:
                    issues.append(f"Possible leakage: {col} has {corr:.4f} correlation with target")

        # Check 6: Data is sorted by date
        dates = pd.to_datetime(fm["GAME_DATE"])
        if not dates.is_monotonic_increasing:
            issues.append("Data is not sorted by date — required for time-series CV")

        return json.dumps({
            "status": "passed" if not issues else "issues_found",
            "total_games": len(fm),
            "features_checked": len(available),
            "issues": issues,
            "home_win_rate": round(home_win_rate, 4),
        })

    def get_feature_info() -> str:
        """Get information about available features and their descriptions."""
        feature_cols = features.get_feature_columns()
        descriptions = {
            "home_elo_pre": "Home team Elo rating before game",
            "away_elo_pre": "Away team Elo rating before game",
            "elo_diff": "Home Elo minus Away Elo",
            "home_elo_expected": "Expected home win probability from Elo",
            "home_win_pct_5": "Home team win% in last 5 games",
            "home_pts_scored_5": "Home team avg points scored in last 5",
            "home_pts_allowed_5": "Home team avg points allowed in last 5",
            "home_net_pts_5": "Home team avg net points in last 5",
            "away_win_pct_5": "Away team win% in last 5 games",
            "away_pts_scored_5": "Away team avg points scored in last 5",
            "away_pts_allowed_5": "Away team avg points allowed in last 5",
            "away_net_pts_5": "Away team avg net points in last 5",
            "home_win_pct_10": "Home team win% in last 10 games",
            "home_pts_scored_10": "Home team avg points scored in last 10",
            "home_pts_allowed_10": "Home team avg points allowed in last 10",
            "home_net_pts_10": "Home team avg net points in last 10",
            "away_win_pct_10": "Away team win% in last 10 games",
            "away_pts_scored_10": "Away team avg points scored in last 10",
            "away_pts_allowed_10": "Away team avg points allowed in last 10",
            "away_net_pts_10": "Away team avg net points in last 10",
            "home_rest_days": "Days since home team's last game",
            "away_rest_days": "Days since away team's last game",
            "home_is_b2b": "Home team on back-to-back (1/0)",
            "away_is_b2b": "Away team on back-to-back (1/0)",
            "rest_advantage": "Home rest days minus away rest days",
        }
        return json.dumps([
            {"feature": f, "description": descriptions.get(f, "No description")}
            for f in feature_cols
        ])

    return [
        Tool(
            name="load_raw_matchups",
            description="Load and inspect the most recent raw NBA matchup data.",
            input_schema={"type": "object", "properties": {}},
            func=load_raw_matchups,
        ),
        Tool(
            name="compute_all_features",
            description="Compute all features (Elo, rolling stats, rest days) from raw matchups. Saves feature matrix.",
            input_schema={"type": "object", "properties": {}},
            func=compute_all_features,
        ),
        Tool(
            name="validate_features",
            description="Run quality and leakage checks on the computed feature matrix.",
            input_schema={"type": "object", "properties": {}},
            func=validate_features,
        ),
        Tool(
            name="get_feature_info",
            description="Get list of all available features with descriptions.",
            input_schema={"type": "object", "properties": {}},
            func=get_feature_info,
        ),
    ]


SYSTEM_PROMPT = """\
You are the Feature Engineering Agent for Predicto, an NBA prediction system.

Your job is to compute features from raw NBA game data and validate them for quality and leakage.

## Your workflow:
1. Load raw matchup data and inspect it
2. Compute all features (Elo ratings, rolling stats, rest days)
3. Validate the feature matrix for data quality and potential leakage
4. Report feature statistics and any issues

## Critical rules:
- **No leakage**: All features must be computed BEFORE each game (using only past data)
- **No future data**: Rolling windows and Elo updates only use games that already happened
- Features with high correlation to the target (>0.8) are suspicious — flag them
- Report null rates — features with >5% nulls need attention
- The feature matrix must be sorted by date for proper time-series cross-validation

After computing and validating features, provide a summary including:
- Number of games and features
- Home win rate (should be ~55-60% in NBA)
- Any quality issues or leakage warnings
- Feature statistics
"""


def create_feature_agent(config: dict) -> Agent:
    """Create and return the Feature Engineering Agent."""
    return Agent(
        name="feature_agent",
        system_prompt=SYSTEM_PROMPT,
        tools=_build_tools(config),
        model=config.get("models", {}).get("agent_model", "claude-sonnet-5"),
    )
