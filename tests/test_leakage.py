"""Data-leakage regression tests.

The invariant every feature must satisfy: the feature row for a game on date
D depends only on games strictly before D. We verify it by mutating all
outcomes AFTER a cutoff date and asserting features at or before the cutoff
are unchanged.
"""
import numpy as np
import pandas as pd
import pytest

from tools import features


TEAMS = ["AAA", "BBB", "CCC", "DDD"]


def make_matchups(n_games: int = 120, seed: int = 7) -> pd.DataFrame:
    """Synthetic round-robin schedule, one game per day."""
    rng = np.random.default_rng(seed)
    rows = []
    date = pd.Timestamp("2025-01-01")
    for i in range(n_games):
        home, away = rng.choice(TEAMS, size=2, replace=False)
        home_pts = int(rng.integers(90, 130))
        away_pts = int(rng.integers(90, 130))
        if home_pts == away_pts:
            home_pts += 1
        rows.append({
            "GAME_ID": f"G{i:04d}",
            "GAME_DATE": date,
            "HOME_TEAM": home,
            "AWAY_TEAM": away,
            "HOME_PTS": home_pts,
            "AWAY_PTS": away_pts,
            "HOME_WIN": int(home_pts > away_pts),
            "SEASON": "2025-26",
        })
        date += pd.Timedelta(days=1)
    return pd.DataFrame(rows)


def flip_future_outcomes(matchups: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Invert every result strictly after the cutoff date."""
    flipped = matchups.copy()
    mask = flipped["GAME_DATE"] > cutoff
    flipped.loc[mask, "HOME_WIN"] = 1 - flipped.loc[mask, "HOME_WIN"]
    # Swap scores so points-based rolling stats change too
    hp = flipped.loc[mask, "HOME_PTS"].copy()
    flipped.loc[mask, "HOME_PTS"] = flipped.loc[mask, "AWAY_PTS"]
    flipped.loc[mask, "AWAY_PTS"] = hp
    return flipped


def build_matrix(matchups: pd.DataFrame) -> pd.DataFrame:
    elo = features.compute_elo_ratings(matchups)
    rolling = features.compute_rolling_stats(matchups)
    rest = features.compute_rest_days(matchups)
    momentum = features.compute_momentum_features(matchups)
    h2h = features.compute_h2h_features(matchups)
    sos = features.compute_strength_of_schedule(matchups, elo)
    fm = features.build_feature_matrix(elo, rolling, rest, momentum, h2h, sos)
    return fm.sort_values("GAME_ID").reset_index(drop=True)


@pytest.fixture(scope="module")
def matchups():
    return make_matchups()


def test_features_do_not_use_future_data(matchups):
    """Mutating games after the cutoff must not change earlier feature rows."""
    cutoff = matchups["GAME_DATE"].iloc[len(matchups) // 2]
    base = build_matrix(matchups)
    mutated = build_matrix(flip_future_outcomes(matchups, cutoff))

    past = base["GAME_DATE"] <= cutoff
    feature_cols = [c for c in base.columns
                    if c not in ("GAME_ID", "GAME_DATE", "HOME_TEAM", "AWAY_TEAM", "HOME_WIN")]
    before = base.loc[past, feature_cols].to_numpy(dtype=float)
    after = mutated.loc[past, feature_cols].to_numpy(dtype=float)

    bad = ~np.isclose(before, after, equal_nan=True)
    if bad.any():
        leaky = sorted({feature_cols[j] for _, j in zip(*np.where(bad))})
        pytest.fail(f"Leaky features (change when future games change): {leaky}")


def test_first_game_has_neutral_elo(matchups):
    """A team's very first game must use the initial rating — its own outcome
    cannot inform its pre-game Elo."""
    elo = features.compute_elo_ratings(matchups)
    first = elo.sort_values("GAME_DATE").iloc[0]
    assert first["home_elo_pre"] == first["away_elo_pre"], (
        "Pre-game Elo of two debut teams should be identical (initial rating)"
    )


def test_rolling_stats_exclude_current_game(matchups):
    """A game's own score must not appear in its rolling features."""
    rolling = features.compute_rolling_stats(matchups)
    merged = matchups.merge(rolling, on="GAME_ID")
    # For every team's FIRST appearance, rolling stats must be null/default —
    # there is no history yet.
    neutral = {"home_win_pct_5": 0.5, "home_net_pts_5": 0.0}
    seen = set()
    for _, row in merged.sort_values("GAME_DATE").iterrows():
        if row["HOME_TEAM"] not in seen:
            for col, default in neutral.items():
                if col in merged.columns:
                    val = row[col]
                    assert pd.isna(val) or val == default, (
                        f"{col} for a debut team should be the neutral prior "
                        f"{default}, got {val}"
                    )
        seen.add(row["HOME_TEAM"])
        seen.add(row["AWAY_TEAM"])
        if len(seen) == len(TEAMS):
            break


def test_rest_days_from_previous_game(matchups):
    rest = features.compute_rest_days(matchups)
    assert (rest.merge(matchups, on="GAME_ID")["home_rest_days"] >= 0).all()


def test_target_not_in_features():
    """HOME_WIN must never be listed as a model feature."""
    cols = features.get_feature_columns()
    assert "HOME_WIN" not in cols
    assert not any("HOME_PTS" == c or "AWAY_PTS" == c for c in cols)
