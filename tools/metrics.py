"""Evaluation metrics: proper scoring rules, calibration, and comparison."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.calibration import calibration_curve

logger = logging.getLogger(__name__)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute proper scoring rules and summary metrics.

    Args:
        y_true: Binary outcomes (0 or 1)
        y_prob: Predicted probabilities for class 1

    Returns:
        Dict with log_loss, brier_score, accuracy, calibration data.
    """
    # Clip probabilities to avoid log(0)
    y_prob = np.clip(y_prob, 1e-6, 1 - 1e-6)

    ll = float(log_loss(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))
    accuracy = float(np.mean((y_prob >= 0.5) == y_true))

    # Calibration curve
    try:
        fraction_pos, mean_predicted = calibration_curve(
            y_true, y_prob, n_bins=10, strategy="uniform"
        )
        calibration = {
            "fraction_positive": fraction_pos.tolist(),
            "mean_predicted": mean_predicted.tolist(),
        }
    except Exception:
        calibration = {}

    # Sharpness (how far predictions are from 0.5 on average)
    sharpness = float(np.mean(np.abs(y_prob - 0.5)))

    return {
        "log_loss": round(ll, 6),
        "brier_score": round(brier, 6),
        "accuracy": round(accuracy, 4),
        "sharpness": round(sharpness, 4),
        "n_samples": int(len(y_true)),
        "base_rate": round(float(np.mean(y_true)), 4),
        "mean_prediction": round(float(np.mean(y_prob)), 4),
        "calibration": calibration,
    }


def compute_naive_baseline(y_true: np.ndarray, home_rate: float = 0.6) -> dict:
    """Compute metrics for a naive baseline that always predicts home_rate."""
    y_prob = np.full_like(y_true, home_rate, dtype=float)
    metrics = compute_metrics(y_true, y_prob)
    metrics["method"] = f"naive_home_{home_rate}"
    return metrics


def compute_market_baseline(
    y_true: np.ndarray,
    market_probs: np.ndarray,
) -> dict:
    """Compute metrics using market-implied probabilities as predictions."""
    # Filter out games without market prices
    valid = ~np.isnan(market_probs) & (market_probs > 0) & (market_probs < 1)
    if valid.sum() < 10:
        return {"method": "market_baseline", "error": "insufficient_market_data",
                "valid_samples": int(valid.sum())}

    metrics = compute_metrics(y_true[valid], market_probs[valid])
    metrics["method"] = "market_baseline"
    metrics["n_matched"] = int(valid.sum())
    return metrics


def compare_to_baselines(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    market_probs: Optional[np.ndarray] = None,
) -> dict:
    """Compare a model's predictions against baselines.

    Returns:
        Dict with model metrics, baseline metrics, and relative improvements.
    """
    model_metrics = compute_metrics(y_true, y_prob)
    model_metrics["method"] = model_name

    naive = compute_naive_baseline(y_true)
    elo_baseline = compute_naive_baseline(y_true, home_rate=float(np.mean(y_true)))

    comparison = {
        "model": model_metrics,
        "naive_baseline": naive,
        "base_rate_baseline": elo_baseline,
        "improvements": {},
    }

    # Compute improvements
    comparison["improvements"]["vs_naive_log_loss"] = round(
        naive["log_loss"] - model_metrics["log_loss"], 6
    )
    comparison["improvements"]["vs_naive_brier"] = round(
        naive["brier_score"] - model_metrics["brier_score"], 6
    )

    # Market baseline if available
    if market_probs is not None:
        market = compute_market_baseline(y_true, market_probs)
        comparison["market_baseline"] = market
        if "log_loss" in market:
            comparison["improvements"]["vs_market_log_loss"] = round(
                market["log_loss"] - model_metrics["log_loss"], 6
            )
            comparison["improvements"]["vs_market_brier"] = round(
                market["brier_score"] - model_metrics["brier_score"], 6
            )

    return comparison


def find_mispricings(
    predictions_df: pd.DataFrame,
    prob_col: str = "model_prob",
    market_col: str = "market_prob",
    min_edge: float = 0.05,
    min_liquidity: float = 100,
) -> pd.DataFrame:
    """Identify potential mispricings where model disagrees with market.

    Args:
        predictions_df: DataFrame with model and market probabilities
        prob_col: Column name for model probability
        market_col: Column name for market probability
        min_edge: Minimum probability difference to flag
        min_liquidity: Minimum market liquidity to consider

    Returns:
        DataFrame of potential mispricings sorted by edge size.
    """
    df = predictions_df.copy()

    # Filter for valid data
    valid = (
        df[prob_col].notna()
        & df[market_col].notna()
        & (df[market_col] > 0)
        & (df[market_col] < 1)
    )
    df = df[valid].copy()

    # Filter by liquidity if column exists
    if "liquidity" in df.columns:
        df = df[df["liquidity"] >= min_liquidity]

    # Compute edge
    df["edge"] = df[prob_col] - df[market_col]
    df["abs_edge"] = df["edge"].abs()

    # Filter by minimum edge
    df = df[df["abs_edge"] >= min_edge]

    # Sort by absolute edge
    df = df.sort_values("abs_edge", ascending=False).reset_index(drop=True)

    return df
