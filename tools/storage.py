"""Storage utilities for Parquet files and DB metadata (Postgres or SQLite)."""
from __future__ import annotations

import json
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from tools import db as dblayer

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent


def get_db_path(config: Optional[dict] = None) -> Path:
    db_rel = "data/predicto.db"
    if config and "storage" in config:
        db_rel = config["storage"].get("db_path", db_rel)
    return BASE_DIR / db_rel


def init_db(db_path: Optional[Path] = None):
    """Open a DB connection (Neon Postgres if DATABASE_URL set, else SQLite)
    and ensure the full schema exists."""
    if db_path is None:
        db_path = get_db_path()
    conn = dblayer.get_conn(db_path)
    dblayer.apply_schema(conn)
    return conn


def save_parquet(df: pd.DataFrame, rel_path: str, timestamp: bool = True) -> Path:
    """Save DataFrame as Parquet file.

    Args:
        df: DataFrame to save
        rel_path: Relative path under project root (e.g., 'data/raw/games_2024.parquet')
        timestamp: If True, append timestamp to filename
    """
    path = BASE_DIR / rel_path
    if timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = path.stem
        path = path.with_name(f"{stem}_{ts}{path.suffix}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info(f"Saved {len(df)} rows to {path}")
    return path


def load_parquet(rel_path: str) -> pd.DataFrame:
    """Load a Parquet file."""
    path = BASE_DIR / rel_path
    if not path.exists():
        logger.warning(f"File not found: {path}")
        return pd.DataFrame()
    return pd.read_parquet(path)


def load_latest_parquet(directory: str, prefix: str) -> pd.DataFrame:
    """Load the most recent Parquet file matching a prefix in a directory."""
    dir_path = BASE_DIR / directory
    if not dir_path.exists():
        return pd.DataFrame()

    matches = sorted(dir_path.glob(f"{prefix}*.parquet"), key=lambda p: p.stat().st_mtime)
    if not matches:
        logger.warning(f"No parquet files matching '{prefix}' in {dir_path}")
        return pd.DataFrame()

    latest = matches[-1]
    logger.info(f"Loading latest: {latest}")
    return pd.read_parquet(latest)


def log_run(conn, run_id: str, config: dict) -> None:
    """Log a pipeline run start."""
    conn.execute(
        "INSERT INTO run_log (run_id, started_at, config_json) VALUES (?, ?, ?)",
        (run_id, datetime.now().isoformat(), json.dumps(config)),
    )
    conn.commit()


def finish_run(conn, run_id: str, status: str, summary: str) -> None:
    """Log a pipeline run completion."""
    conn.execute(
        "UPDATE run_log SET finished_at = ?, status = ?, summary = ? WHERE run_id = ?",
        (datetime.now().isoformat(), status, summary, run_id),
    )
    conn.commit()


def log_experiment(conn, experiment: dict) -> None:
    """Log an experiment to the database."""
    dblayer.insert_or_replace(
        conn,
        "experiment_log",
        ["experiment_id", "run_id", "name", "method", "config_json",
         "features_used", "train_samples", "test_samples", "metrics_json",
         "market_baseline_json", "conclusion", "status", "created_at"],
        (
            experiment["experiment_id"],
            experiment.get("run_id"),
            experiment["name"],
            experiment["method"],
            json.dumps(experiment.get("config", {})),
            json.dumps(experiment.get("features_used", [])),
            experiment.get("train_samples", 0),
            experiment.get("test_samples", 0),
            json.dumps(experiment.get("metrics", experiment.get("overall_metrics", {}))),
            json.dumps(experiment.get("market_baseline", {})),
            experiment.get("conclusion", ""),
            experiment.get("status", "completed"),
            datetime.now().isoformat(),
        ),
    )
    conn.commit()


def get_experiment_history(conn, limit: int = 50) -> list[dict]:
    """Retrieve past experiment results."""
    cursor = conn.execute(
        """SELECT experiment_id, name, method, metrics_json, market_baseline_json,
                  conclusion, status, created_at
           FROM experiment_log ORDER BY created_at DESC LIMIT ?""",
        (limit,),
    )
    rows = cursor.fetchall()
    cols = ["experiment_id", "name", "method", "metrics_json", "market_baseline_json",
            "conclusion", "status", "created_at"]
    experiments = []
    for row in rows:
        exp = dict(zip(cols, row))
        exp["metrics"] = json.loads(exp.pop("metrics_json", "{}"))
        exp["market_baseline"] = json.loads(exp.pop("market_baseline_json", "{}"))
        experiments.append(exp)
    return experiments


# ── Schema v2 ledgers ─────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now().isoformat()


def log_prediction(conn, *, game_date: str, home_team: str, away_team: str,
                   p_home: float, run_id: str = None, experiment_id: str = None,
                   p_home_calibrated: float = None, market_prob: float = None,
                   model_desc: str = None) -> str:
    """Record a prediction the moment it is made — the foundation of CLV
    tracking. One row per (game, model, moment)."""
    pred_id = str(uuid.uuid4())[:12]
    edge = None
    if market_prob is not None:
        edge = (p_home_calibrated if p_home_calibrated is not None else p_home) - market_prob
    conn.execute(
        """INSERT INTO predictions
           (prediction_id, run_id, experiment_id, game_date, home_team, away_team,
            p_home, p_home_calibrated, market_prob_at_pred, edge, model_desc, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (pred_id, run_id, experiment_id, game_date, home_team, away_team,
         float(p_home), p_home_calibrated, market_prob, edge, model_desc, _now()),
    )
    conn.commit()
    return pred_id


def log_odds_snapshot(conn, *, game_date: str, home_team: str, away_team: str,
                      source: str, home_prob: float, volume: float = None,
                      liquidity: float = None) -> str:
    snap_id = str(uuid.uuid4())[:12]
    conn.execute(
        """INSERT INTO odds_snapshots
           (snapshot_id, game_date, home_team, away_team, source, home_prob,
            volume, liquidity, captured_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (snap_id, game_date, home_team, away_team, source,
         float(home_prob) if home_prob is not None else None,
         volume, liquidity, _now()),
    )
    conn.commit()
    return snap_id


def record_outcome(conn, *, game_date: str, home_team: str, away_team: str,
                   home_win: int, home_pts: int = None, away_pts: int = None,
                   closing_prob: float = None) -> None:
    dblayer.insert_or_replace(
        conn, "outcomes",
        ["game_date", "home_team", "away_team", "home_win", "home_pts",
         "away_pts", "closing_prob", "settled_at"],
        (game_date, home_team, away_team, int(home_win), home_pts, away_pts,
         closing_prob, _now()),
        conflict_cols=["game_date", "home_team", "away_team"],
    )
    conn.commit()


def log_paper_trade(conn, *, prediction_id: str, side: str, model_prob: float,
                    odds_taken: float, kelly_fraction: float, stake: float) -> str:
    trade_id = str(uuid.uuid4())[:12]
    conn.execute(
        """INSERT INTO paper_trades
           (trade_id, prediction_id, side, model_prob, odds_taken,
            kelly_fraction, stake, status, placed_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, 'open', ?)""",
        (trade_id, prediction_id, side, float(model_prob), float(odds_taken),
         float(kelly_fraction), float(stake), _now()),
    )
    conn.commit()
    return trade_id


def log_finding(conn, *, claim: str, evidence: str = "", confidence: str = "medium",
                source_agent: str = "meta_scientist") -> str:
    finding_id = str(uuid.uuid4())[:12]
    conn.execute(
        """INSERT INTO findings
           (finding_id, claim, evidence, confidence, status, source_agent, created_at)
           VALUES (?, ?, ?, ?, 'active', ?, ?)""",
        (finding_id, claim, evidence, confidence, source_agent, _now()),
    )
    conn.commit()
    return finding_id


def log_promotion(conn, *, experiment_id: str, model_desc: str, log_loss: float,
                  p_value: float = None, critic_verdict: str = None) -> str:
    promo_id = str(uuid.uuid4())[:12]
    conn.execute(
        """INSERT INTO promotions
           (promotion_id, experiment_id, model_desc, log_loss, p_value,
            critic_verdict, promoted_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (promo_id, experiment_id, model_desc, float(log_loss), p_value,
         critic_verdict, _now()),
    )
    conn.commit()
    return promo_id
