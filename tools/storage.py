"""Storage utilities for Parquet files and SQLite metadata."""
from __future__ import annotations

import json
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent


def get_db_path(config: Optional[dict] = None) -> Path:
    db_rel = "data/predicto.db"
    if config and "storage" in config:
        db_rel = config["storage"].get("db_path", db_rel)
    return BASE_DIR / db_rel


def init_db(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Initialize SQLite database with schema."""
    if db_path is None:
        db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS run_log (
            run_id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            status TEXT DEFAULT 'running',
            config_json TEXT,
            summary TEXT
        );

        CREATE TABLE IF NOT EXISTS experiment_log (
            experiment_id TEXT PRIMARY KEY,
            run_id TEXT,
            name TEXT NOT NULL,
            method TEXT NOT NULL,
            config_json TEXT,
            features_used TEXT,
            train_samples INTEGER,
            test_samples INTEGER,
            metrics_json TEXT,
            market_baseline_json TEXT,
            conclusion TEXT,
            status TEXT DEFAULT 'pending',
            created_at TEXT NOT NULL,
            FOREIGN KEY (run_id) REFERENCES run_log(run_id)
        );

        CREATE TABLE IF NOT EXISTS assumption_ledger (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            assumption TEXT NOT NULL,
            evidence TEXT,
            validated_at TEXT,
            status TEXT DEFAULT 'active',
            breaks_if TEXT
        );

        CREATE TABLE IF NOT EXISTS entity_map (
            source TEXT NOT NULL,
            source_id TEXT NOT NULL,
            canonical_id TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            confidence REAL DEFAULT 1.0,
            last_verified TEXT,
            PRIMARY KEY (source, source_id)
        );
    """)
    conn.commit()
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


def log_run(conn: sqlite3.Connection, run_id: str, config: dict) -> None:
    """Log a pipeline run start."""
    conn.execute(
        "INSERT INTO run_log (run_id, started_at, config_json) VALUES (?, ?, ?)",
        (run_id, datetime.now().isoformat(), json.dumps(config)),
    )
    conn.commit()


def finish_run(conn: sqlite3.Connection, run_id: str, status: str, summary: str) -> None:
    """Log a pipeline run completion."""
    conn.execute(
        "UPDATE run_log SET finished_at = ?, status = ?, summary = ? WHERE run_id = ?",
        (datetime.now().isoformat(), status, summary, run_id),
    )
    conn.commit()


def log_experiment(conn: sqlite3.Connection, experiment: dict) -> None:
    """Log an experiment to the database."""
    conn.execute(
        """INSERT OR REPLACE INTO experiment_log
        (experiment_id, run_id, name, method, config_json, features_used,
         train_samples, test_samples, metrics_json, market_baseline_json,
         conclusion, status, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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


def get_experiment_history(conn: sqlite3.Connection, limit: int = 50) -> list[dict]:
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
