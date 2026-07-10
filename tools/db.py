"""Dual-backend database layer: Neon Postgres in production, SQLite locally.

Set DATABASE_URL (postgres://...) to use Postgres; otherwise falls back to
the local SQLite file. All call sites use the sqlite3 paramstyle (?) — the
Postgres wrapper translates placeholders so the rest of the codebase is
backend-agnostic.
"""
from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent


# ── Schema ───────────────────────────────────────────────────────────────
# {AUTOPK} is replaced per-dialect. Everything else is shared ANSI SQL.

SCHEMA_TABLES = {
    "run_log": """
        CREATE TABLE IF NOT EXISTS run_log (
            run_id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            status TEXT DEFAULT 'running',
            config_json TEXT,
            summary TEXT
        )""",
    "experiment_log": """
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
            created_at TEXT NOT NULL
        )""",
    "assumption_ledger": """
        CREATE TABLE IF NOT EXISTS assumption_ledger (
            id {AUTOPK},
            assumption TEXT NOT NULL,
            evidence TEXT,
            validated_at TEXT,
            status TEXT DEFAULT 'active',
            breaks_if TEXT
        )""",
    "entity_map": """
        CREATE TABLE IF NOT EXISTS entity_map (
            source TEXT NOT NULL,
            source_id TEXT NOT NULL,
            canonical_id TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            confidence REAL DEFAULT 1.0,
            last_verified TEXT,
            PRIMARY KEY (source, source_id)
        )""",
    # ── Schema v2: prediction ledger + market intelligence ──────────────
    "predictions": """
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id TEXT PRIMARY KEY,
            run_id TEXT,
            experiment_id TEXT,
            game_date TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            p_home REAL NOT NULL,
            p_home_calibrated REAL,
            market_prob_at_pred REAL,
            edge REAL,
            model_desc TEXT,
            created_at TEXT NOT NULL
        )""",
    "odds_snapshots": """
        CREATE TABLE IF NOT EXISTS odds_snapshots (
            snapshot_id TEXT PRIMARY KEY,
            game_date TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            source TEXT NOT NULL,
            home_prob REAL,
            volume REAL,
            liquidity REAL,
            captured_at TEXT NOT NULL
        )""",
    "outcomes": """
        CREATE TABLE IF NOT EXISTS outcomes (
            game_date TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            home_win INTEGER,
            home_pts INTEGER,
            away_pts INTEGER,
            closing_prob REAL,
            settled_at TEXT,
            PRIMARY KEY (game_date, home_team, away_team)
        )""",
    "paper_trades": """
        CREATE TABLE IF NOT EXISTS paper_trades (
            trade_id TEXT PRIMARY KEY,
            prediction_id TEXT,
            side TEXT NOT NULL,
            model_prob REAL,
            odds_taken REAL,
            kelly_fraction REAL,
            stake REAL,
            closing_prob REAL,
            clv REAL,
            pnl REAL,
            status TEXT DEFAULT 'open',
            placed_at TEXT NOT NULL,
            settled_at TEXT
        )""",
    "promotions": """
        CREATE TABLE IF NOT EXISTS promotions (
            promotion_id TEXT PRIMARY KEY,
            experiment_id TEXT NOT NULL,
            model_desc TEXT,
            log_loss REAL,
            p_value REAL,
            critic_verdict TEXT,
            promoted_at TEXT NOT NULL,
            retired_at TEXT
        )""",
    "findings": """
        CREATE TABLE IF NOT EXISTS findings (
            finding_id TEXT PRIMARY KEY,
            claim TEXT NOT NULL,
            evidence TEXT,
            confidence TEXT DEFAULT 'medium',
            status TEXT DEFAULT 'active',
            source_agent TEXT,
            created_at TEXT NOT NULL
        )""",
    "hypotheses": """
        CREATE TABLE IF NOT EXISTS hypotheses (
            hypothesis_id TEXT PRIMARY KEY,
            idea TEXT NOT NULL,
            priority INTEGER DEFAULT 3,
            cost_estimate TEXT,
            outcome TEXT,
            status TEXT DEFAULT 'open',
            created_at TEXT NOT NULL
        )""",
    "data_sources": """
        CREATE TABLE IF NOT EXISTS data_sources (
            source_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            kind TEXT,
            schema_json TEXT,
            quality_score REAL,
            marginal_gain REAL,
            status TEXT DEFAULT 'candidate',
            last_fetched_at TEXT,
            created_at TEXT NOT NULL
        )""",
}

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_predictions_game ON predictions (game_date, home_team, away_team)",
    "CREATE INDEX IF NOT EXISTS idx_odds_game ON odds_snapshots (game_date, home_team, away_team, captured_at)",
    "CREATE INDEX IF NOT EXISTS idx_experiments_created ON experiment_log (created_at)",
]


class PGConnection:
    """Adapter giving a psycopg connection the sqlite3 call surface used here.

    Translates '?' placeholders to '%s'. Cursors are consumed eagerly so the
    return object supports fetchone/fetchall like sqlite3's.
    """

    def __init__(self, conn):
        self._conn = conn

    def execute(self, sql: str, params: tuple = ()):  # noqa: A003
        sql = sql.replace("?", "%s")
        # sqlite's INSERT OR REPLACE → postgres upsert can't be translated
        # generically; call sites needing upsert use insert_or_replace().
        cur = self._conn.cursor()
        cur.execute(sql, params)
        return cur

    def executescript(self, script: str):
        cur = self._conn.cursor()
        cur.execute(script)
        return cur

    def commit(self):
        self._conn.commit()

    def rollback(self):
        self._conn.rollback()

    def close(self):
        self._conn.close()


def database_url() -> Optional[str]:
    url = os.environ.get("DATABASE_URL", "").strip()
    return url if url.startswith(("postgres://", "postgresql://")) else None


def is_postgres() -> bool:
    return database_url() is not None


def get_conn(db_path: Optional[Path] = None):
    """Return a DB connection: Postgres if DATABASE_URL is set, else SQLite."""
    url = database_url()
    if url:
        import psycopg

        return PGConnection(psycopg.connect(url))
    if db_path is None:
        db_path = BASE_DIR / "data" / "predicto.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(db_path))


def apply_schema(conn) -> None:
    """Create all tables/indexes, dialect-aware. Idempotent."""
    autopk = (
        "INTEGER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY"
        if is_postgres()
        else "INTEGER PRIMARY KEY AUTOINCREMENT"
    )
    for name, ddl in SCHEMA_TABLES.items():
        conn.execute(ddl.replace("{AUTOPK}", autopk))
    for idx in INDEXES:
        conn.execute(idx)
    conn.commit()
    logger.info(
        "Schema applied on %s (%d tables)",
        "postgres" if is_postgres() else "sqlite",
        len(SCHEMA_TABLES),
    )


def insert_or_replace(conn, table: str, columns: list[str], values: tuple,
                      conflict_cols: Optional[list[str]] = None) -> None:
    """Dialect-aware upsert. conflict_cols defaults to the first column."""
    conflict_cols = conflict_cols or [columns[0]]
    placeholders = ", ".join(["?"] * len(columns))
    cols = ", ".join(columns)
    if is_postgres():
        updates = ", ".join(
            f"{c} = EXCLUDED.{c}" for c in columns if c not in conflict_cols
        )
        sql = (
            f"INSERT INTO {table} ({cols}) VALUES ({placeholders}) "
            f"ON CONFLICT ({', '.join(conflict_cols)}) DO UPDATE SET {updates}"
        )
    else:
        sql = f"INSERT OR REPLACE INTO {table} ({cols}) VALUES ({placeholders})"
    conn.execute(sql, values)
