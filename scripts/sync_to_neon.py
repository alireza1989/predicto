#!/usr/bin/env python3
"""One-way sync: local SQLite → Neon Postgres.

Copies rows that exist locally but not remotely (by primary key). Used for
the initial backfill and as a safety net if a run wrote locally while
DATABASE_URL was unset.
"""
import logging
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv(override=True)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("sync")

from tools import db  # noqa: E402

TABLES_PK = {
    "run_log": ["run_id"],
    "experiment_log": ["experiment_id"],
    "predictions": ["prediction_id"],
    "odds_snapshots": ["snapshot_id"],
    "outcomes": ["game_date", "home_team", "away_team"],
    "paper_trades": ["trade_id"],
    "promotions": ["promotion_id"],
    "findings": ["finding_id"],
    "hypotheses": ["hypothesis_id"],
    "injury_snapshots": ["snapshot_id"],
}


def main():
    if not db.is_postgres():
        log.error("DATABASE_URL not set — nothing to sync to")
        sys.exit(1)

    sqlite_path = db.BASE_DIR / "data" / "predicto.db"
    if not sqlite_path.exists():
        log.error("No local SQLite DB at %s", sqlite_path)
        sys.exit(1)

    local = sqlite3.connect(str(sqlite_path))
    remote = db.get_conn()
    db.apply_schema(remote)

    for table, pk in TABLES_PK.items():
        try:
            rows = local.execute(f"SELECT * FROM {table}").fetchall()
            cols = [d[0] for d in local.execute(f"SELECT * FROM {table} LIMIT 0").description]
        except sqlite3.OperationalError:
            continue  # table doesn't exist locally
        if not rows:
            continue
        existing = {
            tuple(str(v) for v in r)
            for r in remote.execute(f"SELECT {', '.join(pk)} FROM {table}").fetchall()
        }
        pk_idx = [cols.index(c) for c in pk]
        inserted = 0
        for row in rows:
            key = tuple(str(row[i]) for i in pk_idx)
            if key in existing:
                continue
            placeholders = ", ".join(["?"] * len(cols))
            remote.execute(
                f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders})",
                tuple(row),
            )
            inserted += 1
        remote.commit()
        log.info("%s: %d local rows, %d inserted", table, len(rows), inserted)

    local.close()
    remote.close()
    log.info("Sync complete")


if __name__ == "__main__":
    main()
