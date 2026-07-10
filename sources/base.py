"""DataSource contract: every external dataset is a plugin implementing this.

A source must be:
- idempotent and incremental (safe to fetch on every run)
- self-describing (schema, freshness SLA)
- quality-scored, so the platform can auto-archive sources that add no value
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import timedelta

import pandas as pd


class DataSource(ABC):
    name: str = "unnamed"
    kind: str = "generic"          # e.g. injuries, odds, schedule, stats
    freshness_sla: timedelta = timedelta(days=1)

    @abstractmethod
    def fetch(self) -> pd.DataFrame:
        """Fetch current data. Must not raise on transient failures —
        return an empty DataFrame instead."""

    @abstractmethod
    def schema(self) -> dict:
        """Column name → description."""

    def quality_report(self, df: pd.DataFrame) -> dict:
        """Basic quality stats; sources may override with domain checks."""
        if df.empty:
            return {"rows": 0, "ok": False, "reason": "empty fetch"}
        null_rate = float(df.isna().mean().mean())
        return {"rows": len(df), "null_rate": round(null_rate, 4), "ok": null_rate < 0.5}
