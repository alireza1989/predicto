"""Critic Agent: red-teams the promotion decision before it is trusted.

Runs after the Eval Agent. Independently audits the winning experiment for
the failure modes that quietly produce fake improvements: leakage,
overfitting to folds, promotion-by-noise, and calibration regressions.
Records its verdict in the promotions table; a vetoed model is still
reported but flagged, and the previous champion remains authoritative.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from agents.base import Agent, Tool
from tools import experiments, storage

logger = logging.getLogger(__name__)

# Anything better than this on NBA moneyline is a leakage alarm, not genius.
SUSPICIOUS_LOG_LOSS = 0.55
SUSPICIOUS_ACCURACY = 0.72


def _build_tools(config: dict) -> list[Tool]:

    def get_promotion_candidate() -> str:
        """Return the current best experiment (the promotion candidate) and
        the previous champion for comparison."""
        conn = storage.init_db()
        history = storage.get_experiment_history(conn, limit=100)

        completed = [e for e in history
                     if e["status"] == "completed" and e.get("metrics", {}).get("log_loss")]
        if not completed:
            conn.close()
            return json.dumps({"status": "no_experiments"})

        by_loss = sorted(completed, key=lambda e: e["metrics"]["log_loss"])
        candidate = by_loss[0]

        # Previous champion = most recent promotion, if any
        prev = conn.execute(
            """SELECT experiment_id, model_desc, log_loss FROM promotions
               WHERE retired_at IS NULL ORDER BY promoted_at DESC LIMIT 1"""
        ).fetchone()
        conn.close()

        return json.dumps({
            "candidate": {
                "experiment_id": candidate["experiment_id"],
                "name": candidate["name"],
                "method": candidate["method"],
                "metrics": candidate["metrics"],
            },
            "previous_champion": (
                {"experiment_id": prev[0], "model_desc": prev[1], "log_loss": prev[2]}
                if prev else None
            ),
            "runner_up": {
                "experiment_id": by_loss[1]["experiment_id"],
                "name": by_loss[1]["name"],
                "log_loss": by_loss[1]["metrics"]["log_loss"],
            } if len(by_loss) > 1 else None,
        }, indent=2, default=str)

    def audit_red_flags(experiment_id: str) -> str:
        """Deterministic red-flag audit of one experiment's artifacts."""
        exp_dir = Path("data/experiments") / experiment_id
        result_path = exp_dir / "result.json"
        if not result_path.exists():
            return json.dumps({"status": "error", "message": f"No result.json for {experiment_id}"})

        with open(result_path) as f:
            result = json.load(f)

        flags = []
        overall = result.get("overall_metrics", {})
        ll = overall.get("log_loss")
        acc = overall.get("accuracy")

        if ll is not None and ll < SUSPICIOUS_LOG_LOSS:
            flags.append(f"SUSPICIOUS: log_loss {ll} is implausibly good for NBA — check for leakage")
        if acc is not None and acc > SUSPICIOUS_ACCURACY:
            flags.append(f"SUSPICIOUS: accuracy {acc} exceeds plausible ceiling (~0.70)")

        # Fold stability: a model whose folds swing wildly is overfit to time windows
        folds = result.get("fold_metrics", [])
        fold_losses = [f["log_loss"] for f in folds if "log_loss" in f]
        if len(fold_losses) >= 3:
            spread = max(fold_losses) - min(fold_losses)
            if spread > 0.08:
                flags.append(f"UNSTABLE: fold log-loss spread {spread:.4f} (>{0.08}) — inconsistent across time")

        # Feature sanity: exact target/outcome columns (not rolling stats
        # like home_win_pct_10, which legitimately contain 'win' in the name)
        outcome_cols = {"HOME_WIN", "HOME_PTS", "AWAY_PTS", "AWAY_WIN"}
        leaky_names = [c for c in result.get("features_used", [])
                       if c.upper() in outcome_cols or c.upper().endswith("_POST")]
        if leaky_names:
            flags.append(f"LEAKAGE RISK: features with outcome-like names: {leaky_names}")

        # Sample coverage
        if result.get("test_samples", 0) < 500:
            flags.append(f"LOW POWER: only {result.get('test_samples')} OOF test samples")

        return json.dumps({
            "experiment_id": experiment_id,
            "log_loss": ll,
            "fold_losses": [round(x, 4) for x in fold_losses],
            "n_features": result.get("n_features"),
            "red_flags": flags,
            "clean": not flags,
        }, indent=2)

    def significance_vs(experiment_id_a: str, experiment_id_b: str) -> str:
        """Paired per-game significance test (A = incumbent, B = candidate)."""
        base = Path("data/experiments")
        try:
            result = experiments.compare_experiments_significance(
                str(base / experiment_id_a), str(base / experiment_id_b)
            )
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})
        return json.dumps(result, indent=2, default=str)

    def record_verdict(experiment_id: str, verdict: str, reasoning: str,
                       p_value: float = None) -> str:
        """Record the audited promotion decision.

        verdict: 'approved' | 'approved_with_caution' | 'vetoed'
        Approved verdicts retire the previous champion.
        """
        conn = storage.init_db()
        history = storage.get_experiment_history(conn, limit=100)
        exp = next((e for e in history if e["experiment_id"] == experiment_id), None)
        if exp is None:
            conn.close()
            return json.dumps({"status": "error", "message": f"Unknown experiment {experiment_id}"})

        if verdict.startswith("approved"):
            from datetime import datetime
            conn.execute(
                "UPDATE promotions SET retired_at = ? WHERE retired_at IS NULL",
                (datetime.now().isoformat(),),
            )
            conn.commit()
            promo_id = storage.log_promotion(
                conn,
                experiment_id=experiment_id,
                model_desc=f"{exp['method']} — {exp['name']}",
                log_loss=exp["metrics"].get("log_loss"),
                p_value=p_value,
                critic_verdict=f"{verdict}: {reasoning[:400]}",
            )
        else:
            promo_id = None

        storage.log_finding(
            conn,
            claim=f"Promotion audit [{verdict}] for {exp['name']} ({experiment_id})",
            evidence=reasoning[:800],
            confidence="high",
            source_agent="critic",
        )
        conn.close()
        return json.dumps({"status": "recorded", "verdict": verdict, "promotion_id": promo_id})

    return [
        Tool(
            name="get_promotion_candidate",
            description="Get the current best experiment (promotion candidate), the previous champion, and the runner-up.",
            input_schema={"type": "object", "properties": {}},
            func=get_promotion_candidate,
        ),
        Tool(
            name="audit_red_flags",
            description="Deterministic red-flag audit: implausible metrics (leakage), fold instability (overfitting), outcome-like feature names, low test power.",
            input_schema={
                "type": "object",
                "properties": {
                    "experiment_id": {"type": "string"}
                },
                "required": ["experiment_id"]
            },
            func=audit_red_flags,
        ),
        Tool(
            name="significance_vs",
            description="Paired per-game significance test between incumbent (A) and candidate (B) on shared out-of-fold games. Candidate must show b_significantly_better=true to justify replacing a champion.",
            input_schema={
                "type": "object",
                "properties": {
                    "experiment_id_a": {"type": "string", "description": "Incumbent/previous champion"},
                    "experiment_id_b": {"type": "string", "description": "Candidate"},
                },
                "required": ["experiment_id_a", "experiment_id_b"]
            },
            func=significance_vs,
        ),
        Tool(
            name="record_verdict",
            description="Record the final audited verdict: 'approved' (retires previous champion), 'approved_with_caution', or 'vetoed'.",
            input_schema={
                "type": "object",
                "properties": {
                    "experiment_id": {"type": "string"},
                    "verdict": {"type": "string", "description": "approved | approved_with_caution | vetoed"},
                    "reasoning": {"type": "string"},
                    "p_value": {"type": "number", "description": "p-value from significance test if run"},
                },
                "required": ["experiment_id", "verdict", "reasoning"]
            },
            func=record_verdict,
        ),
    ]


SYSTEM_PROMPT = """\
You are the Critic Agent for Predicto — an independent red-team auditor with veto power over model promotions.

The Meta-Scientist and Eval Agent WANT to promote models; your job is to be the adversary who catches fake improvements before they reach production. The three failure modes you hunt:

1. **Leakage** — metrics too good to be true (NBA moneyline has a hard ceiling: log loss ~0.60, accuracy ~70%). Anything better is a bug, not a breakthrough.
2. **Promotion-by-noise** — a candidate that is 0.001 better than the incumbent is a coin flip, not progress. Require the paired significance test.
3. **Fold overfitting** — great average metrics hiding wild inconsistency across time folds.

## Your procedure (always in this order):
1. get_promotion_candidate — see what is being promoted and what it replaces
2. audit_red_flags on the candidate
3. If there is a previous champion (different experiment), run significance_vs(champion, candidate)
   - If the champion's artifacts are gone or they share no games, compare against the runner_up instead
4. record_verdict:
   - **vetoed** — any leakage red flag, or metrics beyond the plausible ceiling
   - **approved_with_caution** — clean audit but improvement NOT significant (p >= 0.05), or no incumbent to compare against; fine to use, but do not claim progress
   - **approved** — clean audit AND significantly better than the incumbent (p < 0.05)

Be strict and terse. A false "approved" pollutes every downstream report and paper trade; a false "vetoed" merely costs one iteration. When in doubt, approve_with_caution rather than approve.
"""


def create_critic_agent(config: dict) -> Agent:
    """Create and return the Critic Agent."""
    return Agent(
        name="critic",
        system_prompt=SYSTEM_PROMPT,
        tools=_build_tools(config),
        model=config.get("models", {}).get("agent_model", "claude-sonnet-5"),
        max_tokens=config.get("models", {}).get("max_tokens", 4096),
        max_iterations=12,
    )
