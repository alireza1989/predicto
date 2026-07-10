"""Evaluation Agent: scores models, compares to baselines, makes promote/reject decisions."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from agents.base import Agent, Tool
from tools import metrics, storage

logger = logging.getLogger(__name__)


def _build_tools(config: dict) -> list[Tool]:
    """Build the tool set for the Eval Agent."""

    def get_experiment_results() -> str:
        """Get all experiment results from the database."""
        conn = storage.init_db()
        history = storage.get_experiment_history(conn, limit=50)
        conn.close()
        if not history:
            return json.dumps({"status": "no_experiments"})
        return json.dumps(history, indent=2, default=str)

    def compare_experiments() -> str:
        """Compare all completed experiments side by side."""
        conn = storage.init_db()
        history = storage.get_experiment_history(conn, limit=50)
        conn.close()

        completed = [e for e in history if e["status"] == "completed"]
        if not completed:
            return json.dumps({"status": "no_completed_experiments"})

        comparison = []
        for exp in completed:
            m = exp.get("metrics", {})
            comparison.append({
                "experiment_id": exp["experiment_id"],
                "name": exp["name"],
                "method": exp["method"],
                "log_loss": m.get("log_loss"),
                "brier_score": m.get("brier_score"),
                "accuracy": m.get("accuracy"),
                "sharpness": m.get("sharpness"),
                "created_at": exp["created_at"],
            })

        # Sort by log_loss (lower is better), handle None values
        comparison.sort(key=lambda x: x.get("log_loss") if x.get("log_loss") is not None else 999)

        return json.dumps({
            "experiments": comparison,
            "best": comparison[0] if comparison else None,
            "total_experiments": len(comparison),
        }, indent=2, default=str)

    def evaluate_against_baselines(experiment_id: str) -> str:
        """Evaluate a specific experiment against naive and market baselines."""
        pred_path = Path(f"data/experiments/{experiment_id}/predictions.parquet")
        if not pred_path.exists():
            return json.dumps({"status": "error", "message": f"No predictions for {experiment_id}"})

        preds = pd.read_parquet(pred_path)
        y_true = preds["y_true"].values
        y_prob = preds["y_prob"].values

        # Get experiment info
        conn = storage.init_db()
        cursor = conn.execute(
            "SELECT name, method FROM experiment_log WHERE experiment_id = ?",
            (experiment_id,)
        )
        row = cursor.fetchone()
        conn.close()
        method_name = row[1] if row else experiment_id

        # Compare to baselines (no market baseline for now)
        comparison = metrics.compare_to_baselines(y_true, y_prob, method_name)

        return json.dumps(comparison, indent=2, default=str)

    def compute_calibration(experiment_id: str) -> str:
        """Compute calibration analysis for a specific experiment."""
        pred_path = Path(f"data/experiments/{experiment_id}/predictions.parquet")
        if not pred_path.exists():
            return json.dumps({"status": "error", "message": f"No predictions for {experiment_id}"})

        preds = pd.read_parquet(pred_path)

        # Bin predictions and compute actual win rates
        preds["prob_bin"] = pd.cut(preds["y_prob"], bins=10, labels=False)
        cal = preds.groupby("prob_bin").agg(
            mean_predicted=("y_prob", "mean"),
            actual_rate=("y_true", "mean"),
            count=("y_true", "count"),
        ).reset_index()

        # Perfect calibration: mean_predicted ≈ actual_rate
        cal_error = abs(cal["mean_predicted"] - cal["actual_rate"]).mean()

        return json.dumps({
            "experiment_id": experiment_id,
            "calibration_table": cal.to_dict(orient="records"),
            "mean_calibration_error": round(float(cal_error), 4),
            "interpretation": (
                "Well calibrated" if cal_error < 0.03
                else "Reasonably calibrated" if cal_error < 0.06
                else "Poorly calibrated — consider recalibration"
            ),
        }, indent=2, default=str)

    def make_promotion_decision() -> str:
        """Decide which model (if any) to promote as the best for predictions."""
        conn = storage.init_db()
        history = storage.get_experiment_history(conn, limit=50)
        conn.close()

        completed = [e for e in history if e["status"] == "completed" and e.get("metrics", {}).get("log_loss")]
        if not completed:
            return json.dumps({"status": "no_candidates", "decision": "none"})

        # Find best by log loss
        best = min(completed, key=lambda e: e["metrics"]["log_loss"])

        # Check if it's meaningfully better than random (log_loss < 0.693)
        ll = best["metrics"]["log_loss"]
        beats_random = ll < 0.693

        # Check if accuracy > 52% (above chance)
        acc = best["metrics"].get("accuracy", 0)
        above_chance = acc > 0.52

        decision = {
            "best_experiment_id": best["experiment_id"],
            "best_method": best["method"],
            "best_name": best["name"],
            "metrics": best["metrics"],
            "beats_random": beats_random,
            "above_chance_accuracy": above_chance,
            "decision": "promote" if (beats_random and above_chance) else "reject",
            "reasoning": [],
        }

        if beats_random:
            decision["reasoning"].append(f"Log loss {ll:.4f} < 0.693 (random baseline)")
        else:
            decision["reasoning"].append(f"Log loss {ll:.4f} >= 0.693 (does not beat random)")

        if above_chance:
            decision["reasoning"].append(f"Accuracy {acc:.4f} > 0.52 (above chance)")
        else:
            decision["reasoning"].append(f"Accuracy {acc:.4f} <= 0.52 (not above chance)")

        # Compare top 3
        top3 = sorted(completed, key=lambda e: e["metrics"]["log_loss"])[:3]
        decision["top_3"] = [
            {"name": e["name"], "method": e["method"], "log_loss": e["metrics"]["log_loss"]}
            for e in top3
        ]

        return json.dumps(decision, indent=2, default=str)

    return [
        Tool(
            name="get_experiment_results",
            description="Get all experiment results from the database.",
            input_schema={"type": "object", "properties": {}},
            func=get_experiment_results,
        ),
        Tool(
            name="compare_experiments",
            description="Compare all completed experiments side by side, ranked by log loss.",
            input_schema={"type": "object", "properties": {}},
            func=compare_experiments,
        ),
        Tool(
            name="evaluate_against_baselines",
            description="Evaluate a specific experiment against naive baselines.",
            input_schema={
                "type": "object",
                "properties": {
                    "experiment_id": {"type": "string", "description": "The experiment ID to evaluate"}
                },
                "required": ["experiment_id"]
            },
            func=evaluate_against_baselines,
        ),
        Tool(
            name="compute_calibration",
            description="Compute calibration analysis for a specific experiment.",
            input_schema={
                "type": "object",
                "properties": {
                    "experiment_id": {"type": "string", "description": "The experiment ID to analyze"}
                },
                "required": ["experiment_id"]
            },
            func=compute_calibration,
        ),
        Tool(
            name="make_promotion_decision",
            description="Decide which model to promote as the best based on all evidence.",
            input_schema={"type": "object", "properties": {}},
            func=make_promotion_decision,
        ),
    ]


SYSTEM_PROMPT = """\
You are the Evaluation Agent for Predicto, an NBA prediction system.

Your job is to **rigorously evaluate model experiments and decide which model (if any) is good enough to use for predictions**.

## Your workflow:
1. Review all completed experiments
2. Compare them side by side on proper scoring rules (log loss, Brier score)
3. Evaluate the best model against baselines (naive, market)
4. Check calibration — are the probabilities trustworthy?
5. Make a promote/reject decision with reasoning

## Evaluation standards:
- **Log loss < 0.693** = better than random coin flip
- **Log loss < 0.68** = meaningfully better than naive
- **Log loss < 0.66** = potentially competitive with market
- **Accuracy > 55%** = useful for binary prediction
- **Good calibration** = when the model says 70%, it should be right ~70% of the time

## Critical rules:
- Be HONEST. If no model is good enough, say so.
- The market is a strong baseline — beating it is hard and claims of doing so need strong evidence
- Check for overfitting: if train metrics are much better than test metrics, the model is overfitting
- Consider model complexity vs improvement tradeoff
- Report confidence intervals / variance across folds

Your output should be a clear evaluation with:
1. Ranking of all methods tried
2. The best method with evidence
3. Promote/reject decision with reasoning
4. Recommendations for improvement
"""


def create_eval_agent(config: dict) -> Agent:
    """Create and return the Evaluation Agent."""
    return Agent(
        name="eval_agent",
        system_prompt=SYSTEM_PROMPT,
        tools=_build_tools(config),
        model=config.get("models", {}).get("agent_model", "claude-sonnet-5"),
    )
