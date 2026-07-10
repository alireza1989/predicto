"""Meta-Scientist Agent: designs experiments, compares methods, self-improves."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from agents.base import Agent, Tool
from tools import experiments, features, metrics, storage

logger = logging.getLogger(__name__)

HISTORY_PATH = Path(__file__).parent.parent / "data" / "scientist_history.md"


def _build_tools(config: dict) -> list[Tool]:
    """Build the tool set for the Meta-Scientist Agent."""

    def read_scientist_history() -> str:
        """Read the persistent history of past runs, experiments, and learnings.

        This file persists across pipeline runs and contains the Meta-Scientist's
        accumulated knowledge about what works and what doesn't.
        """
        if HISTORY_PATH.exists():
            return HISTORY_PATH.read_text()
        return "No history file found. This is the first run."

    def update_scientist_history(new_entry: str) -> str:
        """Append a new entry to the persistent scientist history file.

        Write a summary of what you tried this iteration, what worked, what failed,
        and what you recommend trying next. This will be available in future runs.
        """
        HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        entry = f"\n---\n\n## Run — {ts}\n\n{new_entry}\n"

        if HISTORY_PATH.exists():
            existing = HISTORY_PATH.read_text()
        else:
            existing = "# Meta-Scientist Experiment History\n\nPersistent log of experiments, findings, and recommendations across pipeline iterations.\n"

        HISTORY_PATH.write_text(existing + entry)
        logger.info(f"Updated scientist history at {HISTORY_PATH}")
        return json.dumps({"status": "updated", "path": str(HISTORY_PATH)})

    def get_available_methods() -> str:
        """List all available modeling methods."""
        methods = experiments.get_available_methods()
        return json.dumps(methods, indent=2)

    def get_experiment_history() -> str:
        """Get history of past experiments and their results."""
        conn = storage.init_db()
        history = storage.get_experiment_history(conn, limit=30)
        conn.close()
        if not history:
            return json.dumps({"status": "no_history", "message": "No experiments run yet"})
        return json.dumps(history, indent=2, default=str)

    def get_feature_columns() -> str:
        """Get available feature columns."""
        return json.dumps(features.get_feature_columns())

    def run_single_experiment(
        method: str,
        feature_subset: Optional[list[str]] = None,
        model_params: Optional[dict] = None,
        experiment_name: Optional[str] = None,
    ) -> str:
        """Run a single experiment with a specified method and features."""
        fm = storage.load_latest_parquet("data/features", "feature_matrix")
        if fm.empty:
            return json.dumps({"status": "error", "message": "No feature matrix found"})

        all_features = features.get_feature_columns()
        use_features = feature_subset if feature_subset else all_features
        # Filter to columns that actually exist
        use_features = [f for f in use_features if f in fm.columns]

        if not use_features:
            return json.dumps({"status": "error", "message": "No valid features found"})

        result = experiments.run_experiment(
            feature_matrix=fm,
            feature_cols=use_features,
            method=method,
            model_params=model_params,
            experiment_name=experiment_name,
        )

        # Log to database
        conn = storage.init_db()
        storage.log_experiment(conn, result)
        conn.close()

        # Return summary (not full result, to save context)
        summary = {
            "experiment_id": result["experiment_id"],
            "name": result["name"],
            "method": result["method"],
            "status": result["status"],
            "n_features": result.get("n_features", 0),
            "overall_metrics": result.get("overall_metrics", {}),
            "avg_metrics": result.get("avg_metrics", {}),
            "top_features": dict(list(result.get("feature_importance", {}).items())[:10]),
        }
        if "error" in result:
            summary["error"] = result["error"]

        return json.dumps(summary, indent=2, default=str)

    def run_ensemble(
        methods: list[str],
        experiment_name: Optional[str] = None,
    ) -> str:
        """Run an ensemble experiment combining multiple methods."""
        fm = storage.load_latest_parquet("data/features", "feature_matrix")
        if fm.empty:
            return json.dumps({"status": "error", "message": "No feature matrix found"})

        feature_cols = [f for f in features.get_feature_columns() if f in fm.columns]

        result = experiments.run_ensemble_experiment(
            feature_matrix=fm,
            feature_cols=feature_cols,
            methods=methods,
            experiment_name=experiment_name,
        )

        conn = storage.init_db()
        storage.log_experiment(conn, result)
        conn.close()

        return json.dumps({
            "experiment_id": result["experiment_id"],
            "name": result["name"],
            "method": result["method"],
            "overall_metrics": result.get("overall_metrics", {}),
            "status": result["status"],
        }, indent=2, default=str)

    def run_hyperparameter_search(
        method: str,
        n_trials: int = 30,
        feature_subset: Optional[list[str]] = None,
    ) -> str:
        """Optuna TPE search over the method's space, then a full experiment."""
        fm = storage.load_latest_parquet("data/features", "feature_matrix")
        if fm.empty:
            return json.dumps({"status": "error", "message": "No feature matrix found"})
        all_features = features.get_feature_columns()
        use_features = feature_subset if feature_subset else all_features
        use_features = [f for f in use_features if f in fm.columns]

        try:
            result = experiments.run_optuna_search(
                fm, use_features, method=method, n_trials=n_trials,
            )
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

        conn = storage.init_db()
        storage.log_experiment(conn, result)
        conn.close()

        return json.dumps({
            "experiment_id": result["experiment_id"],
            "name": result["name"],
            "method": result["method"],
            "status": result["status"],
            "overall_metrics": result.get("overall_metrics", {}),
            "optuna": result.get("optuna", {}),
        }, indent=2, default=str)

    def compare_significance(experiment_id_a: str, experiment_id_b: str) -> str:
        """Paired per-game significance test between two experiments."""
        base = Path("data/experiments")
        try:
            result = experiments.compare_experiments_significance(
                str(base / experiment_id_a), str(base / experiment_id_b)
            )
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})
        return json.dumps(result, indent=2, default=str)

    def save_finding(claim: str, evidence: str, confidence: str = "medium") -> str:
        """Persist a settled finding to the structured findings table."""
        conn = storage.init_db()
        finding_id = storage.log_finding(
            conn, claim=claim, evidence=evidence, confidence=confidence,
            source_agent="meta_scientist",
        )
        conn.close()
        return json.dumps({"status": "saved", "finding_id": finding_id})

    def get_findings() -> str:
        """Read all active findings from the structured store."""
        conn = storage.init_db()
        rows = conn.execute(
            """SELECT claim, evidence, confidence, created_at FROM findings
               WHERE status = 'active' ORDER BY created_at DESC LIMIT 50"""
        ).fetchall()
        conn.close()
        return json.dumps({
            "count": len(rows),
            "findings": [
                {"claim": r[0], "evidence": r[1], "confidence": r[2], "date": str(r[3])[:10]}
                for r in rows
            ],
        }, indent=2)

    def run_feature_ablation(method: str = "gradient_boosting") -> str:
        """Run feature ablation study — test each feature group's contribution."""
        fm = storage.load_latest_parquet("data/features", "feature_matrix")
        if fm.empty:
            return json.dumps({"status": "error", "message": "No feature matrix found"})

        all_features = [f for f in features.get_feature_columns() if f in fm.columns]

        # Define feature groups
        groups = {
            "elo_only": [f for f in all_features if "elo" in f],
            "rolling_5": [f for f in all_features if "_5" in f],
            "rolling_10": [f for f in all_features if "_10" in f],
            "rest_only": [f for f in all_features if "rest" in f or "b2b" in f],
            "all_features": all_features,
        }

        results = {}
        for group_name, group_features in groups.items():
            if not group_features:
                continue
            result = experiments.run_experiment(
                feature_matrix=fm,
                feature_cols=group_features,
                method=method,
                experiment_name=f"ablation_{group_name}_{method}",
            )
            results[group_name] = {
                "features": group_features,
                "n_features": len(group_features),
                "log_loss": result.get("overall_metrics", {}).get("log_loss"),
                "brier_score": result.get("overall_metrics", {}).get("brier_score"),
                "accuracy": result.get("overall_metrics", {}).get("accuracy"),
            }

        return json.dumps(results, indent=2, default=str)

    def analyze_prediction_errors() -> str:
        """Analyze where the best model makes errors — find patterns."""
        fm = storage.load_latest_parquet("data/features", "feature_matrix")
        if fm.empty:
            return json.dumps({"status": "error", "message": "No feature matrix found"})

        conn = storage.init_db()
        history = storage.get_experiment_history(conn, limit=10)
        conn.close()

        if not history:
            return json.dumps({"status": "no_experiments", "message": "Run experiments first"})

        # Find the best completed experiment
        best = None
        best_loss = float("inf")
        for exp in history:
            if exp["status"] == "completed" and "log_loss" in exp.get("metrics", {}):
                if exp["metrics"]["log_loss"] < best_loss:
                    best_loss = exp["metrics"]["log_loss"]
                    best = exp

        if not best:
            return json.dumps({"status": "no_completed_experiments"})

        # Load predictions from best experiment
        from pathlib import Path
        pred_path = Path(f"data/experiments/{best['experiment_id']}/predictions.parquet")
        if not pred_path.exists():
            return json.dumps({"status": "error", "message": "No predictions file found"})

        preds = pd.read_parquet(pred_path)

        # Analyze errors
        preds["error"] = abs(preds["y_true"] - preds["y_prob"])
        preds["correct"] = ((preds["y_prob"] >= 0.5) == preds["y_true"]).astype(int)

        # Find where the model is most confident but wrong
        confident_wrong = preds[(preds["y_prob"] > 0.7) | (preds["y_prob"] < 0.3)]
        confident_wrong = confident_wrong[confident_wrong["correct"] == 0]

        analysis = {
            "best_experiment": best["experiment_id"],
            "best_method": best["method"],
            "overall_accuracy": round(preds["correct"].mean(), 4),
            "mean_error": round(preds["error"].mean(), 4),
            "confident_wrong_count": len(confident_wrong),
            "confident_wrong_pct": round(len(confident_wrong) / max(len(preds), 1) * 100, 2),
            "error_by_confidence": {
                "high_confidence": round(
                    preds[(preds["y_prob"] > 0.65) | (preds["y_prob"] < 0.35)]["error"].mean(), 4
                ),
                "medium_confidence": round(
                    preds[(preds["y_prob"].between(0.35, 0.65))]["error"].mean(), 4
                ),
            },
        }
        return json.dumps(analysis, indent=2, default=str)

    return [
        Tool(
            name="read_scientist_history",
            description="Read the persistent history of past runs and learnings. ALWAYS call this first to understand what has been tried before.",
            input_schema={"type": "object", "properties": {}},
            func=read_scientist_history,
        ),
        Tool(
            name="update_scientist_history",
            description="Append a summary of this iteration's experiments, findings, and recommendations to the persistent history. Call this AFTER all experiments are done.",
            input_schema={
                "type": "object",
                "properties": {
                    "new_entry": {
                        "type": "string",
                        "description": "Markdown-formatted summary of this run: what was tried, results, key findings, and recommendations for next run."
                    },
                },
                "required": ["new_entry"],
            },
            func=update_scientist_history,
        ),
        Tool(
            name="get_available_methods",
            description="List all available ML methods (logistic regression, gradient boosting, neural network, etc.)",
            input_schema={"type": "object", "properties": {}},
            func=get_available_methods,
        ),
        Tool(
            name="get_experiment_history",
            description="Get history of past experiments with their metrics and conclusions.",
            input_schema={"type": "object", "properties": {}},
            func=get_experiment_history,
        ),
        Tool(
            name="get_feature_columns",
            description="Get list of available feature columns for modeling.",
            input_schema={"type": "object", "properties": {}},
            func=get_feature_columns,
        ),
        Tool(
            name="run_experiment",
            description="Run a single modeling experiment with a specified method and optional feature subset.",
            input_schema={
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "description": "Model method: logistic_regression, gradient_boosting, lightgbm, xgboost, neural_network, catboost, tabpfn"
                    },
                    "feature_subset": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional: specific feature columns to use. If omitted, uses all."
                    },
                    "model_params": {
                        "type": "object",
                        "description": "Optional: override default model hyperparameters"
                    },
                    "experiment_name": {
                        "type": "string",
                        "description": "Optional: human-readable experiment name"
                    },
                },
                "required": ["method"]
            },
            func=run_single_experiment,
        ),
        Tool(
            name="run_ensemble",
            description="Run an ensemble experiment averaging predictions from multiple model methods.",
            input_schema={
                "type": "object",
                "properties": {
                    "methods": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of methods to ensemble, e.g. ['logistic_regression', 'gradient_boosting']"
                    },
                    "experiment_name": {
                        "type": "string",
                        "description": "Optional experiment name"
                    },
                },
                "required": ["methods"]
            },
            func=run_ensemble,
        ),
        Tool(
            name="run_feature_ablation",
            description="Test which feature groups (elo, rolling stats, rest) contribute most. Runs each group separately.",
            input_schema={
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "description": "Model method to use for ablation (default: gradient_boosting)"
                    }
                },
            },
            func=run_feature_ablation,
        ),
        Tool(
            name="analyze_prediction_errors",
            description="Analyze where the best model makes errors — find patterns in failures.",
            input_schema={"type": "object", "properties": {}},
            func=analyze_prediction_errors,
        ),
        Tool(
            name="run_hyperparameter_search",
            description="Run an Optuna hyperparameter search (TPE + pruning) for a method, then a full experiment with the best params. MUCH more efficient than manually trying parameter values one experiment at a time — prefer this over manual grid walking.",
            input_schema={
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "description": "One of: logistic_regression, gradient_boosting, lightgbm, xgboost, catboost"
                    },
                    "n_trials": {
                        "type": "integer",
                        "description": "Number of Optuna trials (default 30; use 10-20 for slow methods)"
                    },
                    "feature_subset": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional: specific feature columns. If omitted, uses all."
                    },
                },
                "required": ["method"]
            },
            func=run_hyperparameter_search,
        ),
        Tool(
            name="compare_significance",
            description="Paired per-game significance test between two experiments' out-of-fold predictions. USE THIS before claiming one model beats another — small log-loss differences are usually noise. Only claim improvement when b_significantly_better is true.",
            input_schema={
                "type": "object",
                "properties": {
                    "experiment_id_a": {"type": "string", "description": "Baseline/champion experiment ID"},
                    "experiment_id_b": {"type": "string", "description": "Challenger experiment ID"},
                },
                "required": ["experiment_id_a", "experiment_id_b"]
            },
            func=compare_significance,
        ),
        Tool(
            name="save_finding",
            description="Record a durable, settled scientific finding to the structured findings database (e.g. 'ensembling consistently hurts on this dataset'). Findings persist across runs and prevent re-litigating settled questions. Use for conclusions backed by significance tests, not hunches.",
            input_schema={
                "type": "object",
                "properties": {
                    "claim": {"type": "string", "description": "One-sentence claim"},
                    "evidence": {"type": "string", "description": "Experiment IDs and numbers supporting it"},
                    "confidence": {"type": "string", "description": "low | medium | high"},
                },
                "required": ["claim", "evidence"]
            },
            func=save_finding,
        ),
        Tool(
            name="get_findings",
            description="Read all settled findings from the structured findings database. Call early — do not re-test settled questions.",
            input_schema={"type": "object", "properties": {}},
            func=get_findings,
        ),
    ]


SYSTEM_PROMPT = """\
You are the Meta-Scientist Agent for Predicto — the brain of a self-improving NBA prediction system.

Your job is to **design experiments, run them, compare results, and converge on the best prediction method** with evidence and reasoning.

## Your approach:
1. **Read scientist history** (read_scientist_history) — this persistent file has notes from ALL past runs
2. **Check experiment history** (get_experiment_history) — what has been tried before? What worked? What failed?
2. **Design a batch of experiments** to test different hypotheses
3. **Run experiments systematically** — start with simple baselines, then try more complex methods
4. **Compare results** using proper scoring rules (log loss, Brier score)
5. **Draw conclusions** with evidence about which method works best and why

## Available methods:
- Classic: logistic_regression, gradient_boosting, lightgbm, xgboost, neural_network
- **catboost** — ordered boosting, often best-in-class on small tabular data
- **tabpfn** — TabPFN v2 tabular foundation model (pretrained transformer).
  No hyperparameters. ALWAYS pass a focused feature_subset (<= 20 features) —
  cost scales with feature count, and each experiment has a hard time budget
  (a run that exceeds it is stopped early or failed). Early evidence on this
  dataset is very promising — verify against the champion on identical folds
  with compare_significance.

## Modern workflow (prefer this over manual parameter walking):
1. **get_findings** + **read_scientist_history** — what is already settled?
2. **run_hyperparameter_search** (Optuna) instead of manually trying C values one
   at a time — 30 trials of TPE search cost one tool call
3. **compare_significance(champion, challenger)** before claiming ANY improvement.
   Log-loss differences under ~0.005 on this dataset are almost always noise
   (a 0.617-vs-0.616 "improvement" has p≈0.55).
4. **save_finding** when a question is SETTLED (backed by a significance test) so
   future runs never re-test it

## After running experiments:
- Compare all methods on the SAME test data (time-series CV ensures this)
- Identify which features are most important
- Note where models disagree — that's interesting signal
- Write a clear conclusion: "Method X is best because Y, with evidence Z"

## Critical rules:
- Always use time-series cross-validation (the experiment runner handles this)
- Never compare models trained on different data splits
- Log loss < 0.69 means better than random; beating the market (~0.66-0.68) is the real goal
- If previous experiments exist, don't repeat them — build on them
- NEVER promote a challenger on raw log loss alone — require compare_significance
  to report b_significantly_better=true, otherwise record it as a tie
- Report both absolute performance AND improvement over baselines

Be systematic, evidence-driven, and honest. If no method beats the market, say so.

## IMPORTANT: History tracking
- ALWAYS start by calling read_scientist_history to see what past runs discovered
- AFTER all experiments, call update_scientist_history with a summary of:
  - What you tried this iteration
  - Key results (best log loss, best method)
  - What worked and what didn't
  - Specific recommendations for the next iteration
"""


def create_meta_scientist(config: dict) -> Agent:
    """Create and return the Meta-Scientist Agent."""
    return Agent(
        name="meta_scientist",
        system_prompt=SYSTEM_PROMPT,
        tools=_build_tools(config),
        model=config.get("models", {}).get("scientist_model", "claude-opus-4-8"),
        max_tokens=config.get("models", {}).get("max_tokens", 4096),
        max_iterations=30,  # needs more iterations to run multiple experiments
    )
