"""Report Agent: generates the final analysis report with mispricings and evidence."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from agents.base import Agent, Tool
from tools import storage, metrics, experiments, polymarket, html_report

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent


def _build_tools(config: dict) -> list[Tool]:
    """Build the tool set for the Report Agent."""

    def get_promoted_model() -> str:
        """Get info about the best/promoted model."""
        conn = storage.init_db()
        history = storage.get_experiment_history(conn, limit=50)
        conn.close()

        completed = [e for e in history if e["status"] == "completed" and e.get("metrics", {}).get("log_loss")]
        if not completed:
            return json.dumps({"status": "no_models"})

        best = min(completed, key=lambda e: e["metrics"]["log_loss"])
        return json.dumps(best, indent=2, default=str)

    def get_all_experiment_summary() -> str:
        """Get summary of all experiments for the report."""
        conn = storage.init_db()
        history = storage.get_experiment_history(conn, limit=50)
        conn.close()

        summary = []
        for exp in history:
            m = exp.get("metrics", {})
            summary.append({
                "name": exp["name"],
                "method": exp["method"],
                "log_loss": m.get("log_loss"),
                "brier_score": m.get("brier_score"),
                "accuracy": m.get("accuracy"),
                "status": exp["status"],
            })
        return json.dumps(summary, indent=2, default=str)

    def get_upcoming_predictions() -> str:
        """Get predictions for upcoming games from the best model."""
        conn = storage.init_db()
        history = storage.get_experiment_history(conn, limit=50)
        conn.close()

        completed = [e for e in history if e["status"] == "completed" and e.get("metrics", {}).get("log_loss")]
        if not completed:
            return json.dumps({"status": "no_model"})

        best = min(completed, key=lambda e: e["metrics"]["log_loss"])

        # Load feature matrix for upcoming games
        fm = storage.load_latest_parquet("data/features", "feature_matrix")
        upcoming = storage.load_latest_parquet("data/raw", "upcoming_games")

        if fm.empty or upcoming.empty:
            return json.dumps({"status": "no_data", "message": "Need features and upcoming games"})

        # Try to generate predictions
        artifact_dir = f"data/experiments/{best['experiment_id']}"
        try:
            # Use the latest feature data for prediction
            preds = experiments.predict_upcoming(artifact_dir, fm.tail(30))
            return json.dumps({
                "status": "success",
                "model": best["method"],
                "predictions": preds[["GAME_DATE", "HOME_TEAM", "AWAY_TEAM", "model_prob"]].to_dict(orient="records"),
            }, indent=2, default=str)
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

    def get_market_data() -> str:
        """Get Polymarket NBA market data for comparison."""
        markets = storage.load_latest_parquet("data/raw", "polymarket_nba")
        if markets.empty:
            return json.dumps({"status": "no_market_data"})

        # Summarize
        summary = []
        for _, row in markets.head(20).iterrows():
            summary.append({
                "question": row.get("question", ""),
                "outcome_prices": row.get("outcome_prices", []),
                "volume": row.get("volume", 0),
                "liquidity": row.get("liquidity", 0),
            })
        return json.dumps({
            "status": "success",
            "total_markets": len(markets),
            "markets": summary,
        }, indent=2, default=str)

    def compute_betting_edges() -> str:
        """Compare model predictions to Polymarket game odds and find betting edges.

        Matches upcoming game predictions with live Polymarket moneyline markets,
        computes edge (model_prob - market_prob), and returns ranked opportunities.
        """
        # Load game markets
        game_markets = storage.load_latest_parquet("data/raw", "polymarket_game_markets")
        if game_markets.empty:
            return json.dumps({"status": "no_game_markets", "message": "No Polymarket game markets found. Run data agent with fetch_polymarket_game_markets first."})

        # Get best model and generate predictions
        conn = storage.init_db()
        history = storage.get_experiment_history(conn, limit=50)
        conn.close()

        completed = [e for e in history if e["status"] == "completed" and e.get("metrics", {}).get("log_loss")]
        if not completed:
            return json.dumps({"status": "no_model", "message": "No trained model found"})

        # Find best model that has a saved artifact (ensembles don't save model.pkl)
        import os
        completed_sorted = sorted(completed, key=lambda e: e["metrics"]["log_loss"])
        best = None
        for candidate in completed_sorted:
            artifact_path = f"data/experiments/{candidate['experiment_id']}/model.pkl"
            if os.path.exists(artifact_path):
                best = candidate
                break
        if not best:
            return json.dumps({"status": "no_model", "message": "No model with saved artifact found"})

        artifact_dir = f"data/experiments/{best['experiment_id']}"

        # Load feature matrix for predictions
        fm = storage.load_latest_parquet("data/features", "feature_matrix")
        if fm.empty:
            return json.dumps({"status": "error", "message": "No feature matrix found"})

        try:
            preds = experiments.predict_upcoming(artifact_dir, fm.tail(50))
        except Exception as e:
            return json.dumps({"status": "error", "message": f"Prediction failed: {e}"})

        if preds.empty:
            return json.dumps({"status": "no_predictions"})

        # Match predictions to markets and compute edges
        edges_df = polymarket.match_markets_to_predictions(preds, game_markets)

        if edges_df.empty:
            return json.dumps({
                "status": "no_matches",
                "message": "Could not match any predictions to Polymarket markets",
                "predictions_count": len(preds),
                "markets_count": len(game_markets),
                "pred_teams": preds[["HOME_TEAM", "AWAY_TEAM"]].head(10).to_dict(orient="records"),
                "market_teams": game_markets[["team_a", "team_b"]].head(10).to_dict(orient="records"),
            })

        # Record every matched prediction + open paper trades (CLV ledger).
        # Best-effort: the report must still generate if the ledger write fails.
        try:
            from tools import market
            ledger_conn = storage.init_db()
            market.record_predictions_and_trades(
                ledger_conn, edges_df,
                experiment_id=best["experiment_id"],
                model_desc=f"{best['method']} ({best['metrics']['log_loss']:.4f} LL)",
            )
            ledger_conn.close()
        except Exception as e:
            logger.warning(f"Prediction ledger write failed: {e}")

        # Prepare result
        edges_list = []
        for _, row in edges_df.iterrows():
            edges_list.append({
                "home_team": row["HOME_TEAM"],
                "away_team": row["AWAY_TEAM"],
                "model_prob": row["model_prob"],
                "market_prob": row["market_prob"],
                "edge": row["edge"],
                "abs_edge": row["abs_edge"],
                "bet_direction": row["bet_direction"],
                "confidence": row["confidence"],
                "volume": row.get("market_volume", 0),
                "liquidity": row.get("market_liquidity", 0),
                "game_time": str(row.get("game_time", "")),
            })

        # Summary stats
        high_conf = [e for e in edges_list if e["confidence"] == "HIGH"]
        med_conf = [e for e in edges_list if e["confidence"] == "MEDIUM"]

        return json.dumps({
            "status": "success",
            "model_used": best["method"],
            "model_log_loss": best["metrics"]["log_loss"],
            "total_matched": len(edges_list),
            "high_confidence_edges": len(high_conf),
            "medium_confidence_edges": len(med_conf),
            "edges": edges_list,
            "disclaimer": "These are model estimates, NOT trading recommendations. The model has ~64.5% accuracy — significant uncertainty remains on every prediction.",
        }, indent=2, default=str)

    def save_html_report(
        executive_summary: str,
        analysis_text: str = "",
        next_steps: list[dict] = None,
    ) -> str:
        """Save a structured HTML report using the standard template.

        Gathers all experiment data, run history, and edges automatically from the DB.
        The LLM only needs to provide the narrative text and recommendations.
        """
        import os

        report_dir = BASE_DIR / "data" / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)

        # --- Gather data from DB ---
        conn = storage.init_db()
        all_experiments = storage.get_experiment_history(conn, limit=100)

        # Get run history for convergence chart
        cursor = conn.execute(
            "SELECT run_id, started_at, finished_at, status, summary FROM run_log ORDER BY started_at ASC"
        )
        run_rows = cursor.fetchall()
        conn.close()

        # Build completed experiments list sorted by log_loss
        completed = []
        for exp in all_experiments:
            m = exp.get("metrics", {})
            if exp["status"] == "completed" and m.get("log_loss"):
                # Accuracy is stored as decimal (0.6532) — convert to percentage for display
                raw_acc = m.get("accuracy", 0)
                acc_pct = raw_acc * 100 if raw_acc < 1 else raw_acc
                completed.append({
                    "name": exp["name"],
                    "method": exp["method"],
                    "log_loss": m["log_loss"],
                    "accuracy": acc_pct,
                    "brier_score": m.get("brier_score", 0),
                    "created_at": exp.get("created_at", ""),
                })
        completed.sort(key=lambda x: x["log_loss"])

        # Best model
        best = completed[0] if completed else {"name": "N/A", "method": "N/A", "log_loss": 0, "accuracy": 0, "brier_score": 0}

        # Build convergence data: best log_loss per run using time windows
        runs_data = []
        sorted_runs = [(r[0], r[1], r[2]) for r in run_rows if r[3] == "completed"]

        if sorted_runs and all_experiments:
            # Sort experiments by creation time
            timed_exps = []
            for e in all_experiments:
                m = e.get("metrics", {})
                if e["status"] == "completed" and m.get("log_loss"):
                    timed_exps.append({
                        "created_at": e.get("created_at", ""),
                        "log_loss": m["log_loss"],
                    })
            timed_exps.sort(key=lambda x: x["created_at"])

            cumulative_best = float("inf")
            for i, (rid, started, finished) in enumerate(sorted_runs, 1):
                # Find experiments created within this run's time window
                run_end = finished or datetime.now().isoformat()
                run_exps_in_window = [
                    e for e in timed_exps
                    if e["created_at"] >= started and e["created_at"] <= run_end
                ]

                # Update cumulative best with this run's experiments
                for e in run_exps_in_window:
                    if e["log_loss"] < cumulative_best:
                        cumulative_best = e["log_loss"]

                # Count total experiments up to this run
                total_exps_so_far = len([
                    e for e in timed_exps if e["created_at"] <= run_end
                ])

                runs_data.append({
                    "iteration": i,
                    "best_log_loss": cumulative_best if cumulative_best < float("inf") else 0.693,
                    "experiments": total_exps_so_far,
                })

        # Try to get edges data
        edges_data = []
        try:
            edges_result = json.loads(compute_betting_edges())
            if edges_result.get("status") == "success":
                edges_data = edges_result.get("edges", [])
        except Exception:
            pass

        # Determine iteration number
        iteration = len([r for r in run_rows if r[3] == "completed"])

        # Build the data dict
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        data = {
            "iteration": iteration,
            "timestamp": ts,
            "run_id": run_rows[-1][0] if run_rows else "unknown",
            "best_log_loss": best["log_loss"],
            "best_accuracy": best["accuracy"],
            "total_experiments": len(completed),
            "calibration_error": "N/A",
            "edges_found": len([e for e in edges_data if e.get("confidence") == "HIGH"]),
            "executive_summary": executive_summary,
            "best_model": best,
            "runs": runs_data,
            "experiments": completed,
            "edges": edges_data,
            "next_steps": next_steps or [],
            "analysis_text": analysis_text,
        }

        # Render HTML
        html_content = html_report.render_report(data)

        # Save with timestamp
        filename = f"predicto_report_iter{iteration}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        path = report_dir / filename
        path.write_text(html_content)
        logger.info(f"HTML report saved to {path}")

        return json.dumps({
            "status": "saved",
            "path": str(path),
            "iteration": iteration,
            "total_experiments": len(completed),
            "edges_found": len(edges_data),
        })

    def save_report(report_content: str, report_title: str = "predicto_report") -> str:
        """Save the generated report as a markdown file."""
        report_dir = BASE_DIR / "data" / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{report_title}_{ts}.md"
        path = report_dir / filename

        path.write_text(report_content)
        logger.info(f"Report saved to {path}")

        return json.dumps({"status": "saved", "path": str(path)})

    return [
        Tool(
            name="get_promoted_model",
            description="Get information about the best-performing model.",
            input_schema={"type": "object", "properties": {}},
            func=get_promoted_model,
        ),
        Tool(
            name="get_all_experiment_summary",
            description="Get summary of all experiments that were run.",
            input_schema={"type": "object", "properties": {}},
            func=get_all_experiment_summary,
        ),
        Tool(
            name="get_upcoming_predictions",
            description="Get model predictions for upcoming NBA games.",
            input_schema={"type": "object", "properties": {}},
            func=get_upcoming_predictions,
        ),
        Tool(
            name="get_market_data",
            description="Get Polymarket NBA market data for comparison.",
            input_schema={"type": "object", "properties": {}},
            func=get_market_data,
        ),
        Tool(
            name="compute_betting_edges",
            description="Match model predictions to live Polymarket game odds. Returns edge (model vs market) for each game, ranked by confidence. THIS IS THE KEY TOOL for finding potential mispricings.",
            input_schema={"type": "object", "properties": {}},
            func=compute_betting_edges,
        ),
        Tool(
            name="save_html_report",
            description="Save a structured HTML report using the standard Predicto template. Gathers all experiment history, convergence data, and betting edges automatically. You provide the narrative text. THIS IS THE PREFERRED way to save reports.",
            input_schema={
                "type": "object",
                "properties": {
                    "executive_summary": {
                        "type": "string",
                        "description": "1-3 paragraph executive summary of this iteration's results and key findings"
                    },
                    "analysis_text": {
                        "type": "string",
                        "description": "Detailed analysis text — model performance insights, calibration notes, market comparison"
                    },
                    "next_steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "description": {"type": "string"},
                                "impact": {"type": "string", "enum": ["HIGH", "MEDIUM"]},
                            },
                        },
                        "description": "Recommended next steps with title, description, and impact level"
                    },
                },
                "required": ["executive_summary"]
            },
            func=save_html_report,
        ),
        Tool(
            name="save_report",
            description="Save the final report as a markdown file. Use save_html_report instead for the standard HTML template.",
            input_schema={
                "type": "object",
                "properties": {
                    "report_content": {
                        "type": "string",
                        "description": "The full markdown report content"
                    },
                    "report_title": {
                        "type": "string",
                        "description": "Title for the report file (default: predicto_report)"
                    },
                },
                "required": ["report_content"]
            },
            func=save_report,
        ),
    ]


SYSTEM_PROMPT = """\
You are the Report Agent for Predicto, an NBA prediction system.

Your job is to **generate a comprehensive, evidence-based analysis report**.

## Report structure:
1. **Executive Summary** — one paragraph: what was analyzed, what's the key finding
2. **Data Summary** — games analyzed, date range, data quality
3. **Model Comparison** — table of all methods tried with metrics
4. **Best Model Analysis** — why it's the best, feature importance, calibration
5. **Upcoming Game Predictions** (if available) — ranked by confidence
6. **BETTING EDGE ANALYSIS** — USE compute_betting_edges to compare model vs Polymarket odds. This is the MOST IMPORTANT section. Show every matched game with model prob, market prob, edge, and confidence.
7. **Conclusions & Recommendations** — honest assessment, what to improve next

## CRITICAL: Always call compute_betting_edges before writing the report. This compares your model's predictions against live Polymarket odds and finds where the model disagrees with the market.

## Report guidelines:
- Use markdown formatting with tables
- Include actual numbers — log loss, Brier score, accuracy
- Be HONEST about uncertainty and limitations
- If no edge over the market exists, say so clearly
- Flag low-confidence predictions
- Include spread/liquidity warnings for any potential mispricings

## For potential mispricings, include:
- Game details (teams, date)
- Model probability with uncertainty note
- Market implied probability
- Edge size and direction
- Confidence level (HIGH/MEDIUM/LOW)
- Liquidity/spread context

ALWAYS save the report using save_html_report (not save_report). Provide:
- executive_summary: 1-3 paragraph summary of findings
- analysis_text: detailed analysis
- next_steps: list of recommended improvements with title, description, and impact (HIGH/MEDIUM)

The HTML template automatically pulls all experiment data, convergence history, and betting edges.
"""


def create_report_agent(config: dict) -> Agent:
    """Create and return the Report Agent."""
    return Agent(
        name="report_agent",
        system_prompt=SYSTEM_PROMPT,
        tools=_build_tools(config),
        model=config.get("models", {}).get("agent_model", "claude-sonnet-4-20250514"),
        max_tokens=8192,  # reports can be long
    )
