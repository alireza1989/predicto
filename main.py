#!/usr/bin/env python3
"""Predicto: Multi-Agent NBA Prediction System

Orchestrates 5 agents in sequence:
1. Data Agent — fetches NBA games + Polymarket markets
2. Feature Agent — computes features (Elo, form, rest)
3. Meta-Scientist — designs and runs ML experiments
4. Eval Agent — evaluates models, picks the best
5. Report Agent — generates final analysis report
"""

import argparse
import json
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

from agents.data_agent import create_data_agent
from agents.feature_agent import create_feature_agent
from agents.meta_scientist import create_meta_scientist
from agents.eval_agent import create_eval_agent
from agents.report_agent import create_report_agent
from tools.storage import init_db, log_run, finish_run

# Load .env for ANTHROPIC_API_KEY
load_dotenv(override=True)

BASE_DIR = Path(__file__).parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(BASE_DIR / "data" / "predicto.log"),
    ],
)
logger = logging.getLogger("predicto")


def load_config() -> dict:
    config_path = BASE_DIR / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    logger.warning("No config.yaml found, using defaults")
    return {}


def run_pipeline(config: dict, skip_data: bool = False, skip_markets: bool = False):
    """Run the full Predicto pipeline."""

    run_id = str(uuid.uuid4())[:8]
    started = datetime.now()
    logger.info(f"{'='*60}")
    logger.info(f"Predicto Pipeline Run: {run_id}")
    logger.info(f"Started: {started.isoformat()}")
    logger.info(f"{'='*60}")

    # Ensure data dirs exist
    for d in ["data/raw", "data/features", "data/experiments", "data/reports"]:
        (BASE_DIR / d).mkdir(parents=True, exist_ok=True)

    # Init DB and log run
    conn = init_db()
    log_run(conn, run_id, config)
    conn.close()

    seasons = config.get("seasons", ["2023-24", "2024-25"])
    horizon = config.get("horizon_days", 14)
    agent_results = {}

    # ── Step 1: Data Collection ──────────────────────────────────────────
    if not skip_data:
        logger.info("\n" + "="*60)
        logger.info("STEP 1: Data Collection Agent")
        logger.info("="*60)

        data_agent = create_data_agent(config)

        task = f"""Collect NBA data for the prediction system.

1. Fetch historical NBA game results for seasons: {seasons}
2. Fetch player game logs for the SAME seasons: {seasons} — call fetch_player_logs with the same season list. This is critical for roster-strength features in the Feature Agent.
3. Fetch upcoming NBA games for the next {horizon} days
4. Fetch advanced team stats for the most recent season ({seasons[-1]})"""

        if not skip_markets:
            task += f"""
4. Fetch Polymarket GAME-LEVEL markets using fetch_polymarket_game_markets — this gets actual moneyline odds for specific NBA games (most important for edge analysis)
5. Optionally also search broader Polymarket NBA markets"""

        result = data_agent.run(task)
        agent_results["data_agent"] = result
        logger.info(f"Data Agent completed. Tool calls: {result['tool_calls']}")
        print(f"\n--- Data Agent Summary ---\n{result['response'][:1000]}\n")
    else:
        logger.info("Skipping data collection (--skip-data)")

    # ── Step 2: Feature Engineering ──────────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("STEP 2: Feature Engineering Agent")
    logger.info("="*60)

    feature_agent = create_feature_agent(config)
    result = feature_agent.run(
        "Compute and validate all features from the raw NBA matchup data. "
        "Check for data quality issues and potential leakage."
    )
    agent_results["feature_agent"] = result
    logger.info(f"Feature Agent completed. Tool calls: {result['tool_calls']}")
    print(f"\n--- Feature Agent Summary ---\n{result['response'][:1000]}\n")

    # ── Step 3: Meta-Scientist ───────────────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("STEP 3: Meta-Scientist Agent")
    logger.info("="*60)

    scientist = create_meta_scientist(config)
    max_experiments = config.get("experiments", {}).get("max_per_run", 5)
    result = scientist.run(
        f"""Design and run up to {max_experiments} experiments to find the best NBA prediction model.

Your goal: find the method (or combination of methods) that produces the best-calibrated
win probabilities, as measured by log loss and Brier score using time-series cross-validation.

FIRST: Call read_scientist_history to see accumulated learnings from past runs.
THEN: Check experiment history in the database.

If no history exists, run this sequence:
1. Logistic Regression baseline (all features)
2. Gradient Boosting (all features)
3. LightGBM (all features)
4. Feature ablation study
5. Ensemble of the best methods

If history exists, analyze what worked and design follow-up experiments to improve.

After all experiments:
1. Provide a clear conclusion about the best approach with evidence
2. Call update_scientist_history with a summary of what you tried, results, and recommendations for the next iteration"""
    )
    agent_results["meta_scientist"] = result
    logger.info(f"Meta-Scientist completed. Tool calls: {result['tool_calls']}")
    print(f"\n--- Meta-Scientist Summary ---\n{result['response'][:2000]}\n")

    # ── Step 4: Evaluation ───────────────────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("STEP 4: Evaluation Agent")
    logger.info("="*60)

    eval_agent = create_eval_agent(config)
    result = eval_agent.run(
        "Evaluate all completed experiments. Compare them side by side. "
        "Check calibration of the best model. Make a promote/reject decision "
        "with clear reasoning. Be honest — if no model beats the relevant baselines, say so."
    )
    agent_results["eval_agent"] = result
    logger.info(f"Eval Agent completed. Tool calls: {result['tool_calls']}")
    print(f"\n--- Eval Agent Summary ---\n{result['response'][:1000]}\n")

    # ── Step 5: Report Generation ────────────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("STEP 5: Report Agent")
    logger.info("="*60)

    report_agent = create_report_agent(config)

    # Pass summaries from previous agents as context
    context = {
        "data_summary": agent_results.get("data_agent", {}).get("response", "No data agent run")[:500],
        "feature_summary": agent_results.get("feature_agent", {}).get("response", "")[:500],
        "scientist_summary": agent_results.get("meta_scientist", {}).get("response", "")[:1000],
        "eval_summary": agent_results.get("eval_agent", {}).get("response", "")[:500],
    }

    result = report_agent.run(
        "Generate the final Predicto analysis report. "
        "IMPORTANT: Use compute_betting_edges to compare model predictions against live Polymarket odds — "
        "this is the most valuable section showing where our model disagrees with the market. "
        "Also include experiment results, best model analysis, and upcoming predictions. "
        "Save the report using save_html_report with executive_summary, analysis_text, and next_steps.",
        context=context,
    )
    agent_results["report_agent"] = result
    logger.info(f"Report Agent completed. Tool calls: {result['tool_calls']}")
    print(f"\n--- Report Agent Summary ---\n{result['response'][:1000]}\n")

    # ── Finish ───────────────────────────────────────────────────────────
    elapsed = (datetime.now() - started).total_seconds()
    total_tool_calls = sum(r.get("tool_calls", 0) for r in agent_results.values())

    summary = (
        f"Pipeline completed in {elapsed:.0f}s. "
        f"Agents: {len(agent_results)}, Total tool calls: {total_tool_calls}"
    )

    conn = init_db()
    finish_run(conn, run_id, "completed", summary)
    conn.close()

    logger.info(f"\n{'='*60}")
    logger.info(f"Pipeline Complete: {summary}")
    logger.info(f"{'='*60}")

    return agent_results


def main():
    parser = argparse.ArgumentParser(description="Predicto: NBA Prediction Pipeline")
    parser.add_argument("--skip-data", action="store_true",
                        help="Skip data collection (use existing data)")
    parser.add_argument("--skip-markets", action="store_true",
                        help="Skip Polymarket data (NBA data only)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable debug logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = load_config()
    run_pipeline(config, skip_data=args.skip_data, skip_markets=args.skip_markets)


if __name__ == "__main__":
    main()
