# Predicto

An autonomous, multi-agent system that predicts NBA game outcomes using machine learning, then compares those predictions against live Polymarket betting odds to surface potential mispricings. The system is self-improving: it runs experiments, learns from results, and converges toward better models across iterations — without human intervention.

## Live demo

**Dashboard:** https://predicto-dashboard.vercel.app/?token=predicto-demo-2026

The dashboard is token-gated to keep random visitors and crawlers out — the link
above includes the demo token (`predicto-demo-2026`). If you can see this repo,
you're welcome in. (You can also open the bare URL and paste the token into the
gate page; access persists via cookie for 30 days.)

> **Platform upgrade (July 2026)** — Predicto is now a production meta data-scientist
> platform. See [docs/MASTER_PLAN.md](docs/MASTER_PLAN.md) for the full roadmap. Highlights:
>
> - **Neon Postgres** is the production store (`DATABASE_URL`; SQLite fallback locally)
> - **Live dashboard** (Next.js on Vercel) — link above
> - **Prediction ledger + CLV**: every prediction is recorded with the market price at
>   that moment; paper trades (¼-Kelly) settle against closing lines — the
>   industry-standard test of real edge
> - **Critic agent** red-teams every model promotion (leakage/overfit/noise audit
>   with a paired significance test) and has veto power
> - **Modern model search**: Optuna hyperparameter search, CatBoost, TabPFN v2
>   (tabular foundation model), isotonic calibration on out-of-fold predictions
> - **Injury intelligence**: ESPN injury report with player-impact scoring, accruing
>   a historical archive for future training features
> - **GitHub Actions**: nightly pipeline + 30-minute odds snapshots during game hours
> - **Leakage tests in CI**: features provably use only past data

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Getting Started](#getting-started)
4. [Running the Pipeline](#running-the-pipeline)
5. [Configuration Reference](#configuration-reference)
6. [Data: Types, Sources, and Features](#data-types-sources-and-features)
7. [Pipeline Deep Dive](#pipeline-deep-dive)
8. [Output Files and Where to Find Them](#output-files-and-where-to-find-them)
9. [Interpreting Reports](#interpreting-reports)
10. [Web Dashboard](#web-dashboard)
11. [Project Structure](#project-structure)
12. [Technology Stack](#technology-stack)
13. [Extending the System](#extending-the-system)
14. [Troubleshooting](#troubleshooting)

---

## Overview

Predicto has three core goals:

1. **Predict** the probability that the home team wins an NBA game.
2. **Compare** those model predictions against Polymarket's implied probabilities.
3. **Surface edges** — games where the model's view and the market's view differ enough to be interesting.

The prediction engine is not static. A "Meta-Scientist" agent continuously runs experiments (Logistic Regression, Gradient Boosting, XGBoost, LightGBM, Neural Nets, Ensembles), evaluates them with proper scoring rules, and records its findings to a persistent learning log. Each pipeline run builds on the last.

After 36+ experiments across 5 iterations, the system converged on:

- **Best method:** Logistic Regression with L2 regularization, 8-14 selected features
- **Best log loss:** 0.6206 (vs. 0.693 baseline — random chance)
- **Best accuracy:** 65.3% (vs. 55.3% home-team-always baseline)

---

## Architecture

Three layers: **compute** (the agent pipeline, running on GitHub Actions nightly or
locally), **data** (Neon Postgres — the single source of truth; SQLite fallback when
`DATABASE_URL` is unset), and **presentation** (the Next.js dashboard on Vercel,
read-only). Vercel never runs the pipeline — its function limits can't hold a
15-minute multi-agent run; it only renders whatever is in Neon.

Six autonomous agents run sequentially. Each has its own tools, a Claude model
backing it, and a well-defined input/output contract:

```
┌──────────────────┐
│  Data Agent      │  NBA results (6,100+ games, 2021–26) + schedule + Polymarket
│                  │  odds + ESPN injury report (accrues a historical archive)
└────────┬─────────┘
         │  data/raw/*.parquet + injury_snapshots (DB)
         ▼
┌──────────────────┐
│  Market Ops      │  DETERMINISTIC (no LLM): settles past predictions against
│  (step 1.5)      │  final scores, computes CLV + paper-trade PnL, snapshots odds
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Feature Agent   │  Computes ~88 features (Elo, rolling form, rest, momentum,
│                  │  H2H, SOS, roster strength); validates for leakage and nulls
└────────┬─────────┘
         │  data/features/feature_matrix.parquet
         ▼
┌──────────────────┐
│  Meta-Scientist  │  Designs and runs ML experiments: Optuna searches, CatBoost,
│  Agent           │  TabPFN v2, ensembles. Reads structured findings — never
│                  │  re-tests settled questions. Claims improvements only with
│                  │  a paired significance test.
└────────┬─────────┘
         │  data/experiments/{id}/ + experiment_log (DB)
         ▼
┌──────────────────┐
│  Evaluation      │  Compares all experiments, checks calibration, nominates a
│  Agent           │  winner vs. naive and market baselines
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Critic Agent    │  RED TEAM with veto power: leakage audit (implausibly good
│                  │  metrics), fold-stability check, paired significance test
│                  │  vs. the incumbent champion. Verdicts: approved /
│                  │  approved_with_caution / vetoed → promotions (DB)
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Report Agent    │  Predictions for upcoming games, betting edges vs. Polymarket
│                  │  (injury-annotated), logs every prediction to the CLV ledger,
│                  │  opens ¼-Kelly paper trades on ≥5% edges, HTML report
└────────┬─────────┘
         │  predictions + paper_trades (DB), data/reports/*.html
         ▼
   Neon Postgres ──▶ Next.js dashboard (Vercel) — see Live demo above
```

**Agent models:**
- Data, Feature, Eval, Report, Critic agents → `claude-sonnet-5` (fast, capable)
- Meta-Scientist → `claude-opus-4-8` (deeper reasoning for experiment design)

---

## Getting Started

### Prerequisites

- Python 3.9 or higher
- An [Anthropic API key](https://console.anthropic.com/) (Claude is the reasoning engine for all agents)

### Installation

```bash
git clone <repo-url>
cd predicto
pip install -r requirements.txt
```

### Environment Setup

```bash
cp .env.example .env
# Edit .env and add:
ANTHROPIC_API_KEY=sk-ant-...          # required — agents are Claude-backed
DATABASE_URL=postgres://...           # optional — Neon Postgres; omits → local SQLite
```

**Where runs happen:** the same `python main.py` runs locally or on GitHub Actions
(nightly at 10:07 UTC, or Actions → "Nightly Pipeline" → Run workflow). Both write
to the same database when `DATABASE_URL` is set, so the dashboard reflects either
within 5 minutes. CI needs two repo secrets: `ANTHROPIC_API_KEY` and `DATABASE_URL`.

### Verify Setup

```bash
python -c "import anthropic; print('OK')"
python -c "from nba_api.stats.static import teams; print(len(teams.get_teams()), 'teams')"
```

---

## Running the Pipeline

### Full Run (recommended first time)

```bash
python main.py
```

This runs all five agents end-to-end. Expect 5–10 minutes depending on API speed and how many experiments the Meta-Scientist runs.

### Skipping Stages (faster iteration)

```bash
# Reuse existing NBA/Polymarket data — skips API calls entirely
python main.py --skip-data

# Skip Polymarket fetch only (use if rate-limited or testing offline)
python main.py --skip-markets

# Enable debug logging
python main.py --verbose
```

**When to use `--skip-data`:** After your first successful run the raw data is saved to `data/raw/`. You can skip re-fetching it on subsequent runs to save time and API rate limits while iterating on features or models.

### Watching Progress in Real Time

```bash
tail -f data/predicto.log
```

All agent actions, tool calls, and decisions are written to this file.

---

## Configuration Reference

All pipeline settings live in `config.yaml`:

```yaml
sport: nba

# Seasons to collect. Each season adds ~1,200 games to the dataset.
seasons:
  - "2021-22"
  - "2022-23"
  - "2023-24"
  - "2024-25"
  - "2025-26"

# Days ahead to fetch upcoming games for predictions
horizon_days: 14

models:
  # Most agents use Sonnet (faster, cheaper)
  agent_model: "claude-sonnet-4-20250514"
  # Meta-Scientist uses Opus for deeper reasoning
  scientist_model: "claude-opus-4-20250514"
  max_tokens: 4096

polymarket:
  gamma_url: "https://gamma-api.polymarket.com"
  clob_url: "https://clob.polymarket.com"
  request_delay: 0.5   # seconds between requests (respect rate limits)

nba:
  request_delay: 1.0   # seconds between NBA API calls
  timeout: 30

experiments:
  max_per_run: 5        # max experiments the Meta-Scientist runs per pipeline execution
  min_test_samples: 50  # minimum test-set size in each CV fold
  significance_level: 0.05

storage:
  db_path: "data/predicto.db"
  raw_dir: "data/raw"
  features_dir: "data/features"
  experiments_dir: "data/experiments"
  reports_dir: "data/reports"
```

**Common config changes:**
- **Add a new season:** Append to `seasons`. Each season takes ~30 seconds to fetch.
- **More experiments per run:** Increase `max_per_run` (costs more API tokens).
- **Faster iteration:** Decrease `max_per_run` to 1–2 while debugging.
- **Closer horizon:** Decrease `horizon_days` if you only care about tomorrow's games.

---

## Data: Types, Sources, and Features

### External Data Sources

| Source | What it provides | How accessed |
|--------|-----------------|--------------|
| NBA Stats API (`nba_api`) | Historical game results, box scores | `tools/nba.py` |
| NBA CDN (JSON endpoint) | Upcoming game schedule (next 14 days) | `tools/nba.py` |
| Polymarket Gamma API | Active NBA prediction market listings | `tools/polymarket.py` |
| Polymarket CLOB API | Live moneyline odds and orderbook depth | `tools/polymarket.py` |

### Raw Data Files (`data/raw/`)

| File | Schema | Description |
|------|--------|-------------|
| `nba_matchups.parquet` | `GAME_ID, GAME_DATE, HOME_TEAM, AWAY_TEAM, HOME_PTS, AWAY_PTS, HOME_WIN, SEASON` | ~6,075 historical games (2021–26) |
| `upcoming_games.parquet` | `GAME_DATE, HOME_TEAM, AWAY_TEAM` | Next 14 days of games |
| `polymarket_game_markets.parquet` | `event_title, team_a_price, team_b_price, volume, liquidity` | Live market odds as decimals (0–1) |

`HOME_WIN` is the binary target variable: `1` = home team won, `0` = away team won.

### Feature Matrix (`data/features/feature_matrix.parquet`)

The Feature Agent computes **25 features** from raw matchup data. All features are computed strictly from information available *before* each game (no leakage). The feature matrix is the input to every ML experiment.

#### Feature Categories

**1. Elo Ratings (4 features) — Most predictive group**

Elo is a zero-sum rating system. Each game shifts ratings based on outcome vs. expectation.

| Feature | Description |
|---------|-------------|
| `home_elo_pre` | Home team's Elo rating before the game |
| `away_elo_pre` | Away team's Elo rating before the game |
| `elo_diff` | `home_elo_pre - away_elo_pre` |
| `home_elo_expected` | Expected home win probability: `1 / (1 + 10^(-elo_diff/400))` |

*Parameters: K-factor = 20, home-court advantage = 100 Elo points, season reversion = 75% toward 1500.*

**2. Rolling Form Statistics (16 features)**

Computed over the last 5 and 10 games for both teams:

| Feature (prefix: `home_` or `away_`) | Description |
|--------------------------------------|-------------|
| `*_win_pct_5` / `*_win_pct_10` | Win rate in last 5 / 10 games |
| `*_pts_scored_5` / `*_pts_scored_10` | Avg points scored in last 5 / 10 games |
| `*_pts_allowed_5` / `*_pts_allowed_10` | Avg points allowed in last 5 / 10 games |
| `*_net_pts_5` / `*_net_pts_10` | Net point differential in last 5 / 10 games |

**3. Rest and Schedule (5 features)**

| Feature | Description |
|---------|-------------|
| `home_rest_days` | Days since home team last played |
| `away_rest_days` | Days since away team last played |
| `home_is_b2b` | 1 if home team is on a back-to-back |
| `away_is_b2b` | 1 if away team is on a back-to-back |
| `rest_advantage` | `home_rest_days - away_rest_days` |

**Feature validation** runs after every computation:
- Null rate must be < 5% per feature
- Elo-outcome correlation must be > 0.05 (sanity check)
- All features verified to precede their target game date

Code: `tools/features.py`

---

## Pipeline Deep Dive

### Stage 1: Data Agent (`agents/data_agent.py`)

Calls the NBA Stats API across all configured seasons, normalizes game records, fetches the upcoming schedule, then queries Polymarket for active NBA markets and their current odds.

**Key tools:**
- `fetch_nba_games(season)` → saves `nba_matchups.parquet`
- `fetch_upcoming_nba_games(days)` → saves `upcoming_games.parquet`
- `fetch_polymarket_game_markets()` → saves `polymarket_game_markets.parquet`

**Tips:**
- NBA Stats API can be slow or occasionally rate-limit. If it hangs, try again with `--skip-data` and existing data.
- Polymarket data is only meaningful near game time when market liquidity is high.

### Stage 2: Feature Agent (`agents/feature_agent.py`)

Loads `nba_matchups.parquet`, calls `tools/features.py` to compute the full feature matrix, validates it, and saves `feature_matrix.parquet`.

**Key tools:**
- `load_raw_matchups()` — loads raw data
- `compute_all_features()` — computes all 25 features with leakage checks
- `validate_features()` — checks nulls, ranges, correlations
- `get_feature_info()` — returns schema and statistics

### Stage 3: Meta-Scientist Agent (`agents/meta_scientist.py`)

This is the core ML engine. It:

1. Reads `data/scientist_history.md` — its persistent memory of what worked and what didn't
2. Queries the experiment database for past results
3. Designs and runs up to `max_per_run` new experiments
4. Tries methods: Logistic Regression, Gradient Boosting, LightGBM, XGBoost, MLP, Ensembles
5. Runs feature ablation to find the most predictive subset
6. Analyzes prediction errors
7. Updates `scientist_history.md` with new findings and recommendations for the next run

**Time-series cross-validation:** Experiments use walk-forward CV — folds are sorted chronologically so the model is never trained on future data. This is critical for sports prediction.

**Key tools:**
- `run_single_experiment(method, features, hyperparams)` → trains and evaluates one model
- `run_ensemble(experiment_ids)` → combines multiple models
- `run_feature_ablation(base_experiment_id)` → tests feature subsets
- `analyze_prediction_errors(experiment_id)` → diagnoses where the model fails

**Experiment output:** Each experiment is saved under `data/experiments/{experiment_id}/` and recorded in `data/predicto.db`.

### Stage 4: Evaluation Agent (`agents/eval_agent.py`)

Compares all completed experiments and picks the best model to promote.

**Metrics used:**

| Metric | What it measures | Good value |
|--------|-----------------|------------|
| Log Loss | Probability accuracy (penalizes confident wrong predictions) | < 0.693 (random) |
| Brier Score | Mean squared error of probabilities | < 0.25 |
| Accuracy | % predictions correct | > 55.3% (home-always baseline) |

**Baselines:**
- **Naive baseline:** Always predict home team wins (55.3% accuracy in NBA)
- **Market baseline:** Use Polymarket implied probability as the prediction

**Calibration check:** The agent checks whether a model that says "60% home win" is actually right ~60% of the time. Poor calibration means probabilities can't be trusted.

### Stage 5: Report Agent (`agents/report_agent.py`)

Uses the promoted model and Polymarket data to:
1. Generate predictions for upcoming games
2. Compute betting edges (model probability minus market implied probability)
3. Build and save an HTML report

**Edge threshold:** 5% — a game is flagged as an edge if `|model_prob - market_prob| > 0.05`.

---

## Output Files and Where to Find Them

```
data/
├── raw/
│   ├── nba_matchups.parquet          # Historical game results
│   ├── upcoming_games.parquet        # Next 14 days schedule
│   └── polymarket_game_markets.parquet  # Live market odds
│
├── features/
│   └── feature_matrix.parquet        # 25 computed features, one row per game
│
├── experiments/
│   └── {experiment_id}/              # One directory per experiment
│       ├── model.pkl                 # Serialized model
│       ├── predictions.parquet       # OOF + test predictions
│       └── metadata.json            # Hyperparams, metrics, feature list
│
├── reports/
│   ├── predicto_report_*.html        # Primary interactive report (open in browser)
│   ├── predicto_*.md                 # Markdown text report
│   ├── 01_ARCHITECTURE_DOC.md        # Architecture documentation (auto-generated)
│   └── 02_EXPERIMENTATION_RESULTS.md # Full experimentation history
│
├── predicto.db                       # SQLite: experiment logs, run metadata
├── predicto.log                      # Full application log
└── scientist_history.md              # Persistent learning log (read by Meta-Scientist)
```

**Quickest way to see results:**

```bash
# Open the latest HTML report in your browser
open data/reports/$(ls -t data/reports/*.html | head -1)

# Or check the log for the last run summary
tail -100 data/predicto.log
```

**Querying the database:**

```bash
sqlite3 data/predicto.db

# List all experiments and metrics
SELECT experiment_id, method, log_loss, accuracy FROM experiments ORDER BY log_loss ASC;

# See promoted models
SELECT * FROM promoted_models;
```

---

## Interpreting Reports

The HTML report is divided into five sections:

### 1. KPI Cards (top of report)

| KPI | What it means |
|-----|---------------|
| **Iteration #** | How many full pipeline runs have completed |
| **Best Log Loss** | Best probability accuracy achieved (0.693 = random; lower is better) |
| **Best Accuracy** | Best % correct predictions (55.3% = always-home baseline) |
| **Total Experiments** | Cumulative experiments run across all iterations |
| **Market Edges Found** | Games in next 14 days where model disagrees with Polymarket by >5% |

### 2. Convergence Chart

A line chart showing the best log loss achieved per pipeline iteration. Watch for:
- **Trending down:** The system is improving
- **Plateau:** The system has found its optimum — more iterations won't help
- **Below 0.62:** Excellent; you're approaching market efficiency

### 3. Experiment Summary Table

All experiments sorted by log loss. Columns:
- **Method:** Algorithm used (LR, GB, XGB, LGBM, MLP, Ensemble)
- **Features:** Number of features or feature subset name
- **Log Loss:** Primary metric (lower is better)
- **Brier Score:** Secondary probability metric
- **Accuracy:** Classification accuracy

The promoted model (best log loss) is highlighted.

### 4. Best Model Analysis

- **Feature importances:** Which features drive predictions most (Elo is almost always #1)
- **Calibration curve:** Plots predicted probability on X-axis vs. actual win rate on Y-axis. A perfectly calibrated model follows the diagonal. Deviations indicate systematic over- or under-confidence.

### 5. Upcoming Predictions and Betting Edges

**Predictions table:**

| Column | Description |
|--------|-------------|
| Game | HOME vs AWAY (date) |
| Model Prob | Predicted probability that home team wins |
| Market Prob | Polymarket implied probability |
| Edge | `Model Prob - Market Prob` |
| Direction | `MODEL_FAVORS_HOME` or `MODEL_FAVORS_AWAY` |

**Interpreting edges:**

| Edge magnitude | Interpretation |
|----------------|----------------|
| < 3% | Noise — within model uncertainty |
| 3–5% | Mild disagreement — worth watching |
| 5–10% | Meaningful disagreement — potential opportunity |
| > 10% | Strong disagreement — model very confident vs. market |

**Important caveats:**
- Edges are more actionable on markets with **high liquidity** (high volume in the Polymarket data)
- Polymarket odds change right up to game time — run the pipeline close to tip-off for freshest numbers
- No prediction system beats a well-calibrated market consistently at scale. These edges are signals, not guarantees.

---

## Web Dashboard

### Production (Next.js on Vercel)

The production dashboard lives in `web-next/` and reads Neon directly
(read-only, revalidates every 5 minutes). See **Live demo** at the top for the
URL + access token.

Pages: **Overview** (KPIs, champion model + critic verdict, convergence chart,
scientist findings, recent runs), **Experiments** (all experiments ranked by
walk-forward log loss), **Predictions** (the CLV ledger — model vs. market at
prediction time, settled results), **Performance** (paper-trading ROI, average
CLV, positive-CLV rate).

```bash
cd web-next && npm install && npm run dev   # local dev (needs DATABASE_URL)
vercel deploy --prod                        # deploy
```

Access is gated by `web-next/middleware.ts` (demo token, cookie for 30 days;
override the token with a `DEMO_TOKEN` env var on Vercel).

### Legacy (Flask, local only)

```bash
python web/app.py   # http://localhost:5000 — old report browser / run trigger
```

---

## Project Structure

```
predicto/
├── agents/
│   ├── base.py              # Base agent class: tool execution loop, Claude API calls
│   ├── data_agent.py        # Stage 1: NBA games + Polymarket odds + injury report
│   ├── feature_agent.py     # Stage 2: Compute ~88 features, validate for leakage
│   ├── meta_scientist.py    # Stage 3: Design/run experiments (Optuna, TabPFN, ...)
│   ├── eval_agent.py        # Stage 4: Compare models, nominate winner
│   ├── critic_agent.py      # Stage 4.5: Red-team audit with veto power
│   └── report_agent.py      # Stage 5: Predictions, edges, CLV ledger, HTML report
│
├── sources/                 # Data-source plugins (DataSource contract)
│   ├── base.py              # Plugin contract: fetch / schema / quality_report
│   └── espn_injuries.py     # ESPN injury report + player impact scoring
│
├── tools/
│   ├── db.py                # Dual-backend DB: Neon Postgres or SQLite (schema v2)
│   ├── storage.py           # Parquet I/O + DB writers (predictions, trades, findings)
│   ├── market.py            # Odds snapshots, ¼-Kelly paper trades, CLV settlement
│   ├── nba.py               # NBA Stats API client (wraps nba_api)
│   ├── polymarket.py        # Polymarket Gamma + CLOB API clients (read-only)
│   ├── features.py          # Elo, rolling stats, rest, momentum, H2H, roster strength
│   ├── experiments.py       # Experiment runner: time-series CV, Optuna, calibration,
│   │                        #   paired significance tests, TabPFN/CatBoost support
│   ├── metrics.py           # Log loss, Brier score, calibration analysis
│   └── html_report.py       # Jinja2 HTML report template and builder
│
├── scripts/
│   ├── market_ops.py        # CLI: snapshot | settle | summary (cron-safe, no LLM)
│   └── sync_to_neon.py      # One-way SQLite → Neon backfill
│
├── tests/
│   └── test_leakage.py      # Future-mutation invariance: features use only the past
│
├── web-next/                # Production dashboard (Next.js, deployed on Vercel)
├── web/app.py               # Legacy Flask dashboard (local)
│
├── .github/workflows/
│   ├── pipeline.yml         # Nightly full pipeline (10:07 UTC) + manual dispatch
│   ├── odds-snapshot.yml    # Odds capture every 30 min during game hours (CLV feed)
│   └── ci.yml               # Leakage + unit tests on every push
│
├── data/seed/               # Last-known-good parquets (CI fallback for NBA API blocks)
├── data/                    # Generated artifacts (mostly gitignored)
├── docs/MASTER_PLAN.md      # Full platform roadmap
├── main.py                  # Entry point: orchestrates all 6 agents + market ops
├── config.yaml              # Configuration (seasons, models, limits)
└── requirements.txt         # Python dependencies
```

---

## Technology Stack

| Layer | Library | Purpose |
|-------|---------|---------|
| AI/LLM | `anthropic` | Claude Sonnet 5 / Opus 4.8 as agent backbones |
| ML | `scikit-learn` | LR/Ridge, GBM, CV, metrics, isotonic calibration |
| ML | `lightgbm`, `xgboost`, `catboost` | Tree boosting family |
| ML | `tabpfn` (v2) | Tabular foundation model (pretrained transformer) |
| ML | `optuna` | Hyperparameter search (TPE + pruning) |
| ML | `scipy` | Paired significance tests for promotion gating |
| ML | `torch` | Neural network / TabPFN backend |
| Data | `pandas`, `numpy` | DataFrames and numerics |
| Data | `nba_api` | NBA Stats API wrapper |
| Data | `requests` | Polymarket + ESPN injuries HTTP |
| Storage | `psycopg` → **Neon Postgres** | Production database (`DATABASE_URL`) |
| Storage | `sqlite3` | Local fallback database |
| Storage | `pyarrow` | Parquet file I/O |
| Web | Next.js 15 + `@neondatabase/serverless` | Production dashboard on Vercel |
| Web | `flask` | Legacy local dashboard |
| CI/CD | GitHub Actions | Nightly pipeline, odds snapshots, leakage tests |
| Config | `pyyaml`, `python-dotenv` | Config + `.env` loading |

---

## Extending the System

### Adding a New ML Method

In `tools/experiments.py`, add a new branch in the experiment runner:

```python
elif method == "catboost":
    from catboost import CatBoostClassifier
    model = CatBoostClassifier(iterations=100, verbose=0)
```

The Meta-Scientist will pick it up automatically when you describe it in its system prompt in `agents/meta_scientist.py`.

### Adding a New Feature

In `tools/features.py`, add a new column to the feature computation function. Then:
1. Update `get_feature_info()` to include the new feature's description
2. Run `python main.py --skip-data` to recompute features and test

Follow the leakage convention: features for game on date `D` must only use data from games on dates `< D`.

### Changing Data Sources

To swap NBA data for a different sport:
1. Create a new API client in `tools/` (following `nba.py` as a template)
2. Update `agents/data_agent.py` to use the new client
3. Update `tools/features.py` for sport-specific features
4. Update `config.yaml` → `sport: your_sport`

### Adding a New Agent

1. Create `agents/your_agent.py` inheriting from `agents/base.py`
2. Define `TOOLS` list with `name`, `description`, `input_schema`, and a handler function
3. Add `create_your_agent(config)` factory function
4. Import and call it in `main.py` after the appropriate stage

---

## Troubleshooting

**`NBA API timeout` / hanging on data fetch**

The NBA Stats API is unofficial and occasionally slow. Try:
```bash
# Increase timeout in config.yaml:
nba:
  timeout: 60

# Or skip and reuse cached data:
python main.py --skip-data
```

**`ANTHROPIC_API_KEY not set` error**

Create `.env` in the project root with your key:
```
ANTHROPIC_API_KEY=sk-ant-...
```

**No Polymarket data / empty edges section**

Polymarket NBA markets only exist when games are actively being traded (usually 24–48 hours before tip-off). Run the pipeline closer to game time. Check `data/raw/polymarket_game_markets.parquet` — if it has rows but the report shows no edges, the model and market may genuinely agree.

**`feature_matrix.parquet` not found during experiment**

You skipped the feature stage. Run without `--skip-data` at least once, or run the feature agent manually:
```python
from agents.feature_agent import create_feature_agent
import yaml
config = yaml.safe_load(open("config.yaml"))
agent = create_feature_agent(config)
agent.run("Compute and save the feature matrix.")
```

**Experiments not improving across runs**

Check `data/scientist_history.md` — the Meta-Scientist records its reasoning there. If it says "simple models are best", that is the correct answer: NBA games are hard to predict beyond ~65% accuracy. The system has found the efficient frontier.

**`data/predicto.db` is corrupted or inconsistent**

```bash
# Nuclear option — delete and restart from scratch (keeps raw data)
rm data/predicto.db data/scientist_history.md
rm -rf data/experiments/
python main.py --skip-data
```

---

## Key Design Decisions

1. **Why Logistic Regression beats tree models here:** NBA game outcomes are noisy. With ~6,000 games and 25 features, tree models overfit. Linear models with strong regularization generalize better.

2. **Why Elo dominates:** Elo captures long-run team quality efficiently. Rolling form features add marginal signal but also noise. Feature ablation confirmed that Elo alone achieves near-optimal performance.

3. **Why time-series CV:** A random train/test split would let the model see future data during training, inflating metrics. Walk-forward CV ensures evaluation reflects real deployment conditions.

4. **Why Polymarket over bookmakers:** Polymarket provides machine-readable API access and is a prediction market (not a sportsbook), so odds represent genuine crowd probability estimates rather than vig-adjusted lines.

5. **Why a persistent scientist history:** Without memory, the Meta-Scientist would repeat the same experiments each run. The markdown file lets it build on past findings and avoid dead ends.
