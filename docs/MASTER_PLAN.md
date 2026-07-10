# Predicto Master Plan: From NBA Pipeline to Autonomous Meta Data-Scientist Platform

*Drafted 2026-07. This is the working roadmap for evolving Predicto into a production,
self-improving prediction platform deployed on Vercel + Neon.*

---

## 1. Vision

Predicto today is a 5-agent pipeline that predicts NBA home-win probability and compares
it to Polymarket. After 10 iterations it converged on Ridge/LogisticRegression (C=0.01,
14 features, ~0.616 log loss) and correctly concluded that **further gains require new
data, not new algorithms**.

The next evolution is a **meta data-scientist platform**:

> A system that autonomously *acquires new datasets*, *invents and validates features*,
> *searches model space with modern techniques*, *proves its edge against real markets
> (closing-line value)*, and *promotes/retires models* — continuously, without human
> intervention — with a public web dashboard on Vercel backed by Neon Postgres.

Guiding principles (learned from the first 10 runs):

1. **Data > models.** The scientist history proves the model ceiling was hit at ~0.616.
   Every point of future improvement comes from injuries, lineups, player-level data,
   and market signals — so dataset acquisition is the platform's first-class citizen.
2. **Deterministic compute, LLM orchestration.** Agents (Claude) should design, decide,
   and critique; numpy/sklearn should compute. Never let an LLM do arithmetic a library
   can do.
3. **Statistical rigor or it didn't happen.** Every "improvement" must survive paired
   significance tests across walk-forward folds, and ultimately positive CLV.
4. **Everything is logged forever.** Every prediction, with the market odds at the moment
   it was made, goes to Postgres. This is what makes real-world evaluation possible.

---

## 2. Phased Roadmap

### Phase 0 — Foundation Hardening (prerequisite for everything else)

**Goal: make the current system trustworthy and extensible.**

- [ ] Commit the in-flight work (player game logs, roster-strength features in
      `agents/data_agent.py`, `tools/features.py`, `tools/nba.py`).
- [ ] **Split compute from reasoning.** Extract a deterministic `pipeline/` package
      (data → features → train → eval → predict) that runs with zero LLM calls.
      Agents call into it as tools. Benefits: reproducibility, testability, and the
      nightly production run costs $0 in tokens when no decisions are needed.
- [ ] **Experiment registry v2.** Replace ad-hoc SQLite tables with a proper schema
      (see §5, designed to port 1:1 to Neon): `runs`, `experiments`, `models`,
      `features`, `datasets`, `predictions`, `odds_snapshots`, `promotions`.
- [ ] **Leakage tests as CI.** The feature validation logic in `tools/features.py`
      becomes a pytest suite: every feature must provably use only data dated before
      the target game. Run on every commit via GitHub Actions.
- [ ] **Prediction ledger.** Start recording *every* upcoming-game prediction the moment
      it is made: `(game_id, model_version, p_home, market_prob_at_prediction, ts)`.
      Later joined with closing odds and outcomes. This unlocks CLV (Phase 3) — the
      longer it runs, the more valuable it is, so ship it early.
- [ ] Pin an evaluation protocol document: walk-forward CV with purge gap, fixed fold
      boundaries stored in DB so all experiments are comparable across time.

### Phase 1 — Automatic Dataset Expansion (the biggest lever)

**Goal: the platform discovers, onboards, scores, and maintains data sources on its own.**

**1a. Data-source plugin architecture.** Each source is a module implementing a contract:

```python
class DataSource(Protocol):
    name: str
    def fetch(self, since: date) -> pd.DataFrame  # idempotent, incremental
    def schema(self) -> dict                      # columns, types, join keys
    def freshness(self) -> timedelta              # how stale is acceptable
    def quality_report(self) -> QualityReport     # nulls, coverage, drift
```

A `sources/` registry lets the Data Agent enumerate and health-check all sources each run.

**1b. Concrete new sources to onboard (priority order — each directly addresses the
"new data needed" conclusion from runs 7–10):**

| Priority | Source | Signal | Access |
|---|---|---|---|
| 1 | **Injury reports** | Star player out ≈ 2–8% win-prob swing; the single biggest missing signal | BallDontLie API `/injuries`, or scrape official NBA injury report (published 5pm ET daily) |
| 2 | **Starting lineups / depth charts** | Confirms who actually plays | BallDontLie, Rotowire |
| 3 | **Player-level value aggregation** | Roster strength = sum of player on/off impact (EPM/RAPTOR-style); replaces team Elo when rosters change (trades, rest days) | Already fetching player game logs — build player-Elo/BPM aggregation on top |
| 4 | **Multi-book odds** | Opening + live lines from many sportsbooks; market features are the strongest known predictors | The Odds API (free tier: 500 req/mo), OddsMatrix |
| 5 | **Closing lines archive** | Required for CLV evaluation | Snapshot Polymarket + odds API at T-5min before tip |
| 6 | **Schedule/travel** | Travel distance, timezone shifts, 3-games-in-4-nights | Computable from schedule + arena geo (no API needed) |
| 7 | **Referee assignments** | Refs have measurable home-bias and pace effects | official.nba.com, scraped daily |
| 8 | **Season context** | Tanking incentives, playoff seeding stakes, elimination status | Computable from standings |

**1c. Data Scout Agent (new).** Given the sport config, it periodically: searches for
candidate APIs/datasets, drafts a plugin implementation, backfills history, runs a
*marginal-value experiment* (best current model ± the new source's features), and files
its verdict in the registry. Sources that add no verified signal are auto-archived.
This is the "add more and more datasets automatically" requirement, made safe by the
marginal-value gate.

**1d. Versioned feature store.** Feature definitions get IDs and versions; feature
matrices record which definitions produced them. The Feature Discovery Agent (new)
proposes candidate features from newly onboarded data, runs ablations, and only
registers features whose contribution is statistically significant.

### Phase 2 — Model & Architecture Search (Meta-Scientist v2)

**Goal: the scientist explores the modern frontier cheaply and promotes only proven wins.**

**2a. New model families to add to `tools/experiments.py`:**

- **TabPFN (v2.5 / v3)** — tabular foundation models are now genuinely competitive with
  gradient boosting; TabPFN-2.5 handles 50K rows × 2K features and TabPFN-3 (May 2026)
  up to 1M rows. Our ~6K-game dataset is squarely in its sweet spot, and it needs no
  hyperparameter tuning. This is the single most promising "latest technique" to try.
- **AutoGluon** as a bounded-budget challenger (`presets="best_quality"`, 1h budget):
  it currently tops tabular leaderboards as an ensemble and gives us a strong external
  baseline — if our curated pipeline can't beat AutoGluon, the scientist learns why.
- **CatBoost** (never tried; often best-in-class on small tabular data).
- **Bayesian logistic regression / hierarchical GLM** with team random effects — gives
  principled uncertainty and handles roster churn via partial pooling.
- **Player-composition models**: predict from aggregated player ratings of the *actual
  expected lineup* (requires Phase 1 injuries/lineups) — this is how state-of-the-art
  NBA models (538 RAPTOR lineage) beat team-level Elo.
- Keep the losers list (deep MLPs, unregularized trees) in scientist memory so they are
  only retried when the data regime changes (e.g., 10x more features).

**2b. Proper hyperparameter optimization.** Replace manual C-grid walking with **Optuna**
(TPE sampler + median pruner, budget-capped per run). The scientist sets the search
space; Optuna does the mechanical search.

**2c. Calibration & uncertainty as a pipeline stage.**
- Post-hoc calibration (isotonic / Platt) fitted on held-out folds — the scientist
  recommended this twice and it was never implemented.
- **Conformal prediction** intervals on win probability, so the edge detector can require
  `market_prob outside the model's conformal interval` instead of a raw 5% threshold —
  dramatically cuts false-positive "edges."

**2d. Statistical promotion gate.** A challenger replaces the champion only if:
paired-fold log-loss improvement significant at p<0.05 (with correction for the number
of experiments tried), calibration not degraded, and no loss vs. the market baseline.
Champion/challenger metadata lives in the `promotions` table.

**2e. Multi-sport generalization.** `sport:` in config becomes a plugin (data sources +
feature recipes + baselines per sport). NHL and MLB are natural seconds — same
walk-forward machinery, different features. This is what makes it a *meta* platform
rather than an NBA project.

### Phase 3 — Real-World Evaluation: Market Intelligence

**Goal: measure the thing that actually matters — beating the closing line.**

- **CLV tracking**: for every prediction, compare the odds available when the model
  spoke vs. the closing line. Sustained positive CLV (>1–2%) is the industry-standard
  proof of real edge — far more meaningful than log loss on historical folds.
- **Paper-trading ledger**: simulated bankroll placing fractional-Kelly (¼ Kelly) stakes
  on every detected edge, marked to market with actual outcomes. Dashboard shows equity
  curve, ROI, max drawdown, CLV distribution — segmented by edge size, liquidity, and
  time-to-tip.
- **Market-blend baseline**: the model to beat is not "always home" but
  `0.5·model + 0.5·market`. If blending with the market beats the raw model (it usually
  does), the platform should report blended probabilities.
- **Drift & calibration monitor**: weekly job compares predicted vs. realized frequencies
  on the last N games; sustained miscalibration triggers automatic recalibration and an
  alert.
- Predicto stays **analysis/paper-trading only** — it surfaces edges and tracks
  hypothetical performance; it never places bets.

### Phase 4 — Autonomous Operations (the agent layer)

**Goal: the system runs itself nightly and gets smarter without a human in the loop.**

New/upgraded agents (all inherit `agents/base.py`):

| Agent | Role | Cadence |
|---|---|---|
| **Data Scout** (new) | Discover + onboard + score new data sources | Weekly |
| **Feature Discovery** (new) | Propose/ablate features from new data | After any source change |
| **Meta-Scientist v2** | Experiment design; now with structured memory + web research tool to learn new techniques | Nightly |
| **Critic / Red-Team** (new) | Independently audits every promotion: hunts leakage, overfitting-to-folds, multiple-comparison abuse. Has veto power. | Every promotion |
| **Monitor** (new) | Watches live CLV, calibration, data freshness; opens "incidents" that seed the next scientist run | Hourly (cheap, mostly deterministic) |
| Eval, Report | As today, upgraded to CLV-aware metrics | Nightly |

**Structured scientist memory.** `scientist_history.md` (now 500+ lines of prose) becomes
Postgres tables — `findings(claim, evidence_experiment_ids, confidence, status)` and
`hypotheses(idea, priority, cost_estimate, outcome)` — plus a short generated markdown
digest for prompt context. The markdown file stays as a human-readable log, but the
agent queries structured findings so it stops re-litigating settled questions
(it re-confirmed "C=0.01 is optimal" in four separate runs — wasted tokens).

**Budget-aware orchestration.** `main.py` becomes a DAG runner: independent experiments
execute in parallel (they're CPU-cheap), each run has a token + compute budget, and the
scientist allocates its experiment budget by expected information gain.

### Phase 5 — Production Deployment: Vercel + Neon

**Goal: public, always-on web app with the pipeline running on schedule.**

**Key architectural reality:** Vercel functions cap at ~300s (800s enterprise) — model
training cannot run on Vercel. The correct split:

```
┌────────────────────────────────────────────────────────────────┐
│ VERCEL (Next.js App Router)                                    │
│  • Dashboard: predictions, edges, equity curve, CLV,           │
│    convergence, experiment explorer, agent activity feed       │
│  • API routes: read-only queries against Neon                  │
│  • Vercel Cron: lightweight jobs (odds snapshots T-5min,       │
│    freshness checks) + webhook trigger for heavy runs          │
└──────────────┬─────────────────────────────────────────────────┘
               │ SQL (Drizzle/Prisma, pooled connection)
┌──────────────▼─────────────────────────────────────────────────┐
│ NEON POSTGRES (single source of truth)                         │
│  runs · experiments · models · features · datasets ·           │
│  predictions · odds_snapshots · outcomes · promotions ·        │
│  findings · paper_trades                                       │
│  (branching: every PR gets a DB branch for preview deploys)    │
└──────────────▲─────────────────────────────────────────────────┘
               │ writes results
┌──────────────┴─────────────────────────────────────────────────┐
│ COMPUTE PLANE (long-running Python — pick one)                 │
│  Option A: GitHub Actions scheduled workflow (free, 6h limit,  │
│            fits the 5–10 min pipeline; simplest start) ✅ start │
│  Option B: Modal (serverless GPU/CPU jobs, python-native,      │
│            per-second billing; upgrade path for AutoGluon/     │
│            TabPFN budgets and parallel experiment fan-out)     │
│  Runs: nightly full pipeline + agents; artifacts (model .pkl)  │
│  to Vercel Blob / S3; metrics + predictions to Neon            │
└────────────────────────────────────────────────────────────────┘
```

Deployment steps:

1. **Schema-first migration**: implement the §5 schema in Neon (via Drizzle migrations);
   refactor `tools/storage.py` to speak Postgres (`psycopg`/SQLAlchemy) with SQLite kept
   as a local fallback via a `DATABASE_URL` switch.
2. **Next.js dashboard** (new `web-next/` app, replacing Flask long-term): shadcn/ui +
   server components reading Neon directly. Pages: Today's Edges, Predictions,
   Performance (CLV/equity/calibration), Experiments, Agents (live activity from run
   logs), Settings (prompt editor moves here).
3. **Pipeline containerization**: `Dockerfile` + GitHub Actions workflow
   (`schedule: cron 0 10 * * *` ≈ nightly after games settle + a 5pm-ET injury-report
   run). Secrets: `ANTHROPIC_API_KEY`, `DATABASE_URL` (Neon), odds API keys.
4. **Odds snapshotter**: Vercel Cron hitting an API route every 15 min on game days
   (fast, fits function limits) writing `odds_snapshots` — this feeds CLV.
5. **Auth**: Clerk (Vercel marketplace) if the dashboard should be private.
6. Keep the Flask app during migration; retire it once Next.js reaches parity.

*Note: the Neon and Vercel MCP connectors in this workspace need to be authorized
(claude.ai connector settings / `/mcp` in an interactive session) before Claude can
provision these services directly.*

---

## 3. What "advanced" looks like when done

- Nightly autonomous run: data refresh → scout verdicts → feature ablations → Optuna +
  TabPFN/AutoGluon challengers → critic-audited promotion → calibrated, conformal
  predictions for every upcoming game → edges vs. live markets → paper trades placed →
  dashboard updated. Zero human involvement.
- The scientist's memory is structured; it never re-runs settled experiments and spends
  its budget on the frontier (currently: injury/lineup features).
- Success is measured in **CLV and paper-trading ROI**, not just log loss.
- Adding a new sport = writing one data-source plugin + one feature recipe.

## 4. Sequencing & effort estimate

| Milestone | Depends on | Effort |
|---|---|---|
| M0 Foundation (deterministic core, schema v2, prediction ledger, CI leakage tests) | — | ~1 week |
| M1 Injuries + lineups + player-value features | M0 | ~1 week |
| M2 Optuna + TabPFN + CatBoost + calibration stage + promotion gate | M0 | ~4 days |
| M3 Odds snapshots + CLV + paper-trading ledger | M0 | ~4 days |
| M4 New agents (Scout, Critic, Monitor) + structured memory | M1–M3 | ~1 week |
| M5 Neon migration + Next.js dashboard + GitHub Actions compute | M0 (schema) | ~1.5 weeks |
| M6 Multi-sport plugin refactor (NHL first) | M4, M5 | ~1 week |

M1–M3 are independent after M0 and can run in parallel. Recommended order for fastest
real-world credibility: **M0 → M3 (start logging CLV data now — it only accrues value
with time) → M1 → M2 → M5 → M4 → M6.**

## 5. Neon schema sketch

```sql
datasets(id, name, source, schema_json, freshness_sla, quality_score, status)
features(id, name, version, definition, dataset_deps[], marginal_logloss_gain, status)
runs(id, started_at, finished_at, status, token_cost, summary)
experiments(id, run_id, method, feature_set[], hyperparams_json,
            log_loss, brier, accuracy, calib_error, fold_scores[])
models(id, experiment_id, artifact_url, calibrator_url)
promotions(id, model_id, promoted_at, p_value, critic_verdict, retired_at)
predictions(id, game_id, model_id, p_home, p_home_calibrated,
            conformal_lo, conformal_hi, market_prob_at_pred, created_at)
odds_snapshots(game_id, source, home_prob, volume, liquidity, captured_at)
outcomes(game_id, home_win, closing_prob, settled_at)
paper_trades(id, prediction_id, side, kelly_fraction, stake, odds_taken,
             clv, pnl, settled_at)
findings(id, claim, evidence_experiment_ids[], confidence, status, created_at)
hypotheses(id, idea, priority, cost_estimate, outcome, created_at)
```
