import { neon } from "@neondatabase/serverless";

// The pipeline writes to Neon; the dashboard is read-only.
const url = process.env.DATABASE_URL || "";

export const dbConfigured = url.startsWith("postgres");

const sql = dbConfigured ? neon(url) : null;

export async function q<T = Record<string, unknown>>(
  query: string,
  params: unknown[] = []
): Promise<T[]> {
  if (!sql) return [];
  try {
    return (await sql.query(query, params)) as T[];
  } catch (e) {
    console.error("query failed:", query.slice(0, 80), e);
    return [];
  }
}

export type Experiment = {
  experiment_id: string;
  name: string;
  method: string;
  metrics_json: string;
  conclusion: string;
  status: string;
  created_at: string;
};

export type Prediction = {
  prediction_id: string;
  game_date: string;
  home_team: string;
  away_team: string;
  p_home: number;
  market_prob_at_pred: number | null;
  edge: number | null;
  model_desc: string | null;
  created_at: string;
};

export type PaperTrade = {
  trade_id: string;
  side: string;
  model_prob: number;
  odds_taken: number;
  stake: number;
  clv: number | null;
  pnl: number | null;
  status: string;
  placed_at: string;
  home_team?: string;
  away_team?: string;
  game_date?: string;
};

export function parseMetrics(m: string): {
  log_loss?: number;
  brier_score?: number;
  accuracy?: number;
} {
  try {
    return JSON.parse(m || "{}");
  } catch {
    return {};
  }
}
