import type { ColumnKind, ProblemType } from "../types";

const intFmt = new Intl.NumberFormat("en-US");
const compactFmt = new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 });

export function fmtInt(n: number | null | undefined): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "–";
  return intFmt.format(n);
}

export function fmtCompact(n: number | null | undefined): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "–";
  return Math.abs(n) < 10_000 ? intFmt.format(n) : compactFmt.format(n);
}

/** Format a number with sensible precision for display (stats, params). */
export function fmtNum(n: number | null | undefined, digits = 3): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "–";
  if (!Number.isFinite(n)) return n > 0 ? "∞" : "-∞";
  if (Number.isInteger(n)) return intFmt.format(n);
  const abs = Math.abs(n);
  if (abs >= 1e6 || (abs > 0 && abs < 1e-3)) return n.toExponential(2);
  return Number(n.toFixed(digits)).toLocaleString("en-US", { maximumFractionDigits: digits });
}

/** 0.1667 -> "16.7%" */
export function fmtPct(ratio: number | null | undefined, digits = 1): string {
  if (ratio === null || ratio === undefined || Number.isNaN(ratio)) return "–";
  const pct = ratio * 100;
  if (pct > 0 && pct < 0.1) return "<0.1%";
  return `${pct.toFixed(pct === 0 || pct === 100 ? 0 : digits)}%`;
}

export function fmtMs(ms: number): string {
  if (ms < 1) return "<1 ms";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

export function fmtRelativeTime(iso: string): string {
  const t = new Date(iso).getTime();
  if (Number.isNaN(t)) return "";
  const s = Math.round((Date.now() - t) / 1000);
  if (s < 45) return "just now";
  const m = Math.round(s / 60);
  if (m < 60) return `${m} min ago`;
  const h = Math.round(m / 60);
  if (h < 24) return `${h} h ago`;
  return new Date(iso).toLocaleDateString();
}

export function plural(n: number, word: string, pluralWord?: string): string {
  return `${fmtInt(n)} ${n === 1 ? word : pluralWord ?? word + "s"}`;
}

/** Human-readable names for pipeline step_name values. */
export const STEP_LABELS: Record<string, string> = {
  coerce_types: "Type coercion",
  impute_missing: "Missing values",
  handle_duplicates: "Duplicates",
  handle_outliers: "Outliers",
  encode_categoricals: "Encoding",
  engineer_features: "Feature engineering",
  scale_features: "Scaling",
  combine_datasets: "Combine",
  augment_with_synthetic: "Synthetic augmentation",
  generate_synthetic: "Synthetic generation",
};

export function stepLabel(stepName: string): string {
  return STEP_LABELS[stepName] ?? stepName.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());
}

/** Ordered list of the steps run_pipeline executes, for progress UIs. */
export const PIPELINE_STEPS = [
  "coerce_types",
  "impute_missing",
  "handle_duplicates",
  "handle_outliers",
  "encode_categoricals",
  "engineer_features",
  "scale_features",
] as const;

export const KIND_LABELS: Record<ColumnKind, string> = {
  numeric: "Numeric",
  boolean: "Boolean",
  datetime: "Date/time",
  categorical: "Categorical",
  text: "Text",
  identifier: "Identifier",
  empty: "Empty",
};

export const PROBLEM_TYPE_LABELS: Record<ProblemType, string> = {
  binary_classification: "Binary classification",
  multiclass_classification: "Multiclass classification",
  regression: "Regression",
  clustering: "Clustering (unsupervised)",
};

/** Render a preview cell value as a short string; null/undefined -> null. */
export function cellText(v: unknown): string | null {
  if (v === null || v === undefined) return null;
  if (typeof v === "number") return fmtNum(v, 4);
  if (typeof v === "boolean") return v ? "true" : "false";
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}
