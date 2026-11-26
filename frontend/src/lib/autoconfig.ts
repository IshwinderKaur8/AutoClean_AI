/**
 * Data-driven pipeline configuration + a profile-based "what will this do"
 * estimator. Pure functions, no React.
 *
 * Everything here mirrors backend/app/core/pipeline/*.py exactly (which
 * column kinds each step touches, which strategies are valid, what columns
 * get added). Numbers come from the uploaded data's profile, so anything
 * that the backend recomputes mid-pipeline (outlier counts after imputation,
 * derived 0/1 columns, ...) is flagged as an estimate.
 *
 * Core rule: nothing here ever turns on a row-dropping option (dedupe=true,
 * outlier strategy "remove") automatically. Those are explicit opt-ins.
 */
import type { ColumnKind, ColumnProfile, DatasetSummary, OutlierStrategy, PipelineConfigIn } from "../types";
import { DEFAULT_PIPELINE_CONFIG } from "../types";
import type { Tone } from "../components/ui";
import { PIPELINE_STEPS, fmtInt, fmtNum, fmtPct, plural } from "./format";

export type PipelineStepName = (typeof PIPELINE_STEPS)[number];

/* ======================================================================
 * Backend-valid option values (cleaning.py / outliers.py / scaling.py)
 * ==================================================================== */

/** cleaning.impute_missing: mean/median only apply to numeric columns. */
export const NUMERIC_IMPUTE_OPTIONS = ["median", "mean", "mode", "ffill_bfill", "skip"] as const;
/** Applies to categorical, boolean, text and identifier columns. */
export const CATEGORICAL_IMPUTE_OPTIONS = ["mode", "ffill_bfill", "skip"] as const;
export const DATETIME_IMPUTE_OPTIONS = ["ffill_bfill", "mode", "skip"] as const;

export const IMPUTE_LABELS: Record<string, string> = {
  median: "Median",
  mean: "Mean",
  mode: "Most frequent (mode)",
  ffill_bfill: "Forward / back fill (row order)",
  skip: "Leave missing",
};

export function imputeLabel(strategy: string): string {
  if (strategy.startsWith("constant:")) return `Constant "${strategy.slice(9)}"`;
  return IMPUTE_LABELS[strategy] ?? strategy;
}

export const OUTLIER_LABELS: Record<OutlierStrategy, string> = {
  flag: "Flag",
  clip: "Clip",
  remove: "Remove rows",
  skip: "Skip",
};

export const SCALE_LABELS: Record<string, string> = {
  standard: "Standard",
  minmax: "Min–max",
  robust: "Robust",
  none: "None",
};

/** Max one-hot columns the Smart preset accepts before lowering the limit. */
const ONE_HOT_BUDGET = 50;
const DEFAULT_ONE_HOT_LIMIT = DEFAULT_PIPELINE_CONFIG.one_hot_max_cardinality;
export const ONE_HOT_MIN = 2;
export const ONE_HOT_MAX = 500;

/* ======================================================================
 * Small helpers
 * ==================================================================== */

const CATEGORICAL_LIKE: ColumnKind[] = ["categorical", "boolean", "text", "identifier"];

/** Pearson's second skewness coefficient, 3·(mean − median)/std. 0 if unknown. */
export function skewOf(p: ColumnProfile): number {
  const { mean, median, std } = p.stats;
  if (mean === undefined || median === undefined || !std || !Number.isFinite(std)) return 0;
  return (3 * (mean - median)) / std;
}

/** Numeric column that looks like a row identifier (name + nearly all distinct). */
export function isIdLike(p: ColumnProfile): boolean {
  if (p.kind !== "numeric" || p.non_null_count === 0) return false;
  const n = p.name.trim();
  const named = /^(id|index|idx|row[_\s-]?(id|no|num|number))$/i.test(n) || /[_\s-]id$/i.test(n) || /[a-z]Id$/.test(n);
  return named && p.unique_count / p.non_null_count >= 0.8;
}

/** scaling._is_binary_indicator: every non-null value is 0 or 1. */
function isBinaryNumeric(p: ColumnProfile): boolean {
  const { min, max } = p.stats;
  return p.unique_count <= 2 && min !== undefined && max !== undefined && [0, 1].includes(min) && [0, 1].includes(max);
}

/**
 * profiling.profile_column's string branch (after the boolean / numeric /
 * datetime checks): identifier if every value is distinct and there are >20,
 * categorical if ≤50 distinct or ≤50% distinct, else free text. Later steps
 * re-profile, so filling gaps can change a column's kind mid-pipeline.
 */
function stringKind(unique: number, nonNull: number): ColumnKind {
  if (nonNull === 0) return "empty";
  if (unique === nonNull && nonNull > 20) return "identifier";
  if (unique <= 50 || unique / nonNull <= 0.5) return "categorical";
  return "text";
}

function listNames(names: string[], max = 3): string {
  if (names.length <= max) return names.join(", ");
  return `${names.slice(0, max).join(", ")} +${names.length - max} more`;
}

function stableString(v: unknown): string {
  if (v === null || typeof v !== "object") return JSON.stringify(v ?? null);
  if (Array.isArray(v)) return `[${v.map(stableString).join(",")}]`;
  const rec = v as Record<string, unknown>;
  return `{${Object.keys(rec)
    .sort()
    .map((k) => `${JSON.stringify(k)}:${stableString(rec[k])}`)
    .join(",")}}`;
}

export function sameValue(a: unknown, b: unknown): boolean {
  return stableString(a) === stableString(b);
}

const CONFIG_KEYS = Object.keys(DEFAULT_PIPELINE_CONFIG) as (keyof PipelineConfigIn)[];

export function sameConfig(a: PipelineConfigIn, b: PipelineConfigIn): boolean {
  return CONFIG_KEYS.every((k) => sameValue(a[k], b[k]));
}

/** Only overrides for columns that exist (sanitizeConfig drops the rest before a run). */
function liveOverrides<V>(rec: Record<string, V>, cols: Set<string>): Record<string, V> {
  return Object.fromEntries(Object.entries(rec).filter(([k]) => cols.has(k)));
}

/**
 * Copy `base` but keep the user's per-column choices from `current`
 * (impute/outlier/encoding overrides set in the Columns tab, plus the
 * dedupe/scale column lists). Presets change strategies, not columns.
 */
export function withPreserved(base: PipelineConfigIn, current: PipelineConfigIn): PipelineConfigIn {
  return {
    ...base,
    impute_overrides: { ...current.impute_overrides },
    outlier_overrides: { ...current.outlier_overrides },
    encoding_overrides: { ...current.encoding_overrides },
    dedupe_subset: current.dedupe_subset === null ? null : [...current.dedupe_subset],
    scale_columns: current.scale_columns === null ? null : [...current.scale_columns],
  };
}

/* ======================================================================
 * Smart recommendation
 * ==================================================================== */

export interface ConfigReason {
  field: keyof PipelineConfigIn | string;
  /** Human-readable recommended value. */
  value: string;
  reason: string;
  /** True if applying the recommendation would change this field from `current`. */
  changed: boolean;
}

export interface Recommendation {
  config: PipelineConfigIn;
  reasons: ConfigReason[];
  /** Things worth knowing that the recommendation deliberately does NOT change. */
  notes: string[];
}

function skewPhrase(p: ColumnProfile): string {
  const s = skewOf(p);
  return `${p.name}: mean ${fmtNum(p.stats.mean, 2)} vs median ${fmtNum(p.stats.median, 2)}, skew ≈ ${fmtNum(s, 2)}`;
}

export function recommendConfig(
  profile: ColumnProfile[],
  rows: number,
  current: PipelineConfigIn
): Recommendation {
  const cfg = withPreserved(DEFAULT_PIPELINE_CONFIG, current);
  const cols = new Set(profile.map((p) => p.name));
  const imputeOv = liveOverrides(current.impute_overrides, cols);
  const outlierOv = liveOverrides(current.outlier_overrides, cols);
  const encodeOv = liveOverrides(current.encoding_overrides, cols);
  const reasons: Omit<ConfigReason, "changed">[] = [];
  const notes: string[] = [];

  const numeric = profile.filter((p) => p.kind === "numeric");
  const datetimes = profile.filter((p) => p.kind === "datetime");
  const texts = profile.filter((p) => p.kind === "text");
  const booleans = profile.filter((p) => p.kind === "boolean");
  const categoricals = profile.filter((p) => p.kind === "categorical");

  /* ---- coerce_types ---- */
  {
    const typed = profile.filter((p) => p.kind === "numeric" || p.kind === "boolean" || p.kind === "datetime");
    cfg.coerce_types = true;
    reasons.push({
      field: "coerce_types",
      value: "on",
      reason:
        typed.length > 0
          ? `${plural(typed.length, "column")} (${listNames(typed.map((p) => p.name))}) look numeric, boolean or date-like; converting them to real types is what lets the later steps compute medians, IQR fences and scalers.`
          : "No column needs converting, so this is a harmless no-op — kept on so mixed-type uploads are always typed first.",
    });
  }

  /* ---- numeric_impute_strategy ---- */
  {
    const gaps = numeric.filter((p) => p.null_count > 0 && !(p.name in imputeOv));
    if (gaps.length === 0) {
      cfg.numeric_impute_strategy = "median";
      reasons.push({
        field: "numeric_impute_strategy",
        value: "median",
        reason:
          numeric.some((p) => p.null_count > 0)
            ? "Every numeric column with gaps already has a per-column override, so this default isn't used."
            : "No numeric column has missing values, so this won't be used; median is kept as the robust default.",
      });
    } else {
      const skewed = gaps
        .filter((p) => Math.abs(skewOf(p)) >= 0.5 || p.outlier_count > 0)
        .sort((a, b) => Math.abs(skewOf(b)) - Math.abs(skewOf(a)));
      const few = gaps.some((p) => p.non_null_count < 10);
      const fewNote = few ? " (some columns have under 10 values, so treat this as a rough signal)" : "";
      if (skewed.length > 0) {
        cfg.numeric_impute_strategy = "median";
        const top = skewed.slice(0, 2).map((p) =>
          Math.abs(skewOf(p)) >= 0.5 ? skewPhrase(p) : `${p.name}: ${plural(p.outlier_count, "outlier")}`
        );
        reasons.push({
          field: "numeric_impute_strategy",
          value: "median",
          reason: `${top.join("; ")}${skewed.length > 2 ? ` (+${skewed.length - 2} more)` : ""}. Skewed columns or outliers pull the mean toward extreme values; the median isn't affected${fewNote}.`,
        });
      } else {
        cfg.numeric_impute_strategy = "mean";
        const maxSkew = Math.max(...gaps.map((p) => Math.abs(skewOf(p))));
        reasons.push({
          field: "numeric_impute_strategy",
          value: "mean",
          reason: `All ${plural(gaps.length, "numeric column")} with gaps look roughly symmetric (largest |skew| ≈ ${fmtNum(maxSkew, 2)}, under 0.5) with no outliers, so mean and median nearly agree; filling with the mean leaves each column's average unchanged${fewNote}.`,
        });
      }
    }
  }

  /* ---- categorical_impute_strategy ---- */
  {
    const gaps = profile.filter((p) => CATEGORICAL_LIKE.includes(p.kind) && p.null_count > 0 && !(p.name in imputeOv));
    cfg.categorical_impute_strategy = "mode";
    reasons.push({
      field: "categorical_impute_strategy",
      value: "mode",
      reason:
        gaps.length > 0
          ? `Fills ${listNames(gaps.map((p) => `${p.name} (${fmtInt(p.null_count)})`))} with the most frequent value — the only fill that works for labels without relying on row order.`
          : "No categorical, boolean or text column has gaps; mode is kept as the default.",
    });
    for (const p of gaps) {
      if (p.kind === "text" || p.kind === "identifier") {
        notes.push(
          `${p.name} is ${p.kind === "text" ? "free text" : "an identifier"} with ${plural(p.null_count, "missing value")}; mode fills them with its most common value, which is rarely meaningful. Consider a "Leave missing" override in Columns.`
        );
      } else if (p.null_ratio >= 0.4) {
        notes.push(
          `${p.name} is ${fmtPct(p.null_ratio)} missing; filling that many gaps with one value makes it dominate the column. Consider an override in Columns.`
        );
      }
    }
  }

  /* ---- datetime_impute_strategy ---- */
  {
    const gaps = datetimes.filter((p) => p.null_count > 0 && !(p.name in imputeOv));
    cfg.datetime_impute_strategy = "ffill_bfill";
    reasons.push({
      field: "datetime_impute_strategy",
      value: "ffill_bfill",
      reason:
        gaps.length > 0
          ? `${listNames(gaps.map((p) => `${p.name} (${fmtInt(p.null_count)} missing)`))} will be filled from the neighbouring row (forward, then backward), which suits rows stored in time order. If your rows aren't ordered, pick "Most frequent" instead.`
          : "No date column has gaps; forward/back fill is kept as the default.",
    });
  }

  /* ---- dedupe (never auto-enabled) ---- */
  cfg.dedupe = false;
  reasons.push({
    field: "dedupe",
    value: "detect only",
    reason:
      "Duplicates are always detected and counted in the audit trail. Removing them drops rows, so it stays an explicit opt-in.",
  });

  /* ---- outlier_default_strategy (never "remove") ---- */
  let outlierDefault: OutlierStrategy = "flag";
  {
    const candidates = numeric.filter((p) => !(p.name in outlierOv));
    const withOutliers = candidates.filter((p) => p.outlier_count > 0);
    if (withOutliers.length === 0) {
      outlierDefault = "flag";
      reasons.push({
        field: "outlier_default_strategy",
        value: "flag",
        reason:
          candidates.length === 0
            ? "No numeric columns to check (or all have overrides); flag is the least invasive default."
            : `No values fall outside the IQR fences in the uploaded data (${plural(candidates.length, "numeric column")} checked). Outliers are re-detected after imputation and deduplication, so flag is the safe choice: it only adds an indicator column.`,
      });
    } else {
      const shares = withOutliers.map((p) => ({ p, share: p.outlier_count / Math.max(1, p.non_null_count) }));
      const maxShare = Math.max(...shares.map((s) => s.share));
      const phrase = (s: { p: ColumnProfile; share: number }) =>
        `${s.p.name}: ${fmtInt(s.p.outlier_count)} of ${fmtInt(s.p.non_null_count)} values (${fmtPct(s.share)})`;
      if (maxShare <= 0.05) {
        outlierDefault = "clip";
        reasons.push({
          field: "outlier_default_strategy",
          value: "clip",
          reason: `${listNames(shares.map(phrase), 2)} lie beyond the IQR fences. A share that small (≤5%) is typically extreme values or entry errors; clipping caps them at the fence so they don't dominate scaling — no rows are dropped.`,
        });
      } else {
        outlierDefault = "flag";
        const big = shares.filter((s) => s.share > 0.05);
        reasons.push({
          field: "outlier_default_strategy",
          value: "flag",
          reason: `${listNames(big.map(phrase), 2)} lie beyond the IQR fences. More than 5% usually means a long-tailed distribution rather than errors, so values are kept and an indicator column (e.g. ${big[0].p.name}__is_outlier) is added.`,
        });
      }
    }
    cfg.outlier_default_strategy = outlierDefault;
  }

  /* ---- encode ---- */
  {
    const willFlag =
      outlierDefault === "flag" && numeric.some((p) => !(p.name in outlierOv) && p.outlier_count > 0);
    const flagOverrides = numeric.some((p) => outlierOv[p.name] === "flag" && p.outlier_count > 0);
    const needs = categoricals.length + booleans.length;
    cfg.encode = needs > 0 || willFlag || flagOverrides;
    reasons.push({
      field: "encode",
      value: cfg.encode ? "on" : "off",
      reason: cfg.encode
        ? needs > 0
          ? `${listNames([...categoricals, ...booleans].map((p) => p.name))} ${needs === 1 ? "is" : "are"} non-numeric; most models need categories and booleans as numbers.`
          : "No categorical columns, but outlier flag columns are true/false and get converted to 0/1."
        : "No categorical or boolean columns, so encoding would change nothing.",
    });
  }

  /* ---- one_hot_max_cardinality ---- */
  {
    const auto = categoricals.filter((p) => !(p.name in encodeOv)).sort((a, b) => b.unique_count - a.unique_count);
    const cards = auto.map((p) => p.unique_count);
    const added = (t: number) => cards.filter((k) => k <= t).reduce((a, b) => a + b, 0);
    let limit = DEFAULT_ONE_HOT_LIMIT;
    let reason: string;
    if (auto.length === 0) {
      reason =
        categoricals.length > 0
          ? "Every categorical column has an encoding override, so this limit isn't used."
          : "No categorical columns, so this limit isn't used.";
    } else if (added(limit) <= ONE_HOT_BUDGET) {
      const over = auto.filter((p) => p.unique_count > limit);
      const under = auto.filter((p) => p.unique_count <= limit);
      reason =
        over.length === 0
          ? auto.length === 1
            ? `The only categorical column, ${auto[0].name}, has ${fmtInt(auto[0].unique_count)} distinct values (≤${limit}), so it becomes one-hot (+${fmtInt(added(limit))} columns).`
            : `All ${fmtInt(auto.length)} categorical columns have ≤${limit} distinct values (largest: ${auto[0].name}, ${fmtInt(auto[0].unique_count)}), so each becomes one-hot (+${fmtInt(added(limit))} columns).`
          : `${under.length ? `${listNames(under.map((p) => `${p.name} (${p.unique_count})`))} → one-hot (+${fmtInt(added(limit))} columns); ` : ""}${listNames(over.map((p) => `${p.name} (${fmtInt(p.unique_count)})`))} → integer label codes, avoiding ${fmtInt(over.reduce((a, p) => a + p.unique_count, 0))} sparse columns.`;
    } else {
      const before = added(limit);
      const candidates = [...new Set(cards.filter((k) => k < limit))].sort((a, b) => b - a);
      let next = ONE_HOT_MIN;
      for (const k of candidates) {
        if (added(k) <= ONE_HOT_BUDGET) {
          next = Math.max(ONE_HOT_MIN, k);
          break;
        }
      }
      limit = next;
      reason = `At ≤${DEFAULT_ONE_HOT_LIMIT}, one-hot would add ${fmtInt(before)} columns. Lowering the limit to ${limit} keeps it at ${fmtInt(added(limit))} (budget ~${ONE_HOT_BUDGET}); columns with more than ${limit} values get integer label codes instead.`;
    }
    cfg.one_hot_max_cardinality = limit;
    reasons.push({ field: "one_hot_max_cardinality", value: `≤ ${limit}`, reason });
  }

  /* ---- engineer_features ---- */
  {
    cfg.engineer_features = datetimes.length + texts.length > 0;
    const parts: string[] = [];
    if (datetimes.length)
      parts.push(`${listNames(datetimes.map((p) => p.name))} → year, month, day, weekday (+${datetimes.length * 4} columns)`);
    if (texts.length) parts.push(`${listNames(texts.map((p) => p.name))} → length, word count (+${texts.length * 2})`);
    reasons.push({
      field: "engineer_features",
      value: cfg.engineer_features ? "on" : "off",
      reason: cfg.engineer_features
        ? `${parts.join("; ")}. Models can't use raw dates or text directly; these numeric parts they can.`
        : "No date or free-text columns, so the step would add nothing.",
    });
  }

  /* ---- scale_method ---- */
  {
    const scaled = (current.scale_columns === null
      ? numeric
      : numeric.filter((p) => current.scale_columns?.includes(p.name))
    ).filter((p) => !isBinaryNumeric(p));
    const strategyFor = (p: ColumnProfile) => outlierOv[p.name] ?? outlierDefault;
    const heavy = scaled.filter((p) => Math.abs(skewOf(p)) >= 1).sort((a, b) => Math.abs(skewOf(b)) - Math.abs(skewOf(a)));
    const kept = scaled.filter((p) => p.outlier_count > 0 && (strategyFor(p) === "flag" || strategyFor(p) === "skip"));
    if (scaled.length === 0) {
      cfg.scale_method = "standard";
      reasons.push({
        field: "scale_method",
        value: "standard",
        reason:
          current.scale_columns !== null
            ? "None of the selected scale columns are non-binary numeric source columns, so the method barely matters; standard is kept as the default."
            : "No non-binary numeric source columns; any numeric columns created by earlier steps (label codes, date parts, text lengths) still get standard scaling.",
      });
    } else if (heavy.length > 0 || kept.length > 0) {
      cfg.scale_method = "robust";
      const why = heavy.length
        ? `${heavy.slice(0, 2).map(skewPhrase).join("; ")}${heavy.length > 2 ? ` (+${heavy.length - 2} more)` : ""} — strongly skewed (|skew| ≥ 1)`
        : `${listNames(kept.map((p) => `${p.name} (${plural(p.outlier_count, "outlier")})`))} keep their outliers`;
      reasons.push({
        field: "scale_method",
        value: "robust",
        reason: `${why}. Robust scaling centres on the median and divides by the IQR, so a few extreme values don't squash everything else into a narrow band.`,
      });
    } else {
      cfg.scale_method = "standard";
      const maxSkew = Math.max(...scaled.map((p) => Math.abs(skewOf(p))));
      reasons.push({
        field: "scale_method",
        value: "standard",
        reason: `The ${plural(scaled.length, "numeric column")} to scale look reasonably symmetric (largest |skew| ≈ ${fmtNum(maxSkew, 2)}) with no unclipped outliers, so standard scaling (mean 0, std 1) fits well.`,
      });
    }
  }

  for (const p of numeric.filter(isIdLike)) {
    notes.push(
      `${p.name} looks like a row identifier (name + ${fmtPct(p.unique_count / p.non_null_count, 0)} distinct values) but is numeric, so it will be outlier-checked and scaled like a feature. Exclude it from scaling below, or drop it before training.`
    );
  }
  if (rows > 0 && rows < 30) {
    notes.push(`Only ${plural(rows, "row")}: skew and outlier signals are computed from very few values, so treat them as rough.`);
  }

  return {
    config: cfg,
    reasons: reasons.map((r) => ({
      ...r,
      changed: !sameValue(cfg[r.field as keyof PipelineConfigIn], current[r.field as keyof PipelineConfigIn]),
    })),
    notes,
  };
}

/* ======================================================================
 * Presets
 * ==================================================================== */

export type PresetId = "smart" | "safe" | "minimal" | "thorough";

export interface Preset {
  id: PresetId;
  label: string;
  description: string;
  /** True if the preset itself enables a row-dropping option. */
  dropsRows: boolean;
  build: (profile: ColumnProfile[], rows: number, current: PipelineConfigIn) => PipelineConfigIn;
}

export const PRESETS: Preset[] = [
  {
    id: "smart",
    label: "Smart",
    description: "Rule-based settings derived from this dataset's profile. Every choice is explained.",
    dropsRows: false,
    build: (profile, rows, current) => recommendConfig(profile, rows, current).config,
  },
  {
    id: "safe",
    label: "Safe defaults",
    description: "The backend defaults: median/mode fill, flag outliers, one-hot ≤15, standard scaling.",
    dropsRows: false,
    build: (_profile, _rows, current) => withPreserved(DEFAULT_PIPELINE_CONFIG, current),
  },
  {
    id: "minimal",
    label: "Minimal",
    description: "Fix types and fill missing values only. No encoding, features, scaling or outlier handling.",
    dropsRows: false,
    build: (_profile, _rows, current) =>
      withPreserved(
        {
          ...DEFAULT_PIPELINE_CONFIG,
          encode: false,
          engineer_features: false,
          scale_method: "none",
          outlier_default_strategy: "skip",
        },
        current
      ),
  },
  {
    id: "thorough",
    label: "Thorough",
    description: "Defaults plus duplicate removal and outlier clipping. Can drop duplicate rows.",
    dropsRows: true,
    build: (_profile, _rows, current) =>
      withPreserved({ ...DEFAULT_PIPELINE_CONFIG, dedupe: true, outlier_default_strategy: "clip" }, current),
  },
];

/* ======================================================================
 * Pipeline estimate ("will affect" previews)
 * ==================================================================== */

export interface AffectItem {
  label: string;
  detail?: string;
  tone?: Tone;
}

export interface StepNote {
  tone: Tone;
  text: string;
}

export interface StepEstimate {
  step: PipelineStepName;
  /** Recorded in the audit trail (coerce/encode/engineer are skipped entirely when off). */
  runs: boolean;
  /** Enabled: configured to change data (duplicates counts as enabled; it always runs and reports). */
  active: boolean;
  summary: string;
  items: AffectItem[];
  notes: StepNote[];
  /** Numbers come from the uploaded data and may change once earlier steps run. */
  estimate: boolean;
  dropsRows: boolean;
}

export interface PipelineEstimate {
  steps: Record<PipelineStepName, StepEstimate>;
  rowsMayDrop: boolean;
  rowDropReasons: string[];
  columnsIn: number;
  columnsOut: number;
  activeSteps: number;
}

interface SimCol {
  name: string;
  kind: ColumnKind;
  unique: number;
  nulls: number;
  nonNull: number;
  binary: boolean;
  origin: "source" | "flag" | "onehot" | "label" | "bool" | "derived";
  profile?: ColumnProfile;
}

function defaultImputeFor(kind: ColumnKind, cfg: PipelineConfigIn): string {
  if (kind === "numeric") return cfg.numeric_impute_strategy;
  if (kind === "datetime") return cfg.datetime_impute_strategy;
  if (CATEGORICAL_LIKE.includes(kind)) return cfg.categorical_impute_strategy;
  return "skip";
}

const KIND_TO: Partial<Record<ColumnKind, string>> = { numeric: "number", boolean: "true/false", datetime: "date" };

export function estimatePipeline(profile: ColumnProfile[], rows: number, config: PipelineConfigIn): PipelineEstimate {
  const names = new Set(profile.map((p) => p.name));
  const imputeOv = liveOverrides(config.impute_overrides, names);
  const outlierOv = liveOverrides(config.outlier_overrides, names);
  const encodeOv = liveOverrides(config.encoding_overrides, names);

  let sim: SimCol[] = profile.map((p) => ({
    name: p.name,
    kind: p.kind,
    unique: p.unique_count,
    nulls: p.null_count,
    nonNull: p.non_null_count,
    binary: p.kind === "numeric" && isBinaryNumeric(p),
    origin: "source",
    profile: p,
  }));

  /* ---- coerce_types ---- */
  const coerce: StepEstimate = {
    step: "coerce_types",
    runs: config.coerce_types,
    active: false,
    summary: "",
    items: [],
    notes: [],
    estimate: false,
    dropsRows: false,
  };
  {
    const typed = sim.filter((c) => c.kind === "numeric" || c.kind === "boolean" || c.kind === "datetime");
    if (config.coerce_types) {
      let failTotal = 0;
      for (const c of typed) {
        const p = c.profile as ColumnProfile;
        const ratio = c.kind === "numeric" ? p.numeric_convertible_ratio : c.kind === "datetime" ? p.datetime_convertible_ratio : 1;
        const fails = Math.round(c.nonNull * (1 - ratio));
        if (fails > 0) {
          c.nulls += fails;
          c.nonNull -= fails;
          failTotal += fails;
        }
        coerce.items.push({
          label: c.name,
          detail: `→ ${KIND_TO[c.kind]}${fails ? `, ~${fmtInt(fails)} won't parse` : ""}`,
          tone: fails ? "warning" : "neutral",
        });
      }
      coerce.active = true;
      coerce.summary = typed.length
        ? `Converts ${plural(typed.length, "column")} to real types`
        : "Nothing to convert — every column is already text-like or typed";
      if (failTotal > 0) {
        coerce.estimate = true;
        coerce.notes.push({
          tone: "warning",
          text: `~${plural(failTotal, "value")} can't be parsed; they become missing (never dropped) and are then handled by the missing-values step.`,
        });
      }
    } else {
      coerce.summary = "Off — columns keep their stored types";
      if (typed.length > 0)
        coerce.notes.push({
          tone: "warning",
          text: `${listNames(typed.map((c) => c.name))} look ${[...new Set(typed.map((c) => KIND_TO[c.kind]))].join("/")}-like. CSV and Excel uploads store every value as text, and on text values the backend's mean/median fill, outlier check and boolean encoding raise an error, so the run fails. Only turn this off for data that's already typed (e.g. a previous pipeline output).`,
        });
    }
  }

  /* ---- impute_missing ---- */
  const impute: StepEstimate = {
    step: "impute_missing",
    runs: true,
    active: false,
    summary: "",
    items: [],
    notes: [],
    estimate: coerce.estimate,
    dropsRows: false,
  };
  {
    let filledCols = 0;
    let filledCells = 0;
    let leftCols = 0;
    let leftCells = 0;
    const constantNumeric: string[] = [];
    const rekinded: string[] = [];
    for (const c of sim) {
      if (c.nulls === 0) continue;
      const overridden = c.name in imputeOv;
      const strategy = imputeOv[c.name] ?? defaultImputeFor(c.kind, config);
      let fills = false;
      let why = "";
      if (strategy === "skip" || c.kind === "empty") {
        why = c.kind === "empty" ? "column is empty" : "left missing";
      } else if ((strategy === "mean" || strategy === "median") && c.kind !== "numeric") {
        why = `${strategy} doesn't apply to ${c.kind}`;
      } else if (c.nonNull === 0 && !strategy.startsWith("constant:")) {
        why = "no values to fill from";
      } else {
        fills = true;
      }
      if (fills) {
        filledCols += 1;
        filledCells += c.nulls;
        impute.items.push({
          label: c.name,
          detail: `${fmtInt(c.nulls)} · ${imputeLabel(strategy).toLowerCase()}${overridden ? " (override)" : ""}`,
        });
        const isConstant = strategy.startsWith("constant:");
        c.nonNull += c.nulls;
        c.nulls = 0;
        // A 0/1 column filled with a fractional mean/median is no longer a binary indicator, so it gets scaled.
        if (c.binary && c.profile && (strategy === "mean" || strategy === "median")) {
          const v = strategy === "mean" ? c.profile.stats.mean : c.profile.stats.median;
          if (v !== 0 && v !== 1) c.binary = false;
        }
        if (isConstant && c.kind === "numeric") constantNumeric.push(c.name);
        if (c.kind === "categorical" || c.kind === "text" || c.kind === "identifier") {
          if (isConstant && !(c.profile?.sample_values ?? []).includes(strategy.slice(9))) {
            c.unique += 1;
            impute.estimate = true;
          }
          const next = stringKind(c.unique, c.nonNull);
          if (next !== c.kind) {
            rekinded.push(`${c.name} (${c.kind} → ${next})`);
            c.kind = next;
          }
        }
      } else {
        leftCols += 1;
        leftCells += c.nulls;
        impute.items.push({ label: c.name, detail: `${fmtInt(c.nulls)} · ${why}`, tone: "warning" });
      }
    }
    impute.active =
      [config.numeric_impute_strategy, config.categorical_impute_strategy, config.datetime_impute_strategy].some((s) => s !== "skip") ||
      Object.values(imputeOv).some((s) => s !== "skip");
    if (filledCols === 0 && leftCols === 0) impute.summary = "No missing values — nothing to fill";
    else {
      const parts: string[] = [];
      if (filledCols) parts.push(`Fills ${plural(filledCells, "cell")} across ${plural(filledCols, "column")}`);
      if (leftCols) parts.push(`${filledCols ? "leaves" : "Leaves"} ${plural(leftCells, "cell")} missing in ${plural(leftCols, "column")}`);
      impute.summary = parts.join("; ");
    }
    if (leftCols > 0)
      impute.notes.push({
        tone: "warning",
        text: "Gaps that stay missing are kept as-is (rows are never dropped for them); scaling leaves NaN in those cells.",
      });
    if (config.datetime_impute_strategy === "ffill_bfill" && sim.some((c) => c.kind === "datetime" && c.origin === "source" && c.profile && c.profile.null_count > 0 && !(c.name in imputeOv)))
      impute.notes.push({ tone: "info", text: "Forward/back fill copies the value from the neighbouring row, so it assumes rows are in a meaningful order." });
    if (rekinded.length)
      impute.notes.push({
        tone: "info",
        text: `Later steps re-profile the filled data, and the filled copies change how these columns are classified: ${listNames(rekinded)}. Encoding and feature engineering treat them by their new kind.`,
      });
    if (constantNumeric.some((n) => (outlierOv[n] ?? config.outlier_default_strategy) !== "skip"))
      impute.notes.push({
        tone: "danger",
        text: `The constant override on ${listNames(constantNumeric)} is inserted as text, which leaves a mix of numbers and text in a numeric column. The outlier step then fails on it, so the whole run errors. Use median, mean or mode, or set this column's outlier strategy to Skip.`,
      });
  }

  /* ---- handle_duplicates ---- */
  const dupes: StepEstimate = {
    step: "handle_duplicates",
    runs: true,
    active: true,
    summary: "",
    items: [],
    notes: [],
    estimate: false,
    dropsRows: config.dedupe,
  };
  {
    // sanitizeConfig drops unknown columns and turns an empty subset into null (= all columns).
    const kept = (config.dedupe_subset ?? []).filter((c) => names.has(c));
    const subset = kept.length > 0 ? kept : null;
    const compared = new Set(subset ?? profile.map((p) => p.name));
    const scope = subset ? `on ${listNames(subset)}` : "across all columns";
    dupes.summary = config.dedupe
      ? `Removes rows that repeat ${scope}, keeping the first occurrence`
      : `Counts rows that repeat ${scope} and reports them — keeps every row`;
    if (subset) dupes.items = subset.map((c) => ({ label: c }));
    const unique = profile.find((p) => compared.has(p.name) && p.null_count === 0 && rows > 0 && p.unique_count === rows);
    if (unique)
      dupes.notes.push({
        tone: "success",
        text: `${unique.name} has a distinct value in every row, so no duplicates are expected.`,
      });
    if (config.dedupe && subset)
      dupes.notes.push({
        tone: "warning",
        text: "Rows only need to match on the selected columns to count as duplicates; values in the other columns of a dropped row are lost with it.",
      });
  }

  /* ---- handle_outliers ---- */
  const outl: StepEstimate = {
    step: "handle_outliers",
    runs: true,
    active: false,
    summary: "",
    items: [],
    notes: [],
    estimate: true,
    dropsRows: false,
  };
  {
    const numericCols = sim.filter((c) => c.kind === "numeric");
    const checked = numericCols.filter((c) => (outlierOv[c.name] ?? config.outlier_default_strategy) !== "skip");
    let removeUpper = 0;
    const flagged: string[] = [];
    for (const c of checked) {
      const strategy = outlierOv[c.name] ?? config.outlier_default_strategy;
      const count = c.profile?.outlier_count ?? 0;
      if (strategy === "remove") outl.dropsRows = true;
      if (c.nonNull < 4) {
        outl.items.push({ label: c.name, detail: "too few values (<4), skipped", tone: "neutral" });
        continue;
      }
      if (count === 0) continue;
      const verb = strategy === "flag" ? `adds ${c.name}__is_outlier` : strategy === "clip" ? "clipped to fences" : "rows removed";
      outl.items.push({
        label: c.name,
        detail: `${fmtInt(count)} · ${verb}`,
        tone: strategy === "remove" ? "danger" : strategy === "clip" ? "info" : "neutral",
      });
      if (strategy === "flag") flagged.push(`${c.name}__is_outlier`);
      if (strategy === "remove") removeUpper += count;
    }
    const existing = new Set(sim.map((c) => c.name));
    for (const flag of flagged) {
      if (!existing.has(flag))
        sim.push({ name: flag, kind: "boolean", unique: 2, nulls: 0, nonNull: rows, binary: true, origin: "flag" });
    }
    outl.active = config.outlier_default_strategy !== "skip" || Object.values(outlierOv).some((s) => s !== "skip");
    const withOut = outl.items.filter((i) => i.detail && !i.detail.startsWith("too few")).length;
    if (numericCols.length === 0) outl.summary = "No numeric columns to check";
    else if (checked.length === 0) outl.summary = "Off — numeric values are left untouched";
    else
      outl.summary = withOut
        ? `${plural(withOut, "column")} of ${fmtInt(checked.length)} checked ${withOut === 1 ? "has" : "have"} outliers in the uploaded data`
        : `Checks ${plural(checked.length, "numeric column")}; none have outliers in the uploaded data`;
    if (checked.length > 0)
      outl.notes.push({
        tone: "neutral",
        text: "Counts come from the uploaded data. The step re-detects outliers (beyond Q1 − 1.5·IQR or Q3 + 1.5·IQR) after imputation and deduplication, so the real numbers can differ.",
      });
    if (outl.dropsRows)
      outl.notes.push({
        tone: "danger",
        text:
          removeUpper > 0
            ? `Can drop rows: up to ~${plural(Math.min(rows, removeUpper), "row")} with an outlier in a "remove" column.`
            : `Can drop rows: any row with an outlier in a "remove" column is deleted (none detected yet, but they can appear after imputation).`,
      });
  }

  /* ---- encode_categoricals ---- */
  const enc: StepEstimate = {
    step: "encode_categoricals",
    runs: config.encode,
    active: false,
    summary: "",
    items: [],
    notes: [],
    estimate: false,
    dropsRows: false,
  };
  {
    const limit = config.one_hot_max_cardinality;
    if (!config.encode) {
      const left = sim.filter((c) => c.kind === "categorical" || c.kind === "boolean");
      enc.summary = "Off — categories and booleans stay as-is";
      if (left.length)
        enc.notes.push({
          tone: "warning",
          text: `${listNames(left.map((c) => c.name))} stay non-numeric; most models need numbers.`,
        });
    } else {
      // encode_categoricals re-profiles here, so overrides apply by the kind a column has *now*.
      const kindNow = new Map(sim.map((c) => [c.name, c.kind]));
      const next: SimCol[] = [];
      let oneHotCols = 0;
      let oneHotAdded = 0;
      let labelCols = 0;
      let boolCols = 0;
      let flagBools = 0;
      for (const c of sim) {
        if (c.kind === "boolean") {
          boolCols += 1;
          if (c.origin === "flag") flagBools += 1;
          else enc.items.push({ label: c.name, detail: "→ 0/1" });
          next.push({ ...c, kind: "numeric", binary: true, origin: c.origin === "flag" ? "flag" : "bool" });
          continue;
        }
        if (c.kind === "identifier") {
          enc.items.push({ label: c.name, detail: "identifier, left as-is", tone: "warning" });
          next.push(c);
          continue;
        }
        if (c.kind === "text") {
          enc.items.push({ label: c.name, detail: "free text, left as-is", tone: "neutral" });
          next.push(c);
          continue;
        }
        if (c.kind !== "categorical") {
          next.push(c);
          continue;
        }
        const ov = encodeOv[c.name];
        const strategy = ov ?? (c.unique <= limit ? "onehot" : "label");
        const tag = ov ? " (override)" : "";
        if (strategy === "skip") {
          enc.items.push({ label: c.name, detail: `skipped${tag}`, tone: "neutral" });
          next.push(c);
        } else if (strategy === "onehot") {
          oneHotCols += 1;
          oneHotAdded += c.unique;
          enc.items.push({ label: c.name, detail: `one-hot → ${fmtInt(c.unique)} columns${tag}`, tone: c.unique > ONE_HOT_BUDGET ? "warning" : "accent" });
          for (let i = 0; i < c.unique; i++)
            next.push({ name: `${c.name}_#${i}`, kind: "numeric", unique: 2, nulls: 0, nonNull: rows, binary: true, origin: "onehot" });
        } else {
          labelCols += 1;
          enc.items.push({ label: c.name, detail: `label codes 0–${fmtInt(Math.max(0, c.unique - 1))}${tag}`, tone: "info" });
          // pd.factorize codes missing values as -1, so a 2-category column with gaps isn't 0/1.
          next.push({ ...c, kind: "numeric", binary: c.unique <= 2 && c.nulls === 0, nulls: 0, nonNull: rows, origin: "label" });
        }
        // Case/spacing variants visible in the profile's sample values.
        const samples = c.profile?.sample_values ?? [];
        const seen = new Map<string, string>();
        const variants: string[] = [];
        for (const s of samples) {
          const key = s.trim().toLowerCase();
          const prev = seen.get(key);
          if (prev !== undefined && prev !== s) variants.push(`${prev} / ${s}`);
          else seen.set(key, s);
        }
        if (variants.length)
          enc.notes.push({
            tone: "warning",
            text: `${c.name}: sample values include variants that differ only in case or spacing (${variants.slice(0, 2).join(", ")}); they'll be encoded as separate categories. Normalise them in the source if they mean the same thing.`,
          });
      }
      if (flagBools) enc.items.push({ label: `${plural(flagBools, "outlier flag")}`, detail: "→ 0/1" });
      sim = next;
      const parts: string[] = [];
      if (oneHotCols) parts.push(`${fmtInt(oneHotCols)} one-hot (+${fmtInt(oneHotAdded)} columns, original${oneHotCols > 1 ? "s" : ""} replaced)`);
      if (labelCols) parts.push(`${fmtInt(labelCols)} to label codes`);
      if (boolCols) parts.push(`${fmtInt(boolCols)} boolean → 0/1`);
      enc.active = true;
      enc.summary = parts.length ? `Encodes ${plural(oneHotCols + labelCols + boolCols, "column")}: ${parts.join(", ")}` : "No categorical or boolean columns to encode";
      const ignored = Object.keys(encodeOv).filter((n) => kindNow.get(n) !== "categorical");
      if (ignored.length)
        enc.notes.push({
          tone: "info",
          text: `Encoding override${ignored.length > 1 ? "s" : ""} for ${listNames(ignored)} ${ignored.length > 1 ? "are" : "is"} ignored: the backend only applies them to columns that are categorical when this step runs.`,
        });
    }
  }

  /* ---- engineer_features ---- */
  const eng: StepEstimate = {
    step: "engineer_features",
    runs: config.engineer_features,
    active: false,
    summary: "",
    items: [],
    notes: [],
    estimate: false,
    dropsRows: false,
  };
  {
    const dts = sim.filter((c) => c.kind === "datetime");
    const txt = sim.filter((c) => c.kind === "text");
    if (!config.engineer_features) {
      eng.summary = "Off — no derived columns";
      if (dts.length + txt.length)
        eng.notes.push({ tone: "neutral", text: `${listNames([...dts, ...txt].map((c) => c.name))} won't be expanded into numeric features.` });
    } else {
      const added: SimCol[] = [];
      for (const c of dts) {
        eng.items.push({ label: c.name, detail: "+4: year, month, day, weekday", tone: "accent" });
        for (const part of ["year", "month", "day", "weekday"])
          added.push({ name: `${c.name}__${part}`, kind: "numeric", unique: 3, nulls: c.nulls, nonNull: rows - c.nulls, binary: false, origin: "derived" });
      }
      for (const c of txt) {
        eng.items.push({ label: c.name, detail: "+2: length, word count", tone: "accent" });
        for (const part of ["length", "word_count"])
          added.push({ name: `${c.name}__${part}`, kind: "numeric", unique: 3, nulls: 0, nonNull: rows, binary: false, origin: "derived" });
      }
      const present = new Set(sim.map((c) => c.name));
      const fresh = added.filter((a) => !present.has(a.name));
      sim = [...sim, ...fresh];
      eng.active = true;
      eng.summary = added.length
        ? `Adds ${plural(added.length, "column")}${fresh.length < added.length ? ` (${fmtInt(added.length - fresh.length)} overwrite existing ones)` : ""}`
        : "No date or text columns — adds nothing (the backend logs a warning)";
      if (fresh.length < added.length) eng.estimate = true;
    }
  }

  /* ---- scale_features ---- */
  const scale: StepEstimate = {
    step: "scale_features",
    runs: true,
    active: false,
    summary: "",
    items: [],
    notes: [],
    estimate: true,
    dropsRows: false,
  };
  {
    if (config.scale_method === "none") {
      scale.summary = "Off — values keep their original units";
      scale.estimate = false;
    } else {
      // With scale_columns set, the backend skips its kind filter and force-converts each listed column with to_numeric.
      const chosen = config.scale_columns === null ? null : new Set(config.scale_columns);
      const pool = chosen === null ? sim.filter((c) => c.kind === "numeric") : sim.filter((c) => chosen.has(c.name));
      const dates = pool.filter((c) => c.kind === "datetime");
      const nonNumeric = pool.filter((c) => c.kind !== "numeric" && c.kind !== "datetime" && c.kind !== "boolean" && c.kind !== "empty");
      const numericPool = pool.filter((c) => c.kind === "numeric" || c.kind === "datetime");
      const binary = pool.filter((c) => c.binary || c.kind === "boolean");
      const tooFew = numericPool.filter((c) => !c.binary && c.nonNull < 2);
      const targets = numericPool.filter((c) => !c.binary && c.nonNull >= 2);
      for (const c of targets)
        scale.items.push({ label: c.name, tone: c.origin === "source" ? "neutral" : "accent", detail: c.origin === "derived" ? "derived" : c.origin === "label" ? "label codes" : undefined });
      for (const c of tooFew) scale.items.push({ label: c.name, detail: "too few values, skipped", tone: "warning" });
      scale.active = true;
      scale.summary = targets.length
        ? `Scales ${plural(targets.length, "column")} with ${SCALE_LABELS[config.scale_method].toLowerCase()} scaling`
        : config.scale_columns !== null && config.scale_columns.length === 0
          ? "No columns selected — nothing will be scaled"
          : "No numeric columns to scale";
      if (binary.length)
        scale.notes.push({ tone: "neutral", text: `${plural(binary.length, "0/1 column")} (one-hot, booleans, outlier flags) are left unscaled, as the backend excludes binary indicators.` });
      if (config.scale_columns !== null)
        scale.notes.push({ tone: "info", text: "Only the selected columns are scaled; derived columns (date parts, label codes) are left in their original units." });
      const ids = targets.filter((c) => c.profile && isIdLike(c.profile));
      if (ids.length)
        scale.notes.push({ tone: "warning", text: `${listNames(ids.map((c) => c.name))} look${ids.length === 1 ? "s" : ""} like an ID column but will be scaled as a feature.` });
      if (targets.some((c) => c.nulls > 0))
        scale.notes.push({ tone: "warning", text: "Some scaled columns still have missing values; those cells stay NaN." });
      if (dates.length)
        scale.notes.push({
          tone: "danger",
          text: `${listNames(dates.map((c) => c.name))} ${dates.length === 1 ? "is a date column" : "are date columns"} in the scale list. The backend converts dates to their underlying timestamps and scales those, so the dates are replaced by numbers. Remove ${dates.length === 1 ? "it" : "them"} from the list to keep the dates.`,
        });
      if (nonNumeric.length) {
        for (const c of nonNumeric) scale.items.push({ label: c.name, detail: `${c.kind}, not numeric`, tone: "danger" });
        scale.notes.push({
          tone: "danger",
          text: `${listNames(nonNumeric.map((c) => c.name))} ${nonNumeric.length === 1 ? "isn't" : "aren't"} numeric when scaling runs, but ${nonNumeric.length === 1 ? "is" : "are"} in the scale list. The backend force-converts listed columns to numbers, so values that don't convert become missing. A column with no numeric values at all is left untouched. Remove ${nonNumeric.length === 1 ? "it" : "them"} from the list.`,
        });
      }
    }
  }

  const steps: Record<PipelineStepName, StepEstimate> = {
    coerce_types: coerce,
    impute_missing: impute,
    handle_duplicates: dupes,
    handle_outliers: outl,
    encode_categoricals: enc,
    engineer_features: eng,
    scale_features: scale,
  };

  // Mirrors orchestrator: removal is permitted when dedupe=True or "remove" is the default/any override.
  const rowDropReasons: string[] = [];
  if (config.dedupe) rowDropReasons.push("Duplicate removal is on");
  if (config.outlier_default_strategy === "remove") rowDropReasons.push('Outlier default strategy is "remove"');
  const removeOv = Object.entries(outlierOv).filter(([, v]) => v === "remove").map(([k]) => k);
  if (removeOv.length) rowDropReasons.push(`Outlier override "remove" on ${listNames(removeOv)}`);

  return {
    steps,
    rowsMayDrop: rowDropReasons.length > 0,
    rowDropReasons,
    columnsIn: profile.length,
    columnsOut: sim.length,
    activeSteps: Object.values(steps).filter((s) => s.active).length,
  };
}

/* ======================================================================
 * Pipeline-output detection
 * ==================================================================== */

const DERIVED_SUFFIX = /__(is_outlier|year|month|day|weekday|length|word_count)$/;

export interface PipelineOutputInfo {
  isOutput: boolean;
  markers: string[];
  parent: DatasetSummary | null;
}

export function detectPipelineOutput(dataset: DatasetSummary, datasets: DatasetSummary[]): PipelineOutputInfo {
  const markers = dataset.column_names.filter((c) => DERIVED_SUFFIX.test(c));
  const parent = dataset.parent_id ? datasets.find((d) => d.dataset_id === dataset.parent_id) ?? null : null;
  const namedCleaned = dataset.parent_id !== null && /\(cleaned\)\s*$/i.test(dataset.name);
  return { isOutput: markers.length > 0 || namedCleaned, markers, parent };
}
