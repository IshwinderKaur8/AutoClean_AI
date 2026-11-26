/**
 * Data-quality health score and "smart" insights for the Overview tab.
 *
 * Pure functions, no React. Everything here is rule-based and derived from
 * the backend's own profile (profiling.py) plus the preview rows, and every
 * statement about what the pipeline WILL do mirrors the backend code:
 *
 *  - coerce_types (cleaning.py): converts columns profiled as numeric /
 *    boolean / datetime; values that fail conversion become missing (never
 *    dropped rows). Uploads are read as text (ingestion.py, dtype=str).
 *  - impute_missing (cleaning.py): per-kind strategy, overrides win; mean /
 *    median only apply to numeric columns; empty columns are left as-is.
 *  - handle_duplicates: exact duplicates, removal only when dedupe=true.
 *  - handle_outliers (outliers.py): 1.5×IQR fences, needs ≥4 values and a
 *    non-zero IQR; flag adds `<col>__is_outlier`; re-detected on imputed data.
 *  - encode_categoricals (encoding.py): boolean -> 0/1; categorical one-hot
 *    when unique_count <= one_hot_max_cardinality else label codes;
 *    identifier / text columns left unencoded. No case/whitespace folding.
 *  - engineer_features: datetime -> __year/__month/__day/__weekday;
 *    text -> __length/__word_count. Originals are kept.
 *  - scale_features (scaling.py): every column that is numeric at that point
 *    (incl. label codes and date parts) except 0/1 binary columns.
 *  - target detection mirrors recommender._detect_target.
 */
import type { ColumnKind, ColumnProfile, DatasetSummary, PipelineConfigIn, PreviewResult, TabId } from "../types";
import { fmtInt, fmtNum, fmtPct, plural } from "./format";

/* =========================================================================
   Shared helpers
   ========================================================================= */

/** Backend: profiling._detect_outliers_iqr / outliers._iqr_bounds use 1.5×IQR. */
const IQR_K = 1.5;
/** recommender._TARGET_NAME_HINTS, in the same priority order. */
const TARGET_NAME_HINTS = ["target", "label", "class", "y", "outcome", "result"] as const;
/** profiling.py: an object column becomes numeric/datetime at >= 95% convertibility. */
const TYPE_THRESHOLD = 0.95;

const totalNulls = (profile: ColumnProfile[]) => profile.reduce((n, p) => n + p.null_count, 0);

function listNames(names: string[], max = 4): string {
  const shown = names.slice(0, max).map((n) => `“${n}”`);
  const rest = names.length - shown.length;
  return rest > 0 ? `${shown.join(", ")} and ${rest} more` : shown.join(", ");
}

/** Mutable copy helpers so `apply` / `revert` never mutate the caller's config. */
function withRecord<K extends "impute_overrides" | "outlier_overrides" | "encoding_overrides">(
  c: PipelineConfigIn,
  key: K,
  edit: (rec: Record<string, string>) => void
): PipelineConfigIn {
  const rec = { ...(c[key] as Record<string, string>) };
  edit(rec);
  return { ...c, [key]: rec } as PipelineConfigIn;
}

/* =========================================================================
   Preview-derived signals
   ========================================================================= */

export interface DuplicateScan {
  /** Rows that are an exact repeat of an earlier row (pandas keep="first"). */
  count: number;
  rowsChecked: number;
  /** True when the preview contains every row, so `count` is exact. */
  exact: boolean;
}

/**
 * Duplicate rows in the preview. `subset` mirrors config.dedupe_subset
 * (handle_duplicates passes it straight to pandas `duplicated(subset=...)`);
 * null / empty means "compare every column".
 */
export function scanDuplicates(preview: PreviewResult | null, subset?: string[] | null): DuplicateScan | null {
  if (!preview || preview.rows.length < 2) return null;
  const cols = subset && subset.length > 0 ? subset : preview.columns;
  const seen = new Set<string>();
  let count = 0;
  for (const row of preview.rows) {
    // pandas.duplicated treats missing values as equal, so null == null here too.
    const key = JSON.stringify(cols.map((c) => row[c] ?? null));
    if (seen.has(key)) count++;
    else seen.add(key);
  }
  return { count, rowsChecked: preview.rows.length, exact: preview.rows.length >= preview.total_rows };
}

export interface VariantScan {
  column: string;
  /** Groups of raw values that are equal after trimming + lower-casing. */
  groups: string[][];
  /** Distinct raw values seen in the sample. */
  distinct: number;
}

const canonical = (s: string) => s.trim().replace(/\s+/g, " ").toLowerCase();

/** Categorical values that differ only by case / whitespace (sampled from preview + sample_values). */
export function scanCaseVariants(profile: ColumnProfile[], preview: PreviewResult | null): VariantScan[] {
  const out: VariantScan[] = [];
  for (const p of profile) {
    if (p.kind !== "categorical") continue;
    const values = new Set<string>(p.sample_values);
    if (preview) {
      for (const row of preview.rows) {
        const v = row[p.name];
        if (typeof v === "string") values.add(v);
      }
    }
    const byKey = new Map<string, string[]>();
    for (const v of values) {
      const k = canonical(v);
      const g = byKey.get(k);
      if (g) g.push(v);
      else byKey.set(k, [v]);
    }
    const groups = [...byKey.values()].filter((g) => g.length > 1);
    if (groups.length > 0) out.push({ column: p.name, groups, distinct: values.size });
  }
  return out;
}

/**
 * Roughly what pandas.to_numeric accepts for a string (decimal / exponent
 * notation and inf, surrounding whitespace allowed; no hex, no thousands
 * separators). JS Number() differs ("0x10" -> 16, "inf" -> NaN), so it isn't used.
 */
const NUMERIC_RE = /^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$|^[+-]?(?:inf|infinity)$/i;

/** Example raw values in a numeric-kind text column that won't parse as numbers. */
function nonNumericExamples(column: string, preview: PreviewResult | null, max = 3): string[] {
  if (!preview) return [];
  const found = new Set<string>();
  for (const row of preview.rows) {
    const v = row[column];
    if (typeof v !== "string") continue;
    if (!NUMERIC_RE.test(v.trim())) found.add(v);
    if (found.size >= max) break;
  }
  return [...found];
}

/**
 * Whether a column's values are still stored as text (raw upload / combine
 * output), judged from the first non-missing preview value. JSON from the
 * backend carries real numbers/booleans for typed columns. null = unknown.
 */
function storedAsText(column: string, preview: PreviewResult | null): boolean | null {
  if (!preview) return null;
  for (const row of preview.rows) {
    const v = row[column];
    if (v === null || v === undefined) continue;
    return typeof v === "string";
  }
  return null;
}

/* =========================================================================
   Pipeline predictions (mirror the backend rules)
   ========================================================================= */

/** Strategy impute_missing will pick for a column (cleaning.impute_missing). */
export function imputeStrategyFor(p: ColumnProfile, c: PipelineConfigIn): string {
  const o = c.impute_overrides[p.name];
  if (o) return o;
  if (p.kind === "numeric") return c.numeric_impute_strategy;
  if (p.kind === "datetime") return c.datetime_impute_strategy;
  if (p.kind === "empty") return "skip";
  return c.categorical_impute_strategy;
}

/** Plain-English description of what imputation does to this column, and whether it fills anything. */
export function describeImpute(p: ColumnProfile, c: PipelineConfigIn): { text: string; fills: boolean } {
  const s = imputeStrategyFor(p, c);
  if (p.kind === "empty") return { text: "left empty (nothing to impute from)", fills: false };
  if (s === "skip") return { text: "left missing (strategy: skip)", fills: false };
  if ((s === "mean" || s === "median") && p.kind !== "numeric")
    return { text: `left missing (${s} doesn't apply to ${p.kind} columns)`, fills: false };
  if (s === "median")
    return { text: p.stats.median !== undefined ? `median (${fmtNum(p.stats.median)})` : "median", fills: true };
  if (s === "mean") return { text: p.stats.mean !== undefined ? `mean (${fmtNum(p.stats.mean)})` : "mean", fills: true };
  if (s === "mode") return { text: "most frequent value", fills: true };
  if (s === "ffill_bfill") return { text: "copied from the previous row (forward/backward fill)", fills: true };
  if (s.startsWith("constant:")) return { text: `constant “${s.slice(9)}”`, fills: true };
  return { text: `left missing (unknown strategy “${s}”)`, fills: false };
}

/** encoding.encode_categoricals strategy for a categorical column. */
export function encodingFor(p: ColumnProfile, c: PipelineConfigIn): "onehot" | "label" | "skip" {
  const o = c.encoding_overrides[p.name];
  if (o) return o;
  return p.unique_count <= c.one_hot_max_cardinality ? "onehot" : "label";
}

export function outlierStrategyFor(p: ColumnProfile, c: PipelineConfigIn) {
  return c.outlier_overrides[p.name] ?? c.outlier_default_strategy;
}

/**
 * The kind the later steps (encoding, feature engineering, scaling) will see.
 * They re-profile AFTER imputation, and filling gaps changes the counts the
 * profiler uses: an identifier with missing values gets a repeated value
 * (no longer "all unique"), and a text column's distinct/non-null ratio can
 * drop to <= 0.5. Both then re-profile as categorical (<= 50 distinct or
 * ratio <= 0.5, profiling.py) or text. Other kinds are unaffected by filling.
 */
export function pipelineKind(p: ColumnProfile, c: PipelineConfigIn): ColumnKind {
  if (p.null_count === 0 || (p.kind !== "identifier" && p.kind !== "text")) return p.kind;
  if (!describeImpute(p, c).fills) return p.kind;
  const total = p.non_null_count + p.null_count;
  // A constant fill may introduce one new distinct value; mode / ffill_bfill reuse existing ones.
  const unique = p.unique_count + (imputeStrategyFor(p, c).startsWith("constant:") ? 1 : 0);
  if (unique === total && total > 20) return "identifier";
  if (unique <= 50 || unique / total <= 0.5) return "categorical";
  return "text";
}

/** scaling.scale_features: scale_method "none" skips it; scale_columns (when set) restricts it. */
export function willScale(column: string, c: PipelineConfigIn): boolean {
  return c.scale_method !== "none" && (c.scale_columns === null || c.scale_columns.includes(column));
}

export interface ShapeDelta {
  label: string;
  delta: number;
  columns: string[];
}

const DEDUPE_REASON = "duplicate removal is on";

export interface ShapeEstimate {
  /** Best estimate of output rows. */
  rows: number;
  /** exact: no row-removing setting applies; approx: `rows` is a close estimate; upper: only `<= input rows` is known. */
  rowsCertainty: "exact" | "approx" | "upper";
  /** Upper bound on rows that could be removed (null = unknown amount). */
  rowsMayDrop: { reason: string; upTo: number | null }[];
  columns: number;
  deltas: ShapeDelta[];
}

/**
 * Estimate the output shape of run_pipeline with `config`. It's an estimate:
 * outliers are re-detected after imputation and kinds are re-profiled
 * between steps, so the real numbers can differ slightly.
 */
export function estimateOutputShape(
  profile: ColumnProfile[],
  rows: number,
  c: PipelineConfigIn,
  dups: DuplicateScan | null
): ShapeEstimate {
  const deltas: ShapeDelta[] = [];
  const add = (label: string, delta: number, columns: string[]) => {
    if (delta !== 0) deltas.push({ label, delta, columns });
  };

  const flagged = profile.filter((p) => p.kind === "numeric" && p.outlier_count > 0 && outlierStrategyFor(p, c) === "flag");
  add("outlier flag columns", flagged.length, flagged.map((p) => p.name));

  if (c.encode) {
    const onehot = profile.filter((p) => pipelineKind(p, c) === "categorical" && encodingFor(p, c) === "onehot");
    add(
      "net columns from one-hot encoding",
      onehot.reduce((n, p) => n + p.unique_count - 1, 0),
      onehot.map((p) => p.name)
    );
  }
  if (c.engineer_features) {
    const dt = profile.filter((p) => p.kind === "datetime");
    add("date-part features", dt.length * 4, dt.map((p) => p.name));
    // Only columns that are still "text" after imputation get __length / __word_count.
    const tx = profile.filter((p) => pipelineKind(p, c) === "text");
    add("text-length features", tx.length * 2, tx.map((p) => p.name));
  }

  const rowsMayDrop: ShapeEstimate["rowsMayDrop"] = [];
  let estRows = rows;
  let rowsCertainty: ShapeEstimate["rowsCertainty"] = "exact";
  if (c.dedupe) {
    // handle_duplicates runs AFTER imputation: filled gaps can create new matches
    // (or, with forward fill, break existing ones), so a gap-free dataset is the
    // only case where the preview count is the real count.
    const anyGaps = profile.some((p) => p.null_count > 0);
    if (dups?.exact) {
      if (dups.count > 0 || anyGaps) rowsMayDrop.push({ reason: DEDUPE_REASON, upTo: anyGaps ? null : dups.count });
      estRows -= dups.count;
      if (anyGaps) rowsCertainty = "approx";
    } else {
      rowsMayDrop.push({ reason: DEDUPE_REASON, upTo: null });
      rowsCertainty = "upper";
    }
  }
  const removeCols = profile.filter((p) => p.kind === "numeric" && outlierStrategyFor(p, c) === "remove");
  if (removeCols.length > 0) {
    const known = removeCols.reduce((n, p) => n + p.outlier_count, 0);
    rowsMayDrop.push({ reason: `outlier removal in ${plural(removeCols.length, "column")}`, upTo: known > 0 ? known : null });
    rowsCertainty = "upper";
  }

  return {
    rows: estRows,
    rowsCertainty,
    rowsMayDrop,
    columns: profile.length + deltas.reduce((n, d) => n + d.delta, 0),
    deltas,
  };
}

export interface TargetGuess {
  column: string;
  reason: string;
  problem: "binary_classification" | "multiclass_classification" | "regression";
}

/** Mirrors recommender._detect_target + _classify_problem. */
export function detectTarget(profile: ColumnProfile[]): TargetGuess | null {
  const classify = (p: ColumnProfile): TargetGuess["problem"] =>
    p.kind === "boolean" || (p.kind === "categorical" && p.unique_count === 2)
      ? "binary_classification"
      : p.kind === "categorical"
        ? "multiclass_classification"
        : "regression";
  for (const hint of TARGET_NAME_HINTS) {
    for (const p of profile) {
      const n = p.name.toLowerCase();
      if (n === hint || n.endsWith(`_${hint}`))
        return { column: p.name, reason: `its name matches the common target name “${hint}”`, problem: classify(p) };
    }
  }
  const bool = profile.find((p) => p.kind === "boolean");
  if (bool) return { column: bool.name, reason: "it's the first yes/no (boolean) column", problem: classify(bool) };
  const cat = profile.find((p) => p.kind === "categorical" && p.unique_count >= 2 && p.unique_count <= 20);
  if (cat)
    return {
      column: cat.name,
      reason: `it's the first categorical column with 2–20 classes (${cat.unique_count})`,
      problem: classify(cat),
    };
  return null;
}

/** Pearson's second (median) skewness: 3·(mean − median) / std. */
export function medianSkew(p: ColumnProfile): number | null {
  const { mean, median, std } = p.stats;
  if (mean === undefined || median === undefined || !std || p.non_null_count < 5) return null;
  return (3 * (mean - median)) / std;
}

function looksLikeIdName(name: string): boolean {
  return /(^|[_\s.-])id$/i.test(name) || /^id([_\s.-]|$)/i.test(name) || /[a-z]Id$/.test(name) || /^(index|uuid|guid|key)$/i.test(name);
}

/* =========================================================================
   Health score
   ========================================================================= */

export interface HealthComponent {
  key: "completeness" | "consistency" | "outliers" | "structure" | "duplicates";
  label: string;
  /** 0..100 */
  score: number;
  /** Fraction of the overall score (weights of available components sum to 1). */
  weight: number;
  reason: string;
}

export interface HealthResult {
  score: number;
  grade: string;
  tone: "success" | "info" | "warning" | "danger";
  components: HealthComponent[];
}

const clamp100 = (n: number) => Math.max(0, Math.min(100, n));

export function computeHealth(profile: ColumnProfile[], rows: number, preview?: PreviewResult | null): HealthResult {
  if (profile.length === 0 || rows === 0) {
    return {
      score: 0,
      grade: "No data",
      tone: "danger",
      components: [
        { key: "structure", label: "Structure", score: 0, weight: 1, reason: rows === 0 ? "The dataset has no rows." : "The dataset has no columns." },
      ],
    };
  }

  const comps: Omit<HealthComponent, "weight">[] = [];
  const raw: Record<HealthComponent["key"], number> = { completeness: 0.35, consistency: 0.25, outliers: 0.15, structure: 0.15, duplicates: 0.1 };

  // Completeness — each 1% of missing cells costs 2 points.
  const cells = rows * profile.length;
  const nulls = totalNulls(profile);
  const missRatio = cells ? nulls / cells : 0;
  comps.push({
    key: "completeness",
    label: "Completeness",
    score: clamp100(100 * (1 - 2 * missRatio)),
    reason:
      nulls === 0
        ? "No missing cells."
        : `${fmtInt(nulls)} of ${fmtInt(cells)} cells (${fmtPct(missRatio)}) are missing; each 1% costs 2 points.`,
  });

  // Type consistency — share of each column's values that fit its type.
  const variants = new Map(scanCaseVariants(profile, preview ?? null).map((v) => [v.column, v]));
  const issues: string[] = [];
  let consSum = 0;
  let consN = 0;
  for (const p of profile) {
    if (p.kind === "empty") continue;
    let s = 1;
    if (p.kind === "numeric") s = p.numeric_convertible_ratio;
    else if (p.kind === "datetime") s = p.datetime_convertible_ratio;
    else if (p.kind === "categorical" || p.kind === "text") {
      const r = Math.max(p.numeric_convertible_ratio, p.datetime_convertible_ratio);
      if (r >= 0.5 && r < TYPE_THRESHOLD) s = Math.max(r, 1 - r);
      const v = variants.get(p.name);
      if (v) {
        const merged = v.groups.reduce((n, g) => n + g.length - 1, 0);
        s = Math.min(s, (v.distinct - merged) / v.distinct);
      }
    }
    if (s < 0.9999) issues.push(`“${p.name}” ${fmtPct(s, 0)}`);
    consSum += s;
    consN++;
  }
  comps.push({
    key: "consistency",
    label: "Type consistency",
    score: consN ? clamp100((100 * consSum) / consN) : 100,
    reason:
      issues.length === 0
        ? "Every value fits its column's inferred type, with no case/spacing variants in the sample."
        : `Average share of values that fit their column's type. Below 100%: ${issues.slice(0, 4).join(", ")}${issues.length > 4 ? ` and ${issues.length - 4} more` : ""}.`,
  });

  // Outlier burden — each 1% of numeric values outside the fences costs 5 points.
  const numeric = profile.filter((p) => p.kind === "numeric");
  const numCells = numeric.reduce((n, p) => n + p.non_null_count, 0);
  const outCells = numeric.reduce((n, p) => n + p.outlier_count, 0);
  const outRatio = numCells ? outCells / numCells : 0;
  comps.push({
    key: "outliers",
    label: "Outlier burden",
    score: clamp100(100 * (1 - 5 * outRatio)),
    reason:
      numeric.length === 0
        ? "No numeric columns to check."
        : outCells === 0
          ? `No values outside the ${IQR_K}×IQR fences in ${plural(numeric.length, "numeric column")}.`
          : `${fmtInt(outCells)} of ${fmtInt(numCells)} numeric values (${fmtPct(outRatio)}) are outside the ${IQR_K}×IQR fences; each 1% costs 5 points.`,
  });

  // Structure — empty or constant columns carry no information.
  const unusable = profile.filter((p) => p.kind === "empty" || p.unique_count <= 1);
  comps.push({
    key: "structure",
    label: "Structure",
    score: clamp100((100 * (profile.length - unusable.length)) / profile.length),
    reason:
      unusable.length === 0
        ? "Every column has at least two distinct values."
        : `${plural(unusable.length, "column")} (${listNames(unusable.map((p) => p.name), 3)}) ${unusable.length === 1 ? "is" : "are"} empty or constant: share of usable columns.`,
  });

  // Duplicates — only when the preview has rows to compare; each 1% costs 3 points.
  const dups = scanDuplicates(preview ?? null);
  if (dups) {
    const r = dups.count / dups.rowsChecked;
    comps.push({
      key: "duplicates",
      label: "Uniqueness",
      score: clamp100(100 * (1 - 3 * r)),
      reason:
        dups.count === 0
          ? `No exact duplicate rows ${dups.exact ? "in the dataset" : `in the first ${fmtInt(dups.rowsChecked)} rows`}.`
          : `${plural(dups.count, "exact duplicate row")} ${dups.exact ? "in the dataset" : `in the first ${fmtInt(dups.rowsChecked)} rows`} (${fmtPct(r)}); each 1% costs 3 points.`,
    });
  }

  const wSum = comps.reduce((n, c) => n + raw[c.key], 0);
  const components: HealthComponent[] = comps.map((c) => ({ ...c, weight: raw[c.key] / wSum }));
  const score = Math.round(components.reduce((n, c) => n + c.score * c.weight, 0));
  const [grade, tone]: [string, HealthResult["tone"]] =
    score >= 90 ? ["Excellent", "success"] : score >= 75 ? ["Good", "info"] : score >= 55 ? ["Fair", "warning"] : ["Needs work", "danger"];
  return { score, grade, tone, components };
}

/* =========================================================================
   Insights
   ========================================================================= */

export type Severity = "danger" | "warning" | "info" | "success";

export type InsightAction =
  | { type: "navigate"; label: string; tab: TabId }
  | {
      type: "config";
      label: string;
      apply: (c: PipelineConfigIn) => PipelineConfigIn;
      /** Undo just this change. */
      revert: (c: PipelineConfigIn) => PipelineConfigIn;
      /** Whether the current config already has this change. */
      isApplied: (c: PipelineConfigIn) => boolean;
      /** Applying this removes rows: the UI must confirm and label it. */
      dropsRows?: boolean;
    }
  | { type: "ask"; label: string; question: string };

export interface Insight {
  id: string;
  severity: Severity;
  title: string;
  detail: string;
  columns: string[];
  action?: InsightAction;
  secondary?: InsightAction;
  /** Used to sort within a severity (bigger first). */
  weight?: number;
}

const SEVERITY_RANK: Record<Severity, number> = { danger: 0, warning: 1, info: 2, success: 3 };

const OUTLIER_EFFECT: Record<string, string> = {
  flag: "Current setting “flag”: values stay unchanged and a 0/1 column “{col}__is_outlier” is added.",
  clip: "Current setting “clip”: values will be capped at the fences, rows are kept.",
  remove: "Current setting “remove”: every row with an outlier here will be DROPPED.",
  skip: "Current setting “skip”: outliers in this column are left untouched and not flagged.",
};

export function getInsights(
  profile: ColumnProfile[],
  preview: PreviewResult | null,
  dataset: DatasetSummary,
  c: PipelineConfigIn
): Insight[] {
  const out: Insight[] = [];
  const rows = dataset.rows;
  const byName = new Map(profile.map((p) => [p.name, p]));
  const dups = scanDuplicates(preview);
  const navClean: InsightAction = { type: "navigate", label: "Review in Clean", tab: "clean" };
  const navColumns: InsightAction = { type: "navigate", label: "Review columns", tab: "columns" };

  /* ---------- settings that can break the run or drop rows ---------- */
  if (!c.coerce_types) {
    // Without coerce_types, numeric/boolean columns that are still stored as text
    // reach steps that assume real dtypes. Reproduced against the backend:
    //  - impute_missing: mean/median on a text column -> TypeError ("Cannot perform reduction 'median' with string dtype")
    //  - handle_outliers: IQR on a text column with >= 4 values -> TypeError (str - str)
    //  - encode_categoricals: astype("boolean") on "yes"/"no"/"true"/"1" strings -> TypeError ("Need to pass bool-like values")
    const asText = (p: ColumnProfile) => storedAsText(p.name, preview) ?? !dataset.parent_id;
    const reasons: string[] = [];
    const risky = new Set<string>();
    const imputeFail = profile.filter((p) => {
      const s = imputeStrategyFor(p, c);
      return p.kind === "numeric" && asText(p) && p.null_count > 0 && (s === "mean" || s === "median");
    });
    if (imputeFail.length > 0)
      reasons.push(
        `${[...new Set(imputeFail.map((p) => imputeStrategyFor(p, c)))].join("/")} imputation fails on text (${listNames(imputeFail.map((p) => p.name), 3)})`
      );
    const outlierFail = profile.filter((p) => p.kind === "numeric" && asText(p) && p.non_null_count >= 4 && outlierStrategyFor(p, c) !== "skip");
    if (outlierFail.length > 0)
      reasons.push(`the outlier check can't compute quartiles of text (${listNames(outlierFail.map((p) => p.name), 3)})`);
    const boolFail = c.encode ? profile.filter((p) => p.kind === "boolean" && asText(p)) : [];
    if (boolFail.length > 0) reasons.push(`encoding can't turn text yes/no values into 0/1 (${listNames(boolFail.map((p) => p.name), 3)})`);
    for (const p of [...imputeFail, ...outlierFail, ...boolFail]) risky.add(p.name);
    out.push({
      id: "cfg:coerce-off",
      severity: risky.size > 0 ? "danger" : "warning",
      title: risky.size > 0 ? "Type conversion is off: the run will fail" : "Type conversion is turned off",
      detail:
        "Uploaded files are read as text, so without type conversion numbers, yes/no values and dates stay strings." +
        (reasons.length > 0
          ? ` With these settings the backend raises an error because ${reasons.join("; ")}.`
          : " Nothing here is known to fail, but numbers stay text until scaling converts them, and date columns stay strings."),
      columns: [...risky],
      action: {
        type: "config",
        label: "Turn type conversion on",
        apply: (x) => ({ ...x, coerce_types: true }),
        revert: (x) => ({ ...x, coerce_types: false }),
        isApplied: (x) => x.coerce_types,
      },
    });
  }

  const removeCols = profile.filter((p) => p.kind === "numeric" && outlierStrategyFor(p, c) === "remove");
  if (removeCols.length > 0) {
    const known = removeCols.reduce((n, p) => n + p.outlier_count, 0);
    out.push({
      id: "cfg:outlier-remove",
      severity: "danger",
      title: "Outlier removal will drop rows",
      detail: `Outlier strategy “remove” is set for ${listNames(removeCols.map((p) => p.name))}. Every row with a value outside the ${IQR_K}×IQR fences in those columns is deleted${known > 0 ? ` — about ${plural(known, "row")} based on the current profile (a row counts once even if it's an outlier in several columns)` : ""}. Flag or clip keep every row.`,
      columns: removeCols.map((p) => p.name),
      action: {
        type: "config",
        label: "Flag instead (keep all rows)",
        apply: (x) => {
          const next = withRecord(x, "outlier_overrides", (r) => {
            for (const k of Object.keys(r)) if (r[k] === "remove") r[k] = "flag";
          });
          return next.outlier_default_strategy === "remove" ? { ...next, outlier_default_strategy: "flag" } : next;
        },
        revert: (x) => x,
        isApplied: () => false,
      },
      weight: known,
    });
  }

  /* ---------- duplicates ---------- */
  if (dups && dups.count > 0) {
    const where = dups.exact ? "" : ` in the first ${fmtInt(dups.rowsChecked)} rows`;
    out.push({
      id: "dups",
      severity: "warning",
      title: `${plural(dups.count, "exact duplicate row")}${where}`,
      detail: c.dedupe
        ? `Duplicate removal is ON, so these rows will be dropped (the first copy is kept). The check runs after imputation, so the final count can differ slightly.`
        : `Duplicate removal is off by default: duplicates are detected and reported in the audit trail, but kept. Turning it on drops the repeats (the first copy is kept).`,
      columns: [],
      action: {
        type: "config",
        label: dups.exact ? `Remove duplicates (drops ${plural(dups.count, "row")})` : "Remove duplicates (drops rows)",
        apply: (x) => ({ ...x, dedupe: true }),
        revert: (x) => ({ ...x, dedupe: false }),
        isApplied: (x) => x.dedupe,
        dropsRows: true,
      },
      secondary: { type: "navigate", label: "See rows in Preview", tab: "preview" },
      weight: dups.count,
    });
  } else if (c.dedupe) {
    out.push({
      id: "cfg:dedupe-on",
      severity: "warning",
      title: "Duplicate removal is on",
      detail: dups?.exact
        ? "No exact duplicates exist right now, but rows that become identical after imputation would be dropped."
        : dups
          ? `No duplicates in the first ${fmtInt(dups.rowsChecked)} rows, but any exact duplicates elsewhere will be dropped.`
          : "Any exact duplicate rows will be dropped (the first copy is kept).",
      columns: [],
      action: {
        type: "config",
        label: "Turn off (keep every row)",
        apply: (x) => ({ ...x, dedupe: false }),
        revert: (x) => ({ ...x, dedupe: true }),
        isApplied: (x) => !x.dedupe,
      },
    });
  }

  /* ---------- empty / constant / identifiers ---------- */
  const empty = profile.filter((p) => p.kind === "empty");
  if (empty.length > 0) {
    out.push({
      id: "empty",
      severity: "warning",
      title: `${plural(empty.length, "column")} completely empty`,
      detail:
        "No values at all. The pipeline never drops columns: imputation skips them (with a warning) and encoding/scaling ignore them, so they stay 100% missing in the output. Remove them before training.",
      columns: empty.map((p) => p.name),
      action: navColumns,
    });
  }

  const constant = profile.filter((p) => p.kind !== "empty" && p.unique_count <= 1 && rows > 1);
  if (constant.length > 0) {
    out.push({
      id: "constant",
      severity: "info",
      title: `${plural(constant.length, "constant column")}`,
      detail: `Only one distinct value (${constant
        .slice(0, 3)
        .map((p) => `${p.name} = “${p.sample_values[0] ?? ""}”`)
        .join(", ")}), so ${constant.length === 1 ? "it carries" : "they carry"} no information for a model. The pipeline keeps ${constant.length === 1 ? "it" : "them"}; consider excluding before training.`,
      columns: constant.map((p) => p.name),
    });
  }

  const idents = profile.filter((p) => p.kind === "identifier");
  if (idents.length > 0) {
    out.push({
      id: "identifiers",
      severity: "info",
      title: `${plural(idents.length, "identifier column")} detected`,
      detail:
        "Every value is unique (and there are more than 20 rows), so profiling treats it as an ID. The encoder leaves identifiers as text with a warning rather than dropping them, and scaling skips them. Exclude them before training.",
      columns: idents.map((p) => p.name),
    });
  }

  const numericIds = profile.filter(
    (p) => p.kind === "numeric" && looksLikeIdName(p.name) && p.non_null_count >= 3 && p.unique_count >= 0.8 * p.non_null_count
  );
  if (numericIds.length > 0) {
    out.push({
      id: "numeric-ids",
      severity: "warning",
      title: `${listNames(numericIds.map((p) => p.name), 2)} looks like an ID but is numeric`,
      detail: `The name suggests a row ID and ${numericIds
        .map((p) => `${fmtInt(p.unique_count)} of ${fmtInt(p.non_null_count)} values are unique in “${p.name}”`)
        .join("; ")}. Because the values are numbers, the pipeline treats ${numericIds.length === 1 ? "it" : "them"} as a numeric feature: outlier-checked and ${c.scale_method === "none" ? "kept unscaled (scaling is off)" : `scaled with the ${c.scale_method} scaler`}. IDs carry no predictive signal; exclude before training.`,
      columns: numericIds.map((p) => p.name),
      action: navColumns,
    });
  }

  /* ---------- missing values ---------- */
  const withNulls = profile.filter((p) => p.kind !== "empty" && p.null_count > 0);
  const veryHigh = withNulls.filter((p) => p.null_ratio >= 0.5);
  if (veryHigh.length > 0) {
    out.push({
      id: "missing-high",
      severity: "warning",
      title: `${plural(veryHigh.length, "column")} mostly missing`,
      detail: veryHigh
        .map((p) => {
          const d = describeImpute(p, c);
          return `“${p.name}” is ${fmtPct(p.null_ratio)} missing → ${d.fills ? `${fmtInt(p.null_count)} values filled with ${d.text}, so most of the column will be imputed rather than observed` : d.text}`;
        })
        .join(". ") + ".",
      columns: veryHigh.map((p) => p.name),
      action: navColumns,
      secondary: { type: "ask", label: "Ask about missing data", question: "How much data is missing and how will it be cleaned?" },
      weight: veryHigh.reduce((n, p) => n + p.null_count, 0),
    });
  }
  const moderate = withNulls.filter((p) => p.null_ratio < 0.5);
  if (moderate.length > 0) {
    const plans = moderate.map((p) => ({ p, d: describeImpute(p, c) }));
    const unfilled = plans.filter((x) => !x.d.fills);
    const total = moderate.reduce((n, p) => n + p.null_count, 0);
    out.push({
      id: "missing",
      severity: unfilled.length > 0 ? "warning" : "info",
      title: `${plural(total, "missing value")} in ${plural(moderate.length, "column")}`,
      detail:
        "Rows are never dropped for gaps. Plan: " +
        plans
          .slice(0, 6)
          .map(({ p, d }) => `“${p.name}” (${fmtInt(p.null_count)}) → ${d.text}`)
          .join("; ") +
        (plans.length > 6 ? `; and ${plans.length - 6} more` : "") +
        "." +
        (plans.some(({ p, d }) => d.fills && imputeStrategyFor(p, c) === "ffill_bfill")
          ? " Forward fill depends on row order, so sort the data first if order matters."
          : ""),
      columns: moderate.map((p) => p.name),
      action: { type: "ask", label: "Ask about missing data", question: "How much data is missing and how will it be cleaned?" },
      secondary: navClean,
      weight: total,
    });
  }

  /* ---------- type problems ---------- */
  for (const p of profile) {
    if (p.kind === "numeric" && p.numeric_convertible_ratio < 1 && p.numeric_convertible_ratio >= TYPE_THRESHOLD) {
      const bad = Math.max(1, Math.round((1 - p.numeric_convertible_ratio) * p.non_null_count));
      const ex = nonNumericExamples(p.name, preview);
      out.push({
        id: `mixed-num:${p.name}`,
        severity: "warning",
        title: `“${p.name}” has ≈${fmtInt(bad)} non-numeric ${bad === 1 ? "value" : "values"}`,
        detail: `${fmtPct(p.numeric_convertible_ratio)} of values are numbers, so it's treated as numeric. ${ex.length ? `Values like ${ex.map((e) => `“${e}”`).join(", ")} ` : "The rest "}can't convert: type conversion sets them to missing (never drops rows), then imputation fills them with ${describeImpute(p, c).text}.`,
        columns: [p.name],
        action: { type: "navigate", label: "Inspect in Preview", tab: "preview" },
        weight: bad,
      });
    } else if (p.kind === "datetime" && p.datetime_convertible_ratio < 1 && p.datetime_convertible_ratio >= TYPE_THRESHOLD) {
      const bad = Math.max(1, Math.round((1 - p.datetime_convertible_ratio) * p.non_null_count));
      out.push({
        id: `mixed-dt:${p.name}`,
        severity: "warning",
        title: `“${p.name}” has ≈${fmtInt(bad)} unparseable ${bad === 1 ? "date" : "dates"}`,
        detail: `${fmtPct(p.datetime_convertible_ratio)} of values parse as dates. The rest become missing during type conversion and are then filled with ${describeImpute(p, c).text}.`,
        columns: [p.name],
        action: { type: "navigate", label: "Inspect in Preview", tab: "preview" },
        weight: bad,
      });
    } else if (p.kind === "categorical" || p.kind === "text") {
      const num = p.numeric_convertible_ratio;
      const dt = p.datetime_convertible_ratio;
      const [r, what] = num >= dt ? [num, "numbers"] : [dt, "dates"];
      if (r >= 0.5 && r < TYPE_THRESHOLD) {
        out.push({
          id: `mixed:${p.name}`,
          severity: "warning",
          title: `“${p.name}” mixes ${what} and text`,
          detail: `${fmtPct(r)} of values are ${what}, below the 95% needed to convert the column, so it stays ${p.kind}. The ${what} will be ${p.kind === "categorical" ? "encoded as category labels, not as quantities" : "left as free text"}. Clean the odd values at the source to get a real ${what === "numbers" ? "numeric" : "date"} column.`,
          columns: [p.name],
          action: { type: "navigate", label: "Inspect in Preview", tab: "preview" },
          weight: p.non_null_count,
        });
      }
    }
  }

  /* ---------- categorical encoding ---------- */
  for (const v of scanCaseVariants(profile, preview)) {
    const p = byName.get(v.column);
    if (!p) continue;
    const enc = c.encode ? encodingFor(p, c) : "skip";
    const sample = v.groups
      .slice(0, 3)
      .map((g) => g.map((x) => `“${x}”`).join(" / "))
      .join(", ");
    out.push({
      id: `variants:${v.column}`,
      severity: "warning",
      title: `“${v.column}” has values that differ only by case or spacing`,
      detail: `${sample} are treated as different categories. The pipeline does not normalise case or whitespace, so ${
        enc === "onehot"
          ? `one-hot encoding will create separate columns for each spelling (e.g. ${v.groups[0].map((x) => `“${v.column}_${x}”`).join(", ")})`
          : enc === "label"
            ? "label encoding will give each spelling its own code"
            : "they stay distinct values"
      }. Fix the spelling at the source before uploading. Found in a sample of ${plural(v.distinct, "distinct value")}.`,
      columns: [v.column],
      action: { type: "navigate", label: "Inspect in Preview", tab: "preview" },
      weight: v.groups.length,
    });
  }

  if (c.encode) {
    for (const p of profile) {
      if (p.kind !== "categorical") continue;
      const override = c.encoding_overrides[p.name];
      const auto = p.unique_count <= c.one_hot_max_cardinality ? "onehot" : "label";
      // Only columns that exceed the one-hot limit, and whose encoding the user hasn't set to something else.
      if (auto !== "label") continue;
      if (override && override !== "onehot") continue;
      const canOneHot = p.unique_count <= 50;
      out.push({
        id: `highcard:${p.name}`,
        severity: "info",
        title: `“${p.name}” has ${fmtInt(p.unique_count)} categories`,
        detail:
          `That's above the one-hot limit (${c.one_hot_max_cardinality}), so it gets label encoding: integer codes in alphabetical order, which implies an order that isn't real. ` +
          (c.scale_method !== "none" ? `Those codes are then scaled like any numeric column. ` : "") +
          (canOneHot ? `One-hot instead would add ${fmtInt(p.unique_count - 1)} net columns.` : "One-hot would add too many columns here; consider grouping rare values at the source."),
        columns: [p.name],
        action: canOneHot
          ? {
              type: "config",
              label: `One-hot instead (+${fmtInt(p.unique_count - 1)} cols)`,
              apply: (x) => withRecord(x, "encoding_overrides", (r) => void (r[p.name] = "onehot")),
              revert: (x) => withRecord(x, "encoding_overrides", (r) => void delete r[p.name]),
              isApplied: (x) => x.encoding_overrides[p.name] === "onehot",
            }
          : navClean,
        weight: p.unique_count,
      });
    }
  } else if (profile.some((p) => p.kind === "categorical" || p.kind === "boolean")) {
    out.push({
      id: "cfg:encode-off",
      severity: "info",
      title: "Encoding is turned off",
      detail: "Categorical and yes/no columns will stay as text in the output; most models need them as numbers.",
      columns: profile.filter((p) => p.kind === "categorical" || p.kind === "boolean").map((p) => p.name),
      action: {
        type: "config",
        label: "Turn encoding on",
        apply: (x) => ({ ...x, encode: true }),
        revert: (x) => ({ ...x, encode: false }),
        isApplied: (x) => x.encode,
      },
    });
  }

  /* ---------- outliers ---------- */
  const numeric = profile.filter((p) => p.kind === "numeric");
  for (const p of numeric) {
    if (p.outlier_count === 0) continue;
    const strategy = outlierStrategyFor(p, c);
    if (strategy === "remove") continue; // covered by the row-drop insight above
    const ratio = p.non_null_count ? p.outlier_count / p.non_null_count : 0;
    const override = c.outlier_overrides[p.name];
    let action: InsightAction | undefined;
    if (override === "clip" || (!override && strategy !== "clip")) {
      action = {
        type: "config",
        label: `Clip “${p.name}”`,
        apply: (x) => withRecord(x, "outlier_overrides", (r) => void (r[p.name] = "clip")),
        revert: (x) => withRecord(x, "outlier_overrides", (r) => void delete r[p.name]),
        isApplied: (x) => x.outlier_overrides[p.name] === "clip",
      };
    }
    out.push({
      id: `outliers:${p.name}`,
      severity: ratio >= 0.05 ? "warning" : "info",
      title: `${plural(p.outlier_count, "outlier")} in “${p.name}” (${fmtPct(ratio)})`,
      detail: `Values outside ${IQR_K}×IQR of the quartiles (range ${fmtNum(p.stats.min)} – ${fmtNum(p.stats.max)}, median ${fmtNum(p.stats.median)}). ${(OUTLIER_EFFECT[strategy] ?? "").replace("{col}", p.name)} The pipeline re-detects after imputation, so its count can differ slightly.`,
      columns: [p.name],
      action,
      secondary: action ? { type: "ask", label: "Ask about outliers", question: "Are there any outliers in this dataset?" } : navClean,
      weight: p.outlier_count,
    });
  }
  if (numeric.length > 0 && numeric.every((p) => p.non_null_count < 4) && rows > 0) {
    out.push({
      id: "outliers-too-few",
      severity: "info",
      title: "Too few values for outlier detection",
      detail: "The IQR outlier check needs at least 4 non-missing values per numeric column, so it won't flag anything here.",
      columns: [],
    });
  }

  const heavy = numeric.filter((p) => p.non_null_count > 0 && p.outlier_count / p.non_null_count >= 0.05);
  const skewed = numeric
    .map((p) => ({ p, s: medianSkew(p) }))
    .filter((x): x is { p: ColumnProfile; s: number } => x.s !== null && Math.abs(x.s) >= 1);
  if ((heavy.length > 0 || skewed.length > 0) && (c.scale_method === "standard" || c.scale_method === "robust")) {
    const cols = [...new Set([...heavy, ...skewed.map((x) => x.p)].map((p) => p.name))];
    out.push({
      id: "robust-scaling",
      severity: "info",
      title: "Robust scaling may suit this data",
      detail: `${listNames(cols)} ${cols.length === 1 ? "has" : "have"} ${heavy.length > 0 ? "many outliers" : ""}${heavy.length > 0 && skewed.length > 0 ? " or " : ""}${skewed.length > 0 ? "a skewed distribution" : ""}. The standard scaler uses mean and standard deviation, which extreme values pull around; the robust scaler uses median and IQR instead. ${c.scale_method === "robust" ? "Robust scaling is selected." : ""}`.trim(),
      columns: cols,
      action: {
        type: "config",
        label: "Use robust scaling",
        apply: (x) => ({ ...x, scale_method: "robust" }),
        revert: (x) => ({ ...x, scale_method: "standard" }),
        isApplied: (x) => x.scale_method === "robust",
      },
    });
  }

  if (skewed.length > 0) {
    const meanCandidates = skewed
      .map((x) => x.p)
      .filter((p) => {
        const o = c.impute_overrides[p.name];
        return p.null_count > 0 && (o === "median" || (!o && c.numeric_impute_strategy === "mean"));
      });
    const parts = skewed
      .slice(0, 4)
      .map(({ p, s }) => `“${p.name}” ${s > 0 ? "+" : ""}${fmtNum(s, 2)} (mean ${fmtNum(p.stats.mean)} vs median ${fmtNum(p.stats.median)})`);
    out.push({
      id: "skew",
      severity: meanCandidates.length > 0 && meanCandidates.some((p) => c.impute_overrides[p.name] !== "median") ? "warning" : "info",
      title: `${plural(skewed.length, "skewed numeric column")}`,
      detail:
        `Pearson median skewness 3·(mean − median)/std is at least ±1 for ${parts.join(", ")}. ` +
        (meanCandidates.length > 0
          ? "Mean imputation is selected, and the mean is pulled toward the long tail; median imputation is more robust."
          : skewed.some((x) => x.p.null_count > 0)
            ? "Gaps in these columns are filled with the median, which is robust to the tail."
            : "No gaps to fill here, but consider a log transform before linear models."),
      columns: skewed.map((x) => x.p.name),
      action:
        meanCandidates.length > 0
          ? {
              type: "config",
              label: "Impute these with the median",
              apply: (x) => withRecord(x, "impute_overrides", (r) => meanCandidates.forEach((p) => (r[p.name] = "median"))),
              revert: (x) => withRecord(x, "impute_overrides", (r) => meanCandidates.forEach((p) => delete r[p.name])),
              isApplied: (x) => meanCandidates.every((p) => x.impute_overrides[p.name] === "median"),
            }
          : undefined,
      weight: skewed.length,
    });
  }

  /* ---------- feature engineering ---------- */
  const dts = profile.filter((p) => p.kind === "datetime");
  if (dts.length > 0) {
    out.push({
      id: "datetime",
      severity: "info",
      title: `${plural(dts.length, "date/time column")}`,
      detail: c.engineer_features
        ? `Feature engineering adds 4 columns per date (__year, __month, __day, __weekday), ${fmtInt(dts.length * 4)} in total, and keeps the original.${c.scale_method !== "none" ? " The new parts are numbers, so they're scaled too." : ""}`
        : "Feature engineering is off, so dates stay as a single datetime column. Most models can't use a raw date.",
      columns: dts.map((p) => p.name),
      action: c.engineer_features
        ? undefined
        : {
            type: "config",
            label: "Turn feature engineering on",
            apply: (x) => ({ ...x, engineer_features: true }),
            revert: (x) => ({ ...x, engineer_features: false }),
            isApplied: (x) => x.engineer_features,
          },
    });
  }
  const texts = profile.filter((p) => p.kind === "text");
  if (texts.length > 0) {
    out.push({
      id: "text",
      severity: "info",
      title: `${plural(texts.length, "free-text column")}`,
      detail: `Too many distinct values to be a category, so ${texts.length === 1 ? "it's" : "they're"} left unencoded (no bag-of-words/embedding step exists).${c.engineer_features ? " Feature engineering adds __length and __word_count for each." : ""}`,
      columns: texts.map((p) => p.name),
    });
  }

  /* ---------- target ---------- */
  const target = detectTarget(profile);
  if (target) {
    const tp = byName.get(target.column);
    const problem =
      target.problem === "binary_classification" ? "binary classification" : target.problem === "multiclass_classification" ? "multiclass classification" : "regression";
    let fate = "";
    let action: InsightAction | undefined;
    let sev: Severity = "info";
    if (tp) {
      if (tp.kind === "boolean") fate = c.encode ? "It becomes a 0/1 column, which scaling leaves alone." : "";
      else if (tp.kind === "categorical" && c.encode) {
        const enc = encodingFor(tp, c);
        if (enc === "onehot") {
          fate = `With current settings it would be one-hot encoded into ${fmtInt(tp.unique_count)} separate columns, which breaks it as a single target.`;
          sev = "warning";
        } else if (enc === "label")
          fate = `It would be label-encoded to integer codes${c.scale_method !== "none" && tp.unique_count > 2 ? " and then scaled" : ""}.`;
        else fate = "Encoding is skipped for it, so it keeps its original labels.";
        action = {
          type: "config",
          label: "Keep target unencoded",
          apply: (x) => withRecord(x, "encoding_overrides", (r) => void (r[tp.name] = "skip")),
          revert: (x) => withRecord(x, "encoding_overrides", (r) => void delete r[tp.name]),
          isApplied: (x) => x.encoding_overrides[tp.name] === "skip",
        };
      } else if (tp.kind === "numeric" && c.scale_method !== "none")
        fate = `As a numeric column it will be scaled with the ${c.scale_method} scaler like the features.`;
    }
    out.push({
      id: "target",
      severity: sev,
      title: `Likely target: “${target.column}” (${problem})`,
      detail: `Picked because ${target.reason}, using the same rules as the Models tab. The cleaning pipeline doesn't know about targets and processes it like any other column. ${fate}`.trim(),
      columns: [target.column],
      action: action ?? { type: "ask", label: "Which algorithm?", question: "Which algorithm should I use for this dataset?" },
      secondary: action ? { type: "ask", label: "Which algorithm?", question: "Which algorithm should I use for this dataset?" } : { type: "navigate", label: "Open Models", tab: "models" },
    });
  }

  /* ---------- size ---------- */
  if (rows > 0 && rows < 100) {
    out.push({
      id: "small",
      severity: "info",
      title: `Small dataset: ${plural(rows, "row")}`,
      detail: `With under 100 rows, models overfit easily and the outlier check is noisy. The Synthetic menu in the header can add rows sampled from each column's own distribution (as a new dataset; this one is untouched). Those rows don't preserve relationships between columns, so treat them as a stopgap, not real data.`,
      columns: [],
    });
  }

  if (!out.some((i) => i.severity === "danger" || i.severity === "warning")) {
    out.push({
      id: "ok",
      severity: "success",
      title: "No blocking issues found",
      detail: `Checked ${plural(profile.length, "column")} for missing values, type problems, duplicates, outliers and encoding pitfalls. The default pipeline keeps every row.`,
      columns: [],
      action: navClean,
    });
  }

  return out.sort((a, b) => SEVERITY_RANK[a.severity] - SEVERITY_RANK[b.severity] || (b.weight ?? 0) - (a.weight ?? 0));
}

/** Count of columns per kind, in a fixed display order. */
export const KIND_ORDER: ColumnKind[] = ["numeric", "categorical", "boolean", "datetime", "text", "identifier", "empty"];

export function kindComposition(profile: ColumnProfile[]): { kind: ColumnKind; count: number }[] {
  const counts = new Map<ColumnKind, number>();
  for (const p of profile) counts.set(p.kind, (counts.get(p.kind) ?? 0) + 1);
  return KIND_ORDER.filter((k) => counts.has(k)).map((k) => ({ kind: k, count: counts.get(k) ?? 0 }));
}
