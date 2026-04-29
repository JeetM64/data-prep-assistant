"""
feature_engineering.py
======================
Dataset-agnostic AutoML feature engineering module.

Responsibilities:
  - Select top correlated numeric features (up to 5)
  - Generate pairwise interaction features (multiply, ratio, difference)
  - Validate each interaction via lightweight correlation heuristic
  - Enforce hard limits to prevent feature explosion
  - Return an enriched DataFrame and a structured log list

Design constraints enforced:
  - No cross-validation or model training
  - No hardcoded column names
  - Safety guards: skips if rows < 300 or features > 50
  - Max interactions = min(20, max(5, int(0.2 * n_features)))
"""

from __future__ import annotations

import itertools
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_EPSILON = 1e-8          # guard against division-by-zero in ratio features
_TOP_K = 5               # maximum candidate features drawn from correlation rank
_MIN_ROWS = 300          # skip engineering below this row count
_MAX_FEATURES = 50       # skip engineering above this feature count
_CORR_THRESHOLD = 0.03   # minimum |correlation| for a new feature to be kept


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_constant(series: pd.Series) -> bool:
    """Return True if a Series has zero variance (all values identical)."""
    return series.nunique(dropna=True) <= 1


def _safe_correlation(series_a: pd.Series, series_b: pd.Series) -> float:
    """
    Compute Pearson correlation between two Series.
    Returns 0.0 on any failure (constant input, all-NaN, etc.).
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            val = series_a.corr(series_b)
        return float(val) if np.isfinite(val) else 0.0
    except Exception:
        return 0.0


def _select_candidate_features(
    df: pd.DataFrame,
    target: str,
    top_k: int = _TOP_K,
) -> Tuple[List[str], List[str]]:
    """
    Identify numeric columns (excluding target) and rank them by absolute
    correlation with the target.

    Returns
    -------
    top_features : list[str]
        Up to `top_k` column names with highest |corr| to target.
    logs : list[str]
        Informational messages produced during selection.
    """
    logs: List[str] = []

    numeric_cols = [
        col for col in df.select_dtypes(include=[np.number]).columns
        if col != target
    ]

    if not numeric_cols:
        logs.append("No numeric feature columns found; skipping candidate selection.")
        return [], logs

    target_series = df[target]
    if _is_constant(target_series):
        logs.append("Target column is constant; correlation-based selection is not possible.")
        return [], logs

    # Compute |correlation| for every numeric feature
    corr_map: dict[str, float] = {}
    for col in numeric_cols:
        if _is_constant(df[col]):
            logs.append(f"  Skipping constant column '{col}' from candidate selection.")
            continue
        corr_map[col] = abs(_safe_correlation(df[col], target_series))

    if not corr_map:
        logs.append("All numeric columns are constant; no candidates available.")
        return [], logs

    ranked = sorted(corr_map, key=corr_map.get, reverse=True)
    effective_k = min(top_k, len(ranked))
    top_features = ranked[:effective_k]

    logs.append(
        f"Selected top {effective_k} feature(s) based on |correlation| with target "
        f"(from {len(corr_map)} numeric column(s)): {top_features}"
    )
    return top_features, logs


def _make_interaction_name(col_a: str, col_b: str, op: str) -> str:
    """Create a deterministic, readable name for an interaction feature."""
    op_symbols = {"multiply": "x", "ratio": "div", "difference": "minus"}
    symbol = op_symbols.get(op, op)
    return f"{col_a}__{symbol}__{col_b}"


def _compute_interaction(
    series_a: pd.Series,
    series_b: pd.Series,
    op: str,
) -> pd.Series | None:
    """
    Compute a single interaction between two numeric Series.

    Supported operations
    --------------------
    multiply   : a * b
    ratio      : a / (b + epsilon)
    difference : a - b

    Returns None if the result is constant or all-NaN.
    """
    try:
        if op == "multiply":
            result = series_a * series_b
        elif op == "ratio":
            result = series_a / (series_b + _EPSILON)
        elif op == "difference":
            result = series_a - series_b
        else:
            return None

        # Replace inf/-inf introduced by ratio with NaN
        result = result.replace([np.inf, -np.inf], np.nan)

        if result.isna().all() or _is_constant(result.dropna()):
            return None

        return result
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_interactions(
    df: pd.DataFrame,
    target: str,
    include_difference: bool = False,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Generate interaction features for a pre-cleaned, fully numeric DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame. Must already be numeric (post-cleaning & encoding).
    target : str
        Name of the target column (excluded from feature generation).
    include_difference : bool, default False
        Whether to also generate difference (a - b) interactions in addition
        to multiply and ratio.

    Returns
    -------
    df_out : pd.DataFrame
        Original DataFrame with valid interaction columns appended.
    logs : list[str]
        Human-readable audit trail of every decision made.
    """
    logs: List[str] = []
    df_out = df.copy()

    # ------------------------------------------------------------------
    # 1. Safety guards
    # ------------------------------------------------------------------
    n_rows, n_features = df_out.shape
    feature_cols = [c for c in df_out.columns if c != target]
    n_feature_cols = len(feature_cols)

    if n_rows < _MIN_ROWS:
        logs.append(
            f"[SKIP] Dataset has {n_rows} rows (< {_MIN_ROWS}); "
            "feature engineering disabled."
        )
        return df_out, logs

    if n_feature_cols > _MAX_FEATURES:
        logs.append(
            f"[SKIP] Dataset has {n_feature_cols} feature columns (> {_MAX_FEATURES}); "
            "feature engineering disabled to prevent feature explosion."
        )
        return df_out, logs

    if target not in df_out.columns:
        logs.append(f"[ERROR] Target column '{target}' not found in DataFrame.")
        return df_out, logs

    logs.append(
        f"[START] Feature engineering on {n_rows} rows × {n_feature_cols} feature columns."
    )

    # ------------------------------------------------------------------
    # 2. Candidate feature selection
    # ------------------------------------------------------------------
    top_features, sel_logs = _select_candidate_features(df_out, target, top_k=_TOP_K)
    logs.extend(sel_logs)

    if len(top_features) < 2:
        logs.append("[SKIP] Fewer than 2 candidate features; no interactions possible.")
        return df_out, logs

    # ------------------------------------------------------------------
    # 3. Determine interaction budget
    # ------------------------------------------------------------------
    max_interactions = min(20, max(5, int(0.2 * n_feature_cols)))
    logs.append(f"Interaction budget: max {max_interactions} new features.")

    # ------------------------------------------------------------------
    # 4. Build interaction candidates list
    # ------------------------------------------------------------------
    operations = ["multiply", "ratio"]
    if include_difference:
        operations.append("difference")

    # Pairs are ordered to avoid (a,b) + (b,a) duplicates
    pairs = list(itertools.combinations(top_features, 2))
    candidates = [(a, b, op) for a, b in pairs for op in operations]

    logs.append(
        f"Evaluating {len(candidates)} candidate interaction(s) "
        f"from {len(pairs)} pair(s) × {len(operations)} operation(s)."
    )

    # ------------------------------------------------------------------
    # 5. Evaluate & keep interactions
    # ------------------------------------------------------------------
    target_series = df_out[target]
    existing_corrs = {
        col: abs(_safe_correlation(df_out[col], target_series))
        for col in top_features
    }
    # Baseline: best single-feature correlation among candidates
    baseline_corr = max(existing_corrs.values()) if existing_corrs else 0.0

    added = 0
    seen_names: set[str] = set()

    for col_a, col_b, op in candidates:
        if added >= max_interactions:
            logs.append(
                f"Interaction budget ({max_interactions}) reached; "
                "stopping early."
            )
            break

        feat_name = _make_interaction_name(col_a, col_b, op)

        # Skip duplicates (can arise if columns share a common prefix)
        if feat_name in seen_names or feat_name in df_out.columns:
            logs.append(f"  [SKIP-DUP] '{feat_name}' already exists.")
            continue
        seen_names.add(feat_name)

        interaction = _compute_interaction(df_out[col_a], df_out[col_b], op)
        if interaction is None:
            logs.append(f"  [DROP] '{feat_name}': result was constant or all-NaN.")
            continue

        new_corr = abs(_safe_correlation(interaction, target_series))

        # Keep if correlation beats baseline OR exceeds minimum threshold
        if new_corr > baseline_corr or new_corr > _CORR_THRESHOLD:
            df_out[feat_name] = interaction
            added += 1
            logs.append(
                f"  [ADD]  '{feat_name}' | |corr|={new_corr:.4f} "
                f"(baseline={baseline_corr:.4f})"
            )
        else:
            logs.append(
                f"  [DROP] '{feat_name}' | |corr|={new_corr:.4f} too low "
                f"(threshold={_CORR_THRESHOLD}, baseline={baseline_corr:.4f})."
            )

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    logs.append(
        f"[DONE] Generated {added} interaction feature(s). "
        f"DataFrame shape: {df_out.shape}."
    )

    return df_out, logs


# ---------------------------------------------------------------------------
# Optional: quick smoke-test when run as a script
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic dataset – no real column names assumed
    rng = np.random.default_rng(42)
    n = 500
    a = rng.normal(0, 1, n)
    b = rng.normal(5, 2, n)
    c = rng.uniform(0, 10, n)
    noise = rng.normal(0, 0.5, n)
    y = 3 * a + 0.5 * b * c + noise   # b*c is a planted interaction

    sample_df = pd.DataFrame({"feat_a": a, "feat_b": b, "feat_c": c, "target": y})

    result_df, audit_log = generate_interactions(sample_df, target="target")

    print("\n=== Audit Log ===")
    for line in audit_log:
        print(line)

    print(f"\n=== Final columns ({len(result_df.columns)}) ===")
    print(result_df.columns.tolist())