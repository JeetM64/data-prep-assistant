"""
dataset_comparison.py

Compares two datasets (e.g. train.csv vs test.csv) and detects
distribution drift between them.

Distribution drift = when the statistical properties of test data
are significantly different from training data. This causes models
to fail in production even when they have high training accuracy.

This is a real industry problem — models trained on one distribution
but deployed against a different one silently produce wrong predictions.

Techniques used:
  - Kolmogorov-Smirnov test (numeric columns) — non-parametric,
    detects any difference in distribution shape
  - Chi-squared test (categorical columns) — detects shift in
    category frequencies
  - Population Stability Index (PSI) — industry standard metric
    used in credit risk and ML ops to quantify drift magnitude
  - Jensen-Shannon divergence — information-theoretic drift measure
  - Mean/std/median shift analysis — simple but interpretable
  - Missing value pattern comparison
  - New/disappeared category detection (categorical columns)
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency
from scipy.spatial.distance import jensenshannon


# ── PSI CALCULATION ───────────────────────────────────────────────────────────

def compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """
    Population Stability Index (PSI).

    Industry standard for measuring distribution drift.
    Originally from credit risk modeling, now widely used in MLOps.

    PSI < 0.10  → No significant change (stable)
    PSI 0.10-0.25 → Moderate change (monitor)
    PSI > 0.25  → Significant change (model likely degraded)

    Formula: PSI = sum((actual% - expected%) * ln(actual%/expected%))
    """
    # Remove NaN
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Create bins from expected distribution
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints = np.unique(breakpoints)  # remove duplicates

    if len(breakpoints) < 2:
        return 0.0

    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    # Convert to proportions, add small epsilon to avoid log(0)
    eps = 1e-8
    expected_pct = (expected_counts / len(expected)) + eps
    actual_pct = (actual_counts / len(actual)) + eps

    # Normalize to sum to 1
    expected_pct = expected_pct / expected_pct.sum()
    actual_pct = actual_pct / actual_pct.sum()

    psi = float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))
    return round(abs(psi), 4)


def psi_severity(psi: float) -> str:
    if psi < 0.10:
        return "STABLE"
    elif psi < 0.25:
        return "MODERATE_DRIFT"
    else:
        return "SIGNIFICANT_DRIFT"


# ── JENSEN-SHANNON DIVERGENCE ─────────────────────────────────────────────────

def compute_js_divergence(expected: np.ndarray, actual: np.ndarray, buckets: int = 20) -> float:
    """
    Jensen-Shannon divergence between two distributions.
    Symmetric version of KL divergence. Range: [0, 1].
    0 = identical distributions, 1 = completely different.
    """
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Create shared bins
    combined = np.concatenate([expected, actual])
    bins = np.linspace(combined.min(), combined.max(), buckets + 1)

    p = np.histogram(expected, bins=bins)[0].astype(float) + 1e-8
    q = np.histogram(actual, bins=bins)[0].astype(float) + 1e-8

    p = p / p.sum()
    q = q / q.sum()

    return round(float(jensenshannon(p, q)), 4)


# ── NUMERIC COLUMN DRIFT ──────────────────────────────────────────────────────

def analyze_numeric_drift(col: str, train_vals: np.ndarray, test_vals: np.ndarray) -> dict:
    """
    Full drift analysis for one numeric column.
    Uses KS test + PSI + JS divergence + descriptive stats comparison.
    """
    train_clean = train_vals[~np.isnan(train_vals)]
    test_clean = test_vals[~np.isnan(test_vals)]

    if len(train_clean) == 0 or len(test_clean) == 0:
        return {"column": col, "status": "skipped", "reason": "empty after removing NaN"}

    # KS test
    ks_stat, ks_pvalue = ks_2samp(train_clean, test_clean)

    # PSI
    psi = compute_psi(train_clean, test_clean)

    # JS divergence
    js_div = compute_js_divergence(train_clean, test_clean)

    # Descriptive stats shift
    train_mean = float(np.mean(train_clean))
    test_mean = float(np.mean(test_clean))
    train_std = float(np.std(train_clean))
    test_std = float(np.std(test_clean))
    train_median = float(np.median(train_clean))
    test_median = float(np.median(test_clean))

    mean_shift_pct = abs(test_mean - train_mean) / (abs(train_mean) + 1e-9) * 100
    std_shift_pct = abs(test_std - train_std) / (train_std + 1e-9) * 100
    median_shift_pct = abs(test_median - train_median) / (abs(train_median) + 1e-9) * 100

    # Range comparison
    train_min, train_max = float(np.min(train_clean)), float(np.max(train_clean))
    test_min, test_max = float(np.min(test_clean)), float(np.max(test_clean))

    # Out-of-range values in test (values outside training range)
    out_of_range = int(((test_clean < train_min) | (test_clean > train_max)).sum())
    out_of_range_pct = round(out_of_range / len(test_clean) * 100, 2)

    # Determine drift severity
    # Use PSI as primary signal (industry standard)
    sev = psi_severity(psi)

    # Override to HIGH if KS test is very significant AND mean shifted a lot
    if ks_pvalue < 0.001 and mean_shift_pct > 20:
        sev = "SIGNIFICANT_DRIFT"
    elif ks_pvalue < 0.05 and sev == "STABLE":
        sev = "MODERATE_DRIFT"

    drift_detected = sev != "STABLE"

    return {
        "column": col,
        "dtype": "numeric",
        "drift_detected": drift_detected,
        "drift_severity": sev,
        "tests": {
            "ks_statistic": round(float(ks_stat), 4),
            "ks_pvalue": round(float(ks_pvalue), 4),
            "ks_significant": bool(ks_pvalue < 0.05),
            "psi": psi,
            "psi_interpretation": psi_severity(psi),
            "js_divergence": js_div
        },
        "train_stats": {
            "count": len(train_clean),
            "mean": round(train_mean, 4),
            "std": round(train_std, 4),
            "median": round(train_median, 4),
            "min": round(train_min, 4),
            "max": round(train_max, 4),
            "missing_pct": round((len(train_vals) - len(train_clean)) / len(train_vals) * 100, 2)
        },
        "test_stats": {
            "count": len(test_clean),
            "mean": round(test_mean, 4),
            "std": round(test_std, 4),
            "median": round(test_median, 4),
            "min": round(test_min, 4),
            "max": round(test_max, 4),
            "missing_pct": round((len(test_vals) - len(test_clean)) / len(test_vals) * 100, 2)
        },
        "shift_analysis": {
            "mean_shift_pct": round(mean_shift_pct, 2),
            "std_shift_pct": round(std_shift_pct, 2),
            "median_shift_pct": round(median_shift_pct, 2),
            "out_of_range_in_test": out_of_range,
            "out_of_range_pct": out_of_range_pct
        },
        "interpretation": _numeric_interpretation(col, sev, psi, ks_pvalue, mean_shift_pct, out_of_range_pct)
    }


def _numeric_interpretation(col, sev, psi, ks_pvalue, mean_shift_pct, oor_pct) -> str:
    if sev == "STABLE":
        return (
            f"'{col}' distribution is stable between train and test. "
            f"PSI={psi:.3f} (below 0.10 threshold). Model performance on this feature should hold."
        )
    elif sev == "MODERATE_DRIFT":
        return (
            f"'{col}' shows moderate drift (PSI={psi:.3f}). "
            f"Mean shifted by {mean_shift_pct:.1f}%. "
            f"Monitor model predictions on this feature. Consider retraining if performance degrades."
        )
    else:
        parts = [f"'{col}' has significant distribution drift (PSI={psi:.3f}, KS p={ks_pvalue:.4f})."]
        if mean_shift_pct > 20:
            parts.append(f"Mean shifted by {mean_shift_pct:.1f}%.")
        if oor_pct > 5:
            parts.append(f"{oor_pct:.1f}% of test values are outside the training range.")
        parts.append("Model predictions will likely be unreliable for this feature. Retrain with updated data.")
        return " ".join(parts)


# ── CATEGORICAL COLUMN DRIFT ──────────────────────────────────────────────────

def analyze_categorical_drift(col: str, train_vals: pd.Series, test_vals: pd.Series) -> dict:
    """
    Full drift analysis for one categorical column.
    Uses Chi-squared test + PSI on category frequencies + new/missing category detection.
    """
    train_clean = train_vals.dropna()
    test_clean = test_vals.dropna()

    if len(train_clean) == 0 or len(test_clean) == 0:
        return {"column": col, "status": "skipped", "reason": "empty after removing NaN"}

    train_cats = set(train_clean.unique())
    test_cats = set(test_clean.unique())

    new_categories = sorted(list(test_cats - train_cats))
    missing_categories = sorted(list(train_cats - test_cats))

    # Get all categories
    all_cats = sorted(list(train_cats | test_cats))

    # Frequency distributions
    train_freq = train_clean.value_counts(normalize=True)
    test_freq = test_clean.value_counts(normalize=True)

    # Align to same categories
    train_pct = {cat: float(train_freq.get(cat, 0)) for cat in all_cats}
    test_pct = {cat: float(test_freq.get(cat, 0)) for cat in all_cats}

    # Chi-squared test
    train_counts = np.array([train_clean.value_counts().get(cat, 0) for cat in all_cats])
    test_counts = np.array([test_clean.value_counts().get(cat, 0) for cat in all_cats])

    chi2_stat, chi2_pvalue = None, None
    if len(all_cats) > 1 and train_counts.sum() > 0 and test_counts.sum() > 0:
        try:
            contingency = np.array([train_counts, test_counts])
            chi2_stat, chi2_pvalue, _, _ = chi2_contingency(contingency)
            chi2_stat = round(float(chi2_stat), 4)
            chi2_pvalue = round(float(chi2_pvalue), 4)
        except Exception:
            chi2_stat, chi2_pvalue = None, None

    # JS divergence on frequency distributions
    p = np.array([train_pct[c] for c in all_cats]) + 1e-8
    q = np.array([test_pct[c] for c in all_cats]) + 1e-8
    p = p / p.sum()
    q = q / q.sum()
    js_div = round(float(jensenshannon(p, q)), 4)

    # PSI on top categories
    train_arr = np.array([train_pct[c] for c in all_cats])
    test_arr = np.array([test_pct[c] for c in all_cats])
    psi = compute_psi(
        np.repeat(np.arange(len(all_cats)), np.round(train_arr * 1000).astype(int)),
        np.repeat(np.arange(len(all_cats)), np.round(test_arr * 1000).astype(int)),
        buckets=len(all_cats)
    )

    # Severity
    drift_detected = False
    sev = "STABLE"

    if new_categories or missing_categories:
        sev = "MODERATE_DRIFT"
        drift_detected = True
    if chi2_pvalue is not None and chi2_pvalue < 0.05:
        sev = "MODERATE_DRIFT"
        drift_detected = True
    if chi2_pvalue is not None and chi2_pvalue < 0.001:
        sev = "SIGNIFICANT_DRIFT"
        drift_detected = True
    if len(new_categories) > 3:
        sev = "SIGNIFICANT_DRIFT"

    # Top frequency shifts
    freq_shifts = {}
    for cat in all_cats[:10]:
        train_p = train_pct.get(cat, 0)
        test_p = test_pct.get(cat, 0)
        shift = round((test_p - train_p) * 100, 2)
        freq_shifts[str(cat)] = {
            "train_pct": round(train_p * 100, 2),
            "test_pct": round(test_p * 100, 2),
            "shift": shift
        }

    return {
        "column": col,
        "dtype": "categorical",
        "drift_detected": drift_detected,
        "drift_severity": sev,
        "tests": {
            "chi2_statistic": chi2_stat,
            "chi2_pvalue": chi2_pvalue,
            "chi2_significant": bool(chi2_pvalue < 0.05) if chi2_pvalue is not None else None,
            "js_divergence": js_div,
            "psi": psi,
            "psi_interpretation": psi_severity(psi)
        },
        "category_analysis": {
            "train_unique": len(train_cats),
            "test_unique": len(test_cats),
            "new_categories_in_test": new_categories[:20],
            "new_categories_count": len(new_categories),
            "missing_from_test": missing_categories[:20],
            "missing_categories_count": len(missing_categories)
        },
        "train_stats": {
            "count": len(train_clean),
            "missing_pct": round((len(train_vals) - len(train_clean)) / len(train_vals) * 100, 2),
            "top_category": str(train_freq.index[0]) if len(train_freq) > 0 else "-",
            "top_category_pct": round(float(train_freq.iloc[0]) * 100, 2) if len(train_freq) > 0 else 0
        },
        "test_stats": {
            "count": len(test_clean),
            "missing_pct": round((len(test_vals) - len(test_clean)) / len(test_vals) * 100, 2),
            "top_category": str(test_freq.index[0]) if len(test_freq) > 0 else "-",
            "top_category_pct": round(float(test_freq.iloc[0]) * 100, 2) if len(test_freq) > 0 else 0
        },
        "frequency_shifts": freq_shifts,
        "interpretation": _categorical_interpretation(col, sev, new_categories, missing_categories, chi2_pvalue)
    }


def _categorical_interpretation(col, sev, new_cats, missing_cats, chi2_p) -> str:
    parts = []
    if sev == "STABLE":
        return f"'{col}' category distribution is stable. No significant drift detected."
    if new_cats:
        parts.append(f"{len(new_cats)} new categories in test not seen in training: {new_cats[:3]}{'...' if len(new_cats) > 3 else ''}. Model cannot handle these reliably.")
    if missing_cats:
        parts.append(f"{len(missing_cats)} training categories absent in test: {missing_cats[:3]}.")
    if chi2_p is not None and chi2_p < 0.05:
        parts.append(f"Chi-squared test significant (p={chi2_p:.4f}) — category frequencies differ significantly.")
    return " ".join(parts) if parts else f"'{col}' shows distribution drift."


# ── SCHEMA COMPARISON ─────────────────────────────────────────────────────────

def compare_schemas(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """Compare column presence, types, and count between datasets."""
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)

    only_in_train = sorted(list(train_cols - test_cols))
    only_in_test = sorted(list(test_cols - train_cols))
    common_cols = sorted(list(train_cols & test_cols))

    type_mismatches = []
    for col in common_cols:
        t_dtype = str(train_df[col].dtype)
        te_dtype = str(test_df[col].dtype)
        if t_dtype != te_dtype:
            type_mismatches.append({
                "column": col,
                "train_dtype": t_dtype,
                "test_dtype": te_dtype
            })

    return {
        "train_columns": len(train_cols),
        "test_columns": len(test_cols),
        "common_columns": len(common_cols),
        "only_in_train": only_in_train,
        "only_in_test": only_in_test,
        "type_mismatches": type_mismatches,
        "schema_compatible": len(only_in_train) == 0 and len(only_in_test) == 0 and len(type_mismatches) == 0
    }


# ── MISSING VALUE DRIFT ───────────────────────────────────────────────────────

def compare_missing_patterns(train_df: pd.DataFrame, test_df: pd.DataFrame, common_cols: list) -> dict:
    """Compare missing value rates between train and test."""
    shifts = []
    for col in common_cols:
        train_miss = round(float(train_df[col].isnull().mean() * 100), 2)
        test_miss = round(float(test_df[col].isnull().mean() * 100), 2)
        shift = round(test_miss - train_miss, 2)
        if abs(shift) > 5:
            severity = "HIGH" if abs(shift) > 20 else "MODERATE"
            shifts.append({
                "column": col,
                "train_missing_pct": train_miss,
                "test_missing_pct": test_miss,
                "shift": shift,
                "severity": severity
            })

    return {
        "columns_with_missing_drift": len(shifts),
        "details": sorted(shifts, key=lambda x: abs(x["shift"]), reverse=True)
    }


# ── SIZE COMPARISON ───────────────────────────────────────────────────────────

def compare_sizes(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """Basic size comparison with train/test ratio analysis."""
    train_rows, train_cols = train_df.shape
    test_rows, test_cols = test_df.shape
    ratio = round(test_rows / train_rows, 3) if train_rows > 0 else 0

    warnings = []
    if ratio > 2.0:
        warnings.append("Test dataset is more than 2x larger than train — unusual split")
    if ratio < 0.1:
        warnings.append("Test dataset is very small relative to train — evaluation may be unreliable")
    if test_cols != train_cols:
        warnings.append(f"Column count differs: train={train_cols}, test={test_cols}")

    return {
        "train_rows": train_rows,
        "test_rows": test_rows,
        "train_columns": train_cols,
        "test_columns": test_cols,
        "test_to_train_ratio": ratio,
        "warnings": warnings
    }


# ── OVERALL DRIFT SUMMARY ─────────────────────────────────────────────────────

def compute_drift_summary(column_results: list) -> dict:
    """Aggregate column-level drift into dataset-level summary."""
    total = len(column_results)
    if total == 0:
        return {}

    stable = sum(1 for r in column_results if not r.get("drift_detected", False))
    moderate = sum(1 for r in column_results if r.get("drift_severity") == "MODERATE_DRIFT")
    significant = sum(1 for r in column_results if r.get("drift_severity") == "SIGNIFICANT_DRIFT")

    drifted_cols = [r["column"] for r in column_results if r.get("drift_detected")]
    significant_cols = [r["column"] for r in column_results if r.get("drift_severity") == "SIGNIFICANT_DRIFT"]

    drift_pct = round((total - stable) / total * 100, 1)

    if significant > 0:
        overall_risk = "HIGH"
        verdict = (
            f"{significant} column(s) show significant drift. "
            "Model predictions will likely degrade in production. "
            "Retrain with updated data before deployment."
        )
    elif moderate > 0:
        overall_risk = "MODERATE"
        verdict = (
            f"{moderate} column(s) show moderate drift. "
            "Monitor model performance closely. "
            "Consider retraining if accuracy drops below threshold."
        )
    else:
        overall_risk = "LOW"
        verdict = (
            "No significant drift detected. "
            "Train and test distributions are statistically similar. "
            "Model should perform consistently on test data."
        )

    return {
        "total_columns_compared": total,
        "stable_columns": stable,
        "moderate_drift_columns": moderate,
        "significant_drift_columns": significant,
        "drift_percent": drift_pct,
        "drifted_column_names": drifted_cols,
        "significant_drift_column_names": significant_cols,
        "overall_drift_risk": overall_risk,
        "verdict": verdict
    }


# ── MAIN COMPARE FUNCTION ─────────────────────────────────────────────────────

def compare_datasets(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """
    Main comparison function.

    Compares train and test datasets across all dimensions:
    - Schema (columns, types)
    - Size (rows, ratio)
    - Missing value patterns
    - Per-column distribution drift (numeric: KS + PSI + JS)
    - Per-column category drift (categorical: chi2 + new categories)
    - Overall drift risk assessment

    Returns a complete drift report suitable for:
    - Research paper results section
    - Pre-deployment data quality check
    - MLOps monitoring baseline
    """

    # Schema comparison
    schema = compare_schemas(train_df, test_df)
    common_cols = sorted(list(set(train_df.columns) & set(test_df.columns)))

    # Size comparison
    sizes = compare_sizes(train_df, test_df)

    # Missing value drift
    missing_drift = compare_missing_patterns(train_df, test_df, common_cols)

    # Per-column drift analysis
    column_results = []

    for col in common_cols:
        train_col = train_df[col]
        test_col = test_df[col]

        is_numeric = pd.api.types.is_numeric_dtype(train_col)

        if is_numeric:
            result = analyze_numeric_drift(
                col,
                train_col.values.astype(float),
                test_col.values.astype(float)
            )
        else:
            result = analyze_categorical_drift(col, train_col, test_col)

        column_results.append(result)

    # Sort: most drifted first
    severity_order = {"SIGNIFICANT_DRIFT": 0, "MODERATE_DRIFT": 1, "STABLE": 2}
    column_results.sort(key=lambda x: (
        severity_order.get(x.get("drift_severity", "STABLE"), 3),
        -(x.get("tests", {}).get("psi", 0) or 0)
    ))

    # Overall summary
    drift_summary = compute_drift_summary(column_results)

    # Comparison table (clean version for research paper)
    comparison_table = []
    for r in column_results:
        tests = r.get("tests", {})
        row = {
            "column": r["column"],
            "dtype": r.get("dtype", "-"),
            "drift_detected": r.get("drift_detected", False),
            "drift_severity": r.get("drift_severity", "STABLE"),
            "psi": tests.get("psi", "-"),
            "js_divergence": tests.get("js_divergence", "-"),
        }
        if r.get("dtype") == "numeric":
            row["ks_pvalue"] = tests.get("ks_pvalue", "-")
            row["mean_shift_pct"] = r.get("shift_analysis", {}).get("mean_shift_pct", "-")
        else:
            row["chi2_pvalue"] = tests.get("chi2_pvalue", "-")
            row["new_categories"] = r.get("category_analysis", {}).get("new_categories_count", 0)
        comparison_table.append(row)

    return {
        "schema_comparison": schema,
        "size_comparison": sizes,
        "missing_value_drift": missing_drift,
        "drift_summary": drift_summary,
        "column_drift_analysis": column_results,
        "comparison_table": comparison_table,
        "research_note": (
            "Drift detection uses KS test (numeric) and Chi-squared test (categorical) "
            "for statistical significance, PSI for magnitude (industry standard from "
            "credit risk modeling), and Jensen-Shannon divergence as an "
            "information-theoretic measure. PSI < 0.10 = stable, 0.10-0.25 = monitor, "
            "> 0.25 = significant drift."
        )
    }