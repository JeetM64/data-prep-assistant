import pandas as pd
import numpy as np
from scipy.stats import shapiro, kstest


def _normality_test(series: pd.Series) -> dict:
    """
    Run Shapiro-Wilk (n<5000) or KS test (n>=5000) for normality.
    Returns p-value and verdict.
    """
    clean = series.dropna()
    n = len(clean)
    if n < 3:
        return {"test": "skipped", "p_value": None, "is_normal": None}

    try:
        if n < 5000:
            stat, p = shapiro(clean.sample(min(n, 2000), random_state=42))
            test_name = "Shapiro-Wilk"
        else:
            # Normalize before KS test
            normed = (clean - clean.mean()) / (clean.std() + 1e-9)
            stat, p = kstest(normed, "norm")
            test_name = "Kolmogorov-Smirnov"

        return {
            "test": test_name,
            "p_value": round(float(p), 4),
            "is_normal": bool(p > 0.05)
        }
    except Exception:
        return {"test": "failed", "p_value": None, "is_normal": None}


def _outlier_analysis(series: pd.Series) -> dict:
    """
    Dual-method outlier detection:
    - Z-score (good for normal distributions)
    - IQR (robust, distribution-free)

    Returns count, percentage, severity, and recommended action.
    """
    clean = series.dropna()
    n = len(clean)
    if n == 0:
        return {}

    # Z-score method
    z_scores = np.abs((clean - clean.mean()) / (clean.std() + 1e-9))
    z_outliers = int((z_scores > 3).sum())

    # IQR method
    Q1 = clean.quantile(0.25)
    Q3 = clean.quantile(0.75)
    IQR = Q3 - Q1
    iqr_outliers = int(((clean < Q1 - 1.5 * IQR) | (clean > Q3 + 1.5 * IQR)).sum())

    # Use the average as a consensus estimate
    consensus_outliers = (z_outliers + iqr_outliers) // 2
    outlier_pct = round(consensus_outliers / n * 100, 2)

    if outlier_pct > 10:
        severity = "HIGH"
        action = "Apply RobustScaler or log-transform; consider capping at 1st/99th percentile"
        reason = f"{outlier_pct}% outliers detected (Z-score: {z_outliers}, IQR: {iqr_outliers}); high contamination"
    elif outlier_pct > 2:
        severity = "MODERATE"
        action = "Winsorize (cap) at 5th/95th percentile or use RobustScaler"
        reason = f"{outlier_pct}% outliers detected; moderate contamination may skew model"
    elif consensus_outliers > 0:
        severity = "LOW"
        action = "Monitor; few outliers unlikely to cause issues unless extreme"
        reason = f"{outlier_pct}% outliers — minimal impact expected"
    else:
        severity = "NONE"
        action = "No action needed"
        reason = "No outliers detected by Z-score or IQR methods"

    return {
        "z_score_outliers": z_outliers,
        "iqr_outliers": iqr_outliers,
        "consensus_outlier_count": consensus_outliers,
        "outlier_percent": outlier_pct,
        "severity": severity,
        "recommended_action": action,
        "reason": reason
    }


def _variance_analysis(series: pd.Series, col: str) -> dict:
    """
    Detects low-variance or near-zero-variance columns.
    These contribute very little to model learning.
    """
    clean = series.dropna()
    if len(clean) == 0:
        return {}

    variance = clean.var()
    unique_ratio = clean.nunique() / len(clean)

    if variance < 0.01 or unique_ratio < 0.01:
        return {
            "variance": round(float(variance), 6),
            "unique_ratio": round(unique_ratio, 4),
            "warning": "Near-zero variance detected",
            "reason": f"Column '{col}' has very low variance ({variance:.4f}); likely uninformative for ML",
            "action": "Consider dropping or combining with another feature"
        }
    return {
        "variance": round(float(variance), 4),
        "unique_ratio": round(unique_ratio, 4)
    }


def _missing_value_analysis(series: pd.Series, col: str) -> dict:
    """
    Detailed missing value analysis with mechanism inference.
    """
    missing_count = int(series.isnull().sum())
    missing_pct = round(series.isnull().mean() * 100, 2)

    if missing_pct == 0:
        return {"missing_count": 0, "missing_percent": 0.0, "action": "none"}

    # Try to infer missingness mechanism
    if missing_pct > 60:
        mechanism = "MNAR (Missing Not At Random) likely"
        reason = f"{missing_pct:.1f}% missing — column may be structurally sparse (e.g. optional survey field)"
        action = "DROP_COLUMN — imputation unreliable above 60% threshold"
    elif missing_pct > 20:
        mechanism = "MAR (Missing At Random) possible"
        reason = f"{missing_pct:.1f}% missing — impute with care; consider adding binary missingness indicator"
        action = "IMPUTE + ADD_MISSING_INDICATOR — preserve signal from missingness pattern"
    elif missing_pct > 5:
        mechanism = "MCAR (Missing Completely At Random) likely"
        reason = f"{missing_pct:.1f}% missing — safe to impute"
        action = "IMPUTE"
    else:
        mechanism = "MCAR (rare occurrences)"
        reason = f"Only {missing_pct:.1f}% missing — low impact"
        action = "IMPUTE (low priority)"

    return {
        "missing_count": missing_count,
        "missing_percent": missing_pct,
        "inferred_mechanism": mechanism,
        "reason": reason,
        "recommended_action": action
    }


def _correlation_with_others(col: str, df: pd.DataFrame) -> dict:
    """
    Check if column is highly correlated with any other column.
    High correlation = multicollinearity risk.
    """
    numeric_df = df.select_dtypes(include="number")
    if col not in numeric_df.columns or len(numeric_df.columns) < 2:
        return {}

    corr = numeric_df.corr()[col].drop(col)
    high_corr = corr[corr.abs() > 0.9]

    if len(high_corr) > 0:
        return {
            "multicollinearity_risk": True,
            "highly_correlated_with": high_corr.to_dict(),
            "reason": f"Column '{col}' is >90% correlated with {list(high_corr.index)} — risk of multicollinearity",
            "action": "Consider dropping one or using PCA/dimensionality reduction"
        }
    return {"multicollinearity_risk": False}


def preprocessing_suggestions(df: pd.DataFrame) -> dict:
    """
    Research-grade per-column preprocessing analysis.

    For each column, produces:
    - dtype and cardinality stats
    - Missing value analysis with mechanism inference (MCAR/MAR/MNAR)
    - Normality test (Shapiro-Wilk or KS)
    - Outlier analysis (Z-score + IQR dual method)
    - Variance analysis (near-zero-variance detection)
    - Multicollinearity check
    - Encoding recommendation with reasoning
    - Scaling recommendation with reasoning
    - Transformation recommendation with reasoning

    Every recommendation includes a human-readable 'reason' field
    for explainability and research paper reporting.
    """
    report = {}

    for col in df.columns:
        series = df[col]
        col_report = {}

        # ── BASIC STATS ─────────────────────────────────────────────────
        col_report["dtype"] = str(series.dtype)
        col_report["total_rows"] = len(series)
        col_report["unique_values"] = int(series.nunique())
        col_report["unique_ratio"] = round(series.nunique() / max(len(series), 1), 4)

        # ── MISSING VALUE ANALYSIS ───────────────────────────────────────
        col_report["missing_analysis"] = _missing_value_analysis(series, col)

        # ── NUMERIC COLUMN ANALYSIS ──────────────────────────────────────
        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()

            if len(clean) > 0:
                col_report["descriptive_stats"] = {
                    "mean": round(float(clean.mean()), 4),
                    "median": round(float(clean.median()), 4),
                    "std": round(float(clean.std()), 4),
                    "variance": round(float(clean.var()), 4),
                    "skewness": round(float(clean.skew()), 4),
                    "kurtosis": round(float(clean.kurtosis()), 4),
                    "min": round(float(clean.min()), 4),
                    "max": round(float(clean.max()), 4),
                    "q1": round(float(clean.quantile(0.25)), 4),
                    "q3": round(float(clean.quantile(0.75)), 4)
                }

                # Normality test
                col_report["normality_test"] = _normality_test(clean)

                # Outlier analysis
                col_report["outlier_analysis"] = _outlier_analysis(clean)

                # Variance analysis
                col_report["variance_analysis"] = _variance_analysis(clean, col)

                # Scaling recommendation
                skew = clean.skew()
                iqr = clean.quantile(0.75) - clean.quantile(0.25)
                std = clean.std()
                has_outliers = col_report["outlier_analysis"].get("severity") in ("HIGH", "MODERATE")

                if has_outliers:
                    col_report["scaling_recommendation"] = {
                        "scaler": "RobustScaler",
                        "reason": "Significant outliers detected; RobustScaler uses IQR instead of std — more resistant"
                    }
                elif std > 1.0:
                    if abs(skew) < 1.0:
                        col_report["scaling_recommendation"] = {
                            "scaler": "StandardScaler",
                            "reason": f"Low skew ({skew:.2f}), moderate variance — standardization appropriate"
                        }
                    else:
                        col_report["scaling_recommendation"] = {
                            "scaler": "MinMaxScaler",
                            "reason": f"Skewed distribution (skew={skew:.2f}); MinMaxScaler avoids distorting shape"
                        }
                else:
                    col_report["scaling_recommendation"] = {
                        "scaler": "none",
                        "reason": "Low variance column; scaling unlikely to add value"
                    }

                # Transformation recommendation
                if abs(skew) > 2.0 and clean.min() >= 0:
                    col_report["transformation_recommendation"] = {
                        "transform": "log1p",
                        "reason": f"High skew ({skew:.2f}) with non-negative values; log1p is standard remedy"
                    }
                elif abs(skew) > 1.0 and clean.min() >= 0:
                    col_report["transformation_recommendation"] = {
                        "transform": "sqrt",
                        "reason": f"Moderate skew ({skew:.2f}); sqrt is a lighter transformation option"
                    }
                else:
                    col_report["transformation_recommendation"] = {
                        "transform": "none",
                        "reason": "Skewness within acceptable range (|skew| < 1.0)"
                    }

                # Multicollinearity check
                col_report["multicollinearity_check"] = _correlation_with_others(col, df)

        # ── CATEGORICAL COLUMN ANALYSIS ──────────────────────────────────
        else:
            unique_count = series.nunique()
            n = len(series)
            unique_ratio = unique_count / max(n, 1)

            col_report["top_categories"] = series.value_counts().head(5).to_dict()

            # Encoding recommendation
            if unique_count == 2:
                col_report["encoding_recommendation"] = {
                    "encoding": "LabelEncoding",
                    "reason": "Binary column; LabelEncoding (0/1) is most efficient"
                }
            elif unique_count <= 10:
                col_report["encoding_recommendation"] = {
                    "encoding": "OneHotEncoding",
                    "reason": f"Low cardinality ({unique_count} classes); one-hot avoids imposing ordinal relationship"
                }
            elif unique_count <= 50:
                col_report["encoding_recommendation"] = {
                    "encoding": "TargetEncoding",
                    "reason": f"Medium cardinality ({unique_count}); target encoding captures class-level target signal"
                }
            else:
                col_report["encoding_recommendation"] = {
                    "encoding": "HashEncoding",
                    "reason": f"High cardinality ({unique_count}); hashing avoids curse of dimensionality"
                }

            # Identifier warning
            if unique_ratio > 0.8:
                col_report["identifier_warning"] = {
                    "is_likely_identifier": True,
                    "reason": f"{unique_ratio:.0%} unique values — column likely represents an identifier, not a category",
                    "action": "Consider dropping unless it encodes meaningful text (use NLP then)"
                }

        report[col] = col_report

    return report