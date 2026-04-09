import pandas as pd
import numpy as np
import re


# Patterns that indicate ID/useless columns by name
ID_REGEX_PATTERNS = [
    r"^id$", r".*_id$", r"^id_.*", r".*uuid.*", r".*index.*",
    r"^row.*num.*", r".*serial.*", r".*ticket.*", r".*ref.*num.*"
]

# Patterns for datetime/metadata columns that should be dropped or transformed
META_REGEX_PATTERNS = [
    r".*timestamp.*", r".*created.*at.*", r".*updated.*at.*",
    r".*date.*", r".*time.*"
]

# Known semantically useless text patterns
USELESS_TEXT_PATTERNS = [
    r".*name$", r".*address.*", r".*email.*",
    r".*phone.*", r".*description.*", r".*comment.*", r".*note.*"
]

# Columns with these names often benefit from log transform (right-skewed)
LOG_TRANSFORM_KEYWORDS = [
    "price", "income", "salary", "revenue", "cost", "amount",
    "population", "area", "size", "weight", "distance", "count"
]


def _matches_any_pattern(col: str, patterns: list) -> bool:
    col_lower = col.lower().strip()
    return any(re.search(p, col_lower) for p in patterns)


def _iqr_has_outliers(series: pd.Series) -> bool:
    """Returns True if IQR-based outlier count exceeds 5% of data."""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outlier_count = ((series < lower) | (series > upper)).sum()
    return outlier_count / len(series) > 0.05


def _recommend_scaler(series: pd.Series, col: str) -> tuple:
    """
    Returns (scaler_type, reason) based on distribution analysis.
    - StandardScaler: approximately normal distribution
    - MinMaxScaler: bounded, no significant outliers
    - RobustScaler: heavy outliers detected
    """
    skew = abs(series.skew())
    has_outliers = _iqr_has_outliers(series)

    if has_outliers:
        return "RobustScaler", "IQR outliers detected; robust to extreme values"
    elif skew > 1.0:
        return "MinMaxScaler", f"Skewed distribution (skew={skew:.2f}); MinMax preserves shape"
    else:
        return "StandardScaler", f"Near-normal distribution (skew={skew:.2f}); standardization appropriate"


def _recommend_missing_strategy(series: pd.Series, col: str) -> tuple:
    """
    Returns (strategy, reason) for handling missing values.
    """
    skew = abs(series.skew()) if pd.api.types.is_numeric_dtype(series) else None
    missing_pct = series.isnull().mean() * 100

    if missing_pct > 60:
        return "DROP_COLUMN", f"{missing_pct:.1f}% missing — too sparse to impute reliably"

    if pd.api.types.is_numeric_dtype(series):
        if skew and skew > 1.0:
            return "MEDIAN_IMPUTATION", f"Skewed distribution (skew={skew:.2f}); median is robust to outliers"
        else:
            return "MEAN_IMPUTATION", f"Near-normal distribution; mean imputation appropriate"
    else:
        unique_ratio = series.nunique() / max(len(series), 1)
        if unique_ratio < 0.05:
            return "MODE_IMPUTATION", "Low-cardinality categorical; mode is safe default"
        else:
            return "CONSTANT_FILL", "High-cardinality categorical; fill with placeholder 'Unknown'"


def _recommend_encoding(series: pd.Series, col: str, target_col: str) -> tuple:
    """
    Returns (encoding_type, reason) for categorical columns.
    """
    unique_count = series.nunique()
    unique_ratio = unique_count / max(len(series), 1)

    if unique_count == 2:
        return "LabelEncoding", "Binary column; label encoding (0/1) is sufficient"
    elif unique_count <= 10:
        return "OneHotEncoding", f"Low cardinality ({unique_count} classes); one-hot avoids ordinal assumption"
    elif unique_count <= 50:
        return "TargetEncoding", f"Medium cardinality ({unique_count} classes); target encoding reduces dimensionality"
    else:
        return "HashEncoding", f"High cardinality ({unique_count} classes); hashing prevents dimensionality explosion"


def build_preprocessing_pipeline(df: pd.DataFrame, target_column: str) -> dict:
    """
    Advanced preprocessing pipeline builder.

    For each feature column, determines:
    - Whether to drop (ID, high-missing, useless text, datetime metadata)
    - Missing value strategy (mean/median/mode/constant/drop)
    - Encoding strategy (label/onehot/target/hash)
    - Scaling strategy (standard/minmax/robust)
    - Transformation recommendations (log, sqrt, box-cox)

    Each decision includes a 'reason' field for explainability
    (critical for research paper contributions).

    Returns a structured pipeline dict ready for code generation.
    """
    pipeline = {
        "target_column": target_column,
        "drop_columns": [],
        "missing_value_strategy": {},
        "encoding_strategy": {},
        "scaling_strategy": {},
        "transformation_recommendations": {},
        "feature_flags": {},   # metadata flags per column
        "summary": {
            "total_features": 0,
            "dropped": 0,
            "numeric_features": 0,
            "categorical_features": 0,
            "columns_needing_imputation": 0,
            "columns_needing_encoding": 0,
            "columns_needing_scaling": 0
        }
    }

    n_rows = len(df)

    for col in df.columns:

        if col == target_column:
            continue

        pipeline["summary"]["total_features"] += 1
        series = df[col].dropna()
        missing_pct = df[col].isnull().mean() * 100
        unique_count = df[col].nunique()
        flags = []

        # ── STEP 1: SHOULD THIS COLUMN BE DROPPED? ──────────────────────

        # ID columns
        if _matches_any_pattern(col, ID_REGEX_PATTERNS):
            pipeline["drop_columns"].append({
                "column": col,
                "reason": "Matches ID/index naming pattern — no predictive value"
            })
            pipeline["summary"]["dropped"] += 1
            flags.append("id_column")
            pipeline["feature_flags"][col] = flags
            continue

        # Near-unique (implicit ID even without the name)
        if unique_count / n_rows > 0.95 and df[col].dtype == object:
            pipeline["drop_columns"].append({
                "column": col,
                "reason": f"Near-unique string ({unique_count}/{n_rows} unique values) — likely identifier"
            })
            pipeline["summary"]["dropped"] += 1
            flags.append("implicit_id")
            pipeline["feature_flags"][col] = flags
            continue

        # Useless text (names, emails, addresses, descriptions)
        if _matches_any_pattern(col, USELESS_TEXT_PATTERNS) and df[col].dtype == object:
            pipeline["drop_columns"].append({
                "column": col,
                "reason": "Free-text/metadata column with no ML signal (name, email, address, etc.)"
            })
            pipeline["summary"]["dropped"] += 1
            flags.append("useless_text")
            pipeline["feature_flags"][col] = flags
            continue

        # Datetime metadata (usually should be feature-engineered, not raw)
        if _matches_any_pattern(col, META_REGEX_PATTERNS):
            pipeline["drop_columns"].append({
                "column": col,
                "reason": "Datetime/timestamp column — consider feature engineering (day-of-week, month) instead"
            })
            pipeline["summary"]["dropped"] += 1
            flags.append("datetime_needs_engineering")
            pipeline["feature_flags"][col] = flags
            continue

        # High missing — handled per column
        if missing_pct > 60:
            pipeline["drop_columns"].append({
                "column": col,
                "reason": f"{missing_pct:.1f}% missing — imputation would introduce severe bias"
            })
            pipeline["summary"]["dropped"] += 1
            flags.append("high_missing_dropped")
            pipeline["feature_flags"][col] = flags
            continue

        # Zero-variance (constant column)
        if df[col].nunique() <= 1:
            pipeline["drop_columns"].append({
                "column": col,
                "reason": "Constant column (zero variance) — no information"
            })
            pipeline["summary"]["dropped"] += 1
            flags.append("zero_variance")
            pipeline["feature_flags"][col] = flags
            continue

        # ── STEP 2: MISSING VALUE STRATEGY ──────────────────────────────
        if missing_pct > 0:
            strategy, reason = _recommend_missing_strategy(df[col], col)
            pipeline["missing_value_strategy"][col] = {
                "strategy": strategy,
                "missing_percent": round(missing_pct, 2),
                "reason": reason
            }
            pipeline["summary"]["columns_needing_imputation"] += 1

        # ── STEP 3: ENCODING (CATEGORICAL) ──────────────────────────────
        if not pd.api.types.is_numeric_dtype(df[col]):
            encoding, reason = _recommend_encoding(df[col], col, target_column)
            pipeline["encoding_strategy"][col] = {
                "encoding": encoding,
                "unique_values": unique_count,
                "reason": reason
            }
            pipeline["summary"]["columns_needing_encoding"] += 1
            pipeline["summary"]["categorical_features"] += 1
            flags.append("categorical")

        # ── STEP 4: SCALING (NUMERIC) ────────────────────────────────────
        else:
            pipeline["summary"]["numeric_features"] += 1
            flags.append("numeric")

            if len(series) > 0:
                std = series.std()
                col_range = series.max() - series.min()

                # Only recommend scaling if the column has meaningful variance
                # and isn't binary (0/1)
                is_binary_numeric = set(df[col].dropna().unique()).issubset({0, 1})

                if not is_binary_numeric and (std > 1.0 or col_range > 10):
                    scaler, reason = _recommend_scaler(series, col)
                    pipeline["scaling_strategy"][col] = {
                        "scaler": scaler,
                        "std": round(std, 3),
                        "range": round(col_range, 3),
                        "reason": reason
                    }
                    pipeline["summary"]["columns_needing_scaling"] += 1
                elif is_binary_numeric:
                    flags.append("binary_numeric_no_scaling")

            # ── STEP 5: TRANSFORMATION RECOMMENDATIONS ──────────────────
            if len(series) > 0:
                skew = series.skew()
                col_lower = col.lower()

                if skew > 2.0 and series.min() >= 0:
                    # Log transform for heavy right skew + non-negative
                    transform = "log1p"
                    reason = f"High positive skew ({skew:.2f}); log1p stabilizes variance"

                    # Boost confidence if column name matches known patterns
                    if any(kw in col_lower for kw in LOG_TRANSFORM_KEYWORDS):
                        reason += f" (column '{col}' is a domain-known log-transform candidate)"

                    pipeline["transformation_recommendations"][col] = {
                        "transform": transform,
                        "skewness": round(skew, 3),
                        "reason": reason
                    }

                elif skew < -2.0:
                    pipeline["transformation_recommendations"][col] = {
                        "transform": "reflect_log1p",
                        "skewness": round(skew, 3),
                        "reason": f"High negative skew ({skew:.2f}); reflect then log1p"
                    }

                elif 1.0 < abs(skew) <= 2.0 and series.min() >= 0:
                    pipeline["transformation_recommendations"][col] = {
                        "transform": "sqrt",
                        "skewness": round(skew, 3),
                        "reason": f"Moderate skew ({skew:.2f}); sqrt transform may help"
                    }

        pipeline["feature_flags"][col] = flags

    return pipeline