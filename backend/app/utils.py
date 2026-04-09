import pandas as pd
import numpy as np
import re


# ── PATTERN LISTS (same as pipeline_builder for consistency) ────────────────
ID_REGEX_PATTERNS = [
    r"^id$", r".*_id$", r"^id_.*", r".*uuid.*", r".*index.*",
    r"^row.*num.*", r".*serial.*"
]

META_REGEX_PATTERNS = [
    r".*timestamp.*", r".*created.*at.*", r".*updated.*at.*"
]

USELESS_TEXT_PATTERNS = [
    r".*name$", r".*address.*", r".*email.*",
    r".*phone.*", r".*description.*", r".*comment.*"
]


def _matches_pattern(col: str, patterns: list) -> bool:
    col_lower = col.lower().strip()
    return any(re.search(p, col_lower) for p in patterns)


def prepare_ml_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Smart ML dataset preparation pipeline.

    Current version (basic) problems:
    - Drops ALL near-unique columns including valid high-cardinality features
    - Uses mean imputation for everything (wrong for skewed columns)
    - get_dummies creates too many columns for high-cardinality categoricals
    - No target column protection (can accidentally encode the target)

    Upgraded version:
    1. Smart ID column detection (name pattern + uniqueness combined)
    2. Drop high-missing columns (>60% missing)
    3. Drop zero-variance columns
    4. Smart imputation: median for skewed, mean for normal, mode for categorical
    5. Cardinality-aware encoding:
       - Binary → LabelEncoding
       - Low cardinality (≤10) → OneHotEncoding
       - High cardinality (>10) → drop (avoid dimensionality explosion)
    6. Target column protection — never encodes or drops the target
    """

    df = df.copy()
    n_rows = len(df)

    # Protect target column (last column by convention)
    target_col = df.columns[-1]

    drop_cols = []
    drop_reasons = {}

    for col in df.columns:

        if col == target_col:
            continue

        series = df[col]
        missing_pct = series.isnull().mean()

        # ── DROP: ID columns (name pattern + near-unique) ────────────────
        if _matches_pattern(col, ID_REGEX_PATTERNS):
            drop_cols.append(col)
            drop_reasons[col] = "ID column pattern detected"
            continue

        # Near-unique string columns = implicit IDs
        if series.dtype == object and series.nunique() / n_rows > 0.9:
            drop_cols.append(col)
            drop_reasons[col] = f"Near-unique string ({series.nunique()}/{n_rows}) — implicit ID"
            continue

        # Exact unique numeric = ID (e.g. customer_number)
        if pd.api.types.is_numeric_dtype(series) and series.nunique() == n_rows:
            drop_cols.append(col)
            drop_reasons[col] = "Unique numeric column — likely row index or ID"
            continue

        # ── DROP: High missing ───────────────────────────────────────────
        if missing_pct > 0.6:
            drop_cols.append(col)
            drop_reasons[col] = f"{missing_pct:.0%} missing — too sparse to impute"
            continue

        # ── DROP: Zero variance ──────────────────────────────────────────
        if series.nunique() <= 1:
            drop_cols.append(col)
            drop_reasons[col] = "Zero variance — constant column"
            continue

        # ── DROP: Useless text (names, emails, addresses) ────────────────
        if _matches_pattern(col, USELESS_TEXT_PATTERNS) and series.dtype == object:
            drop_cols.append(col)
            drop_reasons[col] = "Free-text metadata column — no ML signal"
            continue

        # ── DROP: Datetime metadata ──────────────────────────────────────
        if _matches_pattern(col, META_REGEX_PATTERNS):
            drop_cols.append(col)
            drop_reasons[col] = "Datetime column — needs feature engineering first"
            continue

    df = df.drop(columns=drop_cols, errors="ignore")

    # ── IMPUTATION ───────────────────────────────────────────────────────
    for col in df.columns:

        if col == target_col:
            continue

        series = df[col]
        missing_count = series.isnull().sum()

        if missing_count == 0:
            continue

        if pd.api.types.is_numeric_dtype(series):
            skewness = abs(series.skew())
            if skewness > 1.0:
                # Skewed → median is more robust
                df[col] = series.fillna(series.median())
            else:
                # Near-normal → mean is fine
                df[col] = series.fillna(series.mean())
        else:
            # Categorical → mode
            mode_val = series.mode()
            if len(mode_val) > 0:
                df[col] = series.fillna(mode_val[0])
            else:
                df[col] = series.fillna("Unknown")

    # ── ENCODING ─────────────────────────────────────────────────────────
    categorical_cols = [
        col for col in df.columns
        if df[col].dtype == object and col != target_col
    ]

    cols_to_onehot = []
    cols_to_drop_high_cardinality = []

    for col in categorical_cols:
        n_unique = df[col].nunique()

        if n_unique == 2:
            # Binary → LabelEncoding (0/1)
            unique_vals = df[col].dropna().unique()
            df[col] = df[col].map({unique_vals[0]: 0, unique_vals[1]: 1})

        elif n_unique <= 10:
            # Low cardinality → OneHotEncoding
            cols_to_onehot.append(col)

        else:
            # High cardinality → drop to avoid dimensionality explosion
            # (TargetEncoding requires target info — not safe here)
            cols_to_drop_high_cardinality.append(col)

    # Apply OneHotEncoding
    if cols_to_onehot:
        df = pd.get_dummies(df, columns=cols_to_onehot, drop_first=True)

    # Drop high cardinality
    if cols_to_drop_high_cardinality:
        df = df.drop(columns=cols_to_drop_high_cardinality, errors="ignore")

    # ── ENCODE TARGET IF CATEGORICAL ─────────────────────────────────────
    # Only encode target if it's a string classification target
    if target_col in df.columns and df[target_col].dtype == object:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df[target_col] = le.fit_transform(df[target_col].astype(str))

    # ── FINAL CLEANUP ─────────────────────────────────────────────────────
    # Remove any remaining NaNs (edge cases)
    df = df.fillna(0)

    # Ensure all columns are numeric (safety check)
    df = df.select_dtypes(include=np.number)

    return df