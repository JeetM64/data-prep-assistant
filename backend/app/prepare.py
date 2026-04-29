"""
prepare.py — Dual-Mode Dataset Preprocessing Pipeline
======================================================

MODE 1: STRICT CLEAN MODE  →  autofix_dataset(df, target, mode="clean")
  - Human-readable output
  - Keeps original column names, original values where possible
  - Fills missing values (imputation)
  - Removes duplicates and garbage rows
  - Drops pure ID and empty columns
  - NO scaling, NO one-hot explosion, NO feature interactions
  - Categorical columns stay as readable strings (Sex, Embarked, etc.)
  - Output looks like the original dataset but cleaned

MODE 2: ML MODE  →  autofix_dataset(df, target, mode="ml")
  - Fully numeric output for ML training
  - Feature engineering (FamilySize, Title, etc.)
  - Encoding (binary, one-hot for low-cardinality, ordinal for medium)
  - Log transform for skewed features
  - StandardScaler
  - 100% numeric, 0 NaN guaranteed

Both modes return (cleaned_df, log_messages) for full transparency.

Also exported:
  prepare_ml_dataset(df, target)  →  used internally by _prepare_for_ml in main.py
"""

import re
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# PATTERN MATCHING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_ID_NAME_PATTERNS = re.compile(
    r"(^|_)(id)$|^id_|^(row|index|rowid|passengerid|customerid|userid|recordid)$",
    re.IGNORECASE,
)
_META_PATTERNS = re.compile(
    r"(timestamp|created_at|updated_at|modified_at|date_added)",
    re.IGNORECASE,
)


def _is_id_column(col: str, s: pd.Series) -> bool:
    """True if column is a pure identifier with no predictive value."""
    cl = col.lower().replace(" ", "")
    # Name-based ID detection
    if _ID_NAME_PATTERNS.search(cl):
        return True
    # Numeric column where every value is unique (sequential ID)
    if pd.api.types.is_integer_dtype(s) and s.nunique() == len(s) and len(s) > 10:
        return True
    return False


def _is_high_cardinality_text(col: str, s: pd.Series, n: int) -> bool:
    """True if column is a high-cardinality free-text column (Name, Ticket, etc.)."""
    if s.dtype != object:
        return False
    unique_ratio = s.nunique() / max(n, 1)
    avg_len = s.dropna().astype(str).str.len().mean() if s.notna().any() else 0
    # High cardinality + long strings = free text
    return unique_ratio > 0.4 and avg_len > 4


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING  (used in ML mode only)
# ══════════════════════════════════════════════════════════════════════════════

def _engineer_features(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, list[str]]:
    """
    Pattern-based feature engineering. Dataset-agnostic.
    Only adds features when source columns exist.
    """
    log = []
    cls = {c: c.lower() for c in df.columns}

    # ── FamilySize + IsAlone (Titanic pattern) ────────────────────────────
    sib = next((c for c, cl in cls.items() if cl == "sibsp"), None)
    par = next((c for c, cl in cls.items() if cl == "parch"), None)
    if sib and par:
        df["FamilySize"] = df[sib].fillna(0) + df[par].fillna(0) + 1
        df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
        log.append(f"Engineered 'FamilySize' from '{sib}' + '{par}' + 1")
        log.append("Engineered 'IsAlone' from FamilySize")

    # ── Title from Name ───────────────────────────────────────────────────
    name_col = next((c for c, cl in cls.items() if cl in ("name",)), None)
    if name_col and df[name_col].dtype == object:
        extracted = df[name_col].astype(str).str.extract(r" ([A-Za-z]+)\.", expand=False)
        title_map = {
            "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
            "Dr": "Rare", "Rev": "Rare", "Col": "Rare", "Major": "Rare",
            "Mlle": "Miss", "Mme": "Mrs", "Ms": "Miss",
            "Countess": "Rare", "Lady": "Rare", "Sir": "Rare",
            "Capt": "Rare", "Don": "Rare", "Jonkheer": "Rare",
        }
        df["Title"] = extracted.map(title_map).fillna("Rare")
        log.append(f"Engineered 'Title' from '{name_col}' ({df['Title'].nunique()} unique titles)")

    # ── Deck from Cabin ───────────────────────────────────────────────────
    cabin_col = next((c for c, cl in cls.items() if cl == "cabin"), None)
    if cabin_col and df[cabin_col].dtype == object:
        missing_pct = df[cabin_col].isnull().mean()
        if missing_pct < 0.95:
            df["Deck"] = (
                df[cabin_col].astype(str)
                .str.split()
                .str[0]
                .str[0]
                .replace("n", "U")  # handle "nan" string
                .fillna("U")
            )
            df.loc[df[cabin_col].isnull(), "Deck"] = "U"
            log.append(f"Engineered 'Deck' from '{cabin_col}' ({df['Deck'].nunique()} unique decks)")

    # ── AgeBin ───────────────────────────────────────────────────────────
    age_col = next((c for c, cl in cls.items() if cl == "age"), None)
    if age_col and pd.api.types.is_numeric_dtype(df.get(age_col, pd.Series())):
        filled_age = df[age_col].fillna(df[age_col].median())
        df["AgeBin"] = pd.cut(
            filled_age,
            bins=[0, 12, 18, 35, 60, 120],
            labels=[0, 1, 2, 3, 4],
        ).astype(float)
        log.append("Engineered 'AgeBin' from Age (0=child, 1=teen, 2=adult, 3=senior, 4=elderly)")

    return df, log


# ══════════════════════════════════════════════════════════════════════════════
# COLUMN DROPPING  (shared between both modes, different aggressiveness)
# ══════════════════════════════════════════════════════════════════════════════

def _drop_columns(
    df: pd.DataFrame,
    target: str,
    missing_threshold: float = 0.7,
    drop_text: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Drop columns based on statistical criteria only.
    missing_threshold: fraction of missing values above which to drop
    drop_text: if True, drop high-cardinality text columns (ML mode)
               if False, keep them (clean mode)
    """
    log = []
    n = len(df)
    to_drop = []

    for col in df.columns:
        if col == target:
            continue
        s = df[col]

        # 1. ID columns
        if _is_id_column(col, s):
            to_drop.append(col)
            log.append(f"Dropped '{col}' → ID/identifier column (no predictive value)")
            continue

        # 2. Metadata / timestamp
        if _META_PATTERNS.search(col):
            to_drop.append(col)
            log.append(f"Dropped '{col}' → timestamp/metadata column")
            continue

        # 3. Too many missing values
        miss = s.isnull().mean()
        if miss > missing_threshold:
            to_drop.append(col)
            log.append(f"Dropped '{col}' → {miss:.0%} missing values (threshold: {missing_threshold:.0%})")
            continue

        # 4. Zero variance (constant)
        if s.nunique() <= 1:
            to_drop.append(col)
            log.append(f"Dropped '{col}' → constant column (zero variance)")
            continue

        # 5. High-cardinality free text (only in ML mode)
        if drop_text and _is_high_cardinality_text(col, s, n):
            to_drop.append(col)
            log.append(f"Dropped '{col}' → high-cardinality text ({s.nunique()} unique values)")
            continue

    df = df.drop(columns=to_drop, errors="ignore")
    return df, log


# ══════════════════════════════════════════════════════════════════════════════
# IMPUTATION  (shared)
# ══════════════════════════════════════════════════════════════════════════════

def _impute(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, list[str]]:
    """Impute missing values. Numeric→median/mean, Categorical→mode."""
    log = []
    num_filled = 0
    cat_filled = 0

    for col in df.columns:
        if col == target:
            continue
        missing = df[col].isnull().sum()
        if missing == 0:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            skew = abs(df[col].skew()) if df[col].std() > 0 else 0
            fill_val = df[col].median() if skew > 1 else df[col].mean()
            df[col] = df[col].fillna(fill_val)
            num_filled += 1
        else:
            mode = df[col].mode()
            fill_val = mode[0] if len(mode) > 0 else "Unknown"
            df[col] = df[col].fillna(fill_val)
            cat_filled += 1

    total = num_filled + cat_filled
    if total > 0:
        log.append(
            f"Imputed {total} columns: "
            f"{num_filled} numeric (median for skewed, mean otherwise), "
            f"{cat_filled} categorical (mode)"
        )
    return df, log


# ══════════════════════════════════════════════════════════════════════════════
# ENCODING  (ML mode only)
# ══════════════════════════════════════════════════════════════════════════════

def _encode_ml(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, list[str]]:
    """
    Encode categorical columns for ML.
    Binary → 0/1
    Low cardinality (≤10) → OneHot (NO duplicate columns — uses pd.get_dummies once)
    Medium (≤50) → Ordinal by frequency rank
    High (>50) → Frequency encoding
    """
    log = []

    cat_cols = [
        c for c in df.columns
        if c != target and (
            df[c].dtype == object
            or str(df[c].dtype) in ("string", "category")
            or df[c].dtype.name == "bool"
        )
    ]

    ohe_cols = []  # collect for single get_dummies call

    for col in cat_cols:
        s = df[col].astype(str).str.strip()
        df[col] = s
        u = s.nunique()

        if u == 2:
            vals = sorted(s.dropna().unique())
            df[col] = s.map({vals[0]: 0, vals[1]: 1}).fillna(0).astype(int)
            log.append(f"Encoded '{col}' → binary ('{vals[0]}'=0, '{vals[1]}'=1)")

        elif u <= 10:
            ohe_cols.append(col)

        elif u <= 50:
            freq_rank = df[col].value_counts().rank(ascending=False, method="dense")
            df[col] = df[col].map(freq_rank).fillna(0).astype(int)
            log.append(f"Encoded '{col}' → ordinal rank ({u} categories)")

        else:
            freq = df[col].value_counts(normalize=True)
            df[col] = df[col].map(freq).fillna(0).astype(float)
            log.append(f"Encoded '{col}' → frequency ({u} categories)")

    # OneHot all low-cardinality columns in one call to avoid column explosion
    if ohe_cols:
        # Build dummies
        dummies = pd.get_dummies(df[ohe_cols], prefix=ohe_cols, drop_first=True, dtype=int)
        # Remove originals and add dummies
        df = df.drop(columns=ohe_cols)
        df = pd.concat([df, dummies], axis=1)
        for col in ohe_cols:
            n_new = sum(1 for c in df.columns if c.startswith(col + "_"))
            log.append(f"Encoded '{col}' → OneHot ({n_new} binary columns)")

    return df, log


# ══════════════════════════════════════════════════════════════════════════════
# TARGET ENCODING (both modes, only if target is string)
# ══════════════════════════════════════════════════════════════════════════════

def _encode_target(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, list[str]]:
    log = []
    if target not in df.columns:
        return df, log
    if df[target].dtype == object or str(df[target].dtype) in ("string", "category"):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df[target] = le.fit_transform(df[target].astype(str))
        log.append(f"Label-encoded target '{target}'")
    return df, log


# ══════════════════════════════════════════════════════════════════════════════
# NUMERIC SAFETY NET (ML mode only)
# ══════════════════════════════════════════════════════════════════════════════

def _ensure_numeric(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, list[str]]:
    """Final guarantee: output is 100% numeric with zero NaN."""
    log = []

    # Encode target
    df, tlog = _encode_target(df, target)
    log.extend(tlog)

    # Handle booleans
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)

    # Any remaining object/string → frequency encode (safe fallback)
    leftover = [
        c for c in df.columns
        if df[c].dtype == object or str(df[c].dtype) in ("string", "category")
    ]
    for col in leftover:
        u = df[col].nunique()
        if u <= 15:
            dummies = pd.get_dummies(df[[col]], prefix=col, drop_first=True, dtype=int)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
        else:
            freq = df[col].value_counts(normalize=True)
            df[col] = df[col].map(freq).fillna(0)
        log.append(f"Safety encode: '{col}' (was still string)")

    # Fill NaN
    nan_count = int(df.isnull().sum().sum())
    if nan_count > 0:
        df = df.fillna(0)
        log.append(f"Filled {nan_count} residual NaN values with 0")

    # Final nuclear drop
    non_num = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_num:
        df = df.drop(columns=non_num, errors="ignore")
        log.append(f"Final drop of {len(non_num)} non-numeric columns: {non_num}")

    df = df.select_dtypes(include=[np.number])
    return df, log


# ══════════════════════════════════════════════════════════════════════════════
# DUPLICATE REMOVAL (both modes)
# ══════════════════════════════════════════════════════════════════════════════

def _remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    log = []
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_removed = n_before - len(df)
    if n_removed > 0:
        log.append(f"Removed {n_removed} duplicate rows")
    return df, log


# ══════════════════════════════════════════════════════════════════════════════
# ★ MODE 1: STRICT CLEAN MODE — Human-Readable Output
# ══════════════════════════════════════════════════════════════════════════════

def clean_dataset(df: pd.DataFrame, target: str = None) -> tuple[pd.DataFrame, list[str]]:
    """
    STRICT CLEAN MODE — produces a human-readable cleaned dataset.

    What it does:
      ✓ Removes duplicate rows
      ✓ Drops ID columns (PassengerId, RowId, etc.)
      ✓ Drops columns with >70% missing values
      ✓ Drops zero-variance (constant) columns
      ✓ Drops timestamp/metadata columns
      ✓ Imputes missing numeric values (median/mean)
      ✓ Imputes missing categorical values (mode)

    What it does NOT do:
      ✗ Does NOT scale or standardize numbers
      ✗ Does NOT one-hot encode categories
      ✗ Does NOT drop Name, Ticket, or other readable text columns
      ✗ Does NOT add engineered features
      ✗ Does NOT change value ranges

    Output: original-looking dataset with clean, readable values.
    """
    df = df.copy()
    if target is None:
        target = df.columns[-1]
    log = []

    # Step 1: Remove duplicates
    df, step_log = _remove_duplicates(df)
    log.extend(step_log)

    # Step 2: Drop ID, empty, constant columns — keep text columns (drop_text=False)
    df, step_log = _drop_columns(df, target, missing_threshold=0.7, drop_text=False)
    log.extend(step_log)

    # Step 3: Impute missing values (keep original values, just fill gaps)
    df, step_log = _impute(df, target)
    log.extend(step_log)

    total_cols = len(df.columns)
    total_rows = len(df)
    log.append(f"Clean output: {total_rows} rows × {total_cols} columns — human-readable, original values preserved")

    return df, log


# ══════════════════════════════════════════════════════════════════════════════
# ★ MODE 2: ML MODE — Fully Numeric Output for ML Training
# ══════════════════════════════════════════════════════════════════════════════

def autofix_dataset(df: pd.DataFrame, target: str = None, mode: str = "ml") -> tuple[pd.DataFrame, list[str]]:
    """
    DUAL-MODE preprocessing pipeline.

    Parameters
    ----------
    df     : input DataFrame (raw, as loaded from CSV/Excel)
    target : target column name (auto-detected if None)
    mode   : "clean" → human-readable output (no scaling/encoding)
             "ml"    → fully numeric ML-ready output (default)

    Returns
    -------
    (cleaned_df, log_messages)
    """
    if target is None:
        target = df.columns[-1]

    if mode == "clean":
        return clean_dataset(df, target)

    # ── ML MODE ──────────────────────────────────────────────────────────────
    df = df.copy()
    log = []

    # Step 1: Remove duplicates
    df, step_log = _remove_duplicates(df)
    log.extend(step_log)

    # Step 2: Feature engineering (BEFORE dropping, so Name exists for Title extraction)
    df, step_log = _engineer_features(df, target)
    log.extend(step_log)

    # Step 3: Drop useless columns (ID, high-missing, high-cardinality text)
    df, step_log = _drop_columns(df, target, missing_threshold=0.6, drop_text=True)
    log.extend(step_log)

    # Step 4: Impute missing values
    df, step_log = _impute(df, target)
    log.extend(step_log)

    # Step 5: Log-transform skewed features
    skew_fixed = []
    for col in df.select_dtypes(include=[np.number]).columns:
        if col == target:
            continue
        try:
            skew = df[col].skew()
            if abs(skew) > 2 and df[col].min() >= 0:
                df[col] = np.log1p(df[col])
                skew_fixed.append(col)
        except Exception:
            pass
    if skew_fixed:
        log.append(f"Log1p-transformed {len(skew_fixed)} highly skewed features: {skew_fixed}")

    # Step 6: Encode categoricals
    df, step_log = _encode_ml(df, target)
    log.extend(step_log)

    # Step 7: Encode target if categorical
    df, step_log = _encode_target(df, target)
    log.extend(step_log)

    # Step 8: Numeric safety net (catches any edge cases)
    df, step_log = _ensure_numeric(df, target)
    log.extend(step_log)

    # Step 9: Standardize numeric features (exclude target)
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target]
    if num_cols:
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        df[num_cols] = sc.fit_transform(df[num_cols])
        log.append(f"Standardized {len(num_cols)} numeric features → mean=0, std=1")

    # Final validation
    assert df.select_dtypes(exclude=[np.number]).empty, \
        f"Non-numeric columns remain: {df.select_dtypes(exclude=[np.number]).columns.tolist()}"
    assert df.isnull().sum().sum() == 0, "NaN values remain in output"

    log.append(
        f"ML output: {len(df)} rows × {len(df.columns)} columns — "
        f"100% numeric, 0 NaN, ready for sklearn/XGBoost/etc."
    )

    return df, log


# ══════════════════════════════════════════════════════════════════════════════
# QUICK ALIAS for internal _prepare_for_ml calls in main.py
# ══════════════════════════════════════════════════════════════════════════════

def prepare_ml_dataset(df: pd.DataFrame, target: str = None) -> pd.DataFrame:
    """
    Quick ML preprocessing. Returns DataFrame only (no log).
    Used internally by _prepare_for_ml in main.py.
    """
    if target is None:
        target = df.columns[-1]
    cleaned, _ = autofix_dataset(df, target, mode="ml")
    return cleaned