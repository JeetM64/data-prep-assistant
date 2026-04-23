"""prepare.py — Smart ML preprocessing pipeline"""

import pandas as pd
import numpy as np
import re

ID_PATTERNS  = [r"^id$", r".*_id$", r"^id_", r".*uuid.*", r".*index$", r"^row"]
META_PATTERNS = [r".*timestamp.*", r".*created.*at.*", r".*updated.*at.*"]
TEXT_PATTERNS = [r".*name$", r".*address.*", r".*email.*", r".*description.*", r".*comment.*"]

def _matches(col, patterns):
    cl = col.lower()
    return any(re.search(p, cl) for p in patterns)

def prepare_ml_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n = len(df)
    target = df.columns[-1]
    drop = []

    for col in df.columns:
        if col == target: continue
        s = df[col]
        miss = s.isnull().mean()

        if _matches(col, ID_PATTERNS) and s.nunique() / n > 0.9:
            drop.append(col); continue
        if s.dtype == object and s.nunique() / n > 0.9:
            drop.append(col); continue
        if pd.api.types.is_numeric_dtype(s) and s.nunique() == n:
            drop.append(col); continue
        if miss > 0.6:
            drop.append(col); continue
        if s.nunique() <= 1:
            drop.append(col); continue
        if _matches(col, TEXT_PATTERNS) and s.dtype == object:
            drop.append(col); continue
        if _matches(col, META_PATTERNS):
            drop.append(col); continue

    df.drop(columns=drop, errors="ignore", inplace=True)

    # Imputation
    for col in df.columns:
        if col == target: continue
        s = df[col]
        if s.isnull().sum() == 0: continue
        if pd.api.types.is_numeric_dtype(s):
            df[col] = s.fillna(s.median() if abs(s.skew()) > 1 else s.mean())
        else:
            mode = s.mode()
            df[col] = s.fillna(mode[0] if len(mode) > 0 else "Unknown")

    # Encoding
    cat_cols = [c for c in df.columns if df[c].dtype == object and c != target]
    one_hot, high_card = [], []

    for col in cat_cols:
        u = df[col].nunique()
        if u == 2:
            vals = df[col].dropna().unique()
            df[col] = df[col].map({vals[0]: 0, vals[1]: 1})
        elif u <= 10:
            one_hot.append(col)
        else:
            high_card.append(col)

    if one_hot:
        df = pd.get_dummies(df, columns=one_hot, drop_first=True)
    if high_card:
        df.drop(columns=high_card, errors="ignore", inplace=True)

    # Target encoding
    if target in df.columns and df[target].dtype == object:
        from sklearn.preprocessing import LabelEncoder
        df[target] = LabelEncoder().fit_transform(df[target].astype(str))

    df = df.fillna(0)
    df = df.select_dtypes(include=np.number)
    return df

def autofix_dataset(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, list]:
    """
    Auto-cleaning engine that applies a full preprocessing pipeline.
    Returns the cleaned DataFrame and a list of applied fixes (strings).
    """
    df = df.copy()
    fixes = []
    
    # 1. Handle Missing Values
    missing_before = df.isnull().sum().sum()
    if missing_before > 0:
        for col in df.columns:
            if col == target: continue
            if df[col].isnull().sum() == 0: continue
            
            if pd.api.types.is_numeric_dtype(df[col]):
                val = df[col].median()
                df[col] = df[col].fillna(val)
            else:
                mode = df[col].mode()
                val = mode[0] if len(mode) > 0 else "Unknown"
                df[col] = df[col].fillna(val)
        fixes.append(f"Imputed {missing_before} missing values (Median for numerics, Mode for categoricals).")
    
    # 2. Outlier Removal (IQR on numeric columns)
    num_cols = df.select_dtypes(include=["float64", "int64", "float32", "int32"]).columns
    outlier_rows = set()
    for col in num_cols:
        if col == target: continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
        outlier_rows.update(outliers)
    
    if outlier_rows:
        n_outliers = len(outlier_rows)
        # Cap outlier removal to max 15% of dataset to avoid losing too much data
        if n_outliers / len(df) < 0.15:
            df = df.drop(index=list(outlier_rows)).reset_index(drop=True)
            fixes.append(f"Removed {n_outliers} extreme outlier rows using IQR method.")
        else:
            fixes.append(f"Detected {n_outliers} outliers, but kept them to preserve dataset size (>15% of rows).")

    # 3. Encoding
    cat_cols = [c for c in df.columns if df[c].dtype == object and c != target]
    encoded_cols = 0
    dropped_cols = 0
    if cat_cols:
        from sklearn.preprocessing import LabelEncoder
        for col in cat_cols:
            u = df[col].nunique()
            if u == 2:
                vals = df[col].dropna().unique()
                df[col] = df[col].map({vals[0]: 0, vals[1]: 1})
                encoded_cols += 1
            elif u <= 10:
                df = pd.get_dummies(df, columns=[col], drop_first=True)
                encoded_cols += 1
            else:
                df.drop(columns=[col], inplace=True)
                dropped_cols += 1
        
        msg = f"Encoded {encoded_cols} categorical features."
        if dropped_cols > 0:
            msg += f" Dropped {dropped_cols} high-cardinality features."
        fixes.append(msg)
        
    # Target encoding
    if target in df.columns and (df[target].dtype == object or str(df[target].dtype) == 'category'):
        from sklearn.preprocessing import LabelEncoder
        df[target] = LabelEncoder().fit_transform(df[target].astype(str))
        fixes.append("Label encoded target variable.")

    # 4. Scaling
    from sklearn.preprocessing import StandardScaler
    features_to_scale = [c for c in df.select_dtypes(include=np.number).columns if c != target]
    if features_to_scale:
        sc = StandardScaler()
        df[features_to_scale] = sc.fit_transform(df[features_to_scale])
        fixes.append(f"Standardized {len(features_to_scale)} numeric features to mean=0, std=1.")
        
    return df, fixes