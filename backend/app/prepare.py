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