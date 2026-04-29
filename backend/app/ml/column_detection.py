import pandas as pd
import numpy as np
import re
from typing import Optional

def detect_column_types(df: pd.DataFrame) -> dict:
    """
    Returns a dictionary mapping column names to their detected semantic type:
    - numeric_int
    - numeric_float
    - categorical_low
    - categorical_medium
    - categorical_high
    - binary
    - text
    - datetime
    - id
    """
    types = {}
    n_rows = len(df)
    if n_rows == 0:
        return {}
        
    for col in df.columns:
        s = df[col]
        col_lower = str(col).lower().strip()
        n_unique = s.nunique(dropna=True)
        unique_ratio = n_unique / max(n_rows, 1)
        
        # 1. Date/Time Detection
        if pd.api.types.is_datetime64_any_dtype(s):
            types[col] = "datetime"
            continue
            
        # 2. ID Detection
        # Pattern matching id, *_id, *Id (case insensitive)
        is_id_pattern = re.search(r"(^|_)?id$", str(col), re.IGNORECASE) is not None
        
        is_monotonic = False
        if pd.api.types.is_numeric_dtype(s):
            clean_s = s.dropna()
            if len(clean_s) > 1:
                is_monotonic = clean_s.is_monotonic_increasing or clean_s.is_monotonic_decreasing

        if unique_ratio > 0.9 and (is_id_pattern or is_monotonic):
            types[col] = "id"
            continue
            
        # 3. Text Column Detection
        if s.dtype == object:
            clean_s = s.dropna().astype(str)
            if len(clean_s) > 0:
                avg_len = clean_s.str.len().mean()
                contains_spaces = clean_s.str.contains(' ').mean() > 0.5
                if unique_ratio > 0.5 and avg_len > 10 and contains_spaces:
                    types[col] = "text"
                    continue
        
        # 4. Binary Detection
        if n_unique == 2:
            types[col] = "binary"
            continue
            
        # 5. Categorical Detection
        if not pd.api.types.is_numeric_dtype(s) or (pd.api.types.is_integer_dtype(s) and unique_ratio < 0.05):
            if unique_ratio < 0.05 or n_unique < 15:
                types[col] = "categorical_low"
            elif unique_ratio <= 0.3:
                types[col] = "categorical_medium"
            else:
                types[col] = "categorical_high"
            continue
            
        # 6. Numeric Detection
        if pd.api.types.is_numeric_dtype(s):
            if pd.api.types.is_integer_dtype(s) or (s.dropna() % 1 == 0).all():
                types[col] = "numeric_int"
            else:
                types[col] = "numeric_float"
            continue
            
        types[col] = "unknown"

    return types

def _quick_feature_importance(df: pd.DataFrame, target: str, col: str) -> float:
    """Returns a signal strength score (0 to 1) for the column against the target."""
    from sklearn.preprocessing import LabelEncoder
    import warnings
    
    clean_df = df[[col, target]].dropna()
    if len(clean_df) < 10:
        return 0.0
        
    s = clean_df[col]
    t = clean_df[target]
    
    if pd.api.types.is_numeric_dtype(s) and pd.api.types.is_numeric_dtype(t):
        corr = abs(s.corr(t))
        return float(corr) if not np.isnan(corr) else 0.0
        
    sample_size = min(1000, len(clean_df))
    sampled = clean_df.sample(sample_size, random_state=42)
    s_samp = sampled[col]
    t_samp = sampled[target]
    
    if s_samp.dtype == object or str(s_samp.dtype) == 'category':
        s_samp = LabelEncoder().fit_transform(s_samp.astype(str))
    else:
        s_samp = s_samp.values.reshape(-1, 1)
        
    is_target_cat = t_samp.dtype == object or str(t_samp.dtype) == 'category' or t_samp.nunique() < 20
    
    if is_target_cat:
        t_samp = LabelEncoder().fit_transform(t_samp.astype(str))
        from sklearn.feature_selection import mutual_info_classif
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mi = mutual_info_classif(s_samp if len(s_samp.shape)==2 else s_samp.reshape(-1,1), t_samp)
        return min(mi[0] / 0.5, 1.0)
    else:
        from sklearn.feature_selection import mutual_info_regression
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mi = mutual_info_regression(s_samp if len(s_samp.shape)==2 else s_samp.reshape(-1,1), t_samp)
        return min(mi[0] / 0.5, 1.0)

def auto_feature_selection(df: pd.DataFrame, target: Optional[str]) -> list:
    """
    Decides what to do with each column.
    Returns a list of dicts.
    """
    if df.empty: return []
    
    col_types = detect_column_types(df)
    n_rows = len(df)
    decisions = []
    
    for col in df.columns:
        if col == target:
            continue
            
        s = df[col]
        missing_pct = s.isnull().mean()
        unique_ratio = s.nunique(dropna=True) / max(n_rows, 1)
        c_type = col_types.get(col, "unknown")
        n_unique = s.nunique(dropna=True)
        
        stats = {
            "unique_ratio": round(unique_ratio, 4),
            "missing_percent": round(missing_pct * 100, 2),
            "type": c_type
        }
        
        def safe_drop(reason_msg: str, force: bool = False):
            if force:
                return {"column": col, "action": "drop", "reason": reason_msg, "stats": stats}
            if target and target in df.columns and missing_pct < 0.99:
                importance = _quick_feature_importance(df, target, col)
                if importance > 0.1:
                    return {"column": col, "action": "keep", "reason": f"Kept despite {reason_msg.lower()} (high predictive signal: {importance:.2f})", "stats": stats}
            return {"column": col, "action": "drop", "reason": reason_msg, "stats": stats}

        # 1. Constant Column
        if n_unique <= 1:
            decisions.append(safe_drop("Constant column (zero variance)", force=True))
            continue
            
        # 2. High Missing
        if missing_pct > 0.70:
            decisions.append(safe_drop(f"Very high missing values ({missing_pct*100:.1f}%)"))
            continue
        
        # 3. ID Column
        if c_type == "id":
            importance = _quick_feature_importance(df, target, col) if target else 0
            if importance < 0.05:
                decisions.append(safe_drop("ID column (>90% unique or matches pattern)", force=True))
            else:
                decisions.append({"column": col, "action": "keep", "reason": f"Matches ID pattern but has predictive signal ({importance:.2f})", "stats": stats})
            continue
            
        # 4. Text Column
        if c_type == "text":
            decisions.append(safe_drop("Useless free-form text column"))
            continue
            
        # 5. High Cardinality Handling
        if c_type == "categorical_high":
            importance = _quick_feature_importance(df, target, col) if target else 0
            if target and importance < 0.02:
                decisions.append(safe_drop("High cardinality categorical with no predictive signal"))
            else:
                decisions.append({
                    "column": col,
                    "action": "encode",
                    "reason": "High cardinality categorical; recommend Frequency or Target Encoding",
                    "stats": stats
                })
            continue

        # 6. Default Keep / Impute / Flag Weak
        final_action = "keep"
        final_reason = f"Informative {c_type.replace('_', ' ')} feature"
        
        if missing_pct >= 0.40:
            final_reason = f"Weak feature ({missing_pct*100:.1f}% missing); flagged"
            
        if missing_pct > 0:
            final_action = "impute"
            if missing_pct < 0.40:
                final_reason = f"Needs imputation ({missing_pct*100:.1f}% missing)"
            
        decisions.append({
            "column": col,
            "action": final_action,
            "reason": final_reason,
            "stats": stats
        })
        
    return decisions
