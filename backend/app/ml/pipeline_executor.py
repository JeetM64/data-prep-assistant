import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler


def execute_pipeline(df: pd.DataFrame, pipeline: dict) -> dict:
    """
    Actually APPLIES the preprocessing pipeline to the dataset.

    Takes the raw DataFrame and the pipeline dict from pipeline_builder.py
    and returns a fully cleaned, ML-ready DataFrame.

    Steps applied in order:
    1. Drop columns (ID columns, high-missing, useless text)
    2. Impute missing values (mean/median/mode/constant)
    3. Encode categorical columns (label/onehot)
    4. Scale numeric columns (standard/minmax/robust)
    5. Apply transformations (log1p/sqrt)

    Returns:
        cleaned_df: the processed DataFrame
        execution_log: what was done to each column
        stats: before/after comparison
        success: True/False
    """

    df = df.copy()
    execution_log = []
    target_col = pipeline.get("target_column")

    original_shape = df.shape
    original_missing = int(df.isnull().sum().sum())

    # ── STEP 1: DROP COLUMNS ──────────────────────────────────────────────
    drop_entries = pipeline.get("drop_columns", [])
    dropped = []

    for entry in drop_entries:
        col = entry.get("column")
        reason = entry.get("reason", "")

        if col == target_col:
            execution_log.append({
                "column": col,
                "action": "SKIPPED DROP",
                "reason": "Target column — never dropped"
            })
            continue

        if col in df.columns:
            df = df.drop(columns=[col])
            dropped.append(col)
            execution_log.append({
                "column": col,
                "action": "DROPPED",
                "reason": reason
            })

    # ── STEP 2: IMPUTE MISSING VALUES ─────────────────────────────────────
    impute_strategy = pipeline.get("missing_value_strategy", {})

    for col, info in impute_strategy.items():
        if col not in df.columns:
            continue

        strategy = info.get("strategy", "MEAN_IMPUTATION")
        missing_before = int(df[col].isnull().sum())

        if missing_before == 0:
            continue

        if strategy == "DROP_COLUMN":
            if col != target_col:
                df = df.drop(columns=[col])
                execution_log.append({
                    "column": col,
                    "action": "DROPPED (high missing)",
                    "reason": info.get("reason", ""),
                    "missing_filled": missing_before
                })
            continue

        elif strategy == "MEDIAN_IMPUTATION":
            fill_val = df[col].median()
            df[col] = df[col].fillna(fill_val)
            execution_log.append({
                "column": col,
                "action": "IMPUTED (median)",
                "fill_value": round(float(fill_val), 4),
                "cells_filled": missing_before
            })

        elif strategy == "MEAN_IMPUTATION":
            fill_val = df[col].mean()
            df[col] = df[col].fillna(fill_val)
            execution_log.append({
                "column": col,
                "action": "IMPUTED (mean)",
                "fill_value": round(float(fill_val), 4),
                "cells_filled": missing_before
            })

        elif strategy == "MODE_IMPUTATION":
            mode_vals = df[col].mode()
            if len(mode_vals) > 0:
                df[col] = df[col].fillna(mode_vals[0])
                execution_log.append({
                    "column": col,
                    "action": "IMPUTED (mode)",
                    "fill_value": str(mode_vals[0]),
                    "cells_filled": missing_before
                })

        elif strategy == "CONSTANT_FILL":
            df[col] = df[col].fillna("Unknown")
            execution_log.append({
                "column": col,
                "action": "IMPUTED (constant='Unknown')",
                "cells_filled": missing_before
            })

    # ── STEP 3: APPLY TRANSFORMATIONS (before encoding/scaling) ──────────
    transform_recs = pipeline.get("transformation_recommendations", {})

    for col, info in transform_recs.items():
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        transform = info.get("transform", "none")

        if transform == "log1p":
            # Only apply if all values are non-negative after imputation
            if df[col].min() >= 0:
                df[col] = np.log1p(df[col])
                execution_log.append({
                    "column": col,
                    "action": "TRANSFORMED (log1p)",
                    "reason": info.get("reason", "")
                })

        elif transform == "sqrt":
            if df[col].min() >= 0:
                df[col] = np.sqrt(df[col])
                execution_log.append({
                    "column": col,
                    "action": "TRANSFORMED (sqrt)",
                    "reason": info.get("reason", "")
                })

        elif transform == "reflect_log1p":
            max_val = df[col].max()
            df[col] = np.log1p(max_val - df[col])
            execution_log.append({
                "column": col,
                "action": "TRANSFORMED (reflect + log1p)",
                "reason": info.get("reason", "")
            })

    # ── STEP 4: ENCODE CATEGORICAL COLUMNS ───────────────────────────────
    encode_strategy = pipeline.get("encoding_strategy", {})
    onehot_cols = []

    for col, info in encode_strategy.items():
        if col not in df.columns:
            continue
        if col == target_col:
            continue

        encoding = info.get("encoding", "LabelEncoding")

        if encoding == "LabelEncoding":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            execution_log.append({
                "column": col,
                "action": "ENCODED (LabelEncoding)",
                "classes": list(le.classes_[:10])
            })

        elif encoding == "OneHotEncoding":
            onehot_cols.append(col)

        elif encoding in ("TargetEncoding", "HashEncoding"):
            # For high cardinality: use label encoding as safe fallback
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            execution_log.append({
                "column": col,
                "action": f"ENCODED (LabelEncoding — fallback from {encoding})",
                "note": "TargetEncoding/HashEncoding require target info; LabelEncoding used"
            })

    # Apply OneHot last (it changes column structure)
    if onehot_cols:
        df = pd.get_dummies(df, columns=onehot_cols, drop_first=True)
        execution_log.append({
            "columns": onehot_cols,
            "action": "ENCODED (OneHotEncoding)",
            "new_columns_created": len(df.columns) - original_shape[1]
        })

    # ── STEP 5: SCALE NUMERIC COLUMNS ────────────────────────────────────
    scale_strategy = pipeline.get("scaling_strategy", {})

    for col, info in scale_strategy.items():
        if col not in df.columns:
            continue
        if col == target_col:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        scaler_type = info.get("scaler", "StandardScaler")

        try:
            values = df[[col]].values

            if scaler_type == "StandardScaler":
                scaler = StandardScaler()
            elif scaler_type == "MinMaxScaler":
                scaler = MinMaxScaler()
            elif scaler_type == "RobustScaler":
                scaler = RobustScaler()
            else:
                continue

            df[col] = scaler.fit_transform(values).flatten()
            execution_log.append({
                "column": col,
                "action": f"SCALED ({scaler_type})",
                "reason": info.get("reason", "")
            })

        except Exception as e:
            execution_log.append({
                "column": col,
                "action": "SCALING FAILED",
                "error": str(e)
            })

    # ── ENCODE TARGET IF CATEGORICAL ──────────────────────────────────────
    if target_col and target_col in df.columns:
        if df[target_col].dtype == object:
            le = LabelEncoder()
            df[target_col] = le.fit_transform(df[target_col].astype(str))
            execution_log.append({
                "column": target_col,
                "action": "TARGET ENCODED (LabelEncoding)",
                "classes": list(le.classes_)
            })

    # ── FINAL CLEANUP ─────────────────────────────────────────────────────
    # Fill any remaining NaN (edge cases)
    remaining_missing = int(df.isnull().sum().sum())
    if remaining_missing > 0:
        numeric_cols = df.select_dtypes(include="number").columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        cat_cols = df.select_dtypes(exclude="number").columns
        for col in cat_cols:
            if col != target_col:
                df[col] = df[col].fillna("Unknown")

        execution_log.append({
            "action": "FINAL CLEANUP",
            "remaining_missing_filled": remaining_missing
        })

    # ── STATS COMPARISON ──────────────────────────────────────────────────
    final_missing = int(df.isnull().sum().sum())

    stats = {
        "original_shape": {"rows": original_shape[0], "columns": original_shape[1]},
        "final_shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "columns_dropped": len(dropped),
        "dropped_column_names": dropped,
        "missing_values_before": original_missing,
        "missing_values_after": final_missing,
        "missing_reduction": original_missing - final_missing,
        "steps_applied": len(execution_log),
        "all_numeric": bool(df.select_dtypes(exclude="number").empty)
    }

    return {
        "success": True,
        "stats": stats,
        "execution_log": execution_log,
        "cleaned_dataframe": df
    }