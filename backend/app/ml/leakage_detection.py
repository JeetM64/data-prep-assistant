import pandas as pd
import numpy as np


def detect_data_leakage(df):
    report = {
        "target_leakage": [],
        "high_correlation_pairs": [],
        "duplicate_columns": [],
        "derived_feature_leakage": [],
        "summary": {"total_issues": 0, "severity": "NONE"}
    }
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        return {"message": "Not enough numeric features"}
    target_col = df.columns[-1]
    feature_cols = [c for c in numeric_df.columns if c != target_col]
    corr_matrix = numeric_df.corr()
    if target_col in numeric_df.columns:
        for feat in feature_cols:
            corr_val = corr_matrix.loc[feat, target_col]
            if abs(corr_val) > 0.95:
                report["target_leakage"].append({
                    "feature": feat,
                    "target": target_col,
                    "correlation": round(float(corr_val), 4),
                    "severity": "CRITICAL",
                    "reason": str(feat) + " has high correlation with target"
                })
    seen_pairs = set()
    for i, col1 in enumerate(feature_cols):
        for col2 in feature_cols[i + 1:]:
            pk = tuple(sorted([col1, col2]))
            if pk in seen_pairs:
                continue
            seen_pairs.add(pk)
            cv = corr_matrix.loc[col1, col2]
            if abs(cv) > 0.90:
                sev = "HIGH" if abs(cv) > 0.95 else "MODERATE"
                report["high_correlation_pairs"].append({
                    "feature_1": col1,
                    "feature_2": col2,
                    "correlation": round(float(cv), 4),
                    "severity": sev,
                    "reason": col1 + " and " + col2 + " are highly correlated"
                })
    total = (
        len(report["target_leakage"]) +
        len(report["high_correlation_pairs"]) +
        len(report["duplicate_columns"]) +
        len(report["derived_feature_leakage"])
    )
    report["summary"]["total_issues"] = total
    if len(report["target_leakage"]) > 0:
        report["summary"]["severity"] = "CRITICAL"
    elif total > 5:
        report["summary"]["severity"] = "HIGH"
    elif total > 0:
        report["summary"]["severity"] = "MODERATE"
    else:
        report["summary"]["severity"] = "NONE"
        report["summary"]["message"] = "No leakage detected"
    return report