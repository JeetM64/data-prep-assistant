import numpy as np
import pandas as pd


def dataset_quality_score(df):

    scores = {}
    rows, cols = df.shape

    # 1. COMPLETENESS
    total_missing = df.isna().sum().sum()
    total_cells = rows * cols
    missing_percent = total_missing / total_cells * 100
    completeness = 100 - missing_percent
    scores["completeness"] = round(completeness, 2)

    # 2. UNIQUENESS
    duplicate_rows = df.duplicated().sum()
    duplicate_percent = duplicate_rows / rows * 100
    uniqueness = 100 - duplicate_percent
    scores["uniqueness"] = round(uniqueness, 2)

    # 3. CONSISTENCY
    numeric_df = df.select_dtypes(include="number")
    total_outliers = 0
    for col in numeric_df.columns:
        z = np.abs((numeric_df[col] - numeric_df[col].mean()) / (numeric_df[col].std() + 1e-9))
        z_outliers = (z > 3).sum()
        Q1 = numeric_df[col].quantile(0.25)
        Q3 = numeric_df[col].quantile(0.75)
        IQR = Q3 - Q1
        iqr_outliers = ((numeric_df[col] < Q1 - 1.5 * IQR) | (numeric_df[col] > Q3 + 1.5 * IQR)).sum()
        total_outliers += (z_outliers + iqr_outliers) / 2
    outlier_percent = total_outliers / rows * 100
    consistency = max(0, 100 - outlier_percent * 2)
    scores["consistency"] = round(consistency, 2)

    # 4. CLASS BALANCE
    target = df.iloc[:, -1]
    class_counts = target.value_counts(normalize=True)
    gini = 1 - sum(class_counts ** 2)
    max_gini = 1 - (1 / len(class_counts))
    balance = (gini / max_gini * 100) if max_gini > 0 else 100
    scores["class_balance"] = round(balance, 2)

    # 5. FEATURE QUALITY
    penalty = 0
    for col in numeric_df.columns:
        if numeric_df[col].std() < 0.01:
            penalty += 10
    if len(numeric_df.columns) > 1:
        corr_matrix = numeric_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_pairs = (upper > 0.95).sum().sum()
        penalty += high_corr_pairs * 5
    feature_quality = max(0, 100 - penalty)
    scores["feature_quality"] = round(feature_quality, 2)

    # 6. ADEQUACY
    ratio = rows / cols
    if ratio >= 10:
        adequacy = 100
    elif ratio >= 5:
        adequacy = 70
    elif ratio >= 2:
        adequacy = 40
    else:
        adequacy = 10
    scores["adequacy"] = adequacy

    # WEIGHTED FINAL SCORE
    weights = {
        "completeness": 0.25,
        "uniqueness": 0.10,
        "consistency": 0.20,
        "class_balance": 0.15,
        "feature_quality": 0.15,
        "adequacy": 0.15
    }
    final_score = sum(scores[k] * weights[k] for k in weights)

    if final_score >= 85:
        status = "Excellent"
    elif final_score >= 70:
        status = "Good"
    elif final_score >= 50:
        status = "Moderate"
    else:
        status = "Poor — needs cleaning"

    return {
        "overall_score": round(final_score, 2),
        "dimension_scores": scores,
        "status": status
    }