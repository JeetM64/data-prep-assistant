import pandas as pd
import numpy as np


def correlation_analysis(df):
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        return {"error": "No numeric columns found"}

    pearson  = numeric_df.corr(method="pearson")
    spearman = numeric_df.corr(method="spearman")
    kendall  = numeric_df.corr(method="kendall")

    strong_pairs = []
    cols = numeric_df.columns

    for i, col1 in enumerate(cols):
        for col2 in cols[i+1:]:
            p_val = pearson.loc[col1, col2]
            s_val = spearman.loc[col1, col2]
            k_val = kendall.loc[col1, col2]

            if abs(p_val) > 0.8:
                if abs(p_val) > 0.95:
                    severity = "CRITICAL"
                elif abs(p_val) > 0.90:
                    severity = "HIGH"
                else:
                    severity = "MODERATE"

                all_agree = abs(s_val) > 0.8 and abs(k_val) > 0.8
                var1 = numeric_df[col1].var()
                var2 = numeric_df[col2].var()
                drop_suggestion = col1 if var1 < var2 else col2

                reason = (
                    f"'{col1}' and '{col2}' are {p_val:.0%} correlated (Pearson). "
                    f"Severity: {severity}. "
                    f"Suggest dropping '{drop_suggestion}' as it has lower variance."
                )

                strong_pairs.append({
                    "feature_1": col1,
                    "feature_2": col2,
                    "pearson": round(p_val, 4),
                    "spearman": round(s_val, 4),
                    "kendall": round(k_val, 4),
                    "severity": severity,
                    "all_methods_agree": all_agree,
                    "drop_suggestion": drop_suggestion,
                    "reason": reason
                })

    return {
        "strong_correlation_pairs": strong_pairs,
        "total_pairs_flagged": len(strong_pairs),
        "summary": (
            f"{len(strong_pairs)} highly correlated pairs found. "
            "Redundant features may cause multicollinearity."
            if len(strong_pairs) > 0
            else "No strong correlations found. Features appear independent."
        )
    }