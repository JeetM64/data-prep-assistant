import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def _detect_task_type(y: pd.Series) -> str:
    if y.dtype == object or y.dtype == bool:
        return "classification"
    if y.nunique() <= 15:
        return "classification"
    return "regression"


def _importance_tier(importance: float, max_importance: float) -> str:
    """
    Classify a feature's importance relative to the most important feature.
    More meaningful than absolute thresholds.
    """
    ratio = importance / (max_importance + 1e-9)
    if ratio >= 0.5:
        return "HIGH"
    elif ratio >= 0.15:
        return "MODERATE"
    elif ratio >= 0.05:
        return "LOW"
    else:
        return "NEGLIGIBLE"


def _importance_reason(feature: str, imp: float, tier: str,
                        perm_imp: float, agrees: bool) -> str:
    """Plain-English explanation of why a feature is important or not."""

    if tier == "HIGH":
        base = f"'{feature}' is a top predictor (RF importance={imp:.4f})."
        if agrees:
            base += " Permutation importance confirms this — removing it significantly hurts performance."
        else:
            base += (
                " However, permutation importance is lower — may be correlated with another strong feature."
            )
        return base

    elif tier == "MODERATE":
        return (
            f"'{feature}' contributes moderately (importance={imp:.4f}). "
            "Worth keeping — provides some signal without adding much noise."
        )

    elif tier == "LOW":
        return (
            f"'{feature}' has low importance ({imp:.4f}). "
            "Consider keeping only if domain knowledge supports it."
        )

    else:
        return (
            f"'{feature}' is negligible (importance={imp:.4f}). "
            "Recommended for removal — adds noise without predictive value."
        )


def feature_importance_analysis(df: pd.DataFrame, target_column: str) -> dict:
    """
    Research-level feature importance analysis using two methods:

    1. Random Forest / Gradient Boosting impurity importance
       - Fast, built into tree models
       - Weakness: biased toward high-cardinality features

    2. Permutation Importance (model-agnostic)
       - Shuffles each feature and measures performance drop
       - More reliable: directly measures how much each feature contributes
       - Not biased by cardinality

    Both methods are compared per feature:
    - If both agree → high confidence in the ranking
    - If they disagree → flag it (likely correlated features or cardinality bias)

    Each feature gets:
    - importance score from both methods
    - tier (HIGH / MODERATE / LOW / NEGLIGIBLE)
    - agreement flag between methods
    - plain-English reason
    """

    try:
        if target_column not in df.columns:
            return {"error": f"Target column '{target_column}' not found."}

        df = df.dropna()

        if len(df) < 10:
            return {"error": "Not enough rows for importance analysis (need at least 10)."}

        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Encode categorical features
        X = X.copy()
        for col in X.select_dtypes(include="object").columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

        numeric_X = X.select_dtypes(include=np.number)

        if numeric_X.empty:
            return {"error": "No numeric features available for importance analysis."}

        # Encode target if string
        if y.dtype == object:
            y = pd.Series(LabelEncoder().fit_transform(y.astype(str)))

        task = _detect_task_type(df[target_column])

        # ── TRAIN/TEST SPLIT ──────────────────────────────────────────────
        X_train, X_test, y_train, y_test = train_test_split(
            numeric_X, y, test_size=0.2, random_state=42
        )

        # ── METHOD 1: RF IMPURITY IMPORTANCE ─────────────────────────────
        if task == "classification":
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            rf = RandomForestRegressor(n_estimators=100, random_state=42)

        rf.fit(X_train, y_train)
        rf_importances = dict(zip(numeric_X.columns, rf.feature_importances_))
        max_rf_imp = max(rf_importances.values()) if rf_importances else 1.0

        # ── METHOD 2: PERMUTATION IMPORTANCE ─────────────────────────────
        # Shuffles each feature column and measures how much test score drops
        # Drop = how important that feature actually is
        perm_result = permutation_importance(
            rf, X_test, y_test,
            n_repeats=10,
            random_state=42,
            n_jobs=-1
        )
        perm_importances = dict(zip(
            numeric_X.columns,
            perm_result.importances_mean
        ))
        perm_std = dict(zip(
            numeric_X.columns,
            perm_result.importances_std
        ))

        # Normalize permutation importances to 0-1 range for comparison
        perm_vals = list(perm_importances.values())
        perm_min = min(perm_vals)
        perm_max = max(perm_vals) if max(perm_vals) != perm_min else 1.0
        perm_normalized = {
            col: (v - perm_min) / (perm_max - perm_min + 1e-9)
            for col, v in perm_importances.items()
        }

        # ── PER FEATURE ANALYSIS ──────────────────────────────────────────
        feature_report = {}

        for col in numeric_X.columns:
            rf_imp = float(rf_importances.get(col, 0))
            perm_imp = float(perm_importances.get(col, 0))
            perm_norm = float(perm_normalized.get(col, 0))
            perm_s = float(perm_std.get(col, 0))

            tier = _importance_tier(rf_imp, max_rf_imp)

            # Agreement: both methods rank this feature similarly
            rf_norm = rf_imp / (max_rf_imp + 1e-9)
            agrees = abs(rf_norm - perm_norm) < 0.3

            reason = _importance_reason(col, rf_imp, tier, perm_imp, agrees)

            feature_report[col] = {
                "rf_importance": round(rf_imp, 4),
                "permutation_importance": round(perm_imp, 4),
                "permutation_std": round(perm_s, 4),
                "methods_agree": agrees,
                "tier": tier,
                "reason": reason
            }

        # ── SORTED RANKINGS ───────────────────────────────────────────────
        ranked_by_rf = sorted(
            feature_report.items(),
            key=lambda x: x[1]["rf_importance"],
            reverse=True
        )

        ranked_by_perm = sorted(
            feature_report.items(),
            key=lambda x: x[1]["permutation_importance"],
            reverse=True
        )

        # ── TIER GROUPINGS ────────────────────────────────────────────────
        high_impact = [c for c, v in feature_report.items() if v["tier"] == "HIGH"]
        moderate_impact = [c for c, v in feature_report.items() if v["tier"] == "MODERATE"]
        low_impact = [c for c, v in feature_report.items() if v["tier"] == "LOW"]
        negligible = [c for c, v in feature_report.items() if v["tier"] == "NEGLIGIBLE"]

        # Disagreements — worth investigating
        disagreements = [
            {
                "feature": col,
                "rf_importance": v["rf_importance"],
                "permutation_importance": v["permutation_importance"],
                "note": (
                    "High RF importance but low permutation importance — "
                    "likely correlated with another feature or cardinality bias."
                    if v["rf_importance"] > 0.1 and v["permutation_importance"] < 0
                    else "Methods disagree — investigate feature relationships."
                )
            }
            for col, v in feature_report.items()
            if not v["methods_agree"]
        ]

        # ── RECOMMENDATION ────────────────────────────────────────────────
        if len(negligible) == 0:
            recommendation = "All features contribute meaningfully. No removal recommended."
        else:
            recommendation = (
                f"Consider removing {len(negligible)} negligible feature(s): {negligible}. "
                f"Keep all HIGH and MODERATE tier features. "
                f"Review LOW tier features against domain knowledge."
            )

        return {
            "task": task,
            "total_features_analyzed": len(numeric_X.columns),
            "feature_details": feature_report,
            "ranking_by_rf_importance": [
                {"feature": c, "rf_importance": v["rf_importance"], "tier": v["tier"]}
                for c, v in ranked_by_rf
            ],
            "ranking_by_permutation": [
                {"feature": c, "permutation_importance": v["permutation_importance"], "tier": v["tier"]}
                for c, v in ranked_by_perm
            ],
            "tier_summary": {
                "HIGH": high_impact,
                "MODERATE": moderate_impact,
                "LOW": low_impact,
                "NEGLIGIBLE": negligible
            },
            "method_disagreements": disagreements,
            "recommendation": recommendation
        }

    except Exception as e:
        return {"error": str(e)}