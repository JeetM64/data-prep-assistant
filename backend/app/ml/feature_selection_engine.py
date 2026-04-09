import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, f_classif, f_regression
from sklearn.preprocessing import LabelEncoder


def _detect_task_type(y: pd.Series) -> str:
    if y.dtype == object or y.dtype == bool:
        return "classification"
    if y.nunique() <= 15:
        return "classification"
    return "regression"


def smart_feature_selection(df, target_column):
    """
    Research-level feature selection using 4 methods combined:

    1. Random Forest feature importance     — tree-based, captures non-linear
    2. Mutual Information                   — information-theoretic, model-free
    3. High correlation pruning             — removes redundant features
    4. Near-zero variance filter            — removes uninformative features

    Each removed feature includes a reason field explaining why.
    Features are only recommended for removal if flagged by 2+ methods
    (consensus approach — reduces false removals).
    """

    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found in dataset."}

    df = df.dropna()

    if len(df) < 10:
        return {"error": "Not enough rows after dropping missing values (need at least 10)."}

    X = df.drop(columns=[target_column])
    y = df[target_column]

    numeric_X = X.select_dtypes(include=np.number)

    if numeric_X.empty:
        return {"error": "No numeric features available for selection."}

    task = _detect_task_type(y)

    # Encode target if string labels
    if y.dtype == object:
        le = LabelEncoder()
        y_encoded = pd.Series(le.fit_transform(y))
    else:
        y_encoded = y

    report = {}
    remove_reasons = {}   # column → list of reasons it was flagged

    # ── METHOD 1: RANDOM FOREST IMPORTANCE ───────────────────────────────
    try:
        if task == "classification":
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            rf = RandomForestRegressor(n_estimators=100, random_state=42)

        rf.fit(numeric_X, y_encoded)
        importances = dict(zip(numeric_X.columns, rf.feature_importances_))

        low_importance = [f for f, imp in importances.items() if imp < 0.02]
        for f in low_importance:
            remove_reasons.setdefault(f, []).append(
                f"Low Random Forest importance ({importances[f]:.4f} < 0.02 threshold)"
            )

        report["rf_importance"] = {
            col: round(float(imp), 4) for col, imp in
            sorted(importances.items(), key=lambda x: x[1], reverse=True)
        }

    except Exception as e:
        report["rf_importance"] = {"error": str(e)}
        importances = {}
        low_importance = []

    # ── METHOD 2: MUTUAL INFORMATION ─────────────────────────────────────
    # MI measures how much knowing a feature reduces uncertainty about target
    # Model-free — works for any relationship (linear or not)
    try:
        if task == "classification":
            mi_scores = mutual_info_classif(numeric_X, y_encoded, random_state=42)
        else:
            mi_scores = mutual_info_regression(numeric_X, y_encoded, random_state=42)

        mi_dict = dict(zip(numeric_X.columns, mi_scores))

        low_mi = [f for f, score in mi_dict.items() if score < 0.01]
        for f in low_mi:
            remove_reasons.setdefault(f, []).append(
                f"Near-zero mutual information ({mi_dict[f]:.4f}) — feature carries no target signal"
            )

        report["mutual_information"] = {
            col: round(float(score), 4) for col, score in
            sorted(mi_dict.items(), key=lambda x: x[1], reverse=True)
        }

    except Exception as e:
        report["mutual_information"] = {"error": str(e)}
        low_mi = []

    # ── METHOD 3: STATISTICAL TEST (F-test) ──────────────────────────────
    # F-test checks linear relationship between feature and target
    # Complements MI which captures non-linear relationships
    try:
        if task == "classification":
            f_scores, p_values = f_classif(numeric_X, y_encoded)
        else:
            f_scores, p_values = f_regression(numeric_X, y_encoded)

        f_dict = dict(zip(numeric_X.columns, p_values))

        # p > 0.05 means the feature is statistically insignificant
        low_f = [f for f, p in f_dict.items() if p > 0.05]
        for f in low_f:
            remove_reasons.setdefault(f, []).append(
                f"Statistically insignificant (F-test p={f_dict[f]:.4f} > 0.05)"
            )

        report["f_test_p_values"] = {
            col: round(float(p), 4) for col, p in f_dict.items()
        }

    except Exception as e:
        report["f_test_p_values"] = {"error": str(e)}
        low_f = []

    # ── METHOD 4: HIGH CORRELATION PRUNING ───────────────────────────────
    # If two features are >85% correlated, the less important one is redundant
    remove_due_to_corr = set()
    high_corr_pairs = []

    if len(numeric_X.columns) > 1:
        corr_matrix = numeric_X.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        for col in upper.columns:
            for row in upper.index:
                val = upper.loc[row, col]
                if pd.notna(val) and val > 0.85:
                    high_corr_pairs.append({
                        "feature_1": row,
                        "feature_2": col,
                        "correlation": round(float(val), 4)
                    })
                    # Drop the one with lower RF importance
                    imp_row = importances.get(row, 0)
                    imp_col = importances.get(col, 0)
                    drop = col if imp_col <= imp_row else row
                    remove_due_to_corr.add(drop)
                    remove_reasons.setdefault(drop, []).append(
                        f"Highly correlated with '{row if drop == col else col}' "
                        f"({val:.2%}) and has lower predictive importance"
                    )

    report["high_correlation_pairs"] = high_corr_pairs

    # ── METHOD 5: NEAR-ZERO VARIANCE ────────────────────────────────────
    low_variance = []
    for col in numeric_X.columns:
        var = numeric_X[col].var()
        if var < 0.01:
            low_variance.append(col)
            remove_reasons.setdefault(col, []).append(
                f"Near-zero variance ({var:.6f}) — column is nearly constant, no signal"
            )

    # ── CONSENSUS: REMOVE ONLY IF FLAGGED BY 2+ METHODS ─────────────────
    # This prevents aggressively dropping features that only failed one test
    definite_remove = [
        col for col, reasons in remove_reasons.items()
        if len(reasons) >= 2
    ]

    # Always remove zero-variance (they're useless by definition)
    for col in low_variance:
        if col not in definite_remove:
            definite_remove.append(col)

    recommended_features = [
        col for col in numeric_X.columns
        if col not in definite_remove
    ]

    # ── BUILD REMOVAL REPORT ─────────────────────────────────────────────
    removal_report = {}
    for col in definite_remove:
        removal_report[col] = {
            "flagged_by_n_methods": len(remove_reasons.get(col, [])),
            "reasons": remove_reasons.get(col, ["Zero variance"])
        }

    # ── FINAL NOTE ───────────────────────────────────────────────────────
    if len(definite_remove) == 0:
        final_note = "All numeric features passed the consensus filter — no removals recommended."
    else:
        final_note = (
            f"{len(definite_remove)} feature(s) recommended for removal "
            f"(flagged by 2+ selection methods). "
            f"{len(recommended_features)} features retained."
        )

    return {
        "task_type": task,
        "total_features_evaluated": len(numeric_X.columns),
        "recommended_features": recommended_features,
        "features_to_remove": definite_remove,
        "removal_report": removal_report,
        "high_correlation_pairs": high_corr_pairs,
        "low_variance_features": low_variance,
        "final_recommendation": final_note,
        "method_details": report
    }