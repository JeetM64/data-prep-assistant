import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import LabelEncoder


def _detect_task_type(y: pd.Series) -> str:
    if y.dtype == object or y.dtype == bool:
        return "classification"
    if y.nunique() <= 15:
        return "classification"
    return "regression"


def _stability_verdict(cv_pct: float, std: float) -> tuple:
    """
    Returns (stability_level, recommendation) based on
    Coefficient of Variation % and raw std.

    CV% = (std / mean) * 100
    This is more meaningful than std alone because:
    - std=0.05 on mean=0.95 → CV%=5.3% → stable
    - std=0.05 on mean=0.55 → CV%=9.1% → unstable
    """
    if cv_pct < 5:
        return "HIGH", "Model is highly stable across folds. Safe to deploy."
    elif cv_pct < 10:
        return "MODERATE", (
            "Moderate variance across folds. Check if training data distribution "
            "is consistent. Consider more data or feature cleaning."
        )
    elif cv_pct < 20:
        return "LOW", (
            "High variance across folds — model performance is inconsistent. "
            "Possible causes: small dataset, class imbalance, noisy features. "
            "Try StratifiedKFold, feature selection, or ensemble methods."
        )
    else:
        return "VERY LOW", (
            "Extremely unstable — model behaves very differently on different subsets. "
            "Dataset likely has distribution issues or severe class imbalance. "
            "Do not trust this model without further investigation."
        )


def _flag_worst_fold(fold_scores: list) -> dict:
    """Identify the worst performing fold and flag it."""
    worst_idx = int(np.argmin(fold_scores))
    worst_score = fold_scores[worst_idx]
    mean_score = np.mean(fold_scores)
    drop = mean_score - worst_score

    flag = None
    if drop > 0.15:
        flag = (
            f"Fold {worst_idx + 1} dropped to {worst_score:.3f} "
            f"({drop:.3f} below mean) — possible distribution shift or "
            "class imbalance in that subset."
        )
    return {
        "worst_fold_index": worst_idx + 1,
        "worst_fold_score": round(worst_score, 4),
        "drop_from_mean": round(drop, 4),
        "warning": flag
    }


def cross_validation_stability(df: pd.DataFrame, target_column: str) -> dict:
    """
    Research-level cross-validation stability analysis.

    Improvements over basic CV:
    1. StratifiedKFold for classification (preserves class ratios per fold)
    2. Multiple models tested (not just Random Forest)
    3. Coefficient of Variation % (std/mean*100) — more meaningful than raw std
    4. Per-fold score breakdown with worst-fold flagging
    5. Multiple metrics for classification (accuracy + F1)
    6. Plain-English stability verdict + recommendation per model
    7. Dataset-level stability summary across all models
    """

    try:
        if target_column not in df.columns:
            return {"error": f"Target column '{target_column}' not found."}

        df = df.dropna()

        if len(df) < 20:
            return {"error": "Need at least 20 rows for cross-validation."}

        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Encode categorical features
        X = X.copy()
        for col in X.select_dtypes(include="object").columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

        # Encode target if string
        if y.dtype == object:
            y = pd.Series(LabelEncoder().fit_transform(y.astype(str)))

        task = _detect_task_type(df[target_column])

        # ── CV STRATEGY ───────────────────────────────────────────────────
        # StratifiedKFold ensures each fold has same class distribution
        # This is critical — basic KFold can create unbalanced folds
        if task == "classification":
            cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
            }
            primary_metric = "accuracy"
            secondary_metric = "f1_weighted"
        else:
            cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)
            models = {
                "Ridge Regression": Ridge(alpha=1.0),
                "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            }
            primary_metric = "r2"
            secondary_metric = None

        # ── PER MODEL ANALYSIS ────────────────────────────────────────────
        results = {}
        stability_levels = []

        for name, model in models.items():

            try:
                # Primary metric CV
                primary_scores = cross_val_score(
                    model, X, y, cv=cv_strategy, scoring=primary_metric
                )

                mean_score = float(primary_scores.mean())
                std_score = float(primary_scores.std())

                # Coefficient of Variation % — key research metric
                cv_pct = (std_score / (mean_score + 1e-9)) * 100

                stability, recommendation = _stability_verdict(cv_pct, std_score)
                stability_levels.append(stability)

                worst_fold_info = _flag_worst_fold(primary_scores.tolist())

                model_result = {
                    "fold_scores": [round(s, 4) for s in primary_scores.tolist()],
                    "mean_score": round(mean_score, 4),
                    "std_dev": round(std_score, 4),
                    "coefficient_of_variation_pct": round(cv_pct, 2),
                    "stability": stability,
                    "worst_fold": worst_fold_info,
                    "recommendation": recommendation
                }

                # Secondary metric (F1 for classification)
                if secondary_metric:
                    secondary_scores = cross_val_score(
                        model, X, y, cv=cv_strategy, scoring=secondary_metric
                    )
                    model_result["f1_cv_mean"] = round(float(secondary_scores.mean()), 4)
                    model_result["f1_cv_std"] = round(float(secondary_scores.std()), 4)

                results[name] = model_result

            except Exception as e:
                results[name] = {"error": str(e)}

        # ── DATASET-LEVEL STABILITY SUMMARY ──────────────────────────────
        # If majority of models are unstable → dataset has structural issues
        priority = {"VERY LOW": 0, "LOW": 1, "MODERATE": 2, "HIGH": 3}
        if stability_levels:
            worst_overall = min(stability_levels, key=lambda s: priority.get(s, 2))
        else:
            worst_overall = "UNKNOWN"

        if worst_overall in ("VERY LOW", "LOW"):
            dataset_message = (
                "Dataset shows unstable CV across models. "
                "Likely causes: small sample size, class imbalance, "
                "or inconsistent feature distributions. "
                "Address data quality before trusting model results."
            )
        elif worst_overall == "MODERATE":
            dataset_message = (
                "Dataset is reasonably stable. Minor variance may improve "
                "with more data or feature engineering."
            )
        else:
            dataset_message = (
                "Dataset is stable across all models and folds. "
                "Cross-validation results are trustworthy."
            )

        return {
            "task": task,
            "primary_metric": primary_metric,
            "cv_folds": 5,
            "cv_strategy": "StratifiedKFold" if task == "classification" else "KFold",
            "per_model_stability": results,
            "dataset_stability_summary": {
                "overall_stability": worst_overall,
                "message": dataset_message
            }
        }

    except Exception as e:
        return {"error": str(e)}