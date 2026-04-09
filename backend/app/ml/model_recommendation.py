import pandas as pd
import numpy as np


def _detect_task_type(y):
    if y.dtype == object or y.dtype == bool:
        return "classification"
    if y.nunique() <= 15:
        return "classification"
    return "regression"


def recommend_model(df: pd.DataFrame, target_column: str = None) -> dict:
    """
    Research-level model recommendation engine.

    Fix: accepts target_column parameter so class imbalance
    is computed on the correct target (not last column which
    could be a non-target like Embarked).
    """
    rows, cols = df.shape

    # ── USE CORRECT TARGET ────────────────────────────────────────────────
    target = target_column if target_column and target_column in df.columns else df.columns[-1]
    y = df[target]

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()
    missing_percent = float(df.isnull().mean().mean() * 100)
    has_missing = missing_percent > 0
    has_categorical = len(categorical_cols) > 0
    task = _detect_task_type(y)

    # ── DATASET SIZE PROFILE ──────────────────────────────────────────────
    ratio = rows / max(cols, 1)
    if ratio < 5:
        ratio_warning = f"DANGER: Only {ratio:.1f} rows per feature. Severe overfitting risk."
    elif ratio < 10:
        ratio_warning = f"CAUTION: {ratio:.1f} rows per feature. Use regularized models."
    else:
        ratio_warning = f"OK: {ratio:.1f} rows per feature. Sufficient for most models."

    size_profile = {
        "size_category": "small" if rows < 1000 else "medium" if rows < 10000 else "large",
        "rows": int(rows),
        "features": int(cols),
        "rows_per_feature_ratio": round(ratio, 2),
        "ratio_warning": ratio_warning
    }

    # ── CLASS IMBALANCE (on correct target) ───────────────────────────────
    imbalance_info = None
    is_imbalanced = False
    if task == "classification":
        counts = y.value_counts()
        ratio_imb = float(counts.max()) / (float(counts.min()) + 1e-9)
        if ratio_imb > 10:
            sev = "SEVERE"
            strat = "Use SMOTE or class_weight='balanced'. Use F1/AUC-ROC not accuracy."
        elif ratio_imb > 3:
            sev = "MODERATE"
            strat = "Use class_weight='balanced' in models. Report F1 score."
        else:
            sev = "NONE"
            strat = "Classes are balanced. Standard training applies."
        is_imbalanced = sev != "NONE"
        imbalance_info = {
            "imbalance_ratio": round(ratio_imb, 2),
            "severity": sev,
            "recommended_strategy": strat,
            "class_distribution": {str(k): int(v) for k, v in counts.items()}
        }

    # ── MODEL RECOMMENDATIONS ─────────────────────────────────────────────
    if task == "classification":
        models = [
            {
                "model": "Logistic Regression",
                "priority": "baseline",
                "why": (
                    "Fast, interpretable, works well on linearly separable data. "
                    "Use as baseline to compare complex models against."
                    + (" Add class_weight='balanced' for imbalance." if is_imbalanced else "")
                ),
                "when_to_avoid": "Non-linear decision boundaries or very high-dimensional data.",
                "hyperparameters_to_tune": ["C", "penalty", "solver"]
            },
            {
                "model": "Random Forest",
                "priority": "recommended",
                "why": (
                    "Handles non-linear relationships, robust to outliers, "
                    "built-in feature importance."
                    + (" Use class_weight='balanced' for imbalance." if is_imbalanced else "")
                ),
                "when_to_avoid": "Very large datasets or when interpretability is critical.",
                "hyperparameters_to_tune": ["n_estimators", "max_depth", "min_samples_leaf"]
            },
            {
                "model": "Gradient Boosting (XGBoost / LightGBM)",
                "priority": "recommended" if rows > 1000 else "optional",
                "why": "State-of-the-art on tabular data. Handles missing values natively (XGBoost).",
                "when_to_avoid": "Small datasets (< 500 rows) — prone to overfitting.",
                "hyperparameters_to_tune": ["learning_rate", "n_estimators", "max_depth", "subsample"]
            },
            {
                "model": "SVM (Support Vector Machine)",
                "priority": "optional",
                "why": "Effective in high-dimensional spaces with clear class margins.",
                "when_to_avoid": "Large datasets (slow training) or need probability outputs.",
                "hyperparameters_to_tune": ["C", "kernel", "gamma"]
            },
            {
                "model": "Decision Tree",
                "priority": "interpretability",
                "why": "Fully interpretable — can be visualized for non-technical stakeholders.",
                "when_to_avoid": "Production use without pruning — prone to overfitting.",
                "hyperparameters_to_tune": ["max_depth", "min_samples_split", "criterion"]
            }
        ]
        primary_metric = "F1-score (weighted)" if is_imbalanced else "Accuracy + F1"
        secondary_metric = "AUC-ROC (for binary classification)"

    else:
        models = [
            {
                "model": "Linear Regression",
                "priority": "baseline",
                "why": "Simple, fast, interpretable. Use as baseline. Assumes linear relationship.",
                "when_to_avoid": "Non-linear target relationships or multicollinear features.",
                "hyperparameters_to_tune": ["fit_intercept"]
            },
            {
                "model": "Ridge Regression",
                "priority": "recommended",
                "why": "Linear Regression with L2 regularization. Handles multicollinearity well.",
                "when_to_avoid": "When feature selection is needed (use Lasso instead).",
                "hyperparameters_to_tune": ["alpha"]
            },
            {
                "model": "Random Forest Regressor",
                "priority": "recommended",
                "why": "Handles non-linear relationships and interactions. Robust to outliers.",
                "when_to_avoid": "Very large datasets or when interpretability is critical.",
                "hyperparameters_to_tune": ["n_estimators", "max_depth", "min_samples_leaf"]
            },
            {
                "model": "Gradient Boosting Regressor (XGBoost / LightGBM)",
                "priority": "recommended" if rows > 1000 else "optional",
                "why": "Best performance on most tabular regression tasks.",
                "when_to_avoid": "Small datasets — needs careful tuning.",
                "hyperparameters_to_tune": ["learning_rate", "n_estimators", "max_depth"]
            }
        ]
        primary_metric = "R² Score + RMSE"
        secondary_metric = "MAE (interpretable error in original units)"

    # ── PREPROCESSING CHECKLIST ───────────────────────────────────────────
    checklist = []
    if has_missing:
        checklist.append({
            "step": "Handle missing values",
            "reason": f"{missing_percent:.1f}% of data is missing",
            "action": "Median imputation for numeric, mode for categorical"
        })
    if has_categorical:
        checklist.append({
            "step": "Encode categorical features",
            "reason": f"{len(categorical_cols)} categorical columns found",
            "action": "OneHotEncoding for low cardinality, TargetEncoding for high cardinality"
        })
    if len(numeric_cols) > 0:
        checklist.append({
            "step": "Scale numeric features",
            "reason": "Most models sensitive to feature scale",
            "action": "StandardScaler for normal data, RobustScaler if outliers present"
        })
    if size_profile["rows_per_feature_ratio"] < 10:
        checklist.append({
            "step": "Feature selection",
            "reason": f"Low rows/feature ratio ({size_profile['rows_per_feature_ratio']:.1f})",
            "action": "Remove low-importance and highly correlated features first"
        })

    return {
        "task_type": task,
        "target_used": target,
        "dataset_profile": size_profile,
        "class_imbalance": imbalance_info,
        "recommended_models": models,
        "evaluation_metrics": {
            "primary": primary_metric,
            "secondary": secondary_metric
        },
        "preprocessing_checklist": checklist
    }