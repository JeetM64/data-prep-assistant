"""
auto_training.py
================
Production-quality AutoML model training, evaluation, and comparison module.

Responsibilities:
  - Auto-detect task type (classification / regression)
  - Detect and handle class imbalance (balanced weights)
  - Train multiple models in a unified pipeline
  - Evaluate with comprehensive, task-appropriate metrics
  - Cross-validate with StratifiedKFold / KFold (5 folds)
  - Detect overfitting per model (LOW / MODERATE / HIGH)
  - Extract feature importance for tree-based models
  - Return a structured, serialisable result dict ready for any frontend

Design constraints:
  - No hardcoded column names — fully dataset-agnostic
  - No heavy hyperparameter search (Optuna is handled separately)
  - No deep learning
  - Integrates with the existing cleaning → encoding → feature-engineering pipeline
"""

from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVR, SVC

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_CV_FOLDS = 5
_TEST_SIZE = 0.2
_RANDOM_STATE = 42
_IMBALANCE_RATIO_THRESHOLD = 3.0
_OVERFIT_MODERATE = 0.05   # train-CV gap above this → MODERATE
_OVERFIT_HIGH = 0.12       # train-CV gap above this → HIGH
_TOP_N_FEATURES = 15       # max features returned in feature importance


# ===========================================================================
# 1. TASK & TARGET UTILITIES
# ===========================================================================

def _detect_task_type(y: pd.Series) -> str:
    """
    Determine classification vs regression from the target column.

    Rules (in priority order):
      - object / bool dtype           → classification
      - numeric, ≤15 unique values    → classification (ordinal / discrete)
      - numeric, >15 unique values    → regression
    """
    if y.dtype == object or y.dtype == bool:
        return "classification"
    return "classification" if y.nunique() <= 15 else "regression"


def _detect_class_imbalance(y: pd.Series) -> dict:
    """Return imbalance statistics for a classification target."""
    counts = y.value_counts()
    ratio = counts.max() / counts.min()
    is_imbalanced = ratio > _IMBALANCE_RATIO_THRESHOLD

    recommendation = (
        "Class imbalance detected — using class_weight='balanced' in all applicable models. "
        "Consider SMOTE or other oversampling for severe imbalance (ratio > 10)."
        if is_imbalanced
        else "Classes are balanced — no special weighting required."
    )

    return {
        "is_imbalanced": bool(is_imbalanced),
        "imbalance_ratio": round(float(ratio), 2),
        "class_distribution": {str(k): int(v) for k, v in counts.items()},
        "recommendation": recommendation,
    }


def _encode_target(y: pd.Series) -> tuple[pd.Series, LabelEncoder | None]:
    """Label-encode a string/object classification target. Returns (encoded_y, encoder)."""
    if y.dtype == object:
        le = LabelEncoder()
        return pd.Series(le.fit_transform(y), name=y.name), le
    return y, None


# ===========================================================================
# 2. PREPROCESSING
# ===========================================================================

def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
      - StandardScales all numeric columns
      - OneHotEncodes all categorical columns (unknown categories → ignored)
    """
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if categorical_cols:
        transformers.append((
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical_cols,
        ))

    return ColumnTransformer(transformers=transformers, remainder="drop")


# ===========================================================================
# 3. MODEL CATALOGUE
# ===========================================================================

def _get_classification_models(is_imbalanced: bool) -> dict[str, Any]:
    cw = "balanced" if is_imbalanced else None
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight=cw, random_state=_RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, class_weight=cw, random_state=_RANDOM_STATE
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=_RANDOM_STATE
        ),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(probability=True, class_weight=cw, random_state=_RANDOM_STATE),
    }


def _get_regression_models() -> dict[str, Any]:
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=_RANDOM_STATE
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, random_state=_RANDOM_STATE
        ),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "SVR": SVR(kernel="rbf"),
    }


# ===========================================================================
# 4. METRICS
# ===========================================================================

def _classification_metrics(
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    n_classes: int,
) -> dict:
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    auc_roc = None
    if y_proba is not None:
        try:
            if n_classes == 2:
                auc_roc = round(float(roc_auc_score(y_test, y_proba[:, 1])), 4)
            else:
                auc_roc = round(
                    float(roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")),
                    4,
                )
        except Exception:
            auc_roc = None

    return {
        "accuracy": round(float(acc), 4),
        "precision_weighted": round(float(precision), 4),
        "recall_weighted": round(float(recall), 4),
        "f1_weighted": round(float(f1), 4),
        "roc_auc": auc_roc,
    }


def _regression_metrics(y_test: pd.Series, y_pred: np.ndarray) -> dict:
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    return {
        "r2_score": round(float(r2), 4),
        "mse": round(float(mse), 4),
        "rmse": round(float(np.sqrt(mse)), 4),
        "mae": round(float(mae), 4),
    }


# ===========================================================================
# 5. OVERFITTING DETECTION
# ===========================================================================

def _overfitting_flag(train_score: float, cv_mean: float) -> str:
    """
    Compare training score vs cross-validation mean.

    Gap thresholds:
      < 0.05  → LOW
      0.05–0.12 → MODERATE
      > 0.12  → HIGH
    """
    gap = train_score - cv_mean
    if gap > _OVERFIT_HIGH:
        return "HIGH"
    if gap > _OVERFIT_MODERATE:
        return "MODERATE"
    return "LOW"


# ===========================================================================
# 6. FEATURE IMPORTANCE
# ===========================================================================

def _extract_feature_importance(
    pipeline: Pipeline,
    X: pd.DataFrame,
    top_n: int = _TOP_N_FEATURES,
) -> list[dict] | None:
    """
    Extract feature importances from tree-based models inside a pipeline.
    Reconstructs feature names after ColumnTransformer encoding.
    Returns None for models that do not expose feature_importances_.
    """
    model_step = pipeline.named_steps.get("model")
    if model_step is None or not hasattr(model_step, "feature_importances_"):
        return None

    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessing"]

    # Reconstruct transformed feature names
    try:
        feature_names: list[str] = []
        for name, transformer, cols in preprocessor.transformers_:
            if name == "remainder":
                continue
            if hasattr(transformer, "get_feature_names_out"):
                feature_names.extend(transformer.get_feature_names_out(cols).tolist())
            else:
                feature_names.extend(cols if isinstance(cols, list) else list(cols))
    except Exception:
        # Fall back to generic names
        n = len(model_step.feature_importances_)
        feature_names = [f"feature_{i}" for i in range(n)]

    importances = model_step.feature_importances_
    if len(importances) != len(feature_names):
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    ranked = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True,
    )[:top_n]

    return [
        {"feature": name, "importance": round(float(imp), 6)}
        for name, imp in ranked
    ]


# ===========================================================================
# 7. MODEL REASONING
# ===========================================================================

def _model_reasoning(
    name: str,
    primary_score: float,
    cv_mean: float,
    cv_std: float,
    overfitting: str,
    task: str,
) -> str:
    metric_label = "F1" if task == "classification" else "R²"

    base = f"{name}: {metric_label}={primary_score:.3f}, CV={cv_mean:.3f}±{cv_std:.3f}."

    if overfitting == "HIGH":
        base += f" High overfitting detected (train–CV gap > {_OVERFIT_HIGH:.0%}). Apply regularisation or reduce complexity."
    elif overfitting == "MODERATE":
        base += " Moderate overfitting. Consider regularisation or pruning."

    if task == "classification":
        if primary_score > 0.90:
            base += " Excellent classification performance."
        elif primary_score > 0.75:
            base += " Good performance — hyperparameter tuning may improve further."
        else:
            base += " Below-average performance. Review preprocessing and feature engineering."
    else:
        if primary_score > 0.85:
            base += f" Explains {primary_score:.0%} of variance — strong fit."
        elif primary_score > 0.60:
            base += f" Explains {primary_score:.0%} of variance — reasonable fit."
        else:
            base += f" Explains only {primary_score:.0%} of variance — consider more features or target transformation."

    return base


# ===========================================================================
# 8. MAIN ENTRY POINT
# ===========================================================================

def auto_train(
    df: pd.DataFrame,
    target_column: str | None = None,
) -> dict:
    """
    Train, evaluate, and compare multiple models on any tabular dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Fully preprocessed DataFrame (output of the cleaning + feature-engineering
        pipeline). Should contain no raw missing values.
    target_column : str | None
        Name of the target column. Defaults to the last column if not provided.

    Returns
    -------
    dict with keys:
      task_type                  – "classification" or "regression"
      target_column              – resolved target name
      n_samples / n_features     – dataset shape info
      class_imbalance_analysis   – imbalance stats (classification only)
      best_model                 – name of top model
      best_score                 – primary metric score of top model
      primary_metric             – metric used for ranking
      model_metrics              – per-model dict with all metrics + diagnostics
      comparison_table           – list of dicts, one row per model (frontend-ready)
      feature_importance         – top features from best tree model (or None)
    """

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    df = df.copy().dropna()

    if df.shape[1] < 2:
        return {"error": "Dataset must have at least one feature and one target column."}

    target = target_column if (target_column and target_column in df.columns) else df.columns[-1]
    X = df.drop(columns=[target])
    y = df[target]

    if y.nunique() < 2:
        return {"error": "Target column has fewer than 2 unique values — cannot train."}

    # ------------------------------------------------------------------
    # Task detection
    # ------------------------------------------------------------------
    task = _detect_task_type(y)
    imbalance_info = _detect_class_imbalance(y) if task == "classification" else None

    # Encode string targets for classification
    y, _label_encoder = _encode_target(y) if task == "classification" else (y, None)

    n_classes = int(y.nunique()) if task == "classification" else 0

    # ------------------------------------------------------------------
    # Preprocessing & split
    # ------------------------------------------------------------------
    preprocessor = _build_preprocessor(X)

    stratify = y if task == "classification" else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=_TEST_SIZE, random_state=_RANDOM_STATE, stratify=stratify
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=_TEST_SIZE, random_state=_RANDOM_STATE
        )

    # ------------------------------------------------------------------
    # Cross-validation strategy
    # ------------------------------------------------------------------
    if task == "classification":
        cv_strategy = StratifiedKFold(n_splits=_CV_FOLDS, shuffle=True, random_state=_RANDOM_STATE)
        cv_scoring = "f1_weighted"
        primary_metric = "f1_weighted"
    else:
        cv_strategy = KFold(n_splits=_CV_FOLDS, shuffle=True, random_state=_RANDOM_STATE)
        cv_scoring = "r2"
        primary_metric = "r2_score"

    # ------------------------------------------------------------------
    # Model catalogue
    # ------------------------------------------------------------------
    is_imbalanced = imbalance_info["is_imbalanced"] if imbalance_info else False
    models = (
        _get_classification_models(is_imbalanced)
        if task == "classification"
        else _get_regression_models()
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    model_metrics: dict[str, dict] = {}
    best_model_name: str | None = None
    best_score: float = -999.0
    best_pipeline: Pipeline | None = None

    for name, estimator in models.items():
        pipeline = Pipeline([
            ("preprocessing", preprocessor),
            ("model", estimator),
        ])

        try:
            # --- Train ---
            t0 = time.time()
            pipeline.fit(X_train, y_train)
            train_time = round(time.time() - t0, 3)

            y_pred = pipeline.predict(X_test)

            # --- Probabilities (classification) ---
            y_proba = None
            if task == "classification" and hasattr(pipeline, "predict_proba"):
                try:
                    y_proba = pipeline.predict_proba(X_test)
                except Exception:
                    y_proba = None

            # --- Test metrics ---
            if task == "classification":
                test_metrics = _classification_metrics(y_test, y_pred, y_proba, n_classes)
                primary_score = test_metrics["f1_weighted"]
            else:
                test_metrics = _regression_metrics(y_test, y_pred)
                primary_score = test_metrics["r2_score"]

            # --- Training score (for overfitting detection) ---
            y_train_pred = pipeline.predict(X_train)
            if task == "classification":
                train_score = float(f1_score(y_train, y_train_pred, average="weighted", zero_division=0))
            else:
                train_score = float(r2_score(y_train, y_train_pred))

            # --- Cross-validation ---
            cv_raw = cross_val_score(pipeline, X, y, cv=cv_strategy, scoring=cv_scoring)
            cv_mean = round(float(cv_raw.mean()), 4)
            cv_std = round(float(cv_raw.std()), 4)

            # --- Overfitting flag ---
            overfit_flag = _overfitting_flag(train_score, cv_mean)

            # --- Feature importance (tree models only) ---
            importance = _extract_feature_importance(pipeline, X)

            # --- Reasoning ---
            reasoning = _model_reasoning(name, primary_score, cv_mean, cv_std, overfit_flag, task)

            # --- Assemble result ---
            model_metrics[name] = {
                **test_metrics,
                "cv_mean": cv_mean,
                "cv_std": cv_std,
                "train_score": round(train_score, 4),
                "overfitting": overfit_flag,
                "training_time_seconds": train_time,
                "feature_importance": importance,
                "reasoning": reasoning,
            }

            if primary_score > best_score:
                best_score = primary_score
                best_model_name = name
                best_pipeline = pipeline

        except Exception as exc:
            model_metrics[name] = {"error": str(exc)}

    # ------------------------------------------------------------------
    # Comparison table (frontend-ready, flat rows)
    # ------------------------------------------------------------------
    comparison_table = _build_comparison_table(model_metrics, task, best_model_name)

    # ------------------------------------------------------------------
    # Feature importance from best model
    # ------------------------------------------------------------------
    best_feature_importance: list[dict] | None = None
    if best_model_name and "feature_importance" in model_metrics.get(best_model_name, {}):
        best_feature_importance = model_metrics[best_model_name]["feature_importance"]

    return {
        "task_type": task,
        "target_column": target,
        "n_samples": len(df),
        "n_features": X.shape[1],
        "class_imbalance_analysis": imbalance_info,
        "best_model": best_model_name,
        "best_score": round(best_score, 4),
        "primary_metric": primary_metric,
        "model_metrics": model_metrics,
        "comparison_table": comparison_table,
        "feature_importance": best_feature_importance,
    }


# ===========================================================================
# 9. COMPARISON TABLE BUILDER
# ===========================================================================

def _build_comparison_table(
    model_metrics: dict[str, dict],
    task: str,
    best_model_name: str | None,
) -> list[dict]:
    """
    Flatten model_metrics into a list of row dicts suitable for a UI table.
    Marks the best model with `is_best: True`.
    """
    rows = []
    for name, metrics in model_metrics.items():
        if "error" in metrics:
            rows.append({"model": name, "error": metrics["error"], "is_best": False})
            continue

        if task == "classification":
            row = {
                "model": name,
                "accuracy": metrics.get("accuracy"),
                "f1_weighted": metrics.get("f1_weighted"),
                "precision_weighted": metrics.get("precision_weighted"),
                "recall_weighted": metrics.get("recall_weighted"),
                "roc_auc": metrics.get("roc_auc"),
                "cv_mean_f1": metrics.get("cv_mean"),
                "cv_std": metrics.get("cv_std"),
                "overfitting": metrics.get("overfitting"),
                "training_time_seconds": metrics.get("training_time_seconds"),
                "is_best": name == best_model_name,
            }
        else:
            row = {
                "model": name,
                "r2_score": metrics.get("r2_score"),
                "rmse": metrics.get("rmse"),
                "mae": metrics.get("mae"),
                "mse": metrics.get("mse"),
                "cv_mean_r2": metrics.get("cv_mean"),
                "cv_std": metrics.get("cv_std"),
                "overfitting": metrics.get("overfitting"),
                "training_time_seconds": metrics.get("training_time_seconds"),
                "is_best": name == best_model_name,
            }
        rows.append(row)

    # Sort: best first, then by primary metric descending
    def _sort_key(r: dict) -> float:
        if r.get("is_best"):
            return 999.0
        if task == "classification":
            return float(r.get("f1_weighted") or 0)
        return float(r.get("r2_score") or -999)

    rows.sort(key=_sort_key, reverse=True)
    return rows


# ===========================================================================
# Smoke test
# ===========================================================================
if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer, load_diabetes

    # --- Classification ---
    bc = load_breast_cancer(as_frame=True)
    df_cls = bc.frame
    print("=== Classification (breast cancer) ===")
    result = auto_train(df_cls, target_column="target")
    print(f"Task: {result['task_type']}")
    print(f"Best model: {result['best_model']}  ({result['primary_metric']}={result['best_score']})")
    print("\nComparison table:")
    for row in result["comparison_table"]:
        star = " ★" if row.get("is_best") else ""
        print(
            f"  {row['model']:<22} F1={row.get('f1_weighted')!s:<6} "
            f"AUC={row.get('roc_auc')!s:<6} "
            f"Overfit={row.get('overfitting')}{star}"
        )

    if result["feature_importance"]:
        print("\nTop 5 features (best model):")
        for fi in result["feature_importance"][:5]:
            print(f"  {fi['feature']:<35} {fi['importance']:.4f}")

    # --- Regression ---
    diab = load_diabetes(as_frame=True)
    df_reg = diab.frame
    print("\n\n=== Regression (diabetes) ===")
    result_r = auto_train(df_reg, target_column="target")
    print(f"Task: {result_r['task_type']}")
    print(f"Best model: {result_r['best_model']}  ({result_r['primary_metric']}={result_r['best_score']})")
    print("\nComparison table:")
    for row in result_r["comparison_table"]:
        star = " ★" if row.get("is_best") else ""
        print(
            f"  {row['model']:<22} R²={row.get('r2_score')!s:<7} "
            f"RMSE={row.get('rmse')!s:<8} "
            f"Overfit={row.get('overfitting')}{star}"
        )