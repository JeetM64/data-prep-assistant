import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, r2_score, mean_squared_error, mean_absolute_error
)

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier


def _detect_task_type(y: pd.Series) -> str:
    """
    Determine if target is classification or regression.
    - Object/bool dtype → classification
    - Numeric with ≤15 unique values → classification (ordinal/discrete)
    - Numeric with >15 unique values → regression
    """
    if y.dtype == object or y.dtype == bool:
        return "classification"
    if y.nunique() <= 15:
        return "classification"
    return "regression"


def _detect_class_imbalance(y: pd.Series) -> dict:
    """
    Detect class imbalance in classification target.
    Imbalance ratio = majority class count / minority class count.
    """
    counts = y.value_counts()
    ratio = counts.max() / counts.min()
    is_imbalanced = ratio > 3.0

    return {
        "is_imbalanced": bool(is_imbalanced),
        "imbalance_ratio": round(float(ratio), 2),
        "class_distribution": counts.to_dict(),
        "recommendation": (
            "Class imbalance detected — using class_weight='balanced' in models. "
            "Consider SMOTE oversampling for severe imbalance (ratio > 10)."
            if is_imbalanced else "Classes are balanced."
        )
    }


def auto_train(df, target_column: str = None):
    """
    Research-level automated model training and comparison.

    Features:
    - Auto task detection (classification vs regression)
    - Class imbalance detection and handling (balanced weights)
    - Stratified K-Fold cross validation for classification
    - Per-model reasoning (why it performed well/poorly)
    - AUC-ROC for binary classification
    - Comprehensive metrics per task type
    - Best model selection with justification

    Args:
        df: cleaned dataframe (no raw missing values)
        target_column: target column name (uses last column if None)
    """

    df = df.dropna()

    if df.shape[1] < 2:
        return {"error": "Dataset must have at least one feature and one target column."}

    # ── TARGET SETUP ─────────────────────────────────────────────────────
    target = target_column if target_column and target_column in df.columns else df.columns[-1]
    X = df.drop(columns=[target])
    y = df[target]

    if len(y.unique()) < 2:
        return {"error": "Target column has only one unique value — cannot train."}

    # ── TASK TYPE + IMBALANCE ────────────────────────────────────────────
    task = _detect_task_type(y)
    imbalance_info = _detect_class_imbalance(y) if task == "classification" else None

    # Encode target if classification with string labels
    le = None
    if task == "classification" and y.dtype == object:
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), name=target)

    # ── PREPROCESSOR ─────────────────────────────────────────────────────
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
    ])

    # ── TRAIN/TEST SPLIT ─────────────────────────────────────────────────
    stratify = y if task == "classification" and y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    # ── CROSS VALIDATION STRATEGY ────────────────────────────────────────
    if task == "classification":
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    else:
        cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)

    # ── MODEL SELECTION ──────────────────────────────────────────────────
    is_imbalanced = imbalance_info["is_imbalanced"] if imbalance_info else False

    if task == "classification":
        cw = "balanced" if is_imbalanced else None
        models = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000, class_weight=cw, random_state=42
            ),
            "Decision Tree": DecisionTreeClassifier(
                max_depth=10, class_weight=cw, random_state=42
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100, class_weight=cw, random_state=42
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=100, random_state=42
            ),
            "SVM": SVC(
                probability=True, class_weight=cw, random_state=42
            ),
        }
        primary_metric = "f1_weighted"

    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=1.0),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
            if False else RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42),
            "SVR": SVR(kernel="rbf"),
        }
        primary_metric = "r2"

    # ── TRAINING LOOP ─────────────────────────────────────────────────────
    results = {}
    best_model_name = None
    best_score = -999

    for name, model in models.items():

        pipe = Pipeline([
            ("preprocessing", preprocessor),
            ("model", model)
        ])

        try:
            start_time = time.time()
            pipe.fit(X_train, y_train)
            training_time = round(time.time() - start_time, 3)

            preds = pipe.predict(X_test)

            if task == "classification":

                acc = accuracy_score(y_test, preds)
                precision = precision_score(y_test, preds, average="weighted", zero_division=0)
                recall = recall_score(y_test, preds, average="weighted", zero_division=0)
                f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

                # AUC-ROC (binary only)
                auc = None
                if y.nunique() == 2 and hasattr(pipe, "predict_proba"):
                    try:
                        proba = pipe.predict_proba(X_test)[:, 1]
                        auc = round(float(roc_auc_score(y_test, proba)), 3)
                    except Exception:
                        auc = None

                cv_scores = cross_val_score(pipe, X, y, cv=cv_strategy, scoring="f1_weighted")
                cv_mean = round(float(cv_scores.mean()), 3)
                cv_std = round(float(cv_scores.std()), 3)

                score = f1

                results[name] = {
                    "accuracy": round(acc, 3),
                    "precision_weighted": round(precision, 3),
                    "recall_weighted": round(recall, 3),
                    "f1_weighted": round(f1, 3),
                    "auc_roc": auc,
                    "cv_mean_f1": cv_mean,
                    "cv_std": cv_std,
                    "training_time_seconds": training_time,
                    "reasoning": _classify_model_reasoning(name, f1, cv_mean, cv_std)
                }

            else:

                r2 = r2_score(y_test, preds)
                mse = mean_squared_error(y_test, preds)
                rmse = float(np.sqrt(mse))
                mae = mean_absolute_error(y_test, preds)

                cv_scores = cross_val_score(pipe, X, y, cv=cv_strategy, scoring="r2")
                cv_mean = round(float(cv_scores.mean()), 3)
                cv_std = round(float(cv_scores.std()), 3)

                score = r2

                results[name] = {
                    "r2_score": round(r2, 3),
                    "mse": round(mse, 4),
                    "rmse": round(rmse, 4),
                    "mae": round(mae, 4),
                    "cv_mean_r2": cv_mean,
                    "cv_std": cv_std,
                    "training_time_seconds": training_time,
                    "reasoning": _regress_model_reasoning(name, r2, cv_mean, cv_std)
                }

            if score > best_score:
                best_score = score
                best_model_name = name

        except Exception as e:
            results[name] = {"error": str(e)}

    return {
        "task_type": task,
        "target_column": target,
        "best_model": best_model_name,
        "best_score": round(best_score, 3),
        "primary_metric": primary_metric,
        "class_imbalance_analysis": imbalance_info,
        "model_comparison": results
    }


def _classify_model_reasoning(name: str, f1: float, cv_mean: float, cv_std: float) -> str:
    """Generate plain-English reasoning for classification model performance."""
    overfit = (f1 - cv_mean) > 0.1
    unstable = cv_std > 0.08

    if overfit and unstable:
        return f"{name} shows signs of overfitting (test F1 {f1:.2f} vs CV {cv_mean:.2f}) and unstable CV (std={cv_std:.2f}). Consider regularization."
    elif overfit:
        return f"{name} may be overfitting — test F1 ({f1:.2f}) is significantly higher than CV mean ({cv_mean:.2f})."
    elif unstable:
        return f"{name} has unstable cross-validation (std={cv_std:.2f}) — performance varies across folds. Check data distribution."
    elif f1 > 0.85:
        return f"{name} performs strongly (F1={f1:.2f}, CV={cv_mean:.2f}). Stable and reliable."
    elif f1 > 0.70:
        return f"{name} performs adequately (F1={f1:.2f}). May benefit from hyperparameter tuning."
    else:
        return f"{name} underperforms (F1={f1:.2f}). Dataset may need better preprocessing or more features."


def _regress_model_reasoning(name: str, r2: float, cv_mean: float, cv_std: float) -> str:
    """Generate plain-English reasoning for regression model performance."""
    overfit = (r2 - cv_mean) > 0.1
    unstable = cv_std > 0.1

    if overfit:
        return f"{name} shows overfitting — test R² ({r2:.2f}) much higher than CV R² ({cv_mean:.2f}). Try regularization."
    elif unstable:
        return f"{name} is unstable across folds (CV std={cv_std:.2f}). Check for outliers or distribution shift."
    elif r2 > 0.85:
        return f"{name} explains {r2:.0%} of variance. Excellent fit."
    elif r2 > 0.60:
        return f"{name} explains {r2:.0%} of variance. Reasonable — consider feature engineering."
    else:
        return f"{name} explains only {r2:.0%} of variance. Poor fit — review features and target distribution."