import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import LabelEncoder


def _detect_task_type(y: pd.Series) -> str:
    if y.dtype == object or y.dtype == bool:
        return "classification"
    if y.nunique() <= 15:
        return "classification"
    return "regression"


def _overfitting_reason(name: str, gap: float, cv_std: float,
                         lc_gap: float, train_score: float, test_score: float) -> str:
    """Generate plain-English explanation of overfitting diagnosis."""

    reasons = []

    if gap > 0.15:
        reasons.append(
            f"large train/test gap ({train_score:.2f} train vs {test_score:.2f} test) "
            "suggests model memorized training data"
        )
    if cv_std > 0.08:
        reasons.append(
            f"high CV variance (std={cv_std:.3f}) means performance is unstable "
            "across different data splits"
        )
    if lc_gap > 0.15:
        reasons.append(
            f"learning curve gap ({lc_gap:.2f}) does not close with more data — "
            "classic overfitting signature"
        )

    if not reasons:
        if test_score > 0.85:
            return f"{name} generalizes well — train/test gap is small and CV is stable."
        else:
            return (
                f"{name} is stable but underperforms (test={test_score:.2f}). "
                "May be underfitting — try more complex models or better features."
            )

    suggestion_map = {
        "Random Forest": "Reduce n_estimators or max_depth, or increase min_samples_leaf.",
        "Gradient Boosting": "Reduce learning_rate, increase min_samples_leaf, or add subsample < 1.0.",
        "Logistic Regression": "Increase regularization strength (reduce C parameter).",
        "Ridge Regression": "Increase alpha (stronger regularization).",
    }

    suggestion = suggestion_map.get(name, "Try regularization or reduce model complexity.")
    return f"{name} shows overfitting: {'; '.join(reasons)}. Recommendation: {suggestion}"


def _overall_risk(results: dict) -> dict:
    """Aggregate per-model risks into a dataset-level verdict."""
    risk_levels = [r["overfitting_risk"] for r in results.values()]

    if "HIGH" in risk_levels:
        overall = "HIGH"
        message = (
            "At least one model shows strong overfitting. "
            "Dataset likely has too many features relative to samples, "
            "or contains noisy/leaking features."
        )
    elif risk_levels.count("MODERATE") >= 2:
        overall = "MODERATE"
        message = (
            "Multiple models show moderate overfitting. "
            "Consider feature selection, regularization, or collecting more data."
        )
    elif "MODERATE" in risk_levels:
        overall = "LOW-MODERATE"
        message = "Minor overfitting in some models. Dataset is reasonably clean."
    else:
        overall = "LOW"
        message = "All models generalize well. Dataset appears ML-ready."

    return {"risk": overall, "message": message}


def detect_overfitting(df: pd.DataFrame, target_column: str) -> dict:
    """
    Research-level overfitting detection using 3 complementary methods:

    1. Train/test gap        — direct score comparison on held-out data
    2. CV variance           — stability across 5 folds (high std = unstable)
    3. Learning curve        — does performance converge with more data?
                               Non-converging gap = classic overfitting signature

    Multiple models tested — Random Forest alone is misleading
    (it always shows high train score by design).

    Each model gets:
    - per-method scores
    - risk level (LOW / MODERATE / HIGH)
    - plain-English reasoning + specific fix recommendation

    Returns overall dataset-level overfitting risk.
    """

    try:
        if target_column not in df.columns:
            return {"error": f"Target column '{target_column}' not found."}

        df = df.dropna()

        if len(df) < 20:
            return {"error": "Not enough rows for overfitting analysis (need at least 20)."}

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

        # ── MODEL SELECTION ───────────────────────────────────────────────
        if task == "classification":
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
            }
            scoring = "accuracy"
        else:
            models = {
                "Ridge Regression": Ridge(alpha=1.0),
                "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            }
            scoring = "r2"

        # ── TRAIN/TEST SPLIT ──────────────────────────────────────────────
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        results = {}

        for name, model in models.items():

            try:
                # ── METHOD 1: TRAIN/TEST GAP ──────────────────────────────
                model.fit(X_train, y_train)
                train_score = float(model.score(X_train, y_train))
                test_score = float(model.score(X_test, y_test))
                gap = abs(train_score - test_score)

                # ── METHOD 2: CROSS-VALIDATION VARIANCE ───────────────────
                cv_scores = cross_val_score(model, X, y, cv=5, scoring=scoring)
                cv_mean = float(cv_scores.mean())
                cv_std = float(cv_scores.std())

                # ── METHOD 3: LEARNING CURVE ──────────────────────────────
                # Train on increasing data sizes and check if gap closes
                train_sizes_pct = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
                lc_train_sizes, lc_train_scores, lc_test_scores = learning_curve(
                    model, X, y,
                    train_sizes=train_sizes_pct,
                    cv=5,
                    scoring=scoring,
                    n_jobs=-1
                )

                # Take mean across CV folds at each training size
                lc_train_means = lc_train_scores.mean(axis=1).tolist()
                lc_test_means = lc_test_scores.mean(axis=1).tolist()

                # End point = model trained on full data
                lc_train_end = lc_train_means[-1]
                lc_test_end = lc_test_means[-1]
                lc_gap = abs(lc_train_end - lc_test_end)

                # Convergence check: is the gap closing as data increases?
                early_gap = abs(lc_train_means[0] - lc_test_means[0])
                converging = lc_gap < early_gap * 0.7  # gap reduced by 30%+ = converging

                # ── RISK LEVEL ────────────────────────────────────────────
                if gap > 0.15 or (cv_std > 0.08 and lc_gap > 0.15):
                    risk = "HIGH"
                elif gap > 0.05 or cv_std > 0.05 or lc_gap > 0.08:
                    risk = "MODERATE"
                else:
                    risk = "LOW"

                # ── REASONING ─────────────────────────────────────────────
                reason = _overfitting_reason(
                    name, gap, cv_std, lc_gap, train_score, test_score
                )

                results[name] = {
                    "train_score": round(train_score, 4),
                    "test_score": round(test_score, 4),
                    "train_test_gap": round(gap, 4),
                    "cv_mean": round(cv_mean, 4),
                    "cv_std": round(cv_std, 4),
                    "learning_curve": {
                        "train_sizes_percent": [int(s * 100) for s in train_sizes_pct],
                        "train_scores": [round(s, 4) for s in lc_train_means],
                        "test_scores": [round(s, 4) for s in lc_test_means],
                        "final_gap": round(lc_gap, 4),
                        "is_converging": converging
                    },
                    "overfitting_risk": risk,
                    "reason": reason
                }

            except Exception as e:
                results[name] = {"error": str(e)}

        return {
            "task": task,
            "scoring_metric": scoring,
            "per_model_analysis": results,
            "overall_overfitting": _overall_risk(results)
        }

    except Exception as e:
        return {"error": str(e)}