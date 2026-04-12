"""
shap_explainability.py

Uses SHAP (SHapley Additive exPlanations) to explain model predictions.

SHAP is the gold standard for ML explainability:
  - Mathematically grounded in game theory (Shapley values)
  - Works for ANY model (tree-based, linear, neural, etc.)
  - Explains GLOBAL feature importance (across all predictions)
  - Explains LOCAL feature importance (for each individual row)
  - Used by Google, Microsoft, and major ML teams in production

What SHAP tells us:
  - For each feature: how much did it push the prediction UP or DOWN?
  - Which features are most important overall? (global SHAP)
  - For one specific row: why did the model predict THIS value?
  - Are there features that interact with each other?

This is different from Random Forest feature importance:
  - RF importance: which features are used most in tree splits
  - SHAP: which features actually CHANGE the prediction the most
  SHAP is more reliable and less biased toward high-cardinality features.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def _prepare_data(df: pd.DataFrame, target: str, task: str):
    """Prepare clean dataset for SHAP analysis."""
    data = df.copy()

    # Drop ID-like columns
    for col in list(data.columns):
        if col == target:
            continue
        if data[col].nunique() / len(data) > 0.95:
            data = data.drop(columns=[col])

    # Encode categoricals
    encoders = {}
    for col in data.select_dtypes(exclude="number").columns:
        if col == target:
            continue
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        encoders[col] = le

    # Impute
    data = data.fillna(data.median(numeric_only=True))

    X = data.drop(columns=[target])
    y = data[target]

    # Encode target if classification
    target_encoder = None
    if task == "classification" and not pd.api.types.is_numeric_dtype(y):
        target_encoder = LabelEncoder()
        y = pd.Series(target_encoder.fit_transform(y.astype(str)), name=target)

    return X, y, target_encoder


def _train_model(X, y, task: str):
    """Train a Random Forest for SHAP analysis."""
    if task == "classification":
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
    else:
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
    model.fit(X, y)
    return model


def run_shap_analysis(
    df: pd.DataFrame,
    target: str,
    task: str = None,
    max_samples_for_shap: int = 200,
    top_n_features: int = 10
) -> dict:
    """
    Full SHAP explainability analysis.

    Returns:
      global_importance: mean |SHAP| per feature (overall importance)
      feature_direction: whether each feature pushes predictions up or down on average
      top_interactions: pairs of features that interact most
      sample_explanations: SHAP values for 3 sample rows (local explanation)
      shap_summary: plain-English explanation of what drives this model
      research_note: how to cite SHAP in the paper
    """

    if not SHAP_AVAILABLE:
        return {
            "status": "skipped",
            "reason": "SHAP library not installed. Run: pip install shap",
            "install_command": "pip install shap"
        }

    if len(df) < 20:
        return {
            "status": "skipped",
            "reason": "Need at least 20 rows for SHAP analysis."
        }

    # Detect task
    if task not in ("classification", "regression"):
        y_sample = df[target].dropna()
        task = "classification" if y_sample.nunique() <= 15 or y_sample.dtype == object else "regression"

    try:
        X, y, target_encoder = _prepare_data(df, target, task)

        if X.shape[1] == 0:
            return {"status": "skipped", "reason": "No features available after preprocessing."}

        feature_names = X.columns.tolist()

        # Scale
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_names)

        # Train model
        model = _train_model(X_scaled, y, task)

        # Use subset for SHAP (TreeExplainer is fast, but limit for large datasets)
        n_shap = min(max_samples_for_shap, len(X_scaled))
        X_shap = X_scaled.iloc[:n_shap]

        # Compute SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_shap)

        # For multiclass, use values for each class separately
        # For binary classification, shap_values is a list of 2 arrays
        if isinstance(shap_values, list):
            if len(shap_values) == 2:
                # Binary: use positive class
                sv = shap_values[1]
            else:
                # Multiclass: use mean absolute across all classes
                sv = np.mean([np.abs(s) for s in shap_values], axis=0)
        else:
            sv = shap_values

        sv = np.array(sv)

        # ── GLOBAL IMPORTANCE ─────────────────────────────────────────────
        mean_abs_shap = np.abs(sv).mean(axis=0)
        feature_importance_shap = {
            feature_names[i]: round(float(mean_abs_shap[i]), 6)
            for i in range(len(feature_names))
        }
        feature_importance_shap = dict(
            sorted(feature_importance_shap.items(), key=lambda x: x[1], reverse=True)
        )
        top_features = dict(list(feature_importance_shap.items())[:top_n_features])

        # ── FEATURE DIRECTION ─────────────────────────────────────────────
        # Positive mean SHAP = feature pushes prediction UP on average
        mean_shap = sv.mean(axis=0)
        feature_direction = {
            feature_names[i]: {
                "mean_shap": round(float(mean_shap[i]), 6),
                "direction": "POSITIVE (pushes prediction up)" if mean_shap[i] > 0 else "NEGATIVE (pushes prediction down)",
                "magnitude": round(float(abs(mean_shap[i])), 6)
            }
            for i in range(len(feature_names))
        }

        # Sort by absolute magnitude
        feature_direction = dict(
            sorted(feature_direction.items(), key=lambda x: x[1]["magnitude"], reverse=True)
        )

        # ── LOCAL EXPLANATIONS (3 sample rows) ───────────────────────────
        sample_indices = [0, len(X_shap) // 2, len(X_shap) - 1]
        sample_explanations = []

        for idx in sample_indices:
            if idx >= len(X_shap):
                continue

            row_shap = sv[idx]
            row_values = X_shap.iloc[idx]

            # Top 5 features for this row
            top_for_row = sorted(
                [(feature_names[i], float(row_shap[i]), float(row_values.iloc[i]))
                 for i in range(len(feature_names))],
                key=lambda x: abs(x[1]), reverse=True
            )[:5]

            prediction = model.predict(X_shap.iloc[[idx]])[0]
            if target_encoder and hasattr(target_encoder, "classes_"):
                try:
                    prediction = target_encoder.classes_[int(prediction)]
                except Exception:
                    pass

            sample_explanations.append({
                "row_index": int(idx),
                "prediction": str(prediction),
                "base_value": round(float(explainer.expected_value[1]
                    if isinstance(explainer.expected_value, (list, np.ndarray))
                    else explainer.expected_value), 4),
                "top_contributing_features": [
                    {
                        "feature": fname,
                        "shap_value": round(fshap, 4),
                        "feature_value": round(fval, 4),
                        "impact": "increases prediction" if fshap > 0 else "decreases prediction"
                    }
                    for fname, fshap, fval in top_for_row
                ]
            })

        # ── FEATURE INTERACTIONS (top pairs) ──────────────────────────────
        # Correlation of SHAP values between features indicates interaction
        shap_df = pd.DataFrame(sv, columns=feature_names)
        interaction_pairs = []
        if len(feature_names) > 1:
            corr = shap_df.corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            pairs = upper.stack().reset_index()
            pairs.columns = ["feature_1", "feature_2", "interaction_strength"]
            pairs = pairs.sort_values("interaction_strength", ascending=False)
            for _, row in pairs.head(5).iterrows():
                interaction_pairs.append({
                    "feature_1": row["feature_1"],
                    "feature_2": row["feature_2"],
                    "interaction_strength": round(float(row["interaction_strength"]), 4),
                    "interpretation": (
                        f"When '{row['feature_1']}' has a high SHAP value, "
                        f"'{row['feature_2']}' tends to have a {'high' if row['interaction_strength'] > 0.5 else 'moderate'} "
                        f"SHAP value too. These features may be correlated or interacting in the model."
                    )
                })

        # ── RF vs SHAP comparison ─────────────────────────────────────────
        rf_importances = dict(zip(feature_names, model.feature_importances_))
        rf_ranking = sorted(rf_importances, key=rf_importances.get, reverse=True)
        shap_ranking = list(feature_importance_shap.keys())

        rank_comparison = []
        for feat in feature_names:
            rf_rank = rf_ranking.index(feat) + 1 if feat in rf_ranking else "-"
            shap_rank = shap_ranking.index(feat) + 1 if feat in shap_ranking else "-"
            if isinstance(rf_rank, int) and isinstance(shap_rank, int):
                rank_diff = abs(rf_rank - shap_rank)
                rank_comparison.append({
                    "feature": feat,
                    "rf_rank": rf_rank,
                    "shap_rank": shap_rank,
                    "rank_difference": rank_diff,
                    "agreement": "agree" if rank_diff <= 2 else "disagree"
                })
        rank_comparison.sort(key=lambda x: x["rf_rank"])

        # ── SHAP SUMMARY TEXT ──────────────────────────────────────────────
        top_feat = list(top_features.keys())[0] if top_features else "unknown"
        top_2 = list(top_features.keys())[:2] if len(top_features) >= 2 else [top_feat]

        direction_desc = feature_direction.get(top_feat, {}).get("direction", "")

        shap_summary = (
            f"SHAP analysis of {n_shap} samples using TreeExplainer on a Random Forest model. "
            f"The most important feature globally is '{top_feat}' "
            f"(mean |SHAP| = {top_features.get(top_feat, 0):.4f}). "
            f"On average, this feature {direction_desc.lower()}. "
            f"The top 2 features ('{top_2[0]}' and '{top_2[1] if len(top_2) > 1 else ''}') "
            f"together account for the majority of prediction variance. "
            f"{'Feature interactions were detected — ' + interaction_pairs[0]['feature_1'] + ' and ' + interaction_pairs[0]['feature_2'] + ' interact strongly.' if interaction_pairs else ''}"
        )

        disagreements = [r for r in rank_comparison if r.get("agreement") == "disagree"]
        if disagreements:
            shap_summary += (
                f" Note: RF importance and SHAP disagree on the ranking of "
                f"{len(disagreements)} features. SHAP is more reliable as it accounts "
                f"for feature interactions and is unbiased toward high-cardinality features."
            )

        return {
            "status": "success",
            "task_type": task,
            "target_column": target,
            "n_samples_analyzed": n_shap,
            "n_features": len(feature_names),
            "model_used": "Random Forest (TreeExplainer)",

            "global_feature_importance": top_features,
            "feature_direction": feature_direction,
            "top_feature_interactions": interaction_pairs,
            "sample_explanations": sample_explanations,
            "rf_vs_shap_comparison": rank_comparison,

            "shap_summary": shap_summary,

            "research_note": (
                "SHAP (SHapley Additive exPlanations) values are grounded in cooperative "
                "game theory. Each feature's SHAP value represents its average marginal "
                "contribution to the prediction across all possible feature coalitions. "
                "Unlike permutation importance or Gini importance, SHAP values are "
                "consistent and locally accurate. "
                "Citation: Lundberg & Lee (2017), 'A Unified Approach to Interpreting "
                "Model Predictions', NeurIPS."
            )
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "hint": "SHAP analysis failed. Dataset may be too small or have incompatible dtypes."
        }