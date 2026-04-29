"""
shap_explainability.py  — place in backend/app/ml/shap_explainability.py

Uses SHAP (SHapley Additive exPlanations) to explain model predictions.
Run:  pip install shap
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


def _prepare_data(df, target, task):
    data = df.copy()
    for col in list(data.columns):
        if col == target: continue
        if data[col].nunique() / len(data) > 0.95:
            data = data.drop(columns=[col])
    encoders = {}
    for col in data.select_dtypes(exclude="number").columns:
        if col == target: continue
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        encoders[col] = le
    data = data.fillna(data.median(numeric_only=True))
    X = data.drop(columns=[target])
    y = data[target]
    target_encoder = None
    if task == "classification" and not pd.api.types.is_numeric_dtype(y):
        target_encoder = LabelEncoder()
        y = pd.Series(target_encoder.fit_transform(y.astype(str)), name=target)
    return X, y, target_encoder


def run_shap_analysis(df, target, task=None, max_samples_for_shap=200, top_n_features=10):
    if not SHAP_AVAILABLE:
        return {"status": "skipped", "reason": "pip install shap"}
    if len(df) < 20:
        return {"status": "skipped", "reason": "Need ≥ 20 rows."}
    if task not in ("classification", "regression"):
        y_s = df[target].dropna()
        task = "classification" if y_s.nunique() <= 15 or y_s.dtype == object else "regression"
    try:
        X, y, target_encoder = _prepare_data(df, target, task)
        if X.shape[1] == 0:
            return {"status": "skipped", "reason": "No features."}
        feature_names = X.columns.tolist()
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_names)
        model = (RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
                 if task == "classification" else
                 RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1))
        model.fit(X_scaled, y)
        n_shap = min(max_samples_for_shap, len(X_scaled))
        X_shap = X_scaled.iloc[:n_shap]
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_shap)
        if isinstance(shap_values, list):
            sv = shap_values[1] if len(shap_values) == 2 else np.mean([np.abs(s) for s in shap_values], axis=0)
        else:
            sv = shap_values
        sv = np.array(sv)
        mean_abs_shap = np.abs(sv).mean(axis=0)
        feature_importance_shap = dict(sorted(
            {feature_names[i]: round(float(mean_abs_shap[i]), 6) for i in range(len(feature_names))}.items(),
            key=lambda x: -x[1]
        ))
        top_features = dict(list(feature_importance_shap.items())[:top_n_features])
        mean_shap = sv.mean(axis=0)
        feature_direction = {
            feature_names[i]: {
                "mean_shap": round(float(mean_shap[i]), 6),
                "direction": "POSITIVE (pushes prediction up)" if mean_shap[i] > 0 else "NEGATIVE (pushes prediction down)",
                "magnitude": round(float(abs(mean_shap[i])), 6)
            }
            for i in range(len(feature_names))
        }
        sample_indices = [0, len(X_shap) // 2, len(X_shap) - 1]
        sample_explanations = []
        for idx in sample_indices:
            if idx >= len(X_shap): continue
            row_shap = sv[idx]
            row_values = X_shap.iloc[idx]
            top5 = sorted(
                [(feature_names[i], float(row_shap[i]), float(row_values.iloc[i])) for i in range(len(feature_names))],
                key=lambda x: abs(x[1]), reverse=True
            )[:5]
            pred = model.predict(X_shap.iloc[[idx]])[0]
            if target_encoder and hasattr(target_encoder, "classes_"):
                try: pred = target_encoder.classes_[int(pred)]
                except: pass
            sample_explanations.append({
                "row_index": int(idx),
                "prediction": str(pred),
                "top_contributing_features": [
                    {"feature": f, "shap_value": round(s, 4), "feature_value": round(v, 4),
                     "impact": "increases prediction" if s > 0 else "decreases prediction"}
                    for f, s, v in top5
                ]
            })
        shap_df = pd.DataFrame(sv, columns=feature_names)
        corr = shap_df.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        pairs = upper.stack().reset_index()
        pairs.columns = ["f1", "f2", "strength"]
        top_interactions = [
            {"feature_1": r.f1, "feature_2": r.f2, "interaction_strength": round(float(r.strength), 4)}
            for _, r in pairs.nlargest(5, "strength").iterrows()
        ]
        top_feat = list(top_features.keys())[0] if top_features else "N/A"
        top2 = list(top_features.keys())[:2]
        return {
            "status": "success",
            "task_type": task,
            "target_column": target,
            "n_samples_analyzed": n_shap,
            "n_features": len(feature_names),
            "model_used": "Random Forest (TreeExplainer)",
            "global_feature_importance": top_features,
            "feature_direction": feature_direction,
            "top_feature_interactions": top_interactions,
            "sample_explanations": sample_explanations,
            "shap_summary": (
                f"SHAP analysis of {n_shap} samples. Top feature: '{top_feat}' "
                f"(mean |SHAP|={top_features.get(top_feat,0):.4f}). "
                f"Top 2: '{top2[0]}' and '{top2[1] if len(top2)>1 else ''}'. "
                f"{'Interactions: '+top_interactions[0]['feature_1']+' ↔ '+top_interactions[0]['feature_2'] if top_interactions else ''}"
            ),
            "research_note": (
                "SHAP (SHapley Additive exPlanations) — cooperative game theory. "
                "Citation: Lundberg & Lee (2017), NeurIPS."
            )
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}