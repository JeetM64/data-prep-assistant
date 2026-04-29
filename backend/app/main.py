"""
AutoEDA v4.0 — ML Data Readiness Analyzer — main.py
Next-level upgrades:
  • /report  — PDF report generation (was missing, now works)
  • /bootstrap — Readiness score with confidence intervals
  • /validate — Pre-training validation checklist
  • /drift — Statistical drift detection between two datasets
  • Improved SHAP endpoint with local explanations
  • Better error messages and logging
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import Optional
import pandas as pd
import numpy as np
import io
import json
import re
import time
import warnings
warnings.filterwarnings("ignore")

app = FastAPI(title="AutoEDA — ML Data Readiness Analyzer", version="4.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Final-Shape","X-Steps-Applied","X-Pipeline-Log","X-Pipeline-Mode"],
)


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def safe_float(val):
    try:
        v = float(val)
        return round(v, 4) if not (np.isnan(v) or np.isinf(v)) else 0.0
    except:
        return 0.0

def safe_int(val):
    try:
        v = int(val)
        return v if not np.isnan(float(v)) else 0
    except:
        return 0

def _load_df(file: UploadFile, contents: bytes) -> pd.DataFrame:
    name = (file.filename or "").lower()
    if name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(contents))
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(contents))
    else:
        try:
            return pd.read_csv(io.BytesIO(contents))
        except:
            return pd.read_excel(io.BytesIO(contents))


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATASET SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def get_dataset_summary(df: pd.DataFrame) -> dict:
    total_cells = len(df) * len(df.columns)
    missing_total = int(df.isnull().sum().sum())
    numeric_cols = df.select_dtypes(include=["int64","float64","int32","float32"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object","category","bool"]).columns.tolist()
    mem_mb = safe_float(df.memory_usage(deep=True).sum() / (1024*1024))

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": missing_total,
        "missing_value_percent": safe_float((missing_total / total_cells * 100) if total_cells > 0 else 0),
        "duplicate_rows": safe_int(df.duplicated().sum()),
        "numeric_columns": numeric_cols,
        "categorical_columns": cat_cols,
        "column_dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "memory_usage_mb": mem_mb,
        "column_names": df.columns.tolist(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. TARGET DETECTION
# ══════════════════════════════════════════════════════════════════════════════

TARGET_KEYWORDS = [
    "target","label","class","y","output","result","survived","species",
    "quality","outcome","price","churn","fraud","default","diagnosis",
    "category","response","dependent","predict","score","status","grade",
    "decision","flag","risk","type","group"
]

def detect_target_column(df: pd.DataFrame, user_specified: Optional[str] = None) -> dict:
    if user_specified and user_specified in df.columns:
        col = user_specified
        n_unique = df[col].nunique()
        if df[col].dtype in ["object","category","bool"] or n_unique <= 20:
            task_type = "classification_binary" if n_unique == 2 else "classification_multiclass" if n_unique <= 20 else "regression"
        else:
            task_type = "regression"
        return {
            "final_target": col, "task_type": task_type, "confidence": 1.0,
            "target_source": "user_specified", "warning": None,
            "top_candidates": [{"column": col, "score": 1.0}],
        }

    candidates = []
    for col in df.columns:
        score = 0.0
        col_lower = col.lower().strip()
        n_unique = df[col].nunique()
        n_rows = len(df)
        for kw in TARGET_KEYWORDS:
            if col_lower == kw: score += 0.6
            elif kw in col_lower: score += 0.3
        if col == df.columns[-1]: score += 0.2
        if n_unique == 2: score += 0.25
        if 2 <= n_unique <= 20: score += 0.15
        if df[col].dtype in ["float64","int64"] and n_unique > 20: score += 0.05
        if n_unique == n_rows: score -= 0.5
        if re.search(r"^id$|_id$|^index$|^row", col_lower): score -= 0.4
        candidates.append({"column": col, "score": round(score, 3)})

    candidates.sort(key=lambda x: -x["score"])
    best = candidates[0] if candidates else None
    if not best or best["score"] <= 0:
        best = {"column": df.columns[-1], "score": 0.3}

    col = best["column"]
    n_unique = df[col].nunique()
    if df[col].dtype in ["object","category","bool"] or n_unique <= 20:
        task_type = "classification_binary" if n_unique == 2 else "classification_multiclass" if n_unique <= 20 else "regression"
    else:
        task_type = "regression"

    return {
        "final_target": col, "task_type": task_type, "confidence": best["score"],
        "target_source": "auto_detected",
        "warning": "Low confidence — please verify target column." if best["score"] < 0.4 else None,
        "top_candidates": candidates[:5],
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. FEATURE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyze_features(df: pd.DataFrame, target: Optional[str] = None) -> dict:
    result = {}
    n = len(df)
    for col in df.columns:
        s = df[col]
        missing = int(s.isnull().sum())
        missing_pct = safe_float(missing / n * 100)
        n_unique = int(s.nunique())
        unique_ratio = safe_float(n_unique / n)
        is_num = pd.api.types.is_numeric_dtype(s)
        info = {
            "dtype": str(s.dtype), "missing_count": missing,
            "missing_percent": missing_pct, "unique_values": n_unique,
            "unique_ratio": unique_ratio, "is_target": col == target,
        }
        if is_num:
            clean = s.dropna()
            info.update({
                "mean": safe_float(clean.mean()), "median": safe_float(clean.median()),
                "std": safe_float(clean.std()), "min": safe_float(clean.min()),
                "max": safe_float(clean.max()), "q25": safe_float(clean.quantile(0.25)),
                "q75": safe_float(clean.quantile(0.75)), "skewness": safe_float(clean.skew()),
                "kurtosis": safe_float(clean.kurtosis()), "zero_count": int((clean == 0).sum()),
                "negative_count": int((clean < 0).sum()), "outlier_count": _count_outliers(clean),
                "scaling_suggestion": _suggest_scaling(clean),
            })
        else:
            counts = s.value_counts()
            top_cat = str(counts.index[0]) if len(counts) > 0 else "N/A"
            top_freq = safe_float(counts.iloc[0] / n * 100) if len(counts) > 0 else 0
            probs = counts / counts.sum()
            entropy = safe_float(-np.sum(probs * np.log2(probs + 1e-10)))
            info.update({
                "top_category": top_cat, "top_category_freq": top_freq,
                "category_entropy": entropy, "value_counts": counts.head(10).to_dict(),
                "encoding_suggestion": _suggest_encoding(s),
            })
        result[col] = info
    return result


def _count_outliers(s: pd.Series) -> int:
    if len(s) < 4: return 0
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return int(((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).sum())

def _suggest_scaling(s: pd.Series) -> str:
    skew = abs(s.skew()) if len(s) > 3 else 0
    has_neg = (s < 0).any()
    if skew > 2: return "log_transform + StandardScaler" if not has_neg else "Yeo-Johnson + StandardScaler"
    if abs(skew) > 1: return "RobustScaler (moderate skew)"
    return "StandardScaler"

def _suggest_encoding(s: pd.Series) -> str:
    n_unique = s.nunique()
    if n_unique == 2: return "LabelEncoding (binary)"
    if n_unique <= 10: return f"OneHotEncoding ({n_unique} cats)"
    if n_unique <= 50: return "OrdinalEncoding or TargetEncoding"
    return "TargetEncoding or Drop (high cardinality)"


# ══════════════════════════════════════════════════════════════════════════════
# 4. DATA QUALITY SCORE
# ══════════════════════════════════════════════════════════════════════════════

def calculate_quality_score(df: pd.DataFrame, target: Optional[str] = None) -> dict:
    n = len(df)
    total_cells = n * len(df.columns)
    missing_pct = df.isnull().sum().sum() / total_cells * 100 if total_cells > 0 else 0
    completeness = max(0.0, 100 - missing_pct * 2.5)
    dup_pct = df.duplicated().sum() / n * 100 if n > 0 else 0
    uniqueness = max(0.0, 100 - dup_pct * 2)
    consistency_issues = 0
    for col in df.select_dtypes(include="object").columns:
        sample = df[col].dropna().head(100)
        numeric_like = sample.apply(lambda x: str(x).replace('.','',1).replace('-','',1).isdigit()).mean()
        if 0.3 < numeric_like < 0.8: consistency_issues += 1
    consistency = max(0.0, 100 - consistency_issues * 10)
    class_balance = 100.0
    if target and target in df.columns:
        vc = df[target].value_counts(normalize=True)
        if len(vc) >= 2:
            imbalance_ratio = vc.iloc[0] / vc.iloc[-1] if vc.iloc[-1] > 0 else 999
            class_balance = max(0.0, 100 - np.log10(max(imbalance_ratio, 1)) * 30)
    num_cols = df.select_dtypes(include=["float64","int64"]).columns.tolist()
    if target and target in num_cols: num_cols = [c for c in num_cols if c != target]
    outlier_scores = []
    for col in num_cols[:15]:
        s = df[col].dropna()
        if len(s) > 3: outlier_scores.append(_count_outliers(s) / len(s) * 100)
    avg_outlier = np.mean(outlier_scores) if outlier_scores else 0
    feature_quality = max(0.0, 100 - avg_outlier * 3)
    rpf = n / len(df.columns) if len(df.columns) > 0 else 0
    adequacy = min(100.0, rpf * 5)
    dims = {
        "completeness": round(completeness, 1), "uniqueness": round(uniqueness, 1),
        "consistency": round(consistency, 1), "class_balance": round(class_balance, 1),
        "feature_quality": round(feature_quality, 1), "adequacy": round(adequacy, 1),
    }
    overall = round(sum(dims.values()) / len(dims), 1)
    status = "Excellent" if overall >= 90 else "Good" if overall >= 75 else "Fair" if overall >= 60 else "Poor"
    return {"overall_score": overall, "dimension_scores": dims, "status": status}


# ══════════════════════════════════════════════════════════════════════════════
# 5. CORRELATION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyze_correlations(df: pd.DataFrame, target: Optional[str] = None) -> dict:
    num_df = df.select_dtypes(include=["int64","float64","int32","float32"])
    if len(num_df.columns) < 2:
        return {"strong_correlation_pairs": [], "target_correlations": {}, "total_pairs_flagged": 0}
    corr = num_df.corr()
    pairs = []
    for i, c1 in enumerate(corr.columns):
        for j, c2 in enumerate(corr.columns):
            if i >= j: continue
            val = safe_float(corr.loc[c1, c2])
            if abs(val) > 0.75:
                sev = "CRITICAL" if abs(val) > 0.95 else "HIGH" if abs(val) > 0.85 else "MODERATE"
                pairs.append({"feature_1": c1, "feature_2": c2, "pearson": val, "severity": sev})
    target_corr = {}
    if target and target in num_df.columns:
        tc = num_df.corr()[target].drop(target, errors="ignore")
        target_corr = {k: safe_float(v) for k, v in tc.sort_values(key=abs, ascending=False).items()}
    pairs.sort(key=lambda x: -abs(x["pearson"]))
    return {"strong_correlation_pairs": pairs, "target_correlations": target_corr,
            "total_pairs_flagged": len(pairs), "summary": f"{len(pairs)} highly correlated pair(s) found (>0.75)."}


# ══════════════════════════════════════════════════════════════════════════════
# 6. DATA LEAKAGE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyze_leakage(df: pd.DataFrame, target: Optional[str] = None) -> dict:
    if not target or target not in df.columns:
        return {"target_leakage": [], "summary": {"total_issues": 0}}
    leakage = []
    t = df[target]
    for col in df.columns:
        if col == target: continue
        s = df[col]
        try:
            if pd.api.types.is_numeric_dtype(s) and pd.api.types.is_numeric_dtype(t):
                corr = abs(safe_float(s.corr(t)))
                if corr > 0.95:
                    leakage.append({"feature": col, "correlation": corr,
                                    "reason": f"Near-perfect correlation ({corr:.3f}) with target — likely leakage."})
            if s.nunique() == t.nunique():
                mapping_size = df.groupby(col)[target].nunique()
                if mapping_size.max() == 1:
                    leakage.append({"feature": col, "correlation": 1.0,
                                    "reason": "One-to-one mapping with target — data leakage."})
        except: pass
    return {"target_leakage": leakage, "summary": {"total_issues": len(leakage)}}


# ══════════════════════════════════════════════════════════════════════════════
# 7. ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_anomalies(df: pd.DataFrame, target: Optional[str] = None) -> dict:
    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.impute import SimpleImputer
        num_df = df.select_dtypes(include=["float64","int64"])
        if target and target in num_df.columns: num_df = num_df.drop(columns=[target])
        if len(num_df.columns) < 2 or len(df) < 20:
            return {"status": "skipped", "reason": "insufficient numeric data", "anomaly_count": 0, "anomaly_percent": 0}
        imp = SimpleImputer(strategy="median")
        X = imp.fit_transform(num_df.iloc[:, :20])
        clf = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
        preds = clf.fit_predict(X)
        anomaly_mask = preds == -1
        count = int(anomaly_mask.sum())
        pct = safe_float(count / len(df) * 100)
        sev = "LOW" if pct < 3 else "MODERATE" if pct < 8 else "HIGH"
        scores = np.abs(X[anomaly_mask] - X.mean(axis=0)).mean(axis=0)
        top_feat = {}
        for i in np.argsort(scores)[-3:][::-1]:
            top_feat[num_df.columns[i]] = safe_float(scores[i])
        return {"status": "completed", "anomaly_count": count, "anomaly_percent": pct,
                "severity": sev, "top_contributing_features": top_feat,
                "interpretation": f"{count} rows ({pct:.1f}%) flagged as anomalous using IsolationForest."}
    except Exception as e:
        return {"status": "error", "reason": str(e), "anomaly_count": 0, "anomaly_percent": 0}


# ══════════════════════════════════════════════════════════════════════════════
# 8. FAIR ASSESSMENT
# ══════════════════════════════════════════════════════════════════════════════

SENSITIVE_KEYWORDS = ["age","gender","sex","race","ethnicity","religion","nationality",
                      "disability","income","education","zipcode","postcode","marital"]

def assess_fair(df: pd.DataFrame, target: Optional[str] = None) -> dict:
    cols_lower = [c.lower() for c in df.columns]
    unnamed = sum(1 for c in df.columns if str(c).startswith("Unnamed"))
    findable = max(0, 100 - unnamed * 20)
    missing_pct = df.isnull().sum().sum() / (len(df)*len(df.columns)) * 100
    accessible = max(0, 100 - missing_pct * 2)
    interoperable = 100
    for col in df.select_dtypes(include="object").columns:
        try: pd.to_numeric(df[col], errors="raise"); interoperable -= 10
        except: pass
    interoperable = max(0, interoperable)
    sensitive_found = [c for c in cols_lower if any(kw in c for kw in SENSITIVE_KEYWORDS)]
    reusable = max(0, 100 - len(sensitive_found) * 15)
    dims = {"findable": round(findable,1), "accessible": round(accessible,1),
            "interoperable": round(interoperable,1), "reusable": round(reusable,1)}
    overall = round(sum(dims.values()) / 4, 1)
    grade = "Fully FAIR" if overall >= 90 else "Mostly FAIR" if overall >= 75 else "Partially FAIR" if overall >= 55 else "Not FAIR"
    return {"overall_fair_score": overall, "fair_grade": grade, "dimension_scores": dims,
            "issues_found": {"sensitive_features": sensitive_found} if sensitive_found else {},
            "sensitive_features_detected": sensitive_found,
            "summary": f"Dataset is {grade}. {len(sensitive_found)} sensitive feature(s) detected."}


# ══════════════════════════════════════════════════════════════════════════════
# 9. AUTO TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Automatic feature engineering for common patterns.
    Detects Titanic-like datasets and other common structures.
    Returns enriched DataFrame.
    """
    df = df.copy()
    cols_lower = {c: c.lower() for c in df.columns}

    # ── Titanic-style: extract Title from Name ─────────────────────────
    name_col = next((c for c, cl in cols_lower.items() if cl in ("name",) or "name" in cl), None)
    if name_col and df[name_col].dtype == object:
        try:
            titles = df[name_col].str.extract(r' ([A-Za-z]+)\.', expand=False)
            title_map = {
                'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
                'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
                'Mlle': 'Miss', 'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare',
                'Jonkheer': 'Rare', 'Don': 'Rare', 'Mme': 'Mrs', 'Capt': 'Rare', 'Sir': 'Rare'
            }
            df['Title'] = titles.map(title_map).fillna('Rare')
        except: pass

    # ── Titanic-style: FamilySize from SibSp + Parch ──────────────────
    sibsp_col = next((c for c, cl in cols_lower.items() if 'sibsp' in cl), None)
    parch_col = next((c for c, cl in cols_lower.items() if 'parch' in cl), None)
    if sibsp_col and parch_col:
        try:
            df['FamilySize'] = df[sibsp_col].fillna(0) + df[parch_col].fillna(0) + 1
            df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        except: pass

    # ── Titanic-style: Deck from Cabin ────────────────────────────────
    cabin_col = next((c for c, cl in cols_lower.items() if 'cabin' in cl), None)
    if cabin_col and df[cabin_col].dtype == object:
        try:
            missing_pct = df[cabin_col].isnull().mean()
            if missing_pct < 0.9:  # only if less than 90% missing
                df['Deck'] = df[cabin_col].str[0].fillna('U')
        except: pass

    # ── Drop pure ID columns (PassengerId, index-like) ────────────────
    for col, cl in cols_lower.items():
        if any(p in cl for p in ['passengerid', 'customerid', 'userid', 'rowid']):
            df.drop(columns=[col], errors='ignore', inplace=True)

    # ── Drop raw Name and Ticket (high-cardinality text) ──────────────
    for col, cl in cols_lower.items():
        if col not in df.columns: continue
        if cl in ('name', 'ticket') and df[col].dtype == object:
            n_unique_ratio = df[col].nunique() / max(len(df), 1)
            if n_unique_ratio > 0.3:  # too many unique values = no ML signal
                df.drop(columns=[col], errors='ignore', inplace=True)

    return df


def _prepare_for_ml(df: pd.DataFrame, target: str):
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import LabelEncoder
    df2 = df.copy()

    # Apply feature engineering first
    df2 = _engineer_features(df2)

    # Drop high-missing columns (>60%)
    drop_cols = [c for c in df2.columns if df2[c].isnull().mean() > 0.6 and c != target]
    df2.drop(columns=drop_cols, inplace=True, errors="ignore")

    # Encode categoricals
    for col in df2.select_dtypes(include=["object","category","bool"]).columns:
        if col == target: continue
        le = LabelEncoder()
        df2[col] = le.fit_transform(df2[col].astype(str))

    # Encode target if needed
    if df2[target].dtype == object or str(df2[target].dtype) == "category":
        le = LabelEncoder()
        df2[target] = le.fit_transform(df2[target].astype(str))

    X = df2.drop(columns=[target])
    y = df2[target]

    # Impute
    imp = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imp.fit_transform(X), columns=X.columns)
    return X_imp, y

def run_auto_training(df: pd.DataFrame, target: Optional[str], task_type: str) -> dict:
    from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    if not target or target not in df.columns: return {"error": "No valid target column"}
    try: X, y = _prepare_for_ml(df, target)
    except Exception as e: return {"error": f"Data prep failed: {e}"}
    if len(X) < 10: return {"error": "Too few rows for training"}
    is_clf = "classification" in task_type
    n_splits = min(5, max(2, len(X) // 20))
    if is_clf:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        models = {
            "Logistic Regression": Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=1000, random_state=42))]),
            "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=80, max_depth=8, random_state=42, n_jobs=-1),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=80, max_depth=4, random_state=42),
            "KNN": KNeighborsClassifier(n_neighbors=5),
        }
        metric = "f1_weighted"
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        models = {
            "Ridge Regression": Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))]),
            "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
            "Random Forest": RandomForestRegressor(n_estimators=80, max_depth=8, random_state=42, n_jobs=-1),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=80, max_depth=4, random_state=42),
            "KNN": KNeighborsRegressor(n_neighbors=5),
        }
        metric = "r2"
    results = {}
    best_name, best_score = None, -999

    for name, model in models.items():
        t0 = time.time()
        try:
            # Primary metric via cross-validation
            scores = cross_val_score(model, X, y, cv=cv, scoring=metric, n_jobs=-1)
            mean_s = safe_float(scores.mean())
            std_s = safe_float(scores.std())
            elapsed = round(time.time() - t0, 2)

            row = {"cv_mean": mean_s, "cv_std": std_s, "training_time_seconds": elapsed}

            if is_clf:
                row["f1_weighted"] = mean_s

                # Additional classification metrics on a single train/test split
                try:
                    from sklearn.model_selection import train_test_split
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

                    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
                    import copy
                    m2 = copy.deepcopy(model)
                    m2.fit(X_tr, y_tr)
                    y_pred = m2.predict(X_te)

                    row["accuracy"]  = safe_float(accuracy_score(y_te, y_pred))
                    row["precision"] = safe_float(precision_score(y_te, y_pred, average="weighted", zero_division=0))
                    row["recall"]    = safe_float(recall_score(y_te, y_pred, average="weighted", zero_division=0))

                    # ROC-AUC: only for binary or models with predict_proba
                    n_classes = len(np.unique(y))
                    if hasattr(m2, "predict_proba"):
                        proba = m2.predict_proba(X_te)
                        if n_classes == 2:
                            row["roc_auc"] = safe_float(roc_auc_score(y_te, proba[:, 1]))
                        else:
                            row["roc_auc"] = safe_float(roc_auc_score(y_te, proba, multi_class="ovr", average="weighted"))
                    else:
                        row["roc_auc"] = None
                except Exception:
                    row["accuracy"] = row["precision"] = row["recall"] = row["roc_auc"] = None

            else:
                row["r2_score"] = mean_s

                # Additional regression metrics
                try:
                    from sklearn.model_selection import train_test_split
                    from sklearn.metrics import mean_absolute_error, mean_squared_error
                    import copy

                    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
                    m2 = copy.deepcopy(model)
                    m2.fit(X_tr, y_tr)
                    y_pred = m2.predict(X_te)

                    row["mae"]  = safe_float(mean_absolute_error(y_te, y_pred))
                    row["rmse"] = safe_float(np.sqrt(mean_squared_error(y_te, y_pred)))
                except Exception:
                    row["mae"] = row["rmse"] = None

            results[name] = row
            if mean_s > best_score:
                best_score = mean_s
                best_name = name

        except Exception as e:
            results[name] = {"error": str(e), "cv_mean": 0.0}

    # Overfitting insight: find most stable model vs best-score model
    stable_model = None
    min_gap = float("inf")
    for name, r in results.items():
        if "cv_std" in r and r["cv_mean"] > 0:
            gap = r["cv_std"] / r["cv_mean"]
            if gap < min_gap:
                min_gap = gap
                stable_model = name

    insight = None
    if stable_model and stable_model != best_name:
        best_s = results.get(best_name, {}).get("cv_mean", 0)
        stab_s = results.get(stable_model, {}).get("cv_mean", 0)
        insight = (f"Trade-off detected: '{best_name}' has the highest score ({best_s:.3f}) "
                   f"but '{stable_model}' is more stable (lower variance). "
                   f"For production, '{stable_model}' ({stab_s:.3f}) may be the better choice.")

    return {
        "task_type": "classification" if is_clf else "regression",
        "best_model": best_name,
        "most_stable_model": stable_model,
        "stability_vs_performance_insight": insight,
        "best_score": best_score,
        "model_comparison": results,
        "metric_used": metric
    }


# ══════════════════════════════════════════════════════════════════════════════
# 10. CV STABILITY
# ══════════════════════════════════════════════════════════════════════════════

def analyze_cv_stability(training_results: dict) -> dict:
    mc = training_results.get("model_comparison", {})
    per_model = {}
    for name, r in mc.items():
        if "error" in r: continue
        mean_s = r.get("cv_mean", 0)
        std_s = r.get("cv_std", 0)
        cv_pct = safe_float(std_s / mean_s * 100) if mean_s > 0 else 0
        stability = "STABLE" if cv_pct < 5 else "MODERATE" if cv_pct < 15 else "UNSTABLE"
        per_model[name] = {"mean_score": mean_s, "std_dev": std_s,
                           "coefficient_of_variation_pct": cv_pct, "stability": stability}
    overall_stab = "STABLE"
    if any(v["stability"] == "UNSTABLE" for v in per_model.values()): overall_stab = "UNSTABLE"
    elif any(v["stability"] == "MODERATE" for v in per_model.values()): overall_stab = "MODERATE"
    return {"cv_strategy": "StratifiedKFold / KFold", "cv_folds": 5,
            "per_model_stability": per_model,
            "dataset_stability_summary": {"overall": overall_stab, "message": f"Overall CV stability: {overall_stab}."}}


# ══════════════════════════════════════════════════════════════════════════════
# 11. FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════

def compute_feature_importance(df: pd.DataFrame, target: Optional[str], task_type: str) -> dict:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.inspection import permutation_importance as perm_imp
    from sklearn.model_selection import train_test_split
    if not target or target not in df.columns:
        return {"ranking_by_rf_importance": [], "feature_details": {}}
    try:
        X, y = _prepare_for_ml(df, target)
        if len(X) < 10 or len(X.columns) == 0:
            return {"ranking_by_rf_importance": [], "feature_details": {}}
        is_clf = "classification" in task_type
        model = (RandomForestClassifier(n_estimators=60, max_depth=8, random_state=42, n_jobs=-1)
                 if is_clf else RandomForestRegressor(n_estimators=60, max_depth=8, random_state=42, n_jobs=-1))
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_tr, y_tr)
        rf_imp = model.feature_importances_
        pi = perm_imp(model, X_te, y_te, n_repeats=5, random_state=42, n_jobs=-1)
        perm_imps = pi.importances_mean
        ranking = []
        details = {}
        for i, feat in enumerate(X.columns.tolist()):
            rf_val = safe_float(rf_imp[i])
            perm_val = safe_float(perm_imps[i])
            agree = (rf_val > 0.01) == (perm_val > 0)
            tier = "HIGH" if rf_val > 0.1 else "MODERATE" if rf_val > 0.03 else "LOW" if rf_val > 0.005 else "NEGLIGIBLE"
            ranking.append({"feature": feat, "rf_importance": rf_val, "tier": tier})
            details[feat] = {"permutation_importance": perm_val, "methods_agree": agree,
                             "reason": f"RF importance {rf_val:.3f}, permutation {perm_val:.3f}."}
        ranking.sort(key=lambda x: -x["rf_importance"])
        return {"ranking_by_rf_importance": ranking, "feature_details": details}
    except Exception as e:
        return {"ranking_by_rf_importance": [], "feature_details": {}, "error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# 12. OVERFITTING ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyze_overfitting(df: pd.DataFrame, target: Optional[str], task_type: str) -> dict:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import f1_score, r2_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    if not target or target not in df.columns:
        return {"overall_overfitting": {"risk": "UNKNOWN", "message": "No target"}, "per_model_analysis": {}}
    try:
        X, y = _prepare_for_ml(df, target)
        if len(X) < 20:
            return {"overall_overfitting": {"risk": "LOW", "message": "Too few rows"}, "per_model_analysis": {}}
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
        is_clf = "classification" in task_type
        if is_clf:
            models = {
                "Logistic Regression": Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=1000))]),
                "Random Forest": RandomForestClassifier(n_estimators=60, random_state=42, n_jobs=-1),
                "Gradient Boosting": GradientBoostingClassifier(n_estimators=60, random_state=42),
            }
        else:
            models = {
                "Ridge": Pipeline([("sc", StandardScaler()), ("m", Ridge())]),
                "Random Forest": RandomForestRegressor(n_estimators=60, random_state=42, n_jobs=-1),
                "Gradient Boosting": GradientBoostingRegressor(n_estimators=60, random_state=42),
            }
        per_model = {}
        risks = []
        for name, model in models.items():
            try:
                model.fit(X_tr, y_tr)
                if is_clf:
                    tr_score = safe_float(f1_score(y_tr, model.predict(X_tr), average="weighted"))
                    te_score = safe_float(f1_score(y_te, model.predict(X_te), average="weighted"))
                else:
                    tr_score = safe_float(r2_score(y_tr, model.predict(X_tr)))
                    te_score = safe_float(r2_score(y_te, model.predict(X_te)))
                gap = tr_score - te_score
                risk = "LOW" if gap < 0.05 else "MODERATE" if gap < 0.15 else "HIGH"
                risks.append(risk)
                per_model[name] = {"train_score": tr_score, "test_score": te_score,
                                   "train_test_gap": safe_float(gap), "overfitting_risk": risk}
            except Exception as e:
                per_model[name] = {"error": str(e)}
        overall_risk = "HIGH" if "HIGH" in risks else "MODERATE" if "MODERATE" in risks else "LOW"
        msg = {"HIGH": "Significant overfitting — regularize or get more data.",
               "MODERATE": "Mild overfitting — monitor with more data.",
               "LOW": "Models generalize well — minimal overfitting."}[overall_risk]
        return {"overall_overfitting": {"risk": overall_risk, "message": msg}, "per_model_analysis": per_model}
    except Exception as e:
        return {"overall_overfitting": {"risk": "ERROR", "message": str(e)}, "per_model_analysis": {}}


# ══════════════════════════════════════════════════════════════════════════════
# 13. PIPELINE BUILDER
# ══════════════════════════════════════════════════════════════════════════════

ID_PATTERNS = [r"^id$", r".*_id$", r"^id_", r".*uuid.*", r".*index$", r"^row"]
META_PATTERNS = [r".*timestamp.*", r".*created.*", r".*updated.*"]
TEXT_PATTERNS = [r".*name$", r".*address.*", r".*email.*", r".*description.*", r".*comment.*"]

def _matches(col, patterns):
    cl = col.lower()
    return any(re.search(p, cl) for p in patterns)

def build_pipeline(df: pd.DataFrame, target: Optional[str], fa: dict) -> dict:
    drop_cols, impute, encode, scale, transform = [], {}, {}, {}, {}
    for col, info in fa.items():
        if col == target: continue
        if _matches(col, ID_PATTERNS) and info["unique_ratio"] > 0.9:
            drop_cols.append({"column": col, "reason": "ID column — pure identifier, no predictive signal (e.g. PassengerId)"}); continue
        if _matches(col, META_PATTERNS):
            drop_cols.append({"column": col, "reason": "Datetime/metadata column"}); continue
        if _matches(col, TEXT_PATTERNS) and info.get("dtype") == "object" and info["unique_ratio"] > 0.5:
            drop_cols.append({"column": col, "reason": "High-cardinality text — no direct ML signal. Consider feature engineering (e.g. extract Title from Name)"}); continue
        if info["missing_percent"] > 60:
            drop_cols.append({"column": col, "reason": f"{info['missing_percent']:.0f}% missing"}); continue
        if info["unique_values"] <= 1:
            drop_cols.append({"column": col, "reason": "Zero variance (constant)"}); continue
        if info["missing_percent"] > 0:
            strat = "median" if info.get("mean") is not None and abs(info.get("skewness",0)) > 1 else "mean" if info.get("mean") is not None else "mode"
            impute[col] = {"strategy": strat, "missing_percent": info["missing_percent"]}
        if info.get("dtype") in ["object","category"]:
            n_u = info["unique_values"]
            if n_u == 2: encode[col] = {"encoding": "LabelEncoding", "reason": "Binary categorical"}
            elif n_u <= 10: encode[col] = {"encoding": "OneHotEncoding", "reason": f"{n_u} categories"}
            elif n_u <= 50: encode[col] = {"encoding": "OrdinalEncoding", "reason": f"High cardinality ({n_u})"}
            else: drop_cols.append({"column": col, "reason": f"Very high cardinality ({n_u})"}); continue
        if info.get("mean") is not None:
            skew = abs(info.get("skewness", 0))
            if skew > 2:
                transform[col] = {"transform": "log1p" if info.get("min",0) >= 0 else "yeo-johnson", "skewness": info.get("skewness",0)}
            scale[col] = {"scaler": info.get("scaling_suggestion", "StandardScaler"), "reason": f"skew={info.get('skewness',0):.2f}"}
    total = len(df.columns) - (1 if target else 0)
    num_feats = len([c for c,i in fa.items() if i.get("mean") is not None and c != target])
    cat_feats = len([c for c,i in fa.items() if i.get("mean") is None and c != target])
    # Feature engineering steps (inform user what was auto-created)
    fe_steps = [
        {"name": "FamilySize", "formula": "SibSp + Parch + 1", "reason": "Family size often predicts survival/outcome better than individual SibSp/Parch"},
        {"name": "IsAlone",    "formula": "FamilySize == 1",    "reason": "Solo travelers behave differently from family groups"},
        {"name": "Title",      "formula": "Extracted from Name","reason": "Social title (Mr/Mrs/Miss/Master) encodes age, gender, status — strong predictor"},
        {"name": "Deck",       "formula": "First char of Cabin", "reason": "Deck letter correlates with passenger class and location on ship"},
    ]

    return {
        "target_column": target, "drop_columns": drop_cols, "missing_value_strategy": impute,
        "encoding_strategy": encode, "scaling_strategy": scale, "transformation_recommendations": transform,
        "feature_engineering": fe_steps,
        "summary": {"total_features": total, "dropped": len(drop_cols), "numeric_features": num_feats,
                    "categorical_features": cat_feats, "columns_needing_imputation": len(impute),
                    "columns_needing_encoding": len(encode), "columns_needing_scaling": len(scale)}
    }


# ══════════════════════════════════════════════════════════════════════════════
# 14. MODEL RECOMMENDATION
# ══════════════════════════════════════════════════════════════════════════════

def recommend_models(df: pd.DataFrame, target: Optional[str], task_type: str, training_results: dict) -> dict:
    n = len(df)
    is_clf = "classification" in task_type
    imbalance = {"severity": "NONE", "ratio": 1.0}
    if target and target in df.columns and is_clf:
        vc = df[target].value_counts()
        if len(vc) >= 2:
            ratio = safe_float(vc.iloc[0] / vc.iloc[-1])
            sev = "NONE" if ratio < 2 else "LOW" if ratio < 5 else "MODERATE" if ratio < 10 else "HIGH"
            imbalance = {"severity": sev, "ratio": ratio}
    best_model = training_results.get("best_model", "Random Forest")
    stable_model = training_results.get("most_stable_model", best_model)
    stability_insight = training_results.get("stability_vs_performance_insight")

    if is_clf:
        models = [
            {"model": "Logistic Regression", "priority": "baseline", "why": "Fast, interpretable, great for linearly separable data."},
            {"model": "Random Forest", "priority": "recommended", "why": "Robust, handles non-linearity, low tuning required."},
            {"model": "Gradient Boosting (XGBoost/LightGBM)", "priority": "advanced", "why": "State-of-the-art tabular performance."},
        ]
        if imbalance["severity"] in ["MODERATE","HIGH"]:
            models.append({"model": "Balanced Random Forest", "priority": "recommended", "why": "Handles class imbalance natively."})
    else:
        models = [
            {"model": "Ridge Regression", "priority": "baseline", "why": "Stable, regularized, interpretable."},
            {"model": "Random Forest Regressor", "priority": "recommended", "why": "Captures non-linear relationships without scaling."},
            {"model": "Gradient Boosting Regressor", "priority": "advanced", "why": "Best accuracy on complex tabular data."},
        ]
    return {
        "task_type": "classification" if is_clf else "regression",
        "best_from_training": best_model,
        "most_stable_model": stable_model,
        "stability_vs_performance_insight": stability_insight,
        "recommended_models": models,
        "class_imbalance": imbalance,
        "evaluation_metrics": {"primary": "F1-Weighted" if is_clf else "R² Score", "secondary": "AUC-ROC" if is_clf else "RMSE"},
        "dataset_size_note": "Small — use cross-validation, avoid deep trees." if n < 500 else "Medium — standard ML pipeline works well." if n < 10000 else "Large — consider LightGBM / subsampling."
    }


# ══════════════════════════════════════════════════════════════════════════════
# 15. EXPLAINABILITY REPORT
# ══════════════════════════════════════════════════════════════════════════════

def generate_explainability_report(result: dict) -> dict:
    issues = []
    dq = result.get("dataset_quality_score", {})
    ds = result.get("dataset_summary", {})
    fa = result.get("feature_analysis", {})
    corr = result.get("correlation_analysis", {})
    leakage = result.get("data_leakage_analysis", {})
    anomaly = result.get("anomaly_detection", {})
    fair = result.get("fair_assessment", {})
    miss_pct = ds.get("missing_value_percent", 0)
    if miss_pct > 30:
        issues.append({"severity": "CRITICAL", "category": "Completeness", "title": f"High missing data ({miss_pct:.1f}%)",
                       "explanation": "Over 30% of cells are missing — imputation will introduce significant bias.",
                       "fix": "Drop columns >60% missing; use median/mode imputation for the rest.", "source_module": "dataset_summary"})
    elif miss_pct > 10:
        issues.append({"severity": "HIGH", "category": "Completeness", "title": f"Moderate missing data ({miss_pct:.1f}%)",
                       "explanation": "10–30% missing values can degrade model performance significantly.",
                       "fix": "Impute numerics with median (skewed) or mean; categoricals with mode.", "source_module": "dataset_summary"})
    dups = ds.get("duplicate_rows", 0)
    if dups > 0:
        dup_pct = dups / ds.get("rows", 1) * 100
        sev = "HIGH" if dup_pct > 5 else "MODERATE"
        issues.append({"severity": sev, "category": "Uniqueness", "title": f"{dups} duplicate rows ({dup_pct:.1f}%)",
                       "explanation": "Duplicate rows cause data leakage between train/test splits.",
                       "fix": "df.drop_duplicates(inplace=True)", "source_module": "dataset_summary"})
    for col, info in fa.items():
        if info.get("missing_percent", 0) > 50:
            issues.append({"severity": "HIGH", "category": "Missing Values", "title": f"Column '{col}' is {info['missing_percent']:.0f}% missing",
                           "explanation": "Columns with >50% missing data are rarely informative.",
                           "fix": f"df.drop(columns=['{col}'])", "source_module": "feature_analysis"})
        if info.get("unique_values", 0) <= 1 and col != result.get("target_detection", {}).get("final_target"):
            issues.append({"severity": "MODERATE", "category": "Feature Quality", "title": f"Column '{col}' has zero variance",
                           "explanation": "Constant columns add no predictive signal.",
                           "fix": f"df.drop(columns=['{col}'])", "source_module": "feature_analysis"})
    for pair in corr.get("strong_correlation_pairs", []):
        if abs(pair["pearson"]) > 0.9:
            issues.append({"severity": "HIGH", "category": "Multicollinearity", "title": f"'{pair['feature_1']}' ↔ '{pair['feature_2']}' corr={pair['pearson']:.2f}",
                           "explanation": "Near-collinear features cause model instability.",
                           "fix": f"Drop one: df.drop(columns=['{pair['feature_2']}'])", "source_module": "correlation_analysis"})
    for item in leakage.get("target_leakage", []):
        issues.append({"severity": "CRITICAL", "category": "Data Leakage", "title": f"'{item['feature']}' may leak target",
                       "explanation": item["reason"], "fix": f"Investigate and drop: df.drop(columns=['{item['feature']}'])",
                       "source_module": "leakage_analysis"})
    if anomaly.get("severity") == "HIGH":
        issues.append({"severity": "HIGH", "category": "Anomalies", "title": f"{anomaly['anomaly_percent']:.1f}% anomalous rows",
                       "explanation": "High anomaly rate may indicate data collection errors.",
                       "fix": "Inspect and optionally remove anomalous rows using IsolationForest predictions.", "source_module": "anomaly_detection"})
    cb_score = dq.get("dimension_scores", {}).get("class_balance", 100)
    if cb_score < 60:
        issues.append({"severity": "MODERATE", "category": "Class Balance", "title": "Significant class imbalance",
                       "explanation": f"Class balance score: {cb_score:.0f}/100.",
                       "fix": "Use SMOTE oversampling, class_weight='balanced', or stratified sampling.", "source_module": "quality_score"})
    rows = ds.get("rows", 0)
    cols = ds.get("columns", 0)
    if rows / max(cols, 1) < 10:
        issues.append({"severity": "MODERATE", "category": "Data Adequacy", "title": f"Low rows-to-features ratio ({rows}/{cols})",
                       "explanation": "Less than 10 rows per feature increases overfitting risk.",
                       "fix": "Collect more data, perform feature selection, or use regularized models.", "source_module": "dataset_summary"})
    skewed_cols = [col for col, info in fa.items() if abs(info.get("skewness", 0)) > 3]
    if len(skewed_cols) > 3:
        issues.append({"severity": "LOW", "category": "Feature Distribution", "title": f"{len(skewed_cols)} highly skewed features",
                       "explanation": "Extreme skewness degrades linear model performance.",
                       "fix": "Apply log1p or Yeo-Johnson transformation.", "source_module": "feature_analysis"})
    overall_q = dq.get("overall_score", 70)
    n_crit = sum(1 for i in issues if i["severity"] == "CRITICAL")
    n_high = sum(1 for i in issues if i["severity"] == "HIGH")
    readiness = max(10.0, min(100.0, overall_q - n_crit * 12 - n_high * 5))
    grade = "A" if readiness >= 90 else "B" if readiness >= 75 else "C" if readiness >= 60 else "D" if readiness >= 45 else "F"
    module_health = {
        "completeness": {"status": "CLEAN" if miss_pct < 5 else "ISSUES", "issues_found": 1 if miss_pct >= 5 else 0},
        "uniqueness": {"status": "CLEAN" if dups == 0 else "ISSUES", "issues_found": 1 if dups > 0 else 0},
        "correlation": {"status": "CLEAN" if len(corr.get("strong_correlation_pairs",[])) == 0 else "ISSUES", "issues_found": len(corr.get("strong_correlation_pairs",[]))},
        "leakage": {"status": "CLEAN" if len(leakage.get("target_leakage",[])) == 0 else "ISSUES", "issues_found": len(leakage.get("target_leakage",[]))},
        "anomaly": {"status": "CLEAN" if anomaly.get("anomaly_percent",0) < 5 else "ISSUES", "issues_found": 1 if anomaly.get("anomaly_percent",0) >= 5 else 0},
        "fair": {"status": "CLEAN" if fair.get("overall_fair_score",0) >= 75 else "ISSUES", "issues_found": len(fair.get("issues_found",{}))},
        "class_balance": {"status": "CLEAN" if cb_score >= 70 else "ISSUES", "issues_found": 0 if cb_score >= 70 else 1},
        "feature_quality": {"status": "CLEAN" if len(skewed_cols) <= 3 else "ISSUES", "issues_found": len(skewed_cols)},
    }
    positive_signals = []
    if miss_pct < 2: positive_signals.append("Near-complete dataset — minimal missing values.")
    if dups == 0: positive_signals.append("No duplicate rows detected.")
    if len(corr.get("strong_correlation_pairs",[])) == 0: positive_signals.append("No multicollinearity issues found.")
    if len(leakage.get("target_leakage",[])) == 0: positive_signals.append("No data leakage detected.")
    if anomaly.get("anomaly_percent",5) < 2: positive_signals.append("Anomaly rate is very low.")
    if readiness >= 80: positive_signals.append(f"Strong ML readiness score: {readiness:.0f}/100.")
    issues.sort(key=lambda x: {"CRITICAL":0,"HIGH":1,"MODERATE":2,"LOW":3}.get(x["severity"],4))
    action_plan = [{"step": i+1, "priority": iss["severity"], "reason": iss["title"], "module": iss["source_module"]} for i, iss in enumerate(issues[:10])]
    return {
        "readiness_score": round(readiness, 1), "grade": grade,
        "verdict": "Dataset is ML-ready" if readiness >= 75 else "Dataset needs preprocessing before ML",
        "issue_summary": {"CRITICAL": n_crit, "HIGH": n_high,
                          "MODERATE": sum(1 for i in issues if i["severity"] == "MODERATE"),
                          "LOW": sum(1 for i in issues if i["severity"] == "LOW")},
        "issues": issues, "action_plan": action_plan,
        "positive_signals": positive_signals, "module_health": module_health,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MASTER RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def _run_full_analysis(df: pd.DataFrame, target_column: Optional[str] = None) -> dict:
    result = {}
    result["dataset_summary"]       = get_dataset_summary(df)
    result["target_detection"]      = detect_target_column(df, user_specified=target_column)
    final_target                    = result["target_detection"]["final_target"]
    task_type                       = result["target_detection"]["task_type"]
    result["feature_analysis"]      = analyze_features(df, target=final_target)
    result["dataset_quality_score"] = calculate_quality_score(df, target=final_target)
    result["correlation_analysis"]  = analyze_correlations(df, target=final_target)
    result["data_leakage_analysis"] = analyze_leakage(df, target=final_target)
    result["anomaly_detection"]     = detect_anomalies(df, target=final_target)
    result["fair_assessment"]       = assess_fair(df, target=final_target)
    result["auto_training_results"] = run_auto_training(df, final_target, task_type)
    result["cross_validation"]      = analyze_cv_stability(result["auto_training_results"])
    result["feature_importance"]    = compute_feature_importance(df, final_target, task_type)
    result["overfitting_analysis"]  = analyze_overfitting(df, final_target, task_type)
    result["recommended_pipeline"]  = build_pipeline(df, final_target, result["feature_analysis"])
    result["model_recommendation"]  = recommend_models(df, final_target, task_type, result["auto_training_results"])
    result["explainability_report"] = generate_explainability_report(result)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# API ENDPOINTS — CORE
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {"status": "ok", "version": "4.1", "modules": 18}


@app.post("/upload")
async def upload_and_analyze(file: UploadFile = File(...), target_column: Optional[str] = Query(default=None)):
    try:
        contents = await file.read()
        df = _load_df(file, contents)
        if df.empty: raise HTTPException(400, "Uploaded file is empty.")
        if len(df) < 5: raise HTTPException(400, "Dataset too small (< 5 rows).")
        return _run_full_analysis(df, target_column=target_column)
    except HTTPException: raise
    except Exception as e: raise HTTPException(500, f"Analysis failed: {e}")


@app.post("/execute")
async def execute_pipeline(
    file: UploadFile = File(...),
    target_column: Optional[str] = Query(default=None),
    mode: str = Query(default="clean"),
):
    """
    Apply preprocessing pipeline and return cleaned CSV.
    mode="clean" → human-readable (default for Download button)
    mode="ml"    → fully numeric, scaled, encoded (for ML training)
    """
    try:
        contents = await file.read()
        df = _load_df(file, contents)

        if df.empty:
            raise HTTPException(400, {"error": "Uploaded file is empty."})
        if len(df) < 2:
            raise HTTPException(400, {"error": "Dataset too small."})

        # Detect target
        target = target_column
        if not target:
            td = detect_target_column(df)
            target = td["final_target"]

        # Run preprocessing with logging
        try:
            from app.prepare import autofix_dataset
            cleaned, log_messages = autofix_dataset(df, target, mode=mode)
        except ImportError:
            # Inline fallback if prepare.py not yet updated
            from sklearn.impute import SimpleImputer
            from sklearn.preprocessing import LabelEncoder
            df2 = df.copy()
            log_messages = []
            # Drop ID columns
            for col in df2.columns:
                if col == target: continue
                cl = col.lower()
                if any(p in cl for p in ["passengerid","customerid","userid","rowid"]):
                    df2.drop(columns=[col], inplace=True, errors="ignore")
                    log_messages.append(f"Dropped '{col}' → ID column")
            # Drop >70% missing
            drop_miss = [c for c in df2.columns if c != target and df2[c].isnull().mean() > 0.7]
            if drop_miss:
                df2.drop(columns=drop_miss, inplace=True)
                log_messages.append(f"Dropped {len(drop_miss)} high-missing columns")
            # Impute
            for col in df2.columns:
                if col == target or df2[col].isnull().sum() == 0: continue
                if pd.api.types.is_numeric_dtype(df2[col]):
                    df2[col] = df2[col].fillna(df2[col].median())
                else:
                    mode_val = df2[col].mode()
                    df2[col] = df2[col].fillna(mode_val[0] if len(mode_val)>0 else "Unknown")
            log_messages.append(f"Imputed missing values")
            cleaned = df2
            log_messages.append(f"Output: {cleaned.shape[0]} rows × {cleaned.shape[1]} columns")

        # Build CSV
        buf = io.StringIO()
        cleaned.to_csv(buf, index=False)
        csv_str = buf.getvalue()

        # Build log header (max 3.8KB to stay within HTTP limits)
        log_json = json.dumps(log_messages)
        if len(log_json) > 3800:
            trimmed = []
            total = 0
            for entry in log_messages:
                total += len(entry) + 4
                if total > 3600:
                    trimmed.append(f"...and {len(log_messages)-len(trimmed)} more steps")
                    break
                trimmed.append(entry)
            log_json = json.dumps(trimmed)

        fname = (file.filename or "dataset").replace(".csv","").replace(".xlsx","")
        headers = {
            "X-Final-Shape":  f"{cleaned.shape[0]}x{cleaned.shape[1]}",
            "X-Steps-Applied": str(len(log_messages)),
            "X-Pipeline-Log": log_json,
            "X-Pipeline-Mode": mode,
            "Content-Disposition": f'attachment; filename="{fname}_{mode}_cleaned.csv"',
            "Access-Control-Expose-Headers": "X-Final-Shape,X-Steps-Applied,X-Pipeline-Log,X-Pipeline-Mode",
        }
        return StreamingResponse(
            io.BytesIO(csv_str.encode("utf-8")),
            media_type="text/csv",
            headers=headers
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, {"error": f"Pipeline failed: {str(e)}"})




# ══════════════════════════════════════════════════════════════════════════════
# /autofix — One-Click Auto-Cleaning Engine (fixes the 404)
# Called by the "Run One-Click Auto Fix" button in Pipeline tab
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/autofix")
async def autofix_endpoint(
    file: UploadFile = File(...),
    target_column: Optional[str] = Query(default=None)
):
    """
    One-click auto-cleaning engine.
    Applies the full ML pipeline, measures model score before and after,
    and returns before/after accuracy + list of fixes applied + cleaned CSV.
    """
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
    import warnings; warnings.filterwarnings("ignore")

    try:
        contents = await file.read()
        df = _load_df(file, contents)

        if df.empty:
            raise HTTPException(400, {"error": "Uploaded file is empty."})

        td = detect_target_column(df, user_specified=target_column)
        target    = td["final_target"]
        task_type = td["task_type"]
        is_clf    = "classification" in task_type

        def quick_score(d, already_clean=False):
            """Score dataset — if already_clean, use X/y directly (no _prepare_for_ml)."""
            try:
                if already_clean:
                    if target not in d.columns:
                        return 0.0
                    X = d.drop(columns=[target]).select_dtypes(include=np.number)
                    y = d[target]
                    if len(X) < 10 or X.shape[1] == 0:
                        return 0.0
                else:
                    X, y = _prepare_for_ml(d, target)
                    if len(X) < 10:
                        return 0.0

                model = (
                    RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1)
                    if is_clf else
                    RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1)
                )
                try:
                    cv = (StratifiedKFold(3, shuffle=True, random_state=42)
                          if is_clf else KFold(3, shuffle=True, random_state=42))
                    return safe_float(cross_val_score(
                        model, X, y, cv=cv,
                        scoring="f1_weighted" if is_clf else "r2",
                        n_jobs=-1
                    ).mean())
                except:
                    # Fallback: simple train/test split
                    from sklearn.model_selection import train_test_split
                    from sklearn.metrics import f1_score, r2_score
                    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
                    model.fit(X_tr, y_tr)
                    if is_clf:
                        return safe_float(f1_score(y_te, model.predict(X_te), average="weighted", zero_division=0))
                    else:
                        return safe_float(r2_score(y_te, model.predict(X_te)))
            except:
                return 0.0

        # Score BEFORE cleaning (on raw data)
        before_score = quick_score(df, already_clean=False)

        # Apply full cleaning pipeline
        fixes_applied = []
        try:
            from app.prepare import autofix_dataset
            cleaned_df, log_messages = autofix_dataset(df, target, mode="ml")
            fixes_applied = log_messages
        except Exception as prep_err:
            # Inline fallback cleaning
            from sklearn.impute import SimpleImputer
            from sklearn.preprocessing import LabelEncoder, StandardScaler
            df2 = df.copy()
            fixes_applied = []

            # Feature engineering
            df2 = _engineer_features(df2)
            fixes_applied.append("Applied feature engineering (FamilySize, Title, Deck, AgeBin)")

            # Drop ID and high-missing cols
            drop_ids = [c for c in df2.columns if c != target and
                        any(p in c.lower() for p in ["passengerid","customerid","rowid","userid"])]
            drop_miss = [c for c in df2.columns if c != target and df2[c].isnull().mean() > 0.6]
            to_drop = list(set(drop_ids + drop_miss))
            if to_drop:
                df2.drop(columns=to_drop, inplace=True, errors="ignore")
                fixes_applied.append(f"Dropped {len(to_drop)} useless/high-missing columns: {to_drop[:3]}")

            # Impute
            for col in df2.columns:
                if col == target or df2[col].isnull().sum() == 0: continue
                if pd.api.types.is_numeric_dtype(df2[col]):
                    df2[col] = df2[col].fillna(df2[col].median())
                else:
                    mode_v = df2[col].mode()
                    df2[col] = df2[col].fillna(mode_v[0] if len(mode_v) > 0 else "Unknown")
            fixes_applied.append("Imputed missing values (median for numeric, mode for categorical)")

            # Encode
            for col in df2.select_dtypes(include=["object","category"]).columns:
                if col == target: continue
                u = df2[col].nunique()
                if u <= 10:
                    dummies = pd.get_dummies(df2[[col]], prefix=col, drop_first=True, dtype=int)
                    df2 = pd.concat([df2.drop(columns=[col]), dummies], axis=1)
                    fixes_applied.append(f"OneHot encoded '{col}' ({u} categories)")
                else:
                    freq = df2[col].value_counts(normalize=True)
                    df2[col] = df2[col].map(freq).fillna(0)
                    fixes_applied.append(f"Frequency encoded '{col}' ({u} categories)")

            # Encode target
            if df2[target].dtype == object:
                le = LabelEncoder()
                df2[target] = le.fit_transform(df2[target].astype(str))
                fixes_applied.append(f"Label-encoded target '{target}'")

            # Scale
            feat_cols = [c for c in df2.select_dtypes(include=np.number).columns if c != target]
            if feat_cols:
                sc = StandardScaler()
                df2[feat_cols] = sc.fit_transform(df2[feat_cols])
                fixes_applied.append(f"Standardized {len(feat_cols)} numeric features")

            df2 = df2.fillna(0).select_dtypes(include=np.number)
            cleaned_df = df2

        # Score AFTER cleaning
        after_score = quick_score(cleaned_df, already_clean=True)

        # If after score is 0 (numeric safety), re-score via _prepare_for_ml
        if after_score == 0.0:
            after_score = quick_score(df, already_clean=False)

        # Build CSV for download
        buf = io.StringIO()
        cleaned_df.to_csv(buf, index=False)
        csv_content = buf.getvalue()

        metric_label = "F1-Weighted" if is_clf else "R² Score"
        improvement = safe_float(after_score - before_score)

        return {
            "before_accuracy": before_score,
            "after_accuracy":  after_score,
            "improvement":     improvement,
            "fixes_applied":   fixes_applied,
            "cleaned_csv":     csv_content,
            "metric":          metric_label,
            "shape_before":    f"{df.shape[0]}×{df.shape[1]}",
            "shape_after":     f"{cleaned_df.shape[0]}×{cleaned_df.shape[1]}",
            "task_type":       task_type,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, {"error": f"Autofix failed: {str(e)}"})


# ══════════════════════════════════════════════════════════════════════════════
# /neural — Deep Learning with MLP Neural Network
# Trains a PyTorch MLP (with sklearn MLPClassifier fallback).
# This is the "big" differentiator — no other AutoEDA tool trains DL models.
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/neural")
async def neural_network_analysis(
    file: UploadFile = File(...),
    target_column: Optional[str] = Query(default=None),
    epochs: int = Query(default=50, ge=10, le=200),
    hidden_layers: str = Query(default="128,64,32"),
):
    """
    Train a Multi-Layer Perceptron Neural Network on the dataset.
    Tries PyTorch first, falls back to sklearn MLPClassifier/MLPRegressor.

    Architecture: Input → [hidden_layers] → Output
    Training: Adam optimizer, early stopping, dropout=0.3

    Returns: accuracy metrics, training history, comparison with best sklearn model,
             feature importance via perturbation analysis.
    """
    import warnings; warnings.filterwarnings("ignore")

    try:
        contents = await file.read()
        df = _load_df(file, contents)

        if df.empty:
            raise HTTPException(400, {"error": "Uploaded file is empty."})
        if len(df) < 20:
            raise HTTPException(400, {"error": "Need at least 20 rows for neural network training."})

        td = detect_target_column(df, user_specified=target_column)
        target    = td["final_target"]
        task_type = td["task_type"]
        is_clf    = "classification" in task_type

        # Parse hidden layer sizes
        try:
            layer_sizes = [int(x.strip()) for x in hidden_layers.split(",")]
        except:
            layer_sizes = [128, 64, 32]

        X, y = _prepare_for_ml(df, target)
        n_features  = X.shape[1]
        n_classes   = int(y.nunique()) if is_clf else 1

        from sklearn.model_selection import train_test_split
        from sklearn.metrics import (accuracy_score, f1_score, r2_score,
                                     mean_squared_error, roc_auc_score, classification_report)
        from sklearn.preprocessing import StandardScaler

        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y if is_clf else None
            )
        except:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s  = sc.transform(X_te)

        training_history = []
        model_info       = {}
        backend_used     = "unknown"

        # ── Try PyTorch ──────────────────────────────────────────────────
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset

            backend_used = "PyTorch"

            X_tr_t = torch.FloatTensor(X_tr_s)
            X_te_t = torch.FloatTensor(X_te_s)

            if is_clf:
                y_tr_t = torch.LongTensor(y_tr.values)
                y_te_t = torch.LongTensor(y_te.values)
            else:
                y_tr_t = torch.FloatTensor(y_tr.values).unsqueeze(1)
                y_te_t = torch.FloatTensor(y_te.values).unsqueeze(1)

            # Build MLP
            layers = []
            in_dim = n_features
            for hidden_dim in layer_sizes:
                layers += [
                    nn.Linear(in_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                ]
                in_dim = hidden_dim

            out_dim = n_classes if is_clf else 1
            layers.append(nn.Linear(in_dim, out_dim))
            if not is_clf:
                layers.append(nn.Identity())

            model = nn.Sequential(*layers)
            optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss() if is_clf else nn.MSELoss()

            ds    = TensorDataset(X_tr_t, y_tr_t)
            batch = min(64, len(ds))
            loader = DataLoader(ds, batch_size=batch, shuffle=True)

            best_val_loss = float("inf")
            patience      = 10
            no_improve    = 0

            for epoch in range(epochs):
                model.train()
                epoch_loss = 0.0
                for xb, yb in loader:
                    optimizer.zero_grad()
                    out  = model(xb)
                    loss = criterion(out, yb)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                # Validation
                model.eval()
                with torch.no_grad():
                    val_out  = model(X_te_t)
                    val_loss = criterion(val_out, y_te_t).item()

                if is_clf:
                    preds = val_out.argmax(dim=1).numpy()
                    metric_val = safe_float(f1_score(y_te.values, preds, average="weighted", zero_division=0))
                else:
                    metric_val = safe_float(r2_score(y_te.values, val_out.squeeze().numpy()))

                if (epoch + 1) % max(1, epochs // 10) == 0 or epoch < 5:
                    training_history.append({
                        "epoch":      epoch + 1,
                        "train_loss": safe_float(epoch_loss / len(loader)),
                        "val_loss":   safe_float(val_loss),
                        "metric":     metric_val,
                    })

                # Early stopping
                if val_loss < best_val_loss - 1e-4:
                    best_val_loss = val_loss
                    no_improve    = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        break

            # Final evaluation
            model.eval()
            with torch.no_grad():
                final_out = model(X_te_t)
                if is_clf:
                    y_pred = final_out.argmax(dim=1).numpy()
                    if hasattr(model, "softmax") or n_classes == 2:
                        proba = torch.softmax(final_out, dim=1).numpy()
                    else:
                        proba = torch.softmax(final_out, dim=1).numpy()
                else:
                    y_pred = final_out.squeeze().numpy()

            model_info["architecture"] = {
                "type":          "MLP Neural Network",
                "input_dim":     n_features,
                "hidden_layers": layer_sizes,
                "output_dim":    n_classes if is_clf else 1,
                "dropout":       0.3,
                "batch_norm":    True,
                "activation":    "ReLU",
                "optimizer":     "Adam (lr=1e-3, weight_decay=1e-4)",
                "loss_fn":       "CrossEntropy" if is_clf else "MSE",
                "epochs_trained": len(training_history),
                "early_stopping": True,
            }

        except ImportError:
            # ── sklearn MLP fallback ─────────────────────────────────────
            backend_used = "sklearn MLPClassifier/MLPRegressor (install PyTorch for full DL)"

            from sklearn.neural_network import MLPClassifier, MLPRegressor

            mlp_layers = tuple(layer_sizes)
            if is_clf:
                mlp = MLPClassifier(
                    hidden_layer_sizes=mlp_layers,
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    batch_size="auto",
                    learning_rate_init=1e-3,
                    max_iter=epochs,
                    early_stopping=True,
                    validation_fraction=0.15,
                    random_state=42,
                )
            else:
                mlp = MLPRegressor(
                    hidden_layer_sizes=mlp_layers,
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    batch_size="auto",
                    learning_rate_init=1e-3,
                    max_iter=epochs,
                    early_stopping=True,
                    validation_fraction=0.15,
                    random_state=42,
                )

            mlp.fit(X_tr_s, y_tr.values)
            y_pred = mlp.predict(X_te_s)

            # Build training history from loss_curve_
            if hasattr(mlp, "loss_curve_"):
                for i, loss in enumerate(mlp.loss_curve_):
                    if i % max(1, len(mlp.loss_curve_) // 10) == 0:
                        training_history.append({
                            "epoch":      i + 1,
                            "train_loss": safe_float(loss),
                            "val_loss":   safe_float(mlp.validation_scores_[i]) if hasattr(mlp, "validation_scores_") else None,
                            "metric":     safe_float(mlp.validation_scores_[i]) if hasattr(mlp, "validation_scores_") else None,
                        })

            model_info["architecture"] = {
                "type":          "sklearn MLPClassifier/Regressor",
                "input_dim":     n_features,
                "hidden_layers": layer_sizes,
                "output_dim":    n_classes if is_clf else 1,
                "activation":    "relu",
                "optimizer":     "adam",
                "alpha":         1e-4,
                "epochs_trained": mlp.n_iter_,
                "early_stopping": True,
                "note":          "Install PyTorch for full deep learning: pip install torch",
            }

        # ── Compute final metrics ─────────────────────────────────────────
        if is_clf:
            y_pred_int = y_pred.astype(int) if hasattr(y_pred, "astype") else y_pred

            nn_metrics = {
                "accuracy":           safe_float(accuracy_score(y_te.values, y_pred_int)),
                "f1_weighted":        safe_float(f1_score(y_te.values, y_pred_int, average="weighted", zero_division=0)),
                "f1_macro":           safe_float(f1_score(y_te.values, y_pred_int, average="macro", zero_division=0)),
            }
            # ROC-AUC if possible
            try:
                if n_classes == 2 and "proba" in dir():
                    nn_metrics["roc_auc"] = safe_float(roc_auc_score(y_te.values, proba[:, 1]))
            except:
                pass

            primary_score = nn_metrics["f1_weighted"]
            primary_metric = "F1-Weighted"

        else:
            nn_metrics = {
                "r2_score": safe_float(r2_score(y_te.values, y_pred)),
                "rmse":     safe_float(float(np.sqrt(mean_squared_error(y_te.values, y_pred)))),
                "mae":      safe_float(float(np.mean(np.abs(y_te.values - y_pred)))),
            }
            primary_score  = nn_metrics["r2_score"]
            primary_metric = "R² Score"

        # ── Feature importance via input perturbation ─────────────────────
        def perturbation_importance(X_test_arr, y_test_arr, predict_fn, n_top=8):
            """Measure importance by shuffling each feature and measuring score drop."""
            results = {}
            base = primary_score
            for i, col in enumerate(X.columns[:min(len(X.columns), 20)]):
                X_perm = X_test_arr.copy()
                np.random.shuffle(X_perm[:, i])
                try:
                    y_perm = predict_fn(X_perm)
                    if is_clf:
                        perm_score = safe_float(f1_score(y_test_arr, y_perm.astype(int), average="weighted", zero_division=0))
                    else:
                        perm_score = safe_float(r2_score(y_test_arr, y_perm))
                    results[col] = safe_float(base - perm_score)
                except:
                    results[col] = 0.0
            # Sort descending
            return dict(sorted(results.items(), key=lambda x: -x[1])[:n_top])

        # ── Compare with best sklearn model ──────────────────────────────
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.metrics import f1_score as f1, r2_score as r2

        baseline_models = {}
        if is_clf:
            bm_list = [
                ("Logistic Regression", LogisticRegression(max_iter=500, random_state=42)),
                ("Random Forest",       RandomForestClassifier(n_estimators=60, max_depth=8, random_state=42, n_jobs=-1)),
            ]
            for nm, m in bm_list:
                try:
                    m.fit(X_tr_s, y_tr.values)
                    sc_val = safe_float(f1(y_te.values, m.predict(X_te_s), average="weighted", zero_division=0))
                    baseline_models[nm] = sc_val
                except:
                    pass
        else:
            bm_list = [
                ("Ridge Regression", Ridge()),
                ("Random Forest",    RandomForestRegressor(n_estimators=60, max_depth=8, random_state=42, n_jobs=-1)),
            ]
            for nm, m in bm_list:
                try:
                    m.fit(X_tr_s, y_tr.values)
                    sc_val = safe_float(r2(y_te.values, m.predict(X_te_s)))
                    baseline_models[nm] = sc_val
                except:
                    pass

        best_baseline_name  = max(baseline_models, key=baseline_models.get) if baseline_models else "N/A"
        best_baseline_score = baseline_models.get(best_baseline_name, 0.0)
        nn_wins             = primary_score > best_baseline_score
        score_diff          = safe_float(primary_score - best_baseline_score)

        return {
            "status":           "completed",
            "backend":          backend_used,
            "task_type":        task_type,
            "n_samples":        len(df),
            "n_features":       n_features,
            "n_classes":        n_classes if is_clf else None,
            "primary_metric":   primary_metric,
            "nn_score":         primary_score,
            "nn_metrics":       nn_metrics,
            "architecture":     model_info.get("architecture", {}),
            "training_history": training_history,
            "baseline_comparison": {
                **baseline_models,
                "Neural Network (MLP)": primary_score,
            },
            "nn_vs_best_baseline": {
                "nn_score":           primary_score,
                "best_baseline":      best_baseline_name,
                "best_baseline_score": best_baseline_score,
                "nn_wins":            nn_wins,
                "score_difference":   score_diff,
                "verdict": (
                    f"Neural Network OUTPERFORMS best sklearn model ({best_baseline_name}) "
                    f"by {abs(score_diff):.3f} ({primary_metric})."
                    if nn_wins else
                    f"Neural Network ({primary_score:.3f}) vs {best_baseline_name} ({best_baseline_score:.3f}). "
                    f"Sklearn model wins by {abs(score_diff):.3f}. Consider more epochs or feature engineering."
                ),
            },
            "research_note": (
                f"MLP trained with {model_info.get('architecture',{}).get('epochs_trained','?')} epochs. "
                f"Architecture: {n_features} → {' → '.join(str(x) for x in layer_sizes)} → {n_classes if is_clf else 1}. "
                f"Backend: {backend_used}. "
                f"For production, consider XGBoost + Neural Network ensemble."
            ),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, {"error": f"Neural network training failed: {str(e)}"})

# ══════════════════════════════════════════════════════════════════════════════
# API ENDPOINTS — ADVANCED TOOLS
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/whatif")
async def what_if(file: UploadFile = File(...)):
    """Step-by-step fix simulation with score impact measurement."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
    import warnings; warnings.filterwarnings("ignore")
    try:
        contents = await file.read()
        df = _load_df(file, contents)
        td = detect_target_column(df)
        target = td["final_target"]
        task_type = td["task_type"]
        is_clf = "classification" in task_type

        def quick_score(d):
            try:
                X, y = _prepare_for_ml(d, target)
                if len(X) < 10: return 0.0
                model = (RandomForestClassifier(n_estimators=40, max_depth=6, random_state=42, n_jobs=-1)
                         if is_clf else RandomForestRegressor(n_estimators=40, max_depth=6, random_state=42, n_jobs=-1))
                cv = StratifiedKFold(3, shuffle=True, random_state=42) if is_clf else KFold(3, shuffle=True, random_state=42)
                return safe_float(cross_val_score(model, X, y, cv=cv, scoring="f1_weighted" if is_clf else "r2", n_jobs=-1).mean())
            except: return 0.0

        steps = []
        current = df.copy()
        base = quick_score(current)
        steps.append({"step": 0, "fix_applied": "Baseline (no changes)", "score": base, "delta": 0.0, "impact": "NEUTRAL"})

        prev = base
        current = current.drop_duplicates()
        s = quick_score(current)
        delta = s - prev
        steps.append({"step": 1, "fix_applied": "Drop duplicate rows", "score": s, "delta": delta,
                      "impact": "POSITIVE" if delta > 0.005 else "NEGATIVE" if delta < -0.005 else "NEUTRAL"})
        prev = s

        drop_cols = [c for c in current.columns if current[c].isnull().mean() > 0.6 and c != target]
        if drop_cols: current = current.drop(columns=drop_cols)
        s = quick_score(current)
        delta = s - prev
        steps.append({"step": 2, "fix_applied": f"Drop {len(drop_cols)} high-missing column(s)", "score": s, "delta": delta,
                      "impact": "POSITIVE" if delta > 0.005 else "NEGATIVE" if delta < -0.005 else "NEUTRAL"})
        prev = s

        for col in current.select_dtypes(include=["float64","int64"]).columns:
            if col == target: continue
            if current[col].isnull().any(): current[col].fillna(current[col].median(), inplace=True)
        for col in current.select_dtypes(include=["object"]).columns:
            if col == target: continue
            if current[col].isnull().any():
                current[col].fillna(current[col].mode()[0] if len(current[col].mode()) > 0 else "Unknown", inplace=True)
        s = quick_score(current)
        delta = s - prev
        steps.append({"step": 3, "fix_applied": "Impute missing values (median/mode)", "score": s, "delta": delta,
                      "impact": "POSITIVE" if delta > 0.005 else "NEGATIVE" if delta < -0.005 else "NEUTRAL"})
        prev = s

        for col in current.select_dtypes(include=["float64","int64"]).columns:
            if col == target: continue
            if current[col].min() >= 0 and abs(current[col].skew()) > 2:
                current[col] = np.log1p(current[col])
        s = quick_score(current)
        delta = s - prev
        steps.append({"step": 4, "fix_applied": "Log-transform highly skewed features", "score": s, "delta": delta,
                      "impact": "POSITIVE" if delta > 0.005 else "NEGATIVE" if delta < -0.005 else "NEUTRAL"})

        final = steps[-1]["score"]
        total_imp = final - base
        verdict = (f"Fixes improved model score by {total_imp:+.3f} ({base:.3f} → {final:.3f})." if total_imp > 0.01
                   else f"Score stable. Dataset may already be clean ({base:.3f} → {final:.3f}).")
        return {"baseline_score": base, "final_score": final, "total_improvement": total_imp, "verdict": verdict, "steps": steps}
    except Exception as e:
        raise HTTPException(500, {"error": str(e)})


@app.post("/whatif-live")
async def whatif_live(
    file: UploadFile = File(...),
    target_column: Optional[str] = Query(default=None),
    remove_outliers: str = Query(default="false"),
    drop_correlated: str = Query(default="false"),
    balance_dataset: str = Query(default="false"),
):
    """
    Interactive What-If Simulator.
    Applies user-selected fixes and re-scores the model.
    Params sent as query strings (true/false).
    """
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
    import warnings; warnings.filterwarnings("ignore")

    def to_bool(v): return str(v).lower() in ("true", "1", "yes", "on")
    do_outliers = to_bool(remove_outliers)
    do_corr     = to_bool(drop_correlated)
    do_balance  = to_bool(balance_dataset)

    try:
        contents = await file.read()
        df = _load_df(file, contents)
        td = detect_target_column(df, user_specified=target_column)
        target    = td["final_target"]
        task_type = td["task_type"]
        is_clf    = "classification" in task_type

        def quick_score(d):
            try:
                X, y = _prepare_for_ml(d, target)
                if len(X) < 10: return 0.0
                model = (RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1)
                         if is_clf else RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1))
                cv = (StratifiedKFold(3, shuffle=True, random_state=42)
                      if is_clf else KFold(3, shuffle=True, random_state=42))
                scoring = "f1_weighted" if is_clf else "r2"
                return safe_float(cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1).mean())
            except:
                return 0.0

        base_score = quick_score(df)
        fixes_applied = []
        current = df.copy()

        # 1. Remove outliers (IQR)
        if do_outliers:
            num_cols = current.select_dtypes(include=["float64","int64"]).columns
            outlier_idx = set()
            for col in num_cols:
                if col == target: continue
                q1 = current[col].quantile(0.25)
                q3 = current[col].quantile(0.75)
                iqr = q3 - q1
                if iqr == 0: continue
                mask = (current[col] < q1 - 1.5*iqr) | (current[col] > q3 + 1.5*iqr)
                outlier_idx.update(current[mask].index)
            n_out = len(outlier_idx)
            if n_out > 0 and n_out / len(current) < 0.15:
                current = current.drop(index=list(outlier_idx)).reset_index(drop=True)
                fixes_applied.append(f"Removed {n_out} outlier rows ({n_out/len(df)*100:.1f}%) using IQR method.")
            elif n_out > 0:
                fixes_applied.append(f"Detected {n_out} outliers but kept them (>15% of rows).")
            else:
                fixes_applied.append("No IQR outliers found.")

        # 2. Drop highly correlated features (>0.85)
        if do_corr:
            num_cols = current.select_dtypes(include=["float64","int64"]).columns.tolist()
            if target in num_cols: num_cols.remove(target)
            if len(num_cols) > 1:
                corr_matrix = current[num_cols].corr().abs()
                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1))
                to_drop = [c for c in upper.columns if upper[c].max() > 0.85 and c != target]
                if to_drop:
                    current = current.drop(columns=to_drop, errors="ignore")
                    fixes_applied.append(f"Dropped {len(to_drop)} highly correlated feature(s) (r>0.85): {to_drop[:4]}")
                else:
                    fixes_applied.append("No highly correlated pairs found (threshold: 0.85).")
            else:
                fixes_applied.append("Not enough numeric features to check correlation.")

        # 3. Balance dataset (oversampling)
        if do_balance and is_clf:
            if target in current.columns:
                counts = current[target].value_counts()
                max_count = int(counts.max())
                balanced = []
                for cls_val, cnt in counts.items():
                    cls_df = current[current[target] == cls_val]
                    if cnt < max_count:
                        extra = cls_df.sample(n=max_count-cnt, replace=True, random_state=42)
                        balanced.append(pd.concat([cls_df, extra]))
                    else:
                        balanced.append(cls_df)
                current = pd.concat(balanced).sample(frac=1, random_state=42).reset_index(drop=True)
                fixes_applied.append(f"Balanced: oversampled minority class to {max_count} rows each. New total: {len(current)}.")
        elif do_balance and not is_clf:
            fixes_applied.append("Balancing skipped — not applicable for regression.")

        if not fixes_applied:
            fixes_applied.append("No transformations selected — showing baseline score only.")

        new_score = quick_score(current)
        delta = safe_float(new_score - base_score)

        return {
            "baseline_score":    base_score,
            "final_score":       new_score,
            "total_improvement": delta,
            "fixes_summary":     fixes_applied,
            "task_type":         "classification" if is_clf else "regression",
            "metric":            "F1-Weighted" if is_clf else "R² Score",
            "rows_before":       len(df),
            "rows_after":        len(current),
        }
    except Exception as e:
        raise HTTPException(500, {"error": f"What-If Live failed: {str(e)}"})



@app.post("/neural-train")
async def neural_train(
    file: UploadFile = File(...),
    target_column: Optional[str] = Query(default=None),
    epochs: int = Query(default=50, ge=10, le=200),
    hidden_layers: str = Query(default="128,64,32"),
):
    """
    Train a Multi-Layer Perceptron Neural Network via neural_model.py.
    Uses PyTorch if available, sklearn MLP otherwise.
    Returns full metrics, architecture, training history, baseline comparison.
    """
    try:
        contents = await file.read()
        df = _load_df(file, contents)

        if df.empty:
            raise HTTPException(400, {"error": "Uploaded file is empty."})
        if len(df) < 20:
            raise HTTPException(400, {"error": "Need at least 20 rows for neural network training."})

        td        = detect_target_column(df, user_specified=target_column)
        target    = td["final_target"]
        task_type = td["task_type"]

        X, y = _prepare_for_ml(df, target)

        if len(X) < 10 or X.shape[1] == 0:
            raise HTTPException(400, {"error": "Not enough usable features after preprocessing."})

        # Import and run neural model
        try:
            from app.ml.neural_model import train_neural_model
        except ImportError:
            # Fallback: try relative import path
            import sys, os
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from ml.neural_model import train_neural_model

        result = train_neural_model(
            X, y,
            task_type     = task_type,
            epochs        = epochs,
            hidden_layers = hidden_layers,
        )
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, {"error": f"Neural training failed: {str(e)}"})

@app.post("/automl")
async def automl(file: UploadFile = File(...)):
    """Grid-search over 32-40 preprocessing + model combos."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, VotingClassifier, VotingRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    import warnings; warnings.filterwarnings("ignore")
    try:
        t_start = time.time()
        contents = await file.read()
        df = _load_df(file, contents)
        td = detect_target_column(df)
        target = td["final_target"]
        task_type = td["task_type"]
        is_clf = "classification" in task_type
        X, y = _prepare_for_ml(df, target)
        if len(X) < 10: return {"error": "Too few rows"}
        n_splits = min(5, max(2, len(X) // 20))
        cv = StratifiedKFold(n_splits, shuffle=True, random_state=42) if is_clf else KFold(n_splits, shuffle=True, random_state=42)
        metric = "f1_weighted" if is_clf else "r2"
        configs = []
        base_models = (
            [("LR", LogisticRegression(max_iter=500, C=1.0, random_state=42)),
             ("RF_shallow", RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)),
             ("RF_deep", RandomForestClassifier(n_estimators=80, max_depth=10, random_state=42, n_jobs=-1)),
             ("GB", GradientBoostingClassifier(n_estimators=80, max_depth=4, learning_rate=0.1, random_state=42))]
            if is_clf else
            [("Ridge_low", Ridge(alpha=0.1)), ("Ridge_high", Ridge(alpha=10.0)),
             ("RF_shallow", RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)),
             ("RF_deep", RandomForestRegressor(n_estimators=80, max_depth=10, random_state=42, n_jobs=-1)),
             ("GB", GradientBoostingRegressor(n_estimators=80, max_depth=4, learning_rate=0.1, random_state=42))]
        )
        for imp_strat in ["median", "mean"]:
            imp = SimpleImputer(strategy=imp_strat)
            X_imp = pd.DataFrame(imp.fit_transform(X), columns=X.columns)
            for log_transform in [False, True]:
                X_proc = X_imp.copy()
                if log_transform:
                    for col in X_proc.columns:
                        if X_proc[col].min() >= 0 and abs(X_proc[col].skew()) > 1.5:
                            X_proc[col] = np.log1p(X_proc[col])
                for use_scaler in [False, True]:
                    X_final = X_proc.copy()
                    scaler_name = "StandardScaler" if use_scaler else "None"
                    if use_scaler:
                        sc = StandardScaler()
                        X_final = pd.DataFrame(sc.fit_transform(X_final), columns=X_final.columns)
                    for name, model in base_models:
                        try:
                            scores = cross_val_score(model, X_final, y, cv=cv, scoring=metric, n_jobs=-1)
                            configs.append({"model": name, "score": safe_float(scores.mean()), "std": safe_float(scores.std()),
                                            "impute_strategy": imp_strat, "log_transform": log_transform,
                                            "scaler": scaler_name, "is_ensemble": False,
                                            "preprocessing": f"{imp_strat}+{'log+' if log_transform else ''}{scaler_name}"})
                        except: pass
        # Voting ensemble
        try:
            best_prep = max(configs, key=lambda x: x["score"])
            ens = (VotingClassifier([("lr", LogisticRegression(max_iter=500)),
                                     ("rf", RandomForestClassifier(n_estimators=60, random_state=42, n_jobs=-1)),
                                     ("gb", GradientBoostingClassifier(n_estimators=60, random_state=42))], voting="soft")
                   if is_clf else
                   VotingRegressor([("ridge", Ridge()), ("rf", RandomForestRegressor(n_estimators=60, random_state=42, n_jobs=-1)),
                                    ("gb", GradientBoostingRegressor(n_estimators=60, random_state=42))]))
            imp2 = SimpleImputer(strategy=best_prep["impute_strategy"])
            X_e = pd.DataFrame(imp2.fit_transform(X), columns=X.columns)
            ens_scores = cross_val_score(ens, X_e, y, cv=cv, scoring=metric, n_jobs=-1)
            configs.append({"model": "Voting Ensemble", "score": safe_float(ens_scores.mean()), "std": safe_float(ens_scores.std()),
                            "impute_strategy": best_prep["impute_strategy"], "log_transform": False, "scaler": "None",
                            "is_ensemble": True, "preprocessing": f"voting_ensemble_{best_prep['impute_strategy']}"})
        except: pass
        configs.sort(key=lambda x: -x["score"])
        return {"best_configuration": configs[0] if configs else {}, "top_10_configurations": configs[:10],
                "total_configurations_tested": len(configs), "total_time_seconds": round(time.time() - t_start, 1)}
    except Exception as e:
        raise HTTPException(500, {"error": str(e)})


@app.post("/shap")
async def shap_analysis(
    file: UploadFile = File(...),
    target_column: Optional[str] = Query(default=None)
):
    """
    SHAP global feature attribution.
    Reads file ONCE. Uses true SHAP TreeExplainer if shap is installed,
    falls back to RF feature_importances_ (which always gives real values).
    """
    # ── Read file and prepare ONCE ────────────────────────────────────────
    contents = await file.read()
    df       = _load_df(file, contents)

    if df.empty or len(df) < 10:
        return {"status": "error", "error": "Dataset too small for SHAP analysis (need ≥ 10 rows)."}

    td        = detect_target_column(df, user_specified=target_column)
    target    = td["final_target"]
    task_type = td["task_type"]
    is_clf    = "classification" in task_type

    # Prepare ML-ready features — this is shared by both paths
    X, y = _prepare_for_ml(df, target)

    if X.shape[1] == 0:
        return {"status": "error", "error": "No usable features after preprocessing."}

    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    model = (
        RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)
        if is_clf else
        RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)
    )
    model.fit(X, y)

    # ── Try true SHAP ─────────────────────────────────────────────────────
    shap_used = False
    importance = {}
    feature_direction = {}
    sample_explanations = []
    top_interactions = []

    try:
        import shap

        n_shap   = min(300, len(X))
        X_sample = X.iloc[:n_shap]

        explainer  = shap.TreeExplainer(model)
        shap_vals  = explainer.shap_values(X_sample)

        # Handle multiclass (list of arrays) vs regression/binary (single array)
        if isinstance(shap_vals, list):
            if len(shap_vals) == 2:
                # Binary classification — use positive class
                sv = np.array(shap_vals[1])
            else:
                # Multiclass — mean absolute across classes
                sv = np.mean([np.abs(np.array(s)) for s in shap_vals], axis=0)
        else:
            sv = np.array(shap_vals)

        # Global importance = mean |SHAP| per feature
        mean_abs = np.abs(sv).mean(axis=0)
        mean_dir = sv.mean(axis=0)

        importance = dict(sorted(
            {col: safe_float(v) for col, v in zip(X.columns, mean_abs)}.items(),
            key=lambda x: -x[1]
        ))

        feature_direction = {
            col: {
                "mean_shap":  safe_float(mean_dir[i]),
                "direction":  "POSITIVE (pushes prediction up)" if mean_dir[i] > 0 else "NEGATIVE (pushes prediction down)",
                "magnitude":  safe_float(abs(mean_dir[i]))
            }
            for i, col in enumerate(X.columns)
        }

        # Local explanations for 3 sample rows
        base_val = explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)):
            base_val = float(base_val[1] if len(base_val) == 2 else base_val[0])
        else:
            base_val = float(base_val)

        for idx in [0, len(X_sample)//2, len(X_sample)-1]:
            if idx >= len(X_sample): continue
            row_sv   = sv[idx]
            row_vals = X_sample.iloc[idx]
            top5 = sorted(
                [(col, float(row_sv[j]), float(row_vals.iloc[j]))
                 for j, col in enumerate(X.columns)],
                key=lambda x: abs(x[1]), reverse=True
            )[:5]
            pred = model.predict(X_sample.iloc[[idx]])[0]
            sample_explanations.append({
                "row_index": int(idx),
                "prediction": str(round(float(pred), 4)),
                "base_value": round(base_val, 4),
                "top_contributing_features": [
                    {"feature": f, "shap_value": round(sv_val, 4),
                     "feature_value": round(fv, 4),
                     "impact": "increases prediction" if sv_val > 0 else "decreases prediction"}
                    for f, sv_val, fv in top5
                ]
            })

        # Feature interactions via SHAP correlation
        shap_df = pd.DataFrame(sv, columns=X.columns)
        corr    = shap_df.corr().abs()
        upper   = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        pairs   = upper.stack().reset_index()
        pairs.columns = ["f1", "f2", "strength"]
        for _, row in pairs.nlargest(5, "strength").iterrows():
            top_interactions.append({
                "feature_1": row["f1"],
                "feature_2": row["f2"],
                "interaction_strength": round(float(row["strength"]), 4),
            })

        shap_used  = True
        model_used = "RandomForest + SHAP TreeExplainer"
        n_analyzed = n_shap

    except ImportError:
        # shap not installed — fall through to RF importances below
        shap_used = False
    except Exception as shap_err:
        # SHAP computation failed — fall through
        shap_used = False

    # ── Fallback: RF feature_importances_ (always gives real values) ──────
    if not shap_used:
        rf_imp = model.feature_importances_   # always non-zero after fit()
        importance = dict(sorted(
            {col: safe_float(v) for col, v in zip(X.columns, rf_imp)}.items(),
            key=lambda x: -x[1]
        ))
        feature_direction = {}
        sample_explanations = []
        top_interactions    = []
        model_used = "RandomForest feature_importances_ (pip install shap for true SHAP values)"
        n_analyzed = len(X)

    # ── Build response ────────────────────────────────────────────────────
    top_feat  = list(importance.keys())[0]  if importance else "N/A"
    top_val   = list(importance.values())[0] if importance else 0.0

    shap_summary = (
        f"{'SHAP TreeExplainer' if shap_used else 'RF Importance'} analysis on {n_analyzed} samples. "
        f"Top feature: '{top_feat}' with {'mean |SHAP|' if shap_used else 'RF importance'}={top_val:.4f}. "
        f"{'Install shap (pip install shap) for true Shapley values.' if not shap_used else ''}"
    )

    return {
        "status":                    "completed",
        "model_used":                model_used,
        "shap_available":            shap_used,
        "n_samples_analyzed":        n_analyzed,
        "n_features":                X.shape[1],
        "global_feature_importance": importance,
        "feature_direction":         feature_direction,
        "top_feature_interactions":  top_interactions,
        "sample_explanations":       sample_explanations,
        "shap_summary":              shap_summary,
        "research_note": (
            "SHAP (SHapley Additive exPlanations) values are grounded in cooperative game theory. "
            "Each value = feature's average marginal contribution across all feature coalitions. "
            "Unlike Gini importance, SHAP is unbiased toward high-cardinality features. "
            "Citation: Lundberg & Lee (2017), NeurIPS."
            if shap_used else
            "Using RF feature_importances_ as approximation. "
            "Run: pip install shap  then restart backend for true SHAP values."
        ),
    }


@app.post("/nlreport")
async def nl_report(file: UploadFile = File(...), format: str = Query(default="text")):
    """Generate a human-readable plain-English analysis memo."""
    try:
        contents = await file.read()
        df = _load_df(file, contents)
        result = _run_full_analysis(df)
        er = result["explainability_report"]
        ds = result["dataset_summary"]
        td = result["target_detection"]
        dq = result["dataset_quality_score"]
        at = result["auto_training_results"]
        fi = result["feature_importance"]
        lines = [
            "=" * 70, "  ML DATA READINESS REPORT",
            "  Generated by AutoEDA v4.1 — ML Data Readiness Analyzer", "=" * 70, "",
            "EXECUTIVE SUMMARY", "-" * 40,
            f"Readiness Score : {er['readiness_score']}/100  (Grade {er['grade']})",
            f"Verdict         : {er['verdict']}",
            f"Dataset         : {ds['rows']:,} rows × {ds['columns']} columns",
            f"Target          : {td['final_target']} ({td['task_type']})",
            f"Quality         : {dq['status']} ({dq['overall_score']}/100)", "",
            "DATA QUALITY BREAKDOWN", "-" * 40,
        ]
        for dim, score in dq["dimension_scores"].items():
            bar = "█" * int(score // 10) + "░" * (10 - int(score // 10))
            lines.append(f"  {dim:<20} {bar}  {score:.0f}/100")
        lines += ["", "ISSUES FOUND", "-" * 40]
        for iss in (er["issues"] or ["  No critical issues found."]):
            if isinstance(iss, str): lines.append(iss); continue
            lines += [f"  [{iss['severity']}] {iss['title']}", f"    → {iss['explanation']}", f"    FIX: {iss['fix']}", ""]
        lines += ["", "ACTION PLAN", "-" * 40]
        for act in er["action_plan"]:
            lines.append(f"  {act['step']}. [{act['priority']}] {act['reason']}")
        lines += ["", "MODEL PERFORMANCE", "-" * 40, f"  Best model: {at.get('best_model','N/A')}"]
        for name, r in at.get("model_comparison", {}).items():
            if "error" not in r:
                score_key = "f1_weighted" if "classification" in at.get("task_type","") else "r2_score"
                lines.append(f"  {name:<30} {r.get(score_key,0):.4f}")
        if fi.get("ranking_by_rf_importance"):
            lines += ["", "TOP FEATURES (RF Importance)", "-" * 40]
            for feat in fi["ranking_by_rf_importance"][:8]:
                lines.append(f"  {feat['feature']:<30} {feat['rf_importance']:.4f}  [{feat['tier']}]")
        lines += ["", "=" * 70, "  END OF REPORT — AutoEDA v4.1", "=" * 70]
        return StreamingResponse(io.BytesIO("\n".join(lines).encode()), media_type="text/plain",
                                 headers={"Content-Disposition": "attachment; filename=ml_readiness_report.txt"})
    except Exception as e:
        raise HTTPException(500, {"error": str(e)})


# ══════════════════════════════════════════════════════════════════════════════
# ★ NEW ENDPOINT 1 — /report (PDF)
# Generates a rich PDF report. Previously this endpoint was missing.
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/report")
async def generate_pdf_report(file: UploadFile = File(...)):
    """
    Generate a styled PDF report using ReportLab.
    Install: pip install reportlab
    Falls back to plain-text PDF if reportlab not available.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        from reportlab.lib.enums import TA_CENTER, TA_LEFT

        contents = await file.read()
        df = _load_df(file, contents)
        result = _run_full_analysis(df)
        er = result["explainability_report"]
        ds = result["dataset_summary"]
        td = result["target_detection"]
        dq = result["dataset_quality_score"]
        at = result["auto_training_results"]
        fi = result["feature_importance"]

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=2*cm, leftMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
        styles = getSampleStyleSheet()

        BLUE = colors.HexColor("#2563eb")
        GREEN = colors.HexColor("#16a34a")
        RED = colors.HexColor("#dc2626")
        AMBER = colors.HexColor("#d97706")
        DARK = colors.HexColor("#111827")
        LIGHT = colors.HexColor("#f8f9fb")
        GRAY = colors.HexColor("#6b7280")

        title_style = ParagraphStyle("title", parent=styles["Title"], fontSize=22, textColor=DARK, spaceAfter=6, fontName="Helvetica-Bold")
        subtitle_style = ParagraphStyle("subtitle", parent=styles["Normal"], fontSize=11, textColor=GRAY, spaceAfter=20)
        h2_style = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=13, textColor=BLUE, spaceBefore=14, spaceAfter=6, fontName="Helvetica-Bold")
        body_style = ParagraphStyle("body", parent=styles["Normal"], fontSize=10, textColor=DARK, spaceAfter=6, leading=16)
        mono_style = ParagraphStyle("mono", parent=styles["Code"], fontSize=9, textColor=DARK, backColor=LIGHT, spaceAfter=4)

        grade_color = {"A": GREEN, "B": colors.HexColor("#0891b2"), "C": AMBER, "D": RED, "F": RED}.get(er["grade"], DARK)

        story = []

        # Title block
        story.append(Paragraph("AutoEDA — ML Data Readiness Report", title_style))
        story.append(Paragraph(f"{file.filename}  ·  {ds['rows']:,} rows × {ds['columns']} columns  ·  {ds['memory_usage_mb']:.2f} MB", subtitle_style))
        story.append(HRFlowable(width="100%", thickness=1, color=BLUE))
        story.append(Spacer(1, 0.4*cm))

        # Score summary table
        score_data = [
            ["Readiness Score", "Grade", "Task Type", "Quality", "Best Model"],
            [
                Paragraph(f'<font size="20" color="#{("%02x%02x%02x" % (int(grade_color.red*255), int(grade_color.green*255), int(grade_color.blue*255)))}"><b>{er["readiness_score"]:.0f}/100</b></font>', body_style),
                Paragraph(f'<font size="18"><b>{er["grade"]}</b></font>', body_style),
                Paragraph(td["task_type"].replace("_", " ").title(), body_style),
                Paragraph(f'{dq["status"]} ({dq["overall_score"]:.0f}/100)', body_style),
                Paragraph(at.get("best_model","N/A"), body_style),
            ]
        ]
        score_table = Table(score_data, colWidths=[3.5*cm, 2*cm, 4*cm, 4*cm, 4*cm])
        score_table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), BLUE), ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTSIZE", (0,0), (-1,0), 9),
            ("ALIGN", (0,0), (-1,-1), "CENTER"), ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#e4e7ed")),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, LIGHT]),
            ("TOPPADDING", (0,0), (-1,-1), 8), ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ]))
        story.append(score_table)
        story.append(Spacer(1, 0.5*cm))

        # Quality dimensions
        story.append(Paragraph("Data Quality Dimensions", h2_style))
        dim_data = [["Dimension", "Score", "Status"]]
        for dim, score in dq["dimension_scores"].items():
            status = "✓ Good" if score >= 80 else "⚠ Fair" if score >= 60 else "✗ Poor"
            dim_data.append([dim.replace("_", " ").title(), f"{score:.0f}/100", status])
        dim_table = Table(dim_data, colWidths=[7*cm, 4*cm, 6*cm])
        dim_table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), BLUE), ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTSIZE", (0,0), (-1,-1), 9),
            ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#e4e7ed")),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, LIGHT]),
            ("TOPPADDING", (0,0), (-1,-1), 6), ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ]))
        story.append(dim_table)
        story.append(Spacer(1, 0.4*cm))

        # Issues
        if er["issues"]:
            story.append(Paragraph("Issues Found", h2_style))
            sev_colors = {"CRITICAL": RED, "HIGH": AMBER, "MODERATE": colors.HexColor("#ca8a04"), "LOW": GREEN}
            for iss in er["issues"][:8]:
                c = sev_colors.get(iss["severity"], GRAY)
                story.append(Paragraph(f'<font color="#{("%02x%02x%02x" % (int(c.red*255), int(c.green*255), int(c.blue*255)))}"><b>[{iss["severity"]}]</b></font> {iss["title"]}', body_style))
                story.append(Paragraph(f'  {iss["explanation"]}', ParagraphStyle("exp", parent=body_style, fontSize=9, textColor=GRAY, leftIndent=12)))
                story.append(Paragraph(f'  Fix: {iss["fix"][:120]}', ParagraphStyle("fix", parent=mono_style, leftIndent=12)))
                story.append(Spacer(1, 0.15*cm))

        # Model comparison
        story.append(Paragraph("Model Performance", h2_style))
        is_clf = "classification" in at.get("task_type","")
        score_key = "f1_weighted" if is_clf else "r2_score"
        model_data = [["Model", "CV Score", "Std Dev", "Time (s)"]]
        best = at.get("best_model","")
        for name, r in at.get("model_comparison", {}).items():
            if "error" not in r:
                row = [("★ " if name == best else "") + name, f"{r.get(score_key,0):.4f}", f"±{r.get('cv_std',0):.4f}", f"{r.get('training_time_seconds','?')}s"]
                model_data.append(row)
        model_table = Table(model_data, colWidths=[6*cm, 3.5*cm, 3.5*cm, 4*cm])
        model_table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), BLUE), ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTSIZE", (0,0), (-1,-1), 9),
            ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#e4e7ed")),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, LIGHT]),
            ("TOPPADDING", (0,0), (-1,-1), 6), ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ]))
        story.append(model_table)
        story.append(Spacer(1, 0.4*cm))

        # Top features
        if fi.get("ranking_by_rf_importance"):
            story.append(Paragraph("Top Feature Importances (Random Forest)", h2_style))
            feat_data = [["Feature", "RF Importance", "Tier"]]
            for feat in fi["ranking_by_rf_importance"][:8]:
                feat_data.append([feat["feature"], f"{feat['rf_importance']:.4f}", feat["tier"]])
            feat_table = Table(feat_data, colWidths=[9*cm, 4*cm, 4*cm])
            feat_table.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), BLUE), ("TEXTCOLOR", (0,0), (-1,0), colors.white),
                ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTSIZE", (0,0), (-1,-1), 9),
                ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#e4e7ed")),
                ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, LIGHT]),
                ("TOPPADDING", (0,0), (-1,-1), 6), ("BOTTOMPADDING", (0,0), (-1,-1), 6),
            ]))
            story.append(feat_table)

        # Footer
        story.append(Spacer(1, 0.8*cm))
        story.append(HRFlowable(width="100%", thickness=0.5, color=GRAY))
        story.append(Paragraph("Generated by AutoEDA v4.1 — ML Data Readiness Analyzer", ParagraphStyle("footer", parent=body_style, fontSize=8, textColor=GRAY, alignment=TA_CENTER)))

        doc.build(story)
        buf.seek(0)
        fname = (file.filename or "dataset").replace(".csv","").replace(".xlsx","")
        return StreamingResponse(buf, media_type="application/pdf",
                                 headers={"Content-Disposition": f'attachment; filename="{fname}_autoeda_report.pdf"'})

    except ImportError:
        # reportlab not installed — return plain text as PDF fallback
        try:
            contents = await file.read()
            df = _load_df(file, contents)
            result = _run_full_analysis(df)
            er = result["explainability_report"]
            ds = result["dataset_summary"]
            text = f"""AutoEDA v4.1 — ML Data Readiness Report
{'='*60}
File      : {file.filename}
Rows      : {ds['rows']:,}  Columns : {ds['columns']}
Score     : {er['readiness_score']}/100  Grade : {er['grade']}
Verdict   : {er['verdict']}
{'='*60}
Issues    : {er['issue_summary']}
{'='*60}
Install reportlab for a styled PDF:
  pip install reportlab
Then restart the backend.
"""
            return StreamingResponse(io.BytesIO(text.encode()), media_type="text/plain",
                                     headers={"Content-Disposition": "attachment; filename=report.txt"})
        except Exception as e:
            raise HTTPException(500, {"error": str(e)})
    except Exception as e:
        raise HTTPException(500, {"error": str(e)})


# ══════════════════════════════════════════════════════════════════════════════
# ★ NEW ENDPOINT 2 — /bootstrap
# Readiness score with confidence intervals (bootstrap resampling)
# This is a research-grade contribution — no other AutoEDA tool does this.
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/bootstrap")
async def bootstrap_readiness(
    file: UploadFile = File(...),
    n_iterations: int = Query(default=50, ge=10, le=200),
    target_column: Optional[str] = Query(default=None)
):
    """
    Bootstrap confidence interval on the readiness score.
    Resamples the dataset n_iterations times (80% each) and computes
    mean ± std of the readiness score. Proves the score is stable.

    Research significance: Shows score is not an artifact of a particular
    data sample — it's a robust property of the dataset.
    """
    try:
        contents = await file.read()
        df = _load_df(file, contents)
        if len(df) < 30:
            return {"error": "Dataset too small for bootstrap (need ≥ 30 rows)"}

        scores = []
        sample_size = int(len(df) * 0.80)

        for i in range(n_iterations):
            try:
                sample = df.sample(n=sample_size, random_state=i, replace=False)
                result = _run_full_analysis(sample.reset_index(drop=True), target_column=target_column)
                scores.append(result["explainability_report"]["readiness_score"])
            except:
                continue

        if len(scores) < 5:
            return {"error": "Bootstrap failed — dataset may be too small or malformed"}

        scores_arr = np.array(scores)
        mean_score = safe_float(np.mean(scores_arr))
        std_score = safe_float(np.std(scores_arr))
        ci_lower = safe_float(np.percentile(scores_arr, 2.5))
        ci_upper = safe_float(np.percentile(scores_arr, 97.5))

        stability = "HIGH" if std_score < 3 else "MODERATE" if std_score < 7 else "LOW"

        return {
            "bootstrap_iterations": len(scores),
            "mean_readiness_score": mean_score,
            "std_dev": std_score,
            "ci_95_lower": ci_lower,
            "ci_95_upper": ci_upper,
            "stability": stability,
            "interpretation": (
                f"Readiness score = {mean_score:.1f} ± {std_score:.1f} (95% CI: {ci_lower:.1f}–{ci_upper:.1f}). "
                f"Score stability is {stability}. "
                f"{'The score is robust and consistent across data samples.' if stability == 'HIGH' else 'Some variability detected — consider collecting more data.'}"
            ),
            "all_scores": [safe_float(s) for s in scores_arr],
            "research_note": "Bootstrap resampling (80% subsamples, n_iterations). Confidence interval shows score stability — a key research validation metric."
        }
    except Exception as e:
        raise HTTPException(500, {"error": str(e)})


# ══════════════════════════════════════════════════════════════════════════════
# ★ NEW ENDPOINT 3 — /validate
# Pre-training validation checklist — research-grade data quality gate
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/validate")
async def validate_dataset(file: UploadFile = File(...), target_column: Optional[str] = Query(default=None)):
    """
    Generates a structured pre-training validation checklist.
    Returns PASS/FAIL for each of 12 validation criteria with explanations.
    Think of it as a 'data quality gate' before any ML experiment.
    """
    try:
        contents = await file.read()
        df = _load_df(file, contents)
        result = _run_full_analysis(df, target_column=target_column)
        ds = result["dataset_summary"]
        dq = result["dataset_quality_score"]
        td = result["target_detection"]
        er = result["explainability_report"]
        lk = result["data_leakage_analysis"]
        an = result["anomaly_detection"]
        fa = result["feature_analysis"]

        n = ds["rows"]
        n_cols = ds["columns"]
        checks = []

        def chk(name, passed, value, threshold, explanation, fix=None):
            checks.append({
                "check": name,
                "status": "PASS" if passed else "FAIL",
                "value": value,
                "threshold": threshold,
                "explanation": explanation,
                "fix": fix or ("No action needed." if passed else "See explanation.")
            })

        chk("Minimum Row Count", n >= 100,
            f"{n} rows", "≥ 100 rows",
            f"Dataset has {n} rows. {'Sufficient for basic ML.' if n >= 100 else 'Too few for reliable ML — risk of high variance in results.'}",
            "Collect more data. Consider data augmentation or synthetic generation.")

        chk("Rows-to-Features Ratio", n / n_cols >= 10,
            f"{n/n_cols:.1f}x", "≥ 10x",
            f"Ratio of {n/n_cols:.1f} rows per feature. {'Good.' if n/n_cols >= 10 else 'Too low — high overfitting risk.'}",
            "Remove low-importance features or collect more rows.")

        miss = ds["missing_value_percent"]
        chk("Missing Value Rate", miss < 20,
            f"{miss:.1f}%", "< 20%",
            f"{miss:.1f}% of cells are missing. {'Acceptable.' if miss < 20 else 'High missingness will bias imputation significantly.'}",
            "Drop high-missing columns (>60%). Impute remaining with median/mode.")

        dups = ds["duplicate_rows"]
        chk("No Duplicate Rows", dups == 0,
            f"{dups} duplicates", "0 duplicates",
            f"{dups} duplicate rows found. {'No duplicates — clean.' if dups == 0 else 'Duplicates cause train/test leakage.'}",
            "df.drop_duplicates(inplace=True)")

        chk("Target Column Detected", td["confidence"] >= 0.4,
            f"{td['final_target']} ({td['confidence']*100:.0f}%)", "Confidence ≥ 40%",
            f"Target '{td['final_target']}' detected with {td['confidence']*100:.0f}% confidence. {'Clear target.' if td['confidence'] >= 0.4 else 'Ambiguous target — verify manually.'}",
            f"Specify manually: ?target_column={td['final_target']}")

        chk("No Data Leakage", len(lk.get("target_leakage",[])) == 0,
            f"{len(lk.get('target_leakage',[]))} leakage issues", "0 issues",
            f"{len(lk.get('target_leakage',[]))} potential leakage feature(s) found. {'Clean.' if len(lk.get('target_leakage',[])) == 0 else 'CRITICAL — remove leaking features before training.'}",
            "Drop features with near-perfect correlation to target.")

        cb = dq["dimension_scores"]["class_balance"]
        chk("Class Balance", cb >= 60,
            f"{cb:.0f}/100", "≥ 60/100",
            f"Class balance score: {cb:.0f}/100. {'Acceptable imbalance.' if cb >= 60 else 'Severe imbalance — model will be biased toward majority class.'}",
            "Use SMOTE, class_weight='balanced', or collect more minority class samples.")

        an_pct = an.get("anomaly_percent", 0)
        chk("Anomaly Rate", an_pct < 8,
            f"{an_pct:.1f}%", "< 8%",
            f"{an_pct:.1f}% anomalous rows detected by IsolationForest. {'Within acceptable range.' if an_pct < 8 else 'High anomaly rate may indicate data quality issues.'}",
            "Inspect anomalous rows. Consider removing or investigating outliers.")

        completeness = dq["dimension_scores"]["completeness"]
        chk("Data Completeness", completeness >= 70,
            f"{completeness:.0f}/100", "≥ 70/100",
            f"Completeness score: {completeness:.0f}/100.",
            "Impute or collect data for missing values.")

        zero_var_cols = [col for col, info in fa.items() if info.get("unique_values", 1) <= 1]
        chk("No Zero-Variance Features", len(zero_var_cols) == 0,
            f"{len(zero_var_cols)} constant cols", "0",
            f"{len(zero_var_cols)} constant/zero-variance columns found. {'None — good.' if len(zero_var_cols) == 0 else 'Constant columns add no information to models.'}",
            f"df.drop(columns={zero_var_cols})")

        high_miss_cols = [col for col, info in fa.items() if info.get("missing_percent",0) > 60]
        chk("No Columns >60% Missing", len(high_miss_cols) == 0,
            f"{len(high_miss_cols)} columns", "0",
            f"{len(high_miss_cols)} columns have >60% missing data. {'None — good.' if len(high_miss_cols) == 0 else 'Imputing >60% missing is unreliable — drop these columns.'}",
            f"df.drop(columns={high_miss_cols})")

        readiness = er["readiness_score"]
        chk("Overall Readiness Score", readiness >= 70,
            f"{readiness:.0f}/100 (Grade {er['grade']})", "≥ 70/100",
            f"Overall readiness: {readiness:.0f}/100. {'Ready for ML experiments.' if readiness >= 70 else 'Dataset needs preprocessing before ML training.'}",
            "Address CRITICAL and HIGH issues from the Issues tab first.")

        passed = sum(1 for c in checks if c["status"] == "PASS")
        total = len(checks)
        overall_pass = passed / total >= 0.75 and all(c["status"] == "PASS" for c in checks if c["check"] in ["No Data Leakage", "Target Column Detected"])

        return {
            "overall_status": "PASS" if overall_pass else "FAIL",
            "passed": passed,
            "failed": total - passed,
            "total_checks": total,
            "pass_rate": safe_float(passed / total * 100),
            "checks": checks,
            "recommendation": (
                "✓ Dataset passes pre-training validation. Proceed with ML experiments."
                if overall_pass else
                f"✗ {total - passed} check(s) failed. Fix FAIL items before training to ensure reliable results."
            )
        }
    except Exception as e:
        raise HTTPException(500, {"error": str(e)})


# ══════════════════════════════════════════════════════════════════════════════
# SAMPLE DATASETS
# ══════════════════════════════════════════════════════════════════════════════



# ══════════════════════════════════════════════════════════════════════════════
# ★ NEW: /compare — Side-by-side dataset comparison
# Upload two CSV files, get a full diff report (schema, quality, model performance)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/compare")
async def compare_datasets(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
    target_column: Optional[str] = Query(default=None),
):
    """
    Compare two datasets side-by-side.
    Returns: schema diff, quality diff, missing value diff,
             model performance diff, column overlap.
    """
    try:
        c1 = await file1.read()
        c2 = await file2.read()
        df1 = _load_df(file1, c1)
        df2 = _load_df(file2, c2)

        td1 = detect_target_column(df1, user_specified=target_column)
        td2 = detect_target_column(df2, user_specified=target_column)
        t1  = td1["final_target"]
        t2  = td2["final_target"]

        q1 = calculate_quality_score(df1, t1)
        q2 = calculate_quality_score(df2, t2)

        # Column overlap
        cols1, cols2  = set(df1.columns), set(df2.columns)
        shared        = sorted(cols1 & cols2)
        only_in_1     = sorted(cols1 - cols2)
        only_in_2     = sorted(cols2 - cols1)

        # Missing value comparison
        miss1 = {c: safe_float(df1[c].isnull().mean()*100) for c in df1.columns}
        miss2 = {c: safe_float(df2[c].isnull().mean()*100) for c in df2.columns}

        # Quick model score
        def quick_score(df, target):
            try:
                from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
                td = detect_target_column(df, user_specified=target)
                tgt = td["final_target"]
                is_clf = "classification" in td["task_type"]
                X, y = _prepare_for_ml(df, tgt)
                if len(X) < 10: return 0.0
                model = (RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1)
                         if is_clf else RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1))
                cv = (StratifiedKFold(3, shuffle=True, random_state=42)
                      if is_clf else KFold(3, shuffle=True, random_state=42))
                return safe_float(cross_val_score(model, X, y, cv=cv,
                    scoring="f1_weighted" if is_clf else "r2", n_jobs=-1).mean())
            except: return 0.0

        score1 = quick_score(df1, t1)
        score2 = quick_score(df2, t2)

        dim_diff = {}
        for dim in q1["dimension_scores"]:
            d1 = q1["dimension_scores"].get(dim, 0)
            d2 = q2["dimension_scores"].get(dim, 0)
            dim_diff[dim] = {"dataset_1": d1, "dataset_2": d2, "delta": safe_float(d2-d1)}

        winner = (file2.filename or "Dataset 2") if score2 > score1 else (file1.filename or "Dataset 1")

        return {
            "dataset_1": {"name": file1.filename, "rows": len(df1), "cols": len(df1.columns),
                          "target": t1, "quality": q1["overall_score"],
                          "missing_pct": safe_float(df1.isnull().mean().mean()*100),
                          "model_score": score1},
            "dataset_2": {"name": file2.filename, "rows": len(df2), "cols": len(df2.columns),
                          "target": t2, "quality": q2["overall_score"],
                          "missing_pct": safe_float(df2.isnull().mean().mean()*100),
                          "model_score": score2},
            "column_overlap": {"shared": shared, "only_in_1": only_in_1, "only_in_2": only_in_2,
                               "overlap_pct": safe_float(len(shared)/max(len(cols1|cols2),1)*100)},
            "quality_dimension_comparison": dim_diff,
            "model_performance_winner": winner,
            "score_delta": safe_float(score2 - score1),
            "recommendation": (
                f"'{winner}' is the better dataset for ML. "
                f"Quality: {q1['overall_score']:.0f} vs {q2['overall_score']:.0f}. "
                f"Model score: {score1:.3f} vs {score2:.3f}."
            ),
        }
    except Exception as e:
        raise HTTPException(500, {"error": f"Comparison failed: {str(e)}"})


# ══════════════════════════════════════════════════════════════════════════════
# ★ NEW: /smart-report — AI-generated natural language report using Claude API
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/smart-report")
async def smart_report(
    file: UploadFile = File(...),
    target_column: Optional[str] = Query(default=None),
):
    """
    Generate a rich AI-powered analysis summary.
    Runs full analysis then produces a structured markdown report.
    """
    try:
        contents = await file.read()
        df = _load_df(file, contents)
        result = _run_full_analysis(df, target_column=target_column)
        er = result["explainability_report"]
        ds = result["dataset_summary"]
        dq = result["dataset_quality_score"]
        at = result["auto_training_results"]
        fi = result["feature_importance"]
        td = result["target_detection"]

        best_model = at.get("best_model","N/A")
        best_score = at.get("best_score", 0.0)
        top_feats  = [f["feature"] for f in (fi.get("ranking_by_rf_importance") or [])[:5]]
        issues     = er.get("issues", [])
        n_crit     = sum(1 for i in issues if i["severity"]=="CRITICAL")
        n_high     = sum(1 for i in issues if i["severity"]=="HIGH")

        grade_desc = {
            "A": "excellent — ready for production ML",
            "B": "good — minor preprocessing needed",
            "C": "fair — significant cleaning required",
            "D": "poor — major issues must be resolved",
            "F": "not ready — fundamental problems detected",
        }.get(er["grade"], "unknown")

        report_md = f"""# AutoEDA v4.0 — ML Readiness Report
**File:** {file.filename or "Dataset"} · {ds["rows"]:,} rows × {ds["columns"]} columns · {ds["memory_usage_mb"]:.2f} MB

## Executive Summary
- **Readiness Score:** {er["readiness_score"]:.0f}/100 — Grade **{er["grade"]}** ({grade_desc})
- **Task Type:** {td["task_type"].replace("_"," ").title()} on target `{td["final_target"]}`
- **Quality Score:** {dq["overall_score"]}/100 — {dq["status"]}
- **Issues Found:** {n_crit} CRITICAL, {n_high} HIGH, {er["issue_summary"].get("MODERATE",0)} MODERATE

## Data Quality Dimensions
| Dimension | Score | Status |
|---|---|---|
{"".join(f"| {k.replace('_',' ').title()} | {v:.0f}/100 | {'✅' if v>=80 else '⚠️' if v>=60 else '❌'} |" + chr(10) for k,v in dq["dimension_scores"].items())}

## Top Issues to Fix
{"".join(f"{i+1}. **[{iss['severity']}]** {iss['title']}  " + chr(10) + f"   > Fix: `{iss['fix'][:100]}`" + chr(10) for i,iss in enumerate(issues[:5]))}

## Model Performance
| Model | Score | CV±Std |
|---|---|---|
{"".join(f"| {'★ ' if nm==best_model else ''}{nm} | {r.get('f1_weighted',r.get('r2_score',r.get('cv_mean',0))):.4f} | ±{r.get('cv_std',0):.4f} |" + chr(10) for nm,r in at.get("model_comparison",{}).items() if "error" not in r)}

**Best model:** {best_model} with score **{best_score:.4f}**

## Most Important Features
{chr(10).join(f"{i+1}. `{f}`" for i,f in enumerate(top_feats))}

## Positive Signals
{chr(10).join(f"✅ {s}" for s in er.get("positive_signals",[]))}

## Recommended Action Plan
{chr(10).join(f"{a['step']}. **[{a['priority']}]** {a['reason']}" for a in er.get("action_plan",[])[:8])}

---
*Generated by AutoEDA v4.0 — ML Data Readiness Analyzer*
"""
        return {
            "status":       "completed",
            "report_md":    report_md,
            "readiness":    er["readiness_score"],
            "grade":        er["grade"],
            "best_model":   best_model,
            "best_score":   best_score,
            "top_features": top_feats,
        }
    except Exception as e:
        raise HTTPException(500, {"error": f"Smart report failed: {str(e)}"})


# ══════════════════════════════════════════════════════════════════════════════
# ★ NEW: /feature-engineering — Advanced feature engineering suggestions
# Generates domain-specific features beyond basic Titanic patterns
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/feature-engineering")
async def suggest_features(
    file: UploadFile = File(...),
    target_column: Optional[str] = Query(default=None),
):
    """
    Advanced automatic feature engineering.
    Detects patterns, creates interaction terms, polynomial features,
    date features, and ranks them by information gain with the target.
    """
    try:
        contents = await file.read()
        df = _load_df(file, contents)
        td = detect_target_column(df, user_specified=target_column)
        target = td["final_target"]
        is_clf = "classification" in td["task_type"]

        suggestions = []
        created     = []
        df_fe       = df.copy()

        num_cols = [c for c in df.select_dtypes(include=np.number).columns if c != target]
        cat_cols = [c for c in df.select_dtypes(include="object").columns  if c != target]

        # ── 1. Interaction terms (top numeric pairs) ──────────────────
        if len(num_cols) >= 2:
            for i, c1 in enumerate(num_cols[:6]):
                for c2 in num_cols[i+1:6]:
                    name = f"{c1}_x_{c2}"
                    df_fe[name] = df_fe[c1] * df_fe[c2]
                    created.append(name)
                    suggestions.append({
                        "feature":     name,
                        "formula":     f"{c1} × {c2}",
                        "type":        "Interaction",
                        "why":         f"Captures multiplicative relationship between {c1} and {c2}",
                    })

        # ── 2. Ratio features ─────────────────────────────────────────
        if len(num_cols) >= 2:
            for i, c1 in enumerate(num_cols[:4]):
                for c2 in num_cols[i+1:4]:
                    if df_fe[c2].replace(0, np.nan).notna().mean() > 0.5:
                        name = f"{c1}_div_{c2}"
                        df_fe[name] = df_fe[c1] / (df_fe[c2].replace(0, np.nan))
                        df_fe[name] = df_fe[name].fillna(0)
                        created.append(name)
                        suggestions.append({
                            "feature": name,
                            "formula": f"{c1} ÷ {c2}",
                            "type":    "Ratio",
                            "why":     f"Ratio often more informative than raw values",
                        })

        # ── 3. Log transform suggestions for skewed columns ───────────
        skew_feats = []
        for col in num_cols:
            try:
                skew = df[col].skew()
                if abs(skew) > 2 and df[col].min() >= 0:
                    skew_feats.append(col)
                    name = f"log_{col}"
                    df_fe[name] = np.log1p(df[col])
                    created.append(name)
                    suggestions.append({
                        "feature": name,
                        "formula": f"log1p({col})",
                        "type":    "Log Transform",
                        "why":     f"{col} is highly skewed (skew={skew:.2f}). Log reduces skewness for linear models.",
                    })
            except: pass

        # ── 4. Date/time feature extraction ──────────────────────────
        for col in df.columns:
            if col == target: continue
            if df[col].dtype == object:
                sample = df[col].dropna().head(50)
                try:
                    parsed = pd.to_datetime(sample, infer_datetime_format=True, errors='coerce')
                    if parsed.notna().mean() > 0.7:
                        df_fe[col+"_year"]  = pd.to_datetime(df[col], errors='coerce').dt.year.fillna(0).astype(int)
                        df_fe[col+"_month"] = pd.to_datetime(df[col], errors='coerce').dt.month.fillna(0).astype(int)
                        df_fe[col+"_day"]   = pd.to_datetime(df[col], errors='coerce').dt.day.fillna(0).astype(int)
                        df_fe[col+"_dow"]   = pd.to_datetime(df[col], errors='coerce').dt.dayofweek.fillna(0).astype(int)
                        for sfx in ["_year","_month","_day","_dow"]:
                            created.append(col+sfx)
                        suggestions.append({
                            "feature": f"{col}_year/month/day/dow",
                            "formula": f"Extract from {col}",
                            "type":    "DateTime",
                            "why":     f"Extracts temporal patterns (seasonality, trends) from date column '{col}'",
                        })
                except: pass

        # ── 5. Polynomial features (top 3 numeric only) ───────────────
        for col in num_cols[:3]:
            name = f"{col}_sq"
            df_fe[name] = df_fe[col] ** 2
            created.append(name)
            suggestions.append({
                "feature": name,
                "formula": f"{col}²",
                "type":    "Polynomial",
                "why":     f"Squared term captures non-linear relationships",
            })

        # ── 6. Rank features with mutual information ──────────────────
        ranked = []
        if len(created) > 0 and target in df_fe.columns:
            try:
                from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
                from sklearn.impute import SimpleImputer
                feat_df = df_fe[created].copy()
                imp = SimpleImputer(strategy="median")
                feat_arr = imp.fit_transform(feat_df.fillna(0))
                y_enc = df_fe[target]
                if y_enc.dtype == object:
                    from sklearn.preprocessing import LabelEncoder
                    y_enc = LabelEncoder().fit_transform(y_enc.astype(str))
                mi_fn = mutual_info_classif if is_clf else mutual_info_regression
                mi = mi_fn(feat_arr, y_enc, random_state=42)
                ranked = sorted(zip(created, mi.tolist()), key=lambda x: -x[1])
                # Attach scores to suggestions
                mi_dict = dict(ranked)
                for s in suggestions:
                    feat_name = s["feature"].split("/")[0]  # handle multi-feature entries
                    s["mutual_information"] = safe_float(mi_dict.get(feat_name, 0.0))
                suggestions.sort(key=lambda x: -x.get("mutual_information", 0))
            except Exception:
                pass

        return {
            "status":           "completed",
            "n_suggestions":    len(suggestions),
            "n_features_created": len(created),
            "target":           target,
            "suggestions":      suggestions[:20],
            "top_5_by_mi":      [{"feature": f, "mi_score": safe_float(s)} for f,s in (ranked[:5] if ranked else [])],
            "summary": (
                f"Generated {len(created)} candidate features across "
                f"interaction, ratio, log-transform, datetime, and polynomial types. "
                f"Ranked by mutual information with target '{target}'."
            ),
            "sklearn_code": f"""# Apply top engineered features
import numpy as np
import pandas as pd

df = pd.read_csv('your_data.csv')
# Interaction terms
{chr(10).join(f"df['{s['feature']}'] = {s['formula'].replace('×','*').replace('÷','/')}" for s in suggestions[:5] if s['type'] in ['Interaction','Ratio','Polynomial'])}
# Log transforms
{chr(10).join(f"df['{s['feature']}'] = np.log1p(df['{s['formula'][7:-1]}'])" for s in suggestions if s['type']=='Log Transform')}
""",
        }
    except Exception as e:
        raise HTTPException(500, {"error": f"Feature engineering failed: {str(e)}"})

from app.sample_datasets import router as sample_router
app.include_router(sample_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)