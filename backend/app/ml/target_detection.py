import pandas as pd
import numpy as np
from scipy.stats import entropy as scipy_entropy


TARGET_NAME_KEYWORDS = [
    "target", "label", "class", "output", "result", "response",
    "y", "outcome", "flag", "status", "survived", "churn",
    "default", "fraud", "price", "salary", "revenue", "sales",
    "score", "rating", "diagnosis"
]

ID_NAME_KEYWORDS = [
    "id", "uuid", "index", "key", "rownum", "row_id",
    "name", "email", "phone", "address", "zipcode",
    "timestamp", "date", "time"
]


def _entropy(series):
    counts = series.value_counts(normalize=True)
    if len(counts) <= 1:
        return 0
    return scipy_entropy(counts) / np.log(len(counts))


def detect_target_column(df: pd.DataFrame) -> dict:
    n_rows = len(df)
    results = []

    for col in df.columns:
        s = df[col].dropna()
        unique = s.nunique()
        missing = df[col].isnull().mean()
        col_lower = col.lower()

        score = 0
        reasons = []

        # ❌ ID detection
        if any(k in col_lower for k in ID_NAME_KEYWORDS):
            results.append((col, -10, "id_column"))
            continue

        if unique / n_rows > 0.95:
            results.append((col, -8, "high_uniqueness"))
            continue

        # ✅ Name match
        if any(k == col_lower for k in TARGET_NAME_KEYWORDS):
            score += 5
            reasons.append("exact_name_match")
        elif any(k in col_lower for k in TARGET_NAME_KEYWORDS):
            score += 3
            reasons.append("partial_name_match")

        # ✅ Type logic
        if s.dtype == object or unique <= 10:
            score += 3
            reasons.append("classification_candidate")
            task = "classification"
        else:
            score += 2
            reasons.append("regression_candidate")
            task = "regression"

        # ✅ entropy
        ent = _entropy(s)
        if 0.2 < ent < 0.8:
            score += 1.5
            reasons.append("good_entropy")

        # ❌ too random
        if ent > 0.95:
            score -= 1
            reasons.append("too_uniform")

        # ❌ missing
        score -= missing * 3

        results.append((col, score, reasons))

    # sort
    results = sorted(results, key=lambda x: x[1], reverse=True)

    best = results[0]
    second = results[1] if len(results) > 1 else best

    # softmax confidence
    scores = np.array([r[1] for r in results])
    exp_scores = np.exp(scores - scores.max())
    probs = exp_scores / exp_scores.sum()

    confidence = float(probs[0])

    return {
        "predicted_target": best[0],
        "confidence": round(confidence, 3),
        "top_candidates": [
            {"column": r[0], "score": round(r[1], 2)} for r in results[:5]
        ],
        "reasoning": best[2],
        "warning": (
            "Low confidence — please specify target manually"
            if confidence < 0.4 else None
        )
    }