import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def detect_anomalies(df: pd.DataFrame) -> dict:
    """
    Multi-variate anomaly detection using Isolation Forest.

    Unlike per-column outlier detection (which checks each feature
    independently), Isolation Forest detects rows that are anomalous
    across ALL features simultaneously.

    A row might look normal in each individual column but be
    statistically anomalous when all columns are considered together.
    This catches hidden data quality problems that per-column checks miss.

    Returns:
        anomaly_count: number of anomalous rows detected
        anomaly_percent: percentage of dataset that is anomalous
        anomaly_indices: row indices of top anomalous rows
        severity: LOW / MODERATE / HIGH
        interpretation: plain-English explanation
        contamination_used: the contamination parameter used
    """

    numeric_df = df.select_dtypes(include="number").dropna(axis=1, how="all")

    if numeric_df.shape[1] < 2:
        return {
            "status": "skipped",
            "reason": "Need at least 2 numeric columns for multivariate anomaly detection"
        }

    if len(df) < 20:
        return {
            "status": "skipped",
            "reason": "Need at least 20 rows for reliable anomaly detection"
        }

    # Fill missing values with median for anomaly detection only
    numeric_filled = numeric_df.fillna(numeric_df.median())

    # Standardize so all features contribute equally
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_filled)

    # Contamination = expected proportion of outliers
    # Use 'auto' for small datasets, 0.05 as default for larger ones
    contamination = min(0.1, max(0.01, 10 / len(df)))

    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100
    )

    predictions = model.fit_predict(X_scaled)
    scores = model.score_samples(X_scaled)  # more negative = more anomalous

    # -1 = anomaly, 1 = normal
    anomaly_mask = predictions == -1
    anomaly_count = int(anomaly_mask.sum())
    anomaly_percent = round(float(anomaly_count / len(df) * 100), 2)

    # Top 10 most anomalous row indices
    top_anomaly_indices = list(
        np.argsort(scores)[:min(10, anomaly_count)]
    )

    # Severity based on percentage
    if anomaly_percent > 15:
        severity = "HIGH"
        interpretation = (
            f"{anomaly_percent}% of rows are multivariate anomalies. "
            "This is unusually high — the dataset may contain data from "
            "multiple sources, collection errors, or significant noise. "
            "Consider investigating and removing the most anomalous rows before training."
        )
    elif anomaly_percent > 5:
        severity = "MODERATE"
        interpretation = (
            f"{anomaly_percent}% of rows are multivariate anomalies. "
            "Some rows behave differently from the rest of the dataset. "
            "Review the flagged rows — they may represent edge cases, errors, or valid rare events."
        )
    else:
        severity = "LOW"
        interpretation = (
            f"Only {anomaly_percent}% of rows flagged as anomalous. "
            "Dataset appears internally consistent. "
            "The few anomalous rows may represent legitimate rare events."
        )

    # Which features contributed most to anomalies
    # Compare mean feature values of anomalous vs normal rows
    anomalous_rows = numeric_filled[anomaly_mask]
    normal_rows = numeric_filled[~anomaly_mask]

    feature_deviation = {}
    for col in numeric_df.columns:
        if len(anomalous_rows) > 0 and len(normal_rows) > 0:
            normal_mean = float(normal_rows[col].mean())
            anomaly_mean = float(anomalous_rows[col].mean())
            normal_std = float(normal_rows[col].std()) + 1e-9
            deviation = abs(anomaly_mean - normal_mean) / normal_std
            feature_deviation[col] = round(deviation, 3)

    # Sort by deviation — most contributing features first
    top_features = dict(
        sorted(feature_deviation.items(), key=lambda x: x[1], reverse=True)[:5]
    )

    return {
        "anomaly_count": anomaly_count,
        "anomaly_percent": anomaly_percent,
        "total_rows_checked": len(df),
        "severity": severity,
        "contamination_used": round(contamination, 3),
        "top_anomalous_row_indices": top_anomaly_indices,
        "top_contributing_features": top_features,
        "interpretation": interpretation,
        "note": (
            "Isolation Forest detects rows that are anomalous across ALL features "
            "simultaneously — this catches issues that per-column outlier detection misses."
        )
    }