import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.utils import resample


def cluster_intelligence(df, max_clusters=6):

    df = df.select_dtypes(include=np.number).dropna()

    if df.shape[1] < 2:
        return {"error": "Need at least 2 numeric features for clustering"}

    scaler = StandardScaler()
    X = scaler.fit_transform(df)

    best_k = 2
    best_score = -1
    silhouette_scores = {}

    # ---------- Optimal Cluster Selection ----------
    for k in range(2, min(max_clusters, len(df))):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        score = silhouette_score(X, labels)
        silhouette_scores[k] = round(score, 3)

        if score > best_score:
            best_score = score
            best_k = k

    final_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = final_model.fit_predict(X)

    sil_score = silhouette_score(X, cluster_labels)
    db_score = davies_bouldin_score(X, cluster_labels)

    # ---------- Cluster Size ----------
    cluster_sizes = pd.Series(cluster_labels).value_counts(normalize=True)
    cluster_distribution = {
        f"Cluster {i}": f"{round(cluster_sizes.get(i,0)*100,2)}%"
        for i in range(best_k)
    }

    # ---------- Feature Influence ----------
    centroids = final_model.cluster_centers_
    centroid_variation = np.std(centroids, axis=0)
    feature_influence = dict(
        sorted(
            zip(df.columns, centroid_variation),
            key=lambda x: x[1],
            reverse=True
        )
    )

    # Normalize influence
    total = sum(feature_influence.values())
    feature_influence = {
        k: round((v / total), 3) for k, v in feature_influence.items()
    }

    # ---------- Cluster Stability ----------
    stability_scores = []

    for _ in range(5):
        sample = resample(X, replace=True)
        temp_model = KMeans(n_clusters=best_k, random_state=None, n_init=10)
        temp_labels = temp_model.fit_predict(sample)
        stability_scores.append(silhouette_score(sample, temp_labels))

    stability = round(np.mean(stability_scores), 3)

    # ---------- Automatic Interpretation ----------
    interpretation = []

    if sil_score > 0.6:
        interpretation.append("Strong natural grouping detected")
    elif sil_score > 0.4:
        interpretation.append("Moderate clustering structure present")
    else:
        interpretation.append("Weak cluster separation")

    if stability > 0.6:
        interpretation.append("Clusters are stable and reliable")
    else:
        interpretation.append("Clusters unstable — dataset may be noisy")

    if best_k > 4:
        interpretation.append("High segmentation detected — possible overfitting risk")

    return {
        "optimal_clusters": best_k,
        "silhouette_score": round(sil_score, 3),
        "davies_bouldin_score": round(db_score, 3),
        "cluster_distribution": cluster_distribution,
        "feature_influence": feature_influence,
        "cluster_stability": stability,
        "interpretation": interpretation
    }