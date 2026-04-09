import pandas as pd
import numpy as np
from typing import List, Dict, Any
import time


def compute_basic_stats(df: pd.DataFrame) -> dict:
    """Fast stats without running all 18 modules."""
    rows, cols = df.shape
    missing_pct = round(float(df.isnull().mean().mean() * 100), 2)
    duplicate_pct = round(float(df.duplicated().sum() / rows * 100), 2)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()

    # Outlier count using IQR on numeric columns
    total_outliers = 0
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum()
        total_outliers += int(outliers)

    return {
        "rows": int(rows),
        "columns": int(cols),
        "numeric_columns": len(numeric_cols),
        "categorical_columns": len(cat_cols),
        "missing_percent": missing_pct,
        "duplicate_percent": duplicate_pct,
        "total_outliers": total_outliers,
        "outlier_percent": round(total_outliers / rows * 100, 2) if rows > 0 else 0
    }


def compute_readiness_score_fast(df: pd.DataFrame) -> dict:
    """
    Compute the 6-dimension readiness score quickly.
    Same formula as data_quality_score.py but standalone.
    """
    rows, cols = df.shape
    scores = {}

    # Completeness
    total_missing = df.isna().sum().sum()
    missing_pct = total_missing / (rows * cols) * 100
    scores["completeness"] = round(max(0, 100 - missing_pct), 2)

    # Uniqueness
    dup_pct = df.duplicated().sum() / rows * 100
    scores["uniqueness"] = round(max(0, 100 - dup_pct), 2)

    # Consistency (IQR + Z-score outliers)
    numeric_df = df.select_dtypes(include="number")
    total_outliers = 0
    for col in numeric_df.columns:
        z = np.abs((numeric_df[col] - numeric_df[col].mean()) / (numeric_df[col].std() + 1e-9))
        z_out = (z > 3).sum()
        Q1 = numeric_df[col].quantile(0.25)
        Q3 = numeric_df[col].quantile(0.75)
        IQR = Q3 - Q1
        iqr_out = ((numeric_df[col] < Q1 - 1.5 * IQR) | (numeric_df[col] > Q3 + 1.5 * IQR)).sum()
        total_outliers += (z_out + iqr_out) / 2
    outlier_pct = total_outliers / rows * 100
    scores["consistency"] = round(max(0, 100 - outlier_pct * 2), 2)

    # Class balance (last column as target)
    target = df.iloc[:, -1]
    class_counts = target.value_counts(normalize=True)
    gini = 1 - sum(class_counts ** 2)
    max_gini = 1 - (1 / len(class_counts)) if len(class_counts) > 1 else 1
    scores["class_balance"] = round((gini / max_gini * 100) if max_gini > 0 else 100, 2)

    # Feature quality
    penalty = 0
    for col in numeric_df.columns:
        if numeric_df[col].std() < 0.01:
            penalty += 10
    if len(numeric_df.columns) > 1:
        corr = numeric_df.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        penalty += int((upper > 0.95).sum().sum()) * 5
    scores["feature_quality"] = round(max(0, 100 - penalty), 2)

    # Adequacy
    ratio = rows / cols
    if ratio >= 10:
        scores["adequacy"] = 100
    elif ratio >= 5:
        scores["adequacy"] = 70
    elif ratio >= 2:
        scores["adequacy"] = 40
    else:
        scores["adequacy"] = 10

    weights = {
        "completeness": 0.25, "uniqueness": 0.10,
        "consistency": 0.20, "class_balance": 0.15,
        "feature_quality": 0.15, "adequacy": 0.15
    }
    overall = round(sum(scores[k] * weights[k] for k in weights), 2)

    grade = "A" if overall >= 85 else "B" if overall >= 70 else "C" if overall >= 55 else "D" if overall >= 40 else "F"

    return {
        "overall_score": overall,
        "grade": grade,
        "dimension_scores": scores
    }


def detect_issues_fast(df: pd.DataFrame) -> dict:
    """Fast issue detection without full module suite."""
    issues = {"critical": [], "high": [], "moderate": []}

    for col in df.columns:
        pct = df[col].isnull().mean() * 100
        if pct > 60:
            issues["critical"].append(f"{col}: {pct:.0f}% missing")
        elif pct > 20:
            issues["high"].append(f"{col}: {pct:.0f}% missing")
        elif pct > 5:
            issues["moderate"].append(f"{col}: {pct:.0f}% missing")

    # Check for ID-like columns
    n_rows = len(df)
    for col in df.columns:
        if df[col].nunique() == n_rows:
            issues["high"].append(f"{col}: likely ID column (all unique)")

    # Check high correlation
    numeric_df = df.select_dtypes(include="number")
    if len(numeric_df.columns) > 1:
        corr = numeric_df.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        high_corr = [(c, r) for c in upper.columns for r in upper.index if pd.notna(upper.loc[r, c]) and upper.loc[r, c] > 0.95 and c != r]
        for c1, c2 in high_corr[:3]:
            issues["high"].append(f"{c1} ↔ {c2}: >95% correlated")

    return {
        "critical_count": len(issues["critical"]),
        "high_count": len(issues["high"]),
        "moderate_count": len(issues["moderate"]),
        "critical_issues": issues["critical"][:5],
        "high_issues": issues["high"][:5],
        "moderate_issues": issues["moderate"][:5],
        "total_issues": len(issues["critical"]) + len(issues["high"]) + len(issues["moderate"])
    }


def rank_datasets(results: List[dict]) -> List[dict]:
    """
    Rank datasets by readiness score.
    Add rank, percentile, and comparison to average.
    """
    valid = [r for r in results if r.get("status") == "success"]
    if not valid:
        return results

    scores = [r["readiness_score"]["overall_score"] for r in valid]
    avg_score = round(float(np.mean(scores)), 2)
    sorted_scores = sorted(scores, reverse=True)

    for r in valid:
        score = r["readiness_score"]["overall_score"]
        r["rank"] = sorted_scores.index(score) + 1
        r["percentile"] = round((1 - sorted_scores.index(score) / len(sorted_scores)) * 100, 1)
        r["vs_average"] = round(score - avg_score, 2)

    return results, avg_score


def generate_benchmark_insights(results: List[dict], avg_score: float) -> dict:
    """
    Generate dataset-level insights from benchmark results.
    This is what goes into the research paper results section.
    """
    valid = [r for r in results if r.get("status") == "success"]
    if not valid:
        return {}

    scores = [r["readiness_score"]["overall_score"] for r in valid]
    grades = [r["readiness_score"]["grade"] for r in valid]

    best = max(valid, key=lambda x: x["readiness_score"]["overall_score"])
    worst = min(valid, key=lambda x: x["readiness_score"]["overall_score"])

    # Dimension averages across all datasets
    dim_keys = ["completeness", "uniqueness", "consistency", "class_balance", "feature_quality", "adequacy"]
    dim_avgs = {}
    for dim in dim_keys:
        vals = [r["readiness_score"]["dimension_scores"].get(dim, 0) for r in valid]
        dim_avgs[dim] = round(float(np.mean(vals)), 2)

    # Most common issue type
    total_critical = sum(r["issues"]["critical_count"] for r in valid)
    total_high = sum(r["issues"]["high_count"] for r in valid)
    total_moderate = sum(r["issues"]["moderate_count"] for r in valid)

    # Score distribution
    grade_dist = {}
    for g in grades:
        grade_dist[g] = grade_dist.get(g, 0) + 1

    return {
        "datasets_analyzed": len(valid),
        "average_readiness_score": avg_score,
        "score_std_dev": round(float(np.std(scores)), 2),
        "score_range": {
            "min": round(min(scores), 2),
            "max": round(max(scores), 2)
        },
        "best_dataset": {
            "name": best["dataset_name"],
            "score": best["readiness_score"]["overall_score"],
            "grade": best["readiness_score"]["grade"]
        },
        "worst_dataset": {
            "name": worst["dataset_name"],
            "score": worst["readiness_score"]["overall_score"],
            "grade": worst["readiness_score"]["grade"]
        },
        "grade_distribution": grade_dist,
        "average_dimension_scores": dim_avgs,
        "weakest_dimension": min(dim_avgs, key=dim_avgs.get),
        "strongest_dimension": max(dim_avgs, key=dim_avgs.get),
        "total_issues_found": {
            "critical": total_critical,
            "high": total_high,
            "moderate": total_moderate
        },
        "research_finding": (
            f"Across {len(valid)} datasets, the average ML readiness score is {avg_score}/100. "
            f"The weakest dimension is '{min(dim_avgs, key=dim_avgs.get)}' "
            f"(avg={dim_avgs[min(dim_avgs, key=dim_avgs.get)]}/100), "
            f"suggesting this is the most common data quality bottleneck. "
            f"{grade_dist.get('A', 0) + grade_dist.get('B', 0)} of {len(valid)} datasets "
            f"are ML-ready (grade A or B)."
        )
    }


def run_benchmark(datasets: List[Dict[str, Any]]) -> dict:
    """
    Main benchmark function.

    Accepts a list of {name, dataframe} dicts.
    Returns a complete comparison report suitable for research paper results section.

    Each dataset gets:
    - Basic stats (rows, cols, missing%, outliers)
    - 6-dimension readiness score
    - Issue summary (critical/high/moderate counts)
    - Rank among all datasets
    - Percentile
    - Comparison to average

    Plus overall insights:
    - Average score across all datasets
    - Grade distribution
    - Weakest/strongest dimensions
    - Research finding summary
    """

    results = []

    for item in datasets:
        name = item.get("name", "unknown")
        df = item.get("dataframe")

        if df is None or df.empty:
            results.append({
                "dataset_name": name,
                "status": "error",
                "error": "Empty or invalid dataset"
            })
            continue

        try:
            start = time.time()

            basic = compute_basic_stats(df)
            readiness = compute_readiness_score_fast(df)
            issues = detect_issues_fast(df)

            elapsed = round(time.time() - start, 3)

            results.append({
                "dataset_name": name,
                "status": "success",
                "analysis_time_seconds": elapsed,
                "basic_stats": basic,
                "readiness_score": readiness,
                "issues": issues
            })

        except Exception as e:
            results.append({
                "dataset_name": name,
                "status": "error",
                "error": str(e)
            })

    # Rank all datasets
    valid_results = [r for r in results if r.get("status") == "success"]
    if valid_results:
        ranked_results, avg_score = rank_datasets(results)
        insights = generate_benchmark_insights(results, avg_score)
    else:
        ranked_results = results
        insights = {}

    return {
        "total_datasets": len(datasets),
        "successful": len(valid_results),
        "failed": len(datasets) - len(valid_results),
        "results": ranked_results,
        "benchmark_insights": insights,
        "comparison_table": [
            {
                "rank": r.get("rank", "-"),
                "dataset": r["dataset_name"],
                "rows": r["basic_stats"]["rows"] if r.get("status") == "success" else "-",
                "columns": r["basic_stats"]["columns"] if r.get("status") == "success" else "-",
                "missing_pct": r["basic_stats"]["missing_percent"] if r.get("status") == "success" else "-",
                "readiness_score": r["readiness_score"]["overall_score"] if r.get("status") == "success" else "-",
                "grade": r["readiness_score"]["grade"] if r.get("status") == "success" else "ERR",
                "critical_issues": r["issues"]["critical_count"] if r.get("status") == "success" else "-",
                "vs_average": r.get("vs_average", "-")
            }
            for r in ranked_results
        ]
    }