"""
nl_report_generator.py

Generates a complete natural language explanation of the analysis results.

Every other EDA tool gives you numbers.
This module turns those numbers into readable paragraphs that explain:
  - What the issue is
  - Why it causes ML failure (root cause)
  - What to do about it (specific actionable steps)
  - What the expected improvement will be

This is the module that makes the tool feel intelligent, not just automated.
No existing open-source EDA tool does this.
"""

from typing import Optional


def _grade_prose(grade: str, score: float) -> str:
    """Convert A-F grade into a natural language opening."""
    if grade == "A":
        return (
            f"This dataset is in excellent condition for machine learning "
            f"(readiness score: {score}/100, Grade A). "
            "It requires minimal preprocessing and is ready for model training. "
            "The few issues identified are minor and unlikely to significantly impact performance."
        )
    elif grade == "B":
        return (
            f"This dataset is in good condition for machine learning "
            f"(readiness score: {score}/100, Grade B). "
            "It requires some preprocessing but no critical issues were found. "
            "Following the recommended pipeline will prepare it for reliable model training."
        )
    elif grade == "C":
        return (
            f"This dataset needs cleaning before reliable model training "
            f"(readiness score: {score}/100, Grade C). "
            "Several data quality issues were identified that will degrade model performance "
            "if left unaddressed. The recommended action plan should be completed before training."
        )
    elif grade == "D":
        return (
            f"This dataset has serious quality problems that will produce unreliable models "
            f"(readiness score: {score}/100, Grade D). "
            "Multiple high-severity issues were found including significant missing data "
            "and structural problems. Do not train models on this data without completing "
            "the full action plan first."
        )
    else:
        return (
            f"This dataset is not ready for machine learning in its current state "
            f"(readiness score: {score}/100, Grade F). "
            "Critical data quality failures were detected that would make any trained model "
            "fundamentally unreliable. Substantial data cleaning and restructuring is required."
        )


def _explain_dataset_overview(summary: dict) -> str:
    """Explain the basic dataset structure."""
    rows = summary.get("rows", 0)
    cols = summary.get("columns", 0)
    missing_pct = summary.get("missing_value_percent", 0)
    duplicates = summary.get("duplicate_rows", 0)
    numeric = len(summary.get("numeric_columns", []))
    categorical = len(summary.get("categorical_columns", []))
    memory = summary.get("memory_usage_mb", 0)

    size_desc = (
        "very small" if rows < 200 else
        "small" if rows < 1000 else
        "medium-sized" if rows < 10000 else
        "large"
    )

    parts = [
        f"The dataset contains {rows:,} rows and {cols} columns ({memory} MB in memory), "
        f"making it a {size_desc} dataset for machine learning purposes. "
    ]

    parts.append(
        f"Of the {cols} columns, {numeric} are numeric and {categorical} are categorical. "
    )

    if missing_pct == 0:
        parts.append("There are no missing values, which is excellent. ")
    elif missing_pct < 5:
        parts.append(
            f"Missing values are minimal at {missing_pct}% overall — "
            "straightforward imputation will suffice. "
        )
    elif missing_pct < 20:
        parts.append(
            f"Missing values are moderate at {missing_pct}% overall — "
            "imputation strategy requires careful consideration per column. "
        )
    else:
        parts.append(
            f"Missing values are substantial at {missing_pct}% overall — "
            "this is a significant data quality concern requiring immediate attention. "
        )

    if duplicates == 0:
        parts.append("No duplicate rows were found. ")
    elif duplicates < 10:
        parts.append(f"Only {duplicates} duplicate rows were found — these should be removed. ")
    else:
        parts.append(
            f"{duplicates:,} duplicate rows were found ({duplicates/rows*100:.1f}% of dataset) — "
            "this is a significant problem that will bias model training. "
        )

    # Rows-to-columns ratio
    ratio = rows / cols if cols > 0 else 0
    if ratio < 10:
        parts.append(
            f"The rows-to-columns ratio of {ratio:.1f} is low — "
            "most models will overfit on this dataset. Feature selection is strongly recommended. "
        )
    elif ratio > 100:
        parts.append(
            f"The rows-to-columns ratio of {ratio:.0f} is excellent — "
            "there is sufficient data relative to the number of features. "
        )

    return "".join(parts)


def _explain_target_detection(td: dict) -> str:
    """Explain target column detection results."""
    target = td.get("final_target", "unknown")
    task = td.get("task_type", "unknown")
    source = td.get("target_source", "auto_detected")
    conf = round(td.get("confidence", 0) * 100)

    task_desc = {
        "classification_binary": "binary classification (predicting one of two outcomes)",
        "classification_multiclass": "multiclass classification (predicting one of several outcomes)",
        "regression": "regression (predicting a continuous numeric value)"
    }.get(task, task)

    if source == "user_specified":
        return (
            f"The target column '{target}' was manually specified by the user. "
            f"This is a {task_desc} problem. "
            "All analysis and model recommendations are tailored to this task type."
        )
    else:
        conf_desc = "high" if conf >= 70 else "moderate" if conf >= 40 else "low"
        return (
            f"The target column was automatically detected as '{target}' "
            f"with {conf_desc} confidence ({conf}%). "
            f"This appears to be a {task_desc} problem. "
            + (
                "If this is incorrect, use POST /upload?target_column=NAME to override. "
                if conf < 60 else ""
            )
        )


def _explain_quality_dimensions(qs: dict) -> str:
    """Explain each quality dimension score."""
    dims = qs.get("dimension_scores", {})
    overall = qs.get("overall_score", 0)
    status = qs.get("status", "")

    parts = [
        f"The dataset quality scorecard gives an overall score of {overall:.0f}/100 ({status}). "
        "Here is what each dimension means:\n\n"
    ]

    descriptions = {
        "completeness": (
            "Completeness measures what percentage of cells contain actual values. "
            "Low completeness means too many missing values, which bias model training "
            "because sklearn models cannot handle NaN by default."
        ),
        "uniqueness": (
            "Uniqueness measures the absence of duplicate rows. "
            "Duplicate rows inflate the training set with redundant examples, "
            "causing models to overweight those patterns and overfit."
        ),
        "consistency": (
            "Consistency measures how free the dataset is from outliers. "
            "Extreme values distort model parameters, especially in linear models "
            "and distance-based algorithms like KNN and SVM."
        ),
        "class_balance": (
            "Class balance measures how evenly distributed the target classes are. "
            "Imbalanced classes cause models to ignore the minority class entirely — "
            "a model predicting 'no fraud' 99% of the time looks accurate but is useless."
        ),
        "feature_quality": (
            "Feature quality measures how informative the features are. "
            "Near-zero variance features add noise without signal. "
            "Highly correlated features cause multicollinearity, "
            "making model coefficients unstable and uninterpretable."
        ),
        "adequacy": (
            "Adequacy measures whether there is enough data relative to the number of features. "
            "The rule of thumb is at least 10 rows per feature. "
            "Below this ratio, models memorize training data instead of learning patterns."
        )
    }

    for dim, score in dims.items():
        status_word = "excellent" if score >= 85 else "good" if score >= 70 else "moderate" if score >= 50 else "poor"
        desc = descriptions.get(dim, "")
        parts.append(
            f"  {dim.replace('_', ' ').title()} ({score:.0f}/100 — {status_word}): {desc}\n\n"
        )

    return "".join(parts)


def _explain_issues(issues: list) -> str:
    """Generate detailed plain-English explanation for each issue."""
    if not issues:
        return "No significant issues were detected. The dataset appears clean and well-structured."

    critical = [i for i in issues if i.get("severity") == "CRITICAL"]
    high = [i for i in issues if i.get("severity") == "HIGH"]
    moderate = [i for i in issues if i.get("severity") == "MODERATE"]
    low = [i for i in issues if i.get("severity") == "LOW"]

    parts = []

    if critical:
        parts.append(
            f"CRITICAL ISSUES ({len(critical)} found — must fix before any training):\n\n"
        )
        for issue in critical:
            parts.append(
                f"  {issue.get('title', '')}\n"
                f"  Root cause: {issue.get('explanation', '')}\n"
                f"  Action required: {issue.get('fix', '')}\n\n"
            )

    if high:
        parts.append(f"HIGH SEVERITY ISSUES ({len(high)} found — fix before training):\n\n")
        for issue in high:
            parts.append(
                f"  {issue.get('title', '')}\n"
                f"  Why this matters: {issue.get('explanation', '')}\n"
                f"  Recommended fix: {issue.get('fix', '')}\n\n"
            )

    if moderate:
        parts.append(f"MODERATE ISSUES ({len(moderate)} found — address when possible):\n\n")
        for issue in moderate:
            parts.append(
                f"  {issue.get('title', '')}: {issue.get('explanation', '')}\n\n"
            )

    if low:
        parts.append(
            f"LOW PRIORITY ({len(low)} minor issues) — monitor but not blocking:\n"
        )
        for issue in low:
            parts.append(f"  - {issue.get('title', '')}\n")

    return "".join(parts)


def _explain_pipeline(pipeline: dict) -> str:
    """Explain the preprocessing pipeline in plain English."""
    if not pipeline:
        return ""

    drops = pipeline.get("drop_columns", [])
    impute = pipeline.get("missing_value_strategy", {})
    encode = pipeline.get("encoding_strategy", {})
    scale = pipeline.get("scaling_strategy", {})
    transform = pipeline.get("transformation_recommendations", {})
    summary = pipeline.get("summary", {})

    parts = [
        f"The recommended preprocessing pipeline has {summary.get('total_features', 0)} features "
        f"and requires {len(drops)} drops, "
        f"{summary.get('columns_needing_imputation', 0)} imputations, "
        f"{summary.get('columns_needing_encoding', 0)} encodings, and "
        f"{summary.get('columns_needing_scaling', 0)} scaling operations.\n\n"
    ]

    if drops:
        parts.append(
            f"Step 1 — DROP {len(drops)} columns: "
            + ", ".join(f"'{d['column']}' ({d['reason']})" for d in drops)
            + ". These columns either contain no predictive value or are structurally unusable. "
            "Keeping them would add noise without signal.\n\n"
        )

    if impute:
        parts.append("Step 2 — IMPUTE missing values: ")
        for col, v in impute.items():
            parts.append(
                f"'{col}' will be imputed using {v.get('strategy', 'imputation').replace('_IMPUTATION', '').lower()} "
                f"({v.get('missing_percent', 0):.1f}% missing). "
            )
        parts.append("\n\n")

    if transform:
        parts.append("Step 3 — TRANSFORM skewed features: ")
        for col, v in transform.items():
            parts.append(
                f"'{col}' will receive log1p transformation (skewness={v.get('skewness', 0):.2f}). "
            )
        parts.append(
            "Log transformation reduces right skew, bringing the distribution closer to normal "
            "which helps linear models and distance-based algorithms significantly.\n\n"
        )

    if encode:
        parts.append("Step 4 — ENCODE categorical features: ")
        for col, v in encode.items():
            parts.append(f"'{col}' → {v.get('encoding', 'encoding')}. ")
        parts.append("\n\n")

    if scale:
        parts.append("Step 5 — SCALE numeric features: ")
        scalers_used = set(v.get("scaler", "") for v in scale.values())
        for scaler in scalers_used:
            cols_with_scaler = [c for c, v in scale.items() if v.get("scaler") == scaler]
            if scaler == "StandardScaler":
                parts.append(
                    f"StandardScaler applied to {cols_with_scaler} "
                    "(subtracts mean, divides by std — assumes near-normal distribution). "
                )
            elif scaler == "RobustScaler":
                parts.append(
                    f"RobustScaler applied to {cols_with_scaler} "
                    "(uses median and IQR — resistant to outlier influence). "
                )
        parts.append("\n")

    return "".join(parts)


def _explain_models(at: dict) -> str:
    """Explain model training results."""
    if not at:
        return ""

    task = at.get("task_type", "")
    best = at.get("best_model", "")
    best_score = at.get("best_score", 0)
    models = at.get("model_comparison", {})
    metric = "F1 score (weighted)" if task == "classification" else "R2 score"
    ci = at.get("class_imbalance_analysis", {})

    parts = [
        f"Five models were automatically trained and compared using {metric}. "
        f"The best performing model is {best} with {metric} of {best_score:.3f}.\n\n"
    ]

    if ci.get("is_imbalanced"):
        ratio = ci.get("imbalance_ratio", 0)
        parts.append(
            f"Class imbalance was detected (majority:minority ratio = {ratio:.1f}:1). "
            "All models used class_weight='balanced' to compensate. "
            "Accuracy alone would be misleading here — F1 score is the appropriate metric.\n\n"
        )

    for name, r in models.items():
        if "error" in r:
            continue
        f1 = r.get("f1_weighted") or r.get("r2_score") or r.get("accuracy") or 0
        cv = r.get("cv_mean_f1") or r.get("cv_mean_r2") or 0
        std = r.get("cv_std") or 0
        is_best = name == best
        parts.append(
            f"  {'★ ' if is_best else ''}{name}: {metric} = {f1:.3f}, "
            f"CV = {cv:.3f} ± {std:.3f}. "
            f"{r.get('reasoning', '')}\n"
        )

    parts.append(
        f"\nThe {best} model is recommended as the starting point. "
        "Consider hyperparameter tuning with GridSearchCV or RandomizedSearchCV "
        "to further improve performance.\n"
    )

    return "".join(parts)


def _explain_fair(fair: dict) -> str:
    """Explain FAIR assessment."""
    if not fair:
        return ""

    overall = fair.get("overall_fair_score", 0)
    grade = fair.get("fair_grade", "")
    dims = fair.get("dimension_scores", {})
    issues = fair.get("issues_found", {})

    parts = [
        f"FAIR Assessment: {overall:.0f}/100 — {grade}.\n\n"
        "FAIR principles (Findable, Accessible, Interoperable, Reusable) evaluate "
        "whether this dataset follows open science standards for research publication.\n\n"
    ]

    dim_desc = {
        "findable": "whether the dataset can be identified by its structure and naming conventions",
        "accessible": "whether the dataset is practically usable without major restructuring",
        "interoperable": "whether the dataset uses standard formats that work with existing tools",
        "reusable": "whether the dataset contains enough information to be reused by others"
    }

    for dim, score in dims.items():
        parts.append(
            f"  {dim.title()} ({score:.0f}/100): "
            f"Measures {dim_desc.get(dim, dim)}. "
            f"{'Satisfactory.' if score >= 70 else 'Needs improvement.'}\n"
        )

    if issues:
        parts.append("\nFAIR issues identified:\n")
        for k, v in issues.items():
            parts.append(f"  - {k.replace('_', ' ').title()}: {v}\n")

    return "".join(parts)


def _explain_anomalies(anom: dict) -> str:
    """Explain anomaly detection results."""
    if not anom or anom.get("status") == "skipped":
        return ""

    count = anom.get("anomaly_count", 0)
    pct = anom.get("anomaly_percent", 0)
    sev = anom.get("severity", "LOW")
    top_features = anom.get("top_contributing_features", {})

    parts = [
        f"Isolation Forest multivariate anomaly detection identified {count} anomalous rows "
        f"({pct}% of dataset). Severity: {sev}.\n\n"
        "Unlike per-column outlier detection which checks each feature independently, "
        "Isolation Forest detects rows that are anomalous across ALL features simultaneously. "
        "This catches subtle data quality problems that standard outlier detection misses.\n\n"
    ]

    parts.append(anom.get("interpretation", ""))

    if top_features:
        parts.append(
            f"\n\nThe features that most distinguish anomalous rows from normal rows are: "
            + ", ".join(f"{col} (deviation score: {score:.3f})" for col, score in list(top_features.items())[:3])
            + ". These features should be investigated in the anomalous rows."
        )

    return "".join(parts)


def _explain_leakage(leak: dict) -> str:
    """Explain data leakage findings."""
    if not leak:
        return ""

    total = leak.get("summary", {}).get("total_issues", 0)
    sev = leak.get("summary", {}).get("severity", "NONE")

    if total == 0:
        return (
            "No data leakage was detected. This is an important finding — "
            "data leakage is one of the most dangerous and commonly missed ML problems. "
            "When leakage exists, models appear to perform well in training but "
            "completely fail in production because they learned from information "
            "that would not be available at prediction time. "
            "This dataset is clean from leakage."
        )

    parts = [
        f"DATA LEAKAGE DETECTED — Severity: {sev}. "
        "This is a critical problem. Data leakage occurs when features contain "
        "information that would not be available at prediction time, causing the "
        "model to learn a shortcut that does not generalize.\n\n"
    ]

    for item in leak.get("target_leakage", []):
        parts.append(
            f"Target leakage: '{item.get('feature')}' correlates {item.get('correlation', 0):.1%} "
            f"with the target. "
            f"Reason: {item.get('reason', '')}. "
            "Remove this feature immediately before training.\n\n"
        )

    return "".join(parts)


def _explain_overfitting(ov: dict) -> str:
    """Explain overfitting analysis."""
    if not ov:
        return ""

    overall = ov.get("overall_overfitting", {})
    risk = overall.get("risk", "LOW")
    per_model = ov.get("per_model_analysis", {})

    parts = [
        f"Overfitting risk: {risk}. {overall.get('message', '')}\n\n"
        "Overfitting occurs when a model memorizes training data instead of "
        "learning generalizable patterns. It shows as high training accuracy "
        "but poor performance on new data — the exact failure mode that makes "
        "ML models unreliable in production.\n\n"
    ]

    for name, r in per_model.items():
        if "error" in r:
            continue
        gap = r.get("train_test_gap", 0)
        risk_level = r.get("overfitting_risk", "LOW")
        converging = r.get("learning_curve", {}).get("is_converging", True)

        if risk_level == "HIGH":
            parts.append(
                f"  {name}: HIGH overfitting risk. "
                f"Train score {r.get('train_score', 0):.3f} vs test score {r.get('test_score', 0):.3f} "
                f"(gap = {gap:.3f}). "
                f"Learning curve {'is converging' if converging else 'is NOT converging — classic overfitting signature'}. "
                f"{r.get('reason', '')}\n\n"
            )
        elif risk_level == "MODERATE":
            parts.append(
                f"  {name}: moderate overfitting (gap = {gap:.3f}). "
                f"Consider regularization.\n"
            )

    return "".join(parts)


# ── MAIN GENERATOR ────────────────────────────────────────────────────────────

def generate_nl_report(analysis_data: dict, filename: str = "dataset") -> dict:
    """
    Generate complete natural language report from analysis data.

    Returns a dict with sections, each containing a plain-English
    paragraph explanation of that section's findings.

    The combined report reads like a professional data scientist
    wrote a full analysis memo — not like a list of statistics.
    """

    er = analysis_data.get("explainability_report", {})
    grade = er.get("grade", "?")
    score = er.get("readiness_score", 0)
    issues = er.get("issues", [])
    signals = er.get("positive_signals", [])
    action_plan = er.get("action_plan", [])

    sections = {}

    # 1. Executive summary
    sections["executive_summary"] = (
        f"DATASET: {filename}\n\n"
        + _grade_prose(grade, score)
        + "\n\n"
        + f"Key findings: {er.get('issue_summary', {}).get('CRITICAL', 0)} critical issues, "
        + f"{er.get('issue_summary', {}).get('HIGH', 0)} high severity issues, "
        + f"{len(signals)} positive signals detected."
    )

    # 2. Dataset overview
    sections["dataset_overview"] = _explain_dataset_overview(
        analysis_data.get("dataset_summary", {})
    )

    # 3. Target detection
    sections["target_detection"] = _explain_target_detection(
        analysis_data.get("target_detection", {})
    )

    # 4. Quality dimensions
    sections["quality_scorecard"] = _explain_quality_dimensions(
        analysis_data.get("dataset_quality_score", {})
    )

    # 5. Issues
    sections["issues_analysis"] = _explain_issues(issues)

    # 6. Positive signals
    if signals:
        sections["positive_signals"] = (
            "The following positive attributes were identified:\n\n"
            + "\n".join(f"  + {s}" for s in signals)
        )

    # 7. Pipeline explanation
    sections["preprocessing_pipeline"] = _explain_pipeline(
        analysis_data.get("recommended_pipeline", {})
    )

    # 8. Model results
    sections["model_results"] = _explain_models(
        analysis_data.get("auto_training_results", {})
    )

    # 9. Leakage
    sections["leakage_analysis"] = _explain_leakage(
        analysis_data.get("data_leakage_analysis", {})
    )

    # 10. Overfitting
    sections["overfitting_analysis"] = _explain_overfitting(
        analysis_data.get("overfitting_analysis", {})
    )

    # 11. Anomaly detection
    sections["anomaly_detection"] = _explain_anomalies(
        analysis_data.get("anomaly_detection", {})
    )

    # 12. FAIR
    sections["fair_assessment"] = _explain_fair(
        analysis_data.get("fair_assessment", {})
    )

    # 13. Action plan summary
    if action_plan:
        plan_text = "RECOMMENDED ACTION PLAN (in priority order):\n\n"
        for a in action_plan:
            plan_text += (
                f"  Step {a.get('step', '')}: [{a.get('priority', '')}] "
                f"{a.get('reason', '')} — {a.get('action', '')[:100]}\n"
            )
        sections["action_plan_summary"] = plan_text

    # 14. Full combined report
    full_report = "\n\n" + "=" * 60 + "\n\n"
    full_report = full_report.join([
        f"{'=' * 60}\nML DATA READINESS REPORT\nDataset: {filename}\n{'=' * 60}\n\n",
        f"1. EXECUTIVE SUMMARY\n{sections['executive_summary']}\n",
        f"2. DATASET OVERVIEW\n{sections['dataset_overview']}\n",
        f"3. TARGET DETECTION\n{sections['target_detection']}\n",
        f"4. QUALITY SCORECARD\n{sections['quality_scorecard']}\n",
        f"5. ISSUES ANALYSIS\n{sections['issues_analysis']}\n",
        f"6. PREPROCESSING PIPELINE\n{sections['preprocessing_pipeline']}\n",
        f"7. MODEL RESULTS\n{sections['model_results']}\n",
        f"8. LEAKAGE ANALYSIS\n{sections['leakage_analysis']}\n",
        f"9. OVERFITTING ANALYSIS\n{sections['overfitting_analysis']}\n",
        f"10. ANOMALY DETECTION\n{sections['anomaly_detection']}\n",
        f"11. FAIR ASSESSMENT\n{sections['fair_assessment']}\n",
        f"12. ACTION PLAN\n{sections.get('action_plan_summary', '')}\n",
    ])

    return {
        "filename": filename,
        "grade": grade,
        "readiness_score": score,
        "sections": sections,
        "full_report": full_report,
        "word_count": len(full_report.split()),
        "sections_count": len(sections)
    }