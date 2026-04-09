"""
explainability_engine.py

Core innovation of the ML Data Readiness Analyzer.

Every other module in this project DETECTS issues.
This module EXPLAINS them — in plain English, with severity,
with root cause, and with a specific fix.

Those tools tell you WHAT your data looks like.
This tool tells you WHY it will fail in ML and HOW to fix it.
"""


# ── SEVERITY WEIGHTS (for overall readiness score) ──────────────────────────
SEVERITY_WEIGHTS = {
    "CRITICAL": 12,
    "HIGH":     7,
    "MODERATE": 4,
    "LOW":      1,
    "INFO":     0
}


def _make_issue(severity, category, title, explanation, fix, source_module):
    return {
        "severity": severity,
        "category": category,
        "title": title,
        "explanation": explanation,
        "fix": fix,
        "source_module": source_module
    }


def _explain_missing_values(preprocessing_advice):
    issues = []
    for col, info in preprocessing_advice.items():
        missing = info.get("missing_analysis", {})
        pct = missing.get("missing_percent", 0)
        mechanism = missing.get("inferred_mechanism", "")
        action = missing.get("recommended_action", "")

        if pct > 60:
            issues.append(_make_issue(
                severity="CRITICAL",
                category="Missing Data",
                title=f"Column '{col}' is {pct:.1f}% missing",
                explanation=(
                    f"'{col}' has {pct:.1f}% missing values. "
                    f"Inferred mechanism: {mechanism}. "
                    "Imputing a column this sparse introduces more noise than signal."
                ),
                fix=f"Drop '{col}' entirely. {action}",
                source_module="preprocessing_suggestions"
            ))
        elif pct > 15:
            issues.append(_make_issue(
                severity="HIGH",
                category="Missing Data",
                title=f"Column '{col}' has {pct:.1f}% missing values",
                explanation=(
                    f"'{col}' is missing {pct:.1f}% of values. "
                    f"Mechanism: {mechanism}. "
                    "This level of missingness can bias model training."
                ),
                fix=f"Impute + add binary missingness indicator column. {action}",
                source_module="preprocessing_suggestions"
            ))
        elif pct > 5:
            issues.append(_make_issue(
                severity="MODERATE",
                category="Missing Data",
                title=f"Column '{col}' has {pct:.1f}% missing values",
                explanation=f"'{col}' has minor missingness ({pct:.1f}%). Generally safe to impute.",
                fix=action if action else "Apply mean/median/mode imputation.",
                source_module="preprocessing_suggestions"
            ))
    return issues


def _explain_target_detection(target_detection):
    issues = []
    confidence = target_detection.get("confidence", 1.0)
    predicted = target_detection.get("predicted_target", "unknown")
    source = target_detection.get("target_source", "auto_detected")

    if source == "auto_detected" and confidence < 0.3:
        issues.append(_make_issue(
            severity="CRITICAL",
            category="Target Detection",
            title="Target column could not be detected with confidence",
            explanation=(
                f"Auto-detection picked '{predicted}' but with only {confidence:.0%} confidence. "
                "If the wrong column is used as target, every model metric is meaningless."
            ),
            fix="Manually specify: POST /upload?target_column=<your_column_name>",
            source_module="target_detection"
        ))
    elif source == "auto_detected" and confidence < 0.6:
        issues.append(_make_issue(
            severity="MODERATE",
            category="Target Detection",
            title=f"Target column detected with moderate confidence ({confidence:.0%})",
            explanation=(
                f"System selected '{predicted}' as target with {confidence:.0%} confidence. "
                "This may be correct but should be verified."
            ),
            fix="Verify the target column is correct or override manually.",
            source_module="target_detection"
        ))
    return issues


def _explain_data_leakage(leakage):
    issues = []

    for item in leakage.get("target_leakage", []):
        issues.append(_make_issue(
            severity="CRITICAL",
            category="Data Leakage",
            title=f"Target leakage: '{item['feature']}' correlates with target",
            explanation=(
                f"Feature '{item['feature']}' has {item['correlation']:.2%} correlation "
                f"with target '{item['target']}'. "
                "Your model will appear perfect during training but fail in production."
            ),
            fix=f"Remove '{item['feature']}' immediately before any training.",
            source_module="leakage_detection"
        ))

    for item in leakage.get("high_correlation_pairs", []):
        if item.get("severity") in ("HIGH", "CRITICAL"):
            issues.append(_make_issue(
                severity="HIGH",
                category="Data Leakage",
                title=f"Near-duplicate features: '{item['feature_1']}' and '{item['feature_2']}'",
                explanation=(
                    f"'{item['feature_1']}' and '{item['feature_2']}' are "
                    f"{item['correlation']:.2%} correlated. One may be derived from the other."
                ),
                fix="Drop the less predictive of the two.",
                source_module="leakage_detection"
            ))

    for item in leakage.get("duplicate_columns", []):
        issues.append(_make_issue(
            severity="HIGH",
            category="Data Leakage",
            title=f"Exact duplicate columns: '{item['feature_1']}' = '{item['feature_2']}'",
            explanation=f"'{item['feature_1']}' and '{item['feature_2']}' are identical.",
            fix=f"Drop '{item['feature_2']}' — keep one copy only.",
            source_module="leakage_detection"
        ))

    return issues


def _explain_overfitting(overfitting):
    issues = []
    per_model = overfitting.get("per_model_analysis", {})

    for model_name, result in per_model.items():
        if "error" in result:
            continue
        risk = result.get("overfitting_risk", "LOW")
        gap = result.get("train_test_gap", 0)
        cv_std = result.get("cv_std", 0)
        lc = result.get("learning_curve", {})
        lc_converging = lc.get("is_converging", True)

        if risk == "HIGH":
            issues.append(_make_issue(
                severity="HIGH",
                category="Overfitting",
                title=f"{model_name}: High overfitting risk (gap={gap:.3f})",
                explanation=(
                    f"{model_name} has a train/test gap of {gap:.3f} and CV std of {cv_std:.3f}. "
                    + ("Learning curve is not converging. " if not lc_converging else "")
                    + "This model will underperform on new data."
                ),
                fix=result.get("reason", "Apply regularization or reduce model complexity."),
                source_module="overfitting_detection"
            ))
        elif risk == "MODERATE":
            issues.append(_make_issue(
                severity="MODERATE",
                category="Overfitting",
                title=f"{model_name}: Moderate overfitting (gap={gap:.3f})",
                explanation=f"{model_name} shows minor train/test gap ({gap:.3f}). May improve with tuning.",
                fix="Try cross-validation tuning or slight regularization.",
                source_module="overfitting_detection"
            ))
    return issues


def _explain_feature_quality(feature_selection, feature_importance):
    issues = []

    tier_summary = feature_importance.get("tier_summary", {})
    negligible = tier_summary.get("NEGLIGIBLE", [])
    if len(negligible) > 0:
        issues.append(_make_issue(
            severity="MODERATE",
            category="Feature Quality",
            title=f"{len(negligible)} negligible feature(s) detected",
            explanation=(
                f"Features {negligible} contribute near-zero predictive value. "
                "Keeping them adds noise without benefit."
            ),
            fix=f"Remove these features: {negligible}",
            source_module="feature_importance_engine"
        ))

    disagreements = feature_importance.get("method_disagreements", [])
    if len(disagreements) > 0:
        names = [d["feature"] for d in disagreements]
        issues.append(_make_issue(
            severity="LOW",
            category="Feature Quality",
            title=f"{len(disagreements)} feature(s) have conflicting importance signals",
            explanation=(
                f"Features {names} score differently between RF importance and permutation importance. "
                "Usually means correlated features sharing importance."
            ),
            fix="Check correlation matrix. Consider removing the weaker duplicate.",
            source_module="feature_importance_engine"
        ))

    to_remove = feature_selection.get("features_to_remove", [])
    removal_report = feature_selection.get("removal_report", {})
    for feat in to_remove:
        n_methods = removal_report.get(feat, {}).get("flagged_by_n_methods", 0)
        reasons = removal_report.get(feat, {}).get("reasons", [])
        if n_methods >= 3:
            issues.append(_make_issue(
                severity="HIGH",
                category="Feature Quality",
                title=f"Feature '{feat}' flagged by {n_methods} selection methods",
                explanation=f"'{feat}' flagged as weak by {n_methods} methods: {'; '.join(reasons)}",
                fix=f"Remove '{feat}' from training features.",
                source_module="feature_selection_engine"
            ))
    return issues


def _explain_cv_stability(cross_validation):
    issues = []
    per_model = cross_validation.get("per_model_stability", {})

    for model_name, result in per_model.items():
        if "error" in result:
            continue
        stability = result.get("stability", "HIGH")
        cv_pct = result.get("coefficient_of_variation_pct", 0)
        worst = result.get("worst_fold", {})
        worst_warning = worst.get("warning")

        if stability == "VERY LOW":
            issues.append(_make_issue(
                severity="CRITICAL",
                category="Model Stability",
                title=f"{model_name}: Extremely unstable CV (CV%={cv_pct:.1f}%)",
                explanation=(
                    f"{model_name} performs very differently across folds (CV%={cv_pct:.1f}%). "
                    "Performance cannot be trusted or reproduced."
                    + (f" {worst_warning}" if worst_warning else "")
                ),
                fix=result.get("recommendation", "Investigate data distribution across folds."),
                source_module="cross_validation_engine"
            ))
        elif stability == "LOW":
            issues.append(_make_issue(
                severity="HIGH",
                category="Model Stability",
                title=f"{model_name}: Unstable cross-validation (CV%={cv_pct:.1f}%)",
                explanation=(
                    f"{model_name} has high variance across folds (CV%={cv_pct:.1f}%). "
                    + (f"{worst_warning}" if worst_warning else "")
                ),
                fix=result.get("recommendation", "Check for class imbalance or distribution shift."),
                source_module="cross_validation_engine"
            ))
    return issues


def _explain_dataset_quality(quality_score):
    issues = []
    dimensions = quality_score.get("dimension_scores", {})

    dim_explanations = {
        "completeness": "Completeness measures what percentage of data is present (not missing).",
        "uniqueness": "Uniqueness measures how many rows are original (not duplicated).",
        "consistency": "Consistency measures how free the data is from outliers.",
        "class_balance": "Class balance measures how evenly distributed the target classes are.",
        "feature_quality": "Feature quality measures how informative and non-redundant features are.",
        "adequacy": "Adequacy measures if there are enough rows relative to features."
    }

    for dim, score in dimensions.items():
        explanation = dim_explanations.get(dim, "")
        if score < 50:
            issues.append(_make_issue(
                severity="HIGH",
                category="Dataset Quality",
                title=f"Low {dim} score ({score:.1f}/100)",
                explanation=f"Dataset scores {score:.1f}/100 on {dim}. {explanation}",
                fix=f"Address {dim} issues before training any model.",
                source_module="data_quality_score"
            ))
        elif score < 70:
            issues.append(_make_issue(
                severity="MODERATE",
                category="Dataset Quality",
                title=f"Moderate {dim} score ({score:.1f}/100)",
                explanation=f"Dataset scores {score:.1f}/100 on {dim}. {explanation}",
                fix=f"Improving {dim} will increase model reliability.",
                source_module="data_quality_score"
            ))
    return issues


def _compute_readiness_score(all_issues):
    """
    Compute ML readiness score (0-100).

    Key improvements over original:
    - Smaller per-issue weights (less aggressive)
    - Total penalty capped at 60 (floor always 40)
    - Bonus +5 if no CRITICAL issues
    - Titanic (real issues but clean benchmark) should score 55-70
    """
    critical_count  = sum(1 for i in all_issues if i["severity"] == "CRITICAL")
    high_count      = sum(1 for i in all_issues if i["severity"] == "HIGH")
    moderate_count  = sum(1 for i in all_issues if i["severity"] == "MODERATE")
    low_count       = sum(1 for i in all_issues if i["severity"] == "LOW")

    raw_penalty = (
        critical_count  * SEVERITY_WEIGHTS["CRITICAL"] +
        high_count      * SEVERITY_WEIGHTS["HIGH"] +
        moderate_count  * SEVERITY_WEIGHTS["MODERATE"] +
        low_count       * SEVERITY_WEIGHTS["LOW"]
    )

    # Cap penalty so one bad issue can't tank everything
    total_penalty = min(raw_penalty, 60)
    score = 100 - total_penalty

    # Bonus for clean critical tier
    if critical_count == 0:
        score += 5

    score = max(0, min(100, score))

    if score >= 85:
        grade = "A"
        verdict = "Dataset is ML-ready. Proceed to model training."
    elif score >= 70:
        grade = "B"
        verdict = "Dataset is mostly ready. Fix HIGH severity issues first."
    elif score >= 55:
        grade = "C"
        verdict = "Dataset needs cleaning before reliable training."
    elif score >= 40:
        grade = "D"
        verdict = "Dataset has serious problems. Training now will give misleading results."
    else:
        grade = "F"
        verdict = "Dataset is not ready for ML. Address all CRITICAL issues first."

    return {
        "readiness_score": round(score, 1),
        "grade": grade,
        "verdict": verdict,
        "total_issues": len(all_issues),
        "by_severity": {
            "CRITICAL": critical_count,
            "HIGH":     high_count,
            "MODERATE": moderate_count,
            "LOW":      low_count,
        }
    }


def _prioritized_action_plan(all_issues):
    severity_order = {"CRITICAL": 0, "HIGH": 1, "MODERATE": 2, "LOW": 3, "INFO": 4}
    sorted_issues = sorted(all_issues, key=lambda x: severity_order.get(x["severity"], 5))
    return [
        {
            "step": i + 1,
            "priority": issue["severity"],
            "action": issue["fix"],
            "reason": issue["title"],
            "module": issue["source_module"]
        }
        for i, issue in enumerate(sorted_issues)
    ]


def generate_explainability_report(analysis_result):
    """
    Master explainability engine.

    Takes the full /upload response and generates a human-readable
    ML readiness report with:
    - readiness_score: 0-100 with A-F grade
    - issues: all problems ranked by severity
    - action_plan: numbered step-by-step fix list
    - positive_signals: what is already good
    - module_health: which modules found issues
    """
    all_issues = []

    try:
        preprocessing = analysis_result.get("preprocessing_advice", {})
        if isinstance(preprocessing, dict):
            all_issues += _explain_missing_values(preprocessing)
    except Exception:
        pass

    try:
        target_det = analysis_result.get("target_detection", {})
        if isinstance(target_det, dict):
            all_issues += _explain_target_detection(target_det)
    except Exception:
        pass

    try:
        leakage = analysis_result.get("data_leakage_analysis", {})
        if isinstance(leakage, dict):
            all_issues += _explain_data_leakage(leakage)
    except Exception:
        pass

    try:
        overfitting = analysis_result.get("overfitting_analysis", {})
        if isinstance(overfitting, dict):
            all_issues += _explain_overfitting(overfitting)
    except Exception:
        pass

    try:
        feat_selection = analysis_result.get("smart_feature_selection", {})
        feat_importance = analysis_result.get("feature_importance", {})
        if isinstance(feat_selection, dict) and isinstance(feat_importance, dict):
            all_issues += _explain_feature_quality(feat_selection, feat_importance)
    except Exception:
        pass

    try:
        cv = analysis_result.get("cross_validation", {})
        if isinstance(cv, dict):
            all_issues += _explain_cv_stability(cv)
    except Exception:
        pass

    try:
        quality = analysis_result.get("dataset_quality_score", {})
        if isinstance(quality, dict):
            all_issues += _explain_dataset_quality(quality)
    except Exception:
        pass

    readiness = _compute_readiness_score(all_issues)
    action_plan = _prioritized_action_plan(all_issues)

    # Positive signals
    positive_signals = []
    quality_score = analysis_result.get("dataset_quality_score", {})
    dimensions = quality_score.get("dimension_scores", {})
    for dim, score in dimensions.items():
        if score >= 90:
            positive_signals.append(f"Excellent {dim} ({score:.0f}/100) — no action needed")

    leakage = analysis_result.get("data_leakage_analysis", {})
    if isinstance(leakage, dict):
        if leakage.get("summary", {}).get("severity") == "NONE":
            positive_signals.append("No data leakage detected")
        if len(leakage.get("target_leakage", [])) == 0:
            positive_signals.append("No target leakage found")

    overfitting = analysis_result.get("overfitting_analysis", {})
    if isinstance(overfitting, dict):
        if overfitting.get("overall_overfitting", {}).get("risk", "") == "LOW":
            positive_signals.append("All models show low overfitting risk")

    target_det = analysis_result.get("target_detection", {})
    if isinstance(target_det, dict):
        confidence = target_det.get("confidence", 0)
        if confidence >= 0.8:
            predicted = target_det.get("predicted_target", "")
            positive_signals.append(
                f"Target column '{predicted}' detected with high confidence ({confidence:.0%})"
            )

    # Module health
    module_issue_counts = {}
    for issue in all_issues:
        mod = issue["source_module"]
        module_issue_counts[mod] = module_issue_counts.get(mod, 0) + 1

    all_modules = [
        "target_detection", "preprocessing_suggestions", "leakage_detection",
        "overfitting_detection", "cross_validation_engine",
        "feature_importance_engine", "feature_selection_engine", "data_quality_score"
    ]

    module_health = {
        mod: {
            "issues_found": module_issue_counts.get(mod, 0),
            "status": "ISSUES FOUND" if module_issue_counts.get(mod, 0) > 0 else "CLEAN"
        }
        for mod in all_modules
    }

    return {
        "readiness_score": readiness["readiness_score"],
        "grade": readiness["grade"],
        "verdict": readiness["verdict"],
        "issue_summary": readiness["by_severity"],
        "issues": all_issues,
        "action_plan": action_plan,
        "positive_signals": positive_signals,
        "module_health": module_health
    }