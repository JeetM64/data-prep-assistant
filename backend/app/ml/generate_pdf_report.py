import io
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether
)
from reportlab.platypus import BaseDocTemplate, PageTemplate, Frame
from reportlab.lib.colors import HexColor


# ── COLORS ────────────────────────────────────────────────────────────────────
C_DARK    = HexColor("#2C3E50")
C_BLUE    = HexColor("#2980B9")
C_GREEN   = HexColor("#27AE60")
C_ORANGE  = HexColor("#E67E22")
C_RED     = HexColor("#E74C3C")
C_YELLOW  = HexColor("#F39C12")
C_LIGHT   = HexColor("#ECF0F1")
C_MID     = HexColor("#BDC3C7")
C_WHITE   = colors.white
C_TEAL    = HexColor("#1ABC9C")


def grade_color(grade):
    return {
        "A": C_GREEN, "B": C_TEAL,
        "C": C_YELLOW, "D": C_ORANGE, "F": C_RED
    }.get(grade, C_BLUE)


def severity_color(sev):
    return {
        "CRITICAL": C_RED, "HIGH": C_ORANGE,
        "MODERATE": C_YELLOW, "LOW": C_GREEN
    }.get(sev, C_MID)


# ── STYLES ────────────────────────────────────────────────────────────────────
def make_styles():
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="ReportTitle",
        fontName="Helvetica-Bold",
        fontSize=26,
        textColor=C_WHITE,
        alignment=TA_CENTER,
        spaceAfter=4
    ))
    styles.add(ParagraphStyle(
        name="ReportSubtitle",
        fontName="Helvetica",
        fontSize=12,
        textColor=C_LIGHT,
        alignment=TA_CENTER,
        spaceAfter=2
    ))
    styles.add(ParagraphStyle(
        name="SectionTitle",
        fontName="Helvetica-Bold",
        fontSize=14,
        textColor=C_DARK,
        spaceBefore=14,
        spaceAfter=6,
        borderPadding=(0, 0, 4, 0)
    ))
    styles.add(ParagraphStyle(
        name="SubSection",
        fontName="Helvetica-Bold",
        fontSize=11,
        textColor=C_BLUE,
        spaceBefore=8,
        spaceAfter=4
    ))
    styles.add(ParagraphStyle(
        name="Body",
        fontName="Helvetica",
        fontSize=9,
        textColor=C_DARK,
        spaceAfter=4,
        leading=14,
        alignment=TA_JUSTIFY
    ))
    styles.add(ParagraphStyle(
        name="Small",
        fontName="Helvetica",
        fontSize=8,
        textColor=HexColor("#7F8C8D"),
        spaceAfter=2
    ))
    styles.add(ParagraphStyle(
        name="TableCell",
        fontName="Helvetica",
        fontSize=8,
        textColor=C_DARK,
        leading=11
    ))
    styles.add(ParagraphStyle(
        name="TableHeader",
        fontName="Helvetica-Bold",
        fontSize=8,
        textColor=C_WHITE,
        leading=11
    ))
    styles.add(ParagraphStyle(
        name="Mono",
        fontName="Courier",
        fontSize=8,
        textColor=C_DARK,
        spaceAfter=2
    ))
    return styles


# ── HELPER BUILDERS ───────────────────────────────────────────────────────────
def header_table(title, subtitle=""):
    """Dark header banner with title."""
    content = [[
        Paragraph(f'<font color="white"><b>{title}</b></font>', ParagraphStyle(
            "h", fontName="Helvetica-Bold", fontSize=14, textColor=C_WHITE, alignment=TA_LEFT
        )),
        Paragraph(f'<font color="#BDC3C7">{subtitle}</font>', ParagraphStyle(
            "hs", fontName="Helvetica", fontSize=9, textColor=C_MID, alignment=TA_RIGHT
        ))
    ]]
    t = Table(content, colWidths=[110*mm, 60*mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), C_DARK),
        ("ROWPADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROUNDEDCORNERS", [4, 4, 4, 4]),
    ]))
    return t


def metric_cards(items):
    """
    Row of metric cards. items = [(label, value, color), ...]
    Max 4 per row.
    """
    col_w = 170 * mm / len(items)
    row = []
    for label, value, color in items:
        cell = Table(
            [[Paragraph(str(value), ParagraphStyle(
                "mv", fontName="Helvetica-Bold", fontSize=18,
                textColor=color, alignment=TA_CENTER
            ))],
             [Paragraph(label, ParagraphStyle(
                "ml", fontName="Helvetica", fontSize=8,
                textColor=HexColor("#7F8C8D"), alignment=TA_CENTER
            ))]],
            colWidths=[col_w]
        )
        cell.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), C_LIGHT),
            ("ROWPADDING", (0, 0), (-1, -1), 6),
            ("BOX", (0, 0), (-1, -1), 0.5, C_MID),
            ("ROUNDEDCORNERS", [4, 4, 4, 4]),
        ]))
        row.append(cell)
    t = Table([row], colWidths=[col_w] * len(items), hAlign="LEFT")
    t.setStyle(TableStyle([
        ("COLPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def section_line():
    return HRFlowable(width="100%", thickness=0.5, color=C_MID, spaceAfter=6, spaceBefore=2)


def data_table(headers, rows, col_widths=None):
    """Standard data table with dark header."""
    if not rows:
        return Paragraph("No data available.", ParagraphStyle(
            "nd", fontName="Helvetica", fontSize=8, textColor=HexColor("#999")
        ))
    s = make_styles()
    header_row = [Paragraph(h, s["TableHeader"]) for h in headers]
    data_rows = []
    for i, row in enumerate(rows):
        data_rows.append([
            Paragraph(str(cell) if cell is not None else "-", s["TableCell"])
            for cell in row
        ])

    all_rows = [header_row] + data_rows
    w = col_widths or [170 * mm / len(headers)] * len(headers)
    t = Table(all_rows, colWidths=w, repeatRows=1)

    style = [
        ("BACKGROUND", (0, 0), (-1, 0), C_DARK),
        ("TEXTCOLOR", (0, 0), (-1, 0), C_WHITE),
        ("ROWPADDING", (0, 0), (-1, -1), 5),
        ("GRID", (0, 0), (-1, -1), 0.3, C_MID),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]
    for i in range(1, len(all_rows)):
        if i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), C_LIGHT))
    t.setStyle(TableStyle(style))
    return t


def bar_cell(value, max_val=100, color=C_BLUE, width=60*mm):
    """Inline progress bar for table cells."""
    pct = min(1.0, value / max_val) if max_val > 0 else 0
    bar_w = width * pct
    blank_w = width - bar_w
    row = []
    if bar_w > 0:
        row.append(Table([[""]], colWidths=[bar_w], rowHeights=[8]))
        row[-1].setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), color)]))
    if blank_w > 0:
        row.append(Table([[""]], colWidths=[blank_w], rowHeights=[8]))
        row[-1].setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), C_LIGHT)]))
    t = Table([row], hAlign="LEFT")
    return t


# ── SECTION BUILDERS ──────────────────────────────────────────────────────────

def build_cover_page(story, data, styles, filename):
    """Full cover page with grade, score, and key stats."""
    er = data.get("explainability_report", {})
    ds = data.get("dataset_summary", {})
    grade = er.get("grade", "?")
    score = er.get("readiness_score", 0)
    gc = grade_color(grade)

    # Title banner
    title_table = Table(
        [[Paragraph("ML DATA READINESS ANALYZER", ParagraphStyle(
            "ct", fontName="Helvetica-Bold", fontSize=22,
            textColor=C_WHITE, alignment=TA_CENTER
        ))],
         [Paragraph("Automated Dataset Analysis Report", ParagraphStyle(
            "cs", fontName="Helvetica", fontSize=12,
            textColor=C_LIGHT, alignment=TA_CENTER
        ))],
         [Paragraph(f"Dataset: {filename}", ParagraphStyle(
            "cf", fontName="Helvetica", fontSize=10,
            textColor=C_MID, alignment=TA_CENTER
        ))]],
        colWidths=[170*mm]
    )
    title_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), C_DARK),
        ("ROWPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(title_table)
    story.append(Spacer(1, 8*mm))

    # Grade + Score big display
    grade_table = Table([[
        Table(
            [[Paragraph(grade, ParagraphStyle(
                "gl", fontName="Helvetica-Bold", fontSize=52,
                textColor=gc, alignment=TA_CENTER
            ))],
             [Paragraph("Grade", ParagraphStyle(
                "gs", fontName="Helvetica", fontSize=9,
                textColor=HexColor("#999"), alignment=TA_CENTER
            ))]],
            colWidths=[50*mm]
        ),
        Table(
            [[Paragraph(str(score), ParagraphStyle(
                "sl", fontName="Helvetica-Bold", fontSize=52,
                textColor=C_DARK, alignment=TA_CENTER
            ))],
             [Paragraph("Readiness Score / 100", ParagraphStyle(
                "ss", fontName="Helvetica", fontSize=9,
                textColor=HexColor("#999"), alignment=TA_CENTER
            ))]],
            colWidths=[60*mm]
        ),
        Table(
            [[Paragraph(er.get("verdict", ""), ParagraphStyle(
                "vl", fontName="Helvetica", fontSize=10,
                textColor=C_DARK, alignment=TA_CENTER, leading=14
            ))]],
            colWidths=[60*mm]
        )
    ]], colWidths=[50*mm, 60*mm, 60*mm])
    grade_table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOX", (0, 0), (-1, -1), 0.5, C_MID),
        ("ROWPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(grade_table)
    story.append(Spacer(1, 6*mm))

    # Issue summary badges
    iss = er.get("issue_summary", {})
    issue_data = [
        ("CRITICAL", iss.get("CRITICAL", 0), C_RED),
        ("HIGH", iss.get("HIGH", 0), C_ORANGE),
        ("MODERATE", iss.get("MODERATE", 0), C_YELLOW),
        ("LOW", iss.get("LOW", 0), C_GREEN),
    ]
    story.append(metric_cards([(f"{c} issues", v, col) for c, v, col in issue_data]))
    story.append(Spacer(1, 6*mm))

    # Dataset stats row
    story.append(metric_cards([
        ("Rows", f"{ds.get('rows', 0):,}", C_DARK),
        ("Columns", ds.get("columns", 0), C_DARK),
        ("Missing %", f"{ds.get('missing_value_percent', 0)}%", C_ORANGE if ds.get('missing_value_percent', 0) > 5 else C_GREEN),
        ("Duplicates", ds.get("duplicate_rows", 0), C_RED if ds.get("duplicate_rows", 0) > 0 else C_GREEN),
    ]))
    story.append(Spacer(1, 6*mm))

    # Generated timestamp + positive signals
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  "
        f"Tool: ML Data Readiness Analyzer v3.0  |  "
        f"Modules run: 18",
        styles["Small"]
    ))

    signals = er.get("positive_signals", [])
    if signals:
        story.append(Spacer(1, 4*mm))
        story.append(Paragraph("Positive signals detected:", styles["SubSection"]))
        for s in signals:
            story.append(Paragraph(f"  ✓  {s}", ParagraphStyle(
                "sig", fontName="Helvetica", fontSize=9,
                textColor=C_GREEN, spaceAfter=2
            )))

    story.append(PageBreak())


def build_quality_section(story, data, styles):
    """Dataset quality scorecard section."""
    qs = data.get("dataset_quality_score", {})
    dims = qs.get("dimension_scores", {})
    if not dims:
        return

    story.append(header_table("Dataset Quality Scorecard", f"Overall: {qs.get('overall_score', 0)}/100 — {qs.get('status', '')}"))
    story.append(Spacer(1, 4*mm))

    dim_colors = {
        "completeness": C_BLUE, "uniqueness": C_GREEN,
        "consistency": C_ORANGE, "class_balance": HexColor("#9B59B6"),
        "feature_quality": C_TEAL, "adequacy": C_DARK
    }

    rows = []
    for dim, score in dims.items():
        color = dim_colors.get(dim, C_BLUE)
        score_color = C_GREEN if score >= 80 else C_ORANGE if score >= 60 else C_RED
        rows.append([
            dim.replace("_", " ").title(),
            f"{score:.0f}/100",
            bar_cell(score, 100, color, 60*mm),
            "Good" if score >= 80 else "Moderate" if score >= 60 else "Needs work"
        ])

    story.append(data_table(
        ["Dimension", "Score", "Visual", "Status"],
        rows,
        [40*mm, 20*mm, 65*mm, 35*mm]
    ))
    story.append(Spacer(1, 6*mm))


def build_issues_section(story, data, styles):
    """All issues with severity, explanation, fix."""
    er = data.get("explainability_report", {})
    issues = er.get("issues", [])
    if not issues:
        return

    story.append(header_table("Issues Found", f"{len(issues)} issues across all modules"))
    story.append(Spacer(1, 4*mm))

    for issue in issues:
        sev = issue.get("severity", "LOW")
        sc = severity_color(sev)

        # Issue card
        card_rows = [
            [Paragraph(f'<b>{issue.get("title", "")}</b>', styles["TableCell"]),
             Paragraph(sev, ParagraphStyle(
                 "sevl", fontName="Helvetica-Bold", fontSize=8,
                 textColor=sc, alignment=TA_RIGHT
             ))],
            [Paragraph(issue.get("explanation", ""), styles["TableCell"]),
             Paragraph(issue.get("category", ""), styles["Small"])],
            [Paragraph(f'Fix: {issue.get("fix", "")}', ParagraphStyle(
                "fix", fontName="Helvetica-Oblique", fontSize=8,
                textColor=C_BLUE, leading=11
            )),
             Paragraph(issue.get("source_module", ""), styles["Small"])]
        ]
        card = Table(card_rows, colWidths=[130*mm, 40*mm])
        card.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), HexColor("#FAFAFA")),
            ("LINERIGHT", (0, 0), (0, -1), 3, sc),
            ("ROWPADDING", (0, 0), (-1, -1), 5),
            ("BOX", (0, 0), (-1, -1), 0.3, C_MID),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(KeepTogether([card, Spacer(1, 2*mm)]))

    story.append(Spacer(1, 4*mm))


def build_action_plan_section(story, data, styles):
    """Numbered action plan."""
    er = data.get("explainability_report", {})
    plan = er.get("action_plan", [])
    if not plan:
        return

    story.append(header_table("Action Plan", "Prioritized steps to improve ML readiness"))
    story.append(Spacer(1, 4*mm))

    rows = []
    for a in plan:
        sev = a.get("priority", "LOW")
        sc = severity_color(sev)
        rows.append([
            str(a.get("step", "")),
            a.get("priority", ""),
            a.get("reason", ""),
            a.get("action", "")[:120] + ("..." if len(a.get("action", "")) > 120 else ""),
            a.get("module", "")
        ])

    story.append(data_table(
        ["#", "Priority", "Issue", "Action", "Module"],
        rows,
        [8*mm, 20*mm, 45*mm, 65*mm, 32*mm]
    ))
    story.append(Spacer(1, 6*mm))


def build_features_section(story, data, styles):
    """Feature analysis and importance table."""
    fi = data.get("feature_importance", {})
    ranking = fi.get("ranking_by_rf_importance", [])
    details = fi.get("feature_details", {})
    fa = data.get("feature_analysis", {})

    story.append(header_table("Feature Analysis", f"{len(fa)} features analyzed"))
    story.append(Spacer(1, 4*mm))

    if fa:
        story.append(Paragraph("Column Overview", styles["SubSection"]))
        rows = []
        for col, info in list(fa.items())[:20]:
            missing = info.get("missing_percent", 0)
            unique = info.get("unique_values", 0)
            dtype = info.get("dtype", "")
            is_num = info.get("mean") is not None
            if is_num:
                suggestion = info.get("scaling_suggestion", "-")
                extra = f"mean={info.get('mean',''):.2f}" if isinstance(info.get('mean'), float) else ""
            else:
                suggestion = info.get("encoding_suggestion", "-")
                extra = info.get("top_category", "")
            rows.append([col, dtype, f"{missing}%", str(unique), str(extra)[:20], suggestion[:30]])

        story.append(data_table(
            ["Column", "Type", "Missing", "Unique", "Mean/Top", "Suggestion"],
            rows,
            [35*mm, 20*mm, 18*mm, 18*mm, 30*mm, 49*mm]
        ))
        story.append(Spacer(1, 4*mm))

    if ranking:
        story.append(Paragraph("Feature Importance Ranking", styles["SubSection"]))
        tier_c = {"HIGH": C_GREEN, "MODERATE": C_YELLOW, "LOW": C_ORANGE, "NEGLIGIBLE": C_RED}
        rows = []
        for f in ranking:
            d = details.get(f["feature"], {})
            tier = f.get("tier", "LOW")
            rows.append([
                f["feature"],
                f"{f.get('rf_importance', 0):.4f}",
                f"{d.get('permutation_importance', 0):.4f}",
                tier,
                "Yes" if d.get("methods_agree") else "No"
            ])
        story.append(data_table(
            ["Feature", "RF Importance", "Permutation", "Tier", "Methods Agree"],
            rows,
            [40*mm, 35*mm, 35*mm, 30*mm, 30*mm]
        ))
        story.append(Spacer(1, 6*mm))


def build_models_section(story, data, styles):
    """Model training results."""
    at = data.get("auto_training_results", {})
    models = at.get("model_comparison", {})
    task = at.get("task_type", "")
    best = at.get("best_model", "")
    if not models:
        return

    story.append(header_table("Model Training Results",
        f"Task: {task}  |  Best: {best}  |  Score: {at.get('best_score', 0):.3f}"))
    story.append(Spacer(1, 4*mm))

    metric_key = "f1_weighted" if task == "classification" else "r2_score"
    metric_label = "F1 (weighted)" if task == "classification" else "R2 Score"

    rows = []
    for name, r in models.items():
        if "error" in r:
            continue
        score = r.get(metric_key) or r.get("accuracy") or 0
        is_best = name == best
        rows.append([
            f"★ {name}" if is_best else name,
            f"{score:.3f}",
            f"{r.get('cv_mean_f1') or r.get('cv_mean_r2') or 0:.3f}",
            f"{r.get('cv_std', 0):.3f}",
            f"{r.get('auc_roc', '-')}",
            f"{r.get('training_time_seconds', 0):.2f}s",
            r.get("reasoning", "")[:60]
        ])

    story.append(data_table(
        ["Model", metric_label, "CV Mean", "CV Std", "AUC-ROC", "Time", "Reasoning"],
        rows,
        [35*mm, 18*mm, 18*mm, 15*mm, 18*mm, 14*mm, 52*mm]
    ))
    story.append(Spacer(1, 6*mm))


def build_pipeline_section(story, data, styles):
    """Preprocessing pipeline recommendation."""
    pipeline = data.get("recommended_pipeline", {})
    if not pipeline:
        return

    story.append(header_table("Recommended Preprocessing Pipeline",
        f"Target: {pipeline.get('target_column', '')}"))
    story.append(Spacer(1, 4*mm))

    s = pipeline.get("summary", {})
    story.append(metric_cards([
        ("Features", s.get("total_features", 0), C_DARK),
        ("To drop", s.get("dropped", 0), C_RED),
        ("Need imputation", s.get("columns_needing_imputation", 0), C_ORANGE),
        ("Need encoding", s.get("columns_needing_encoding", 0), C_BLUE),
    ]))
    story.append(Spacer(1, 4*mm))

    rows = []
    for entry in pipeline.get("drop_columns", []):
        rows.append(["DROP", entry.get("column", ""), entry.get("reason", "")])
    for col, v in pipeline.get("missing_value_strategy", {}).items():
        rows.append(["IMPUTE", col, f"{v.get('strategy','')} — {v.get('reason','')}"])
    for col, v in pipeline.get("encoding_strategy", {}).items():
        rows.append(["ENCODE", col, f"{v.get('encoding','')} — {v.get('reason','')}"])
    for col, v in pipeline.get("scaling_strategy", {}).items():
        rows.append(["SCALE", col, f"{v.get('scaler','')} — {v.get('reason','')}"])
    for col, v in pipeline.get("transformation_recommendations", {}).items():
        rows.append(["TRANSFORM", col, f"{v.get('transform','')} (skew={v.get('skewness',0)}) — {v.get('reason','')}"])

    story.append(data_table(
        ["Step", "Column", "Action & Reason"],
        rows,
        [22*mm, 35*mm, 113*mm]
    ))
    story.append(Spacer(1, 6*mm))


def build_fair_section(story, data, styles):
    """FAIR assessment section."""
    fair = data.get("fair_assessment", {})
    if not fair:
        return

    story.append(header_table("FAIR Assessment",
        f"Score: {fair.get('overall_fair_score', 0)}/100 — {fair.get('fair_grade', '')}"))
    story.append(Spacer(1, 4*mm))

    dims = fair.get("dimension_scores", {})
    dim_c = {"findable": C_BLUE, "accessible": C_GREEN, "interoperable": C_ORANGE, "reusable": C_TEAL}
    story.append(metric_cards([
        (k.title(), f"{v:.0f}", dim_c.get(k, C_BLUE))
        for k, v in dims.items()
    ]))
    story.append(Spacer(1, 4*mm))

    issues = fair.get("issues_found", {})
    if issues:
        rows = [[k.replace("_", " ").title(), v] for k, v in issues.items()]
        story.append(data_table(["FAIR Issue", "Description"], rows, [50*mm, 120*mm]))
    story.append(Spacer(1, 6*mm))


def build_anomaly_section(story, data, styles):
    """Anomaly detection section."""
    anom = data.get("anomaly_detection", {})
    if not anom or anom.get("status") == "skipped":
        return

    story.append(header_table("Anomaly Detection (Isolation Forest)",
        f"Severity: {anom.get('severity', '-')}"))
    story.append(Spacer(1, 4*mm))

    story.append(metric_cards([
        ("Anomalous rows", anom.get("anomaly_count", 0), C_ORANGE),
        ("Anomaly %", f"{anom.get('anomaly_percent', 0)}%", C_ORANGE),
        ("Rows checked", anom.get("total_rows_checked", 0), C_DARK),
        ("Severity", anom.get("severity", "-"), severity_color(anom.get("severity", "LOW"))),
    ]))
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph(anom.get("interpretation", ""), styles["Body"]))
    story.append(Spacer(1, 3*mm))

    top_feat = anom.get("top_contributing_features", {})
    if top_feat:
        story.append(Paragraph("Top features contributing to anomalies:", styles["SubSection"]))
        rows = [[col, f"{dev:.3f}"] for col, dev in top_feat.items()]
        story.append(data_table(["Feature", "Deviation Score"], rows, [80*mm, 40*mm]))

    story.append(Spacer(1, 6*mm))


def build_leakage_section(story, data, styles):
    """Data leakage section."""
    leak = data.get("data_leakage_analysis", {})
    total = leak.get("summary", {}).get("total_issues", 0)

    story.append(header_table("Data Leakage Analysis",
        f"Severity: {leak.get('summary', {}).get('severity', 'NONE')}"))
    story.append(Spacer(1, 4*mm))

    if total == 0:
        story.append(Paragraph("No data leakage detected. Dataset is clean.", ParagraphStyle(
            "ok", fontName="Helvetica-Bold", fontSize=10, textColor=C_GREEN
        )))
    else:
        for item in leak.get("target_leakage", []):
            story.append(Paragraph(
                f"CRITICAL — Target leakage: {item.get('feature')} correlates {item.get('correlation'):.2%} with target",
                ParagraphStyle("lk", fontName="Helvetica-Bold", fontSize=9, textColor=C_RED)
            ))
        if leak.get("high_correlation_pairs"):
            rows = [[
                i.get("feature_1", ""), i.get("feature_2", ""),
                f"{i.get('correlation', 0):.4f}", i.get("severity", "")
            ] for i in leak["high_correlation_pairs"]]
            story.append(data_table(
                ["Feature 1", "Feature 2", "Correlation", "Severity"],
                rows, [45*mm, 45*mm, 30*mm, 30*mm]
            ))
    story.append(Spacer(1, 6*mm))


def build_overfitting_section(story, data, styles):
    """Overfitting analysis section."""
    ov = data.get("overfitting_analysis", {})
    if not ov:
        return

    overall = ov.get("overall_overfitting", {})
    story.append(header_table("Overfitting Analysis",
        f"Overall Risk: {overall.get('risk', '-')}"))
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph(overall.get("message", ""), styles["Body"]))
    story.append(Spacer(1, 3*mm))

    per_model = ov.get("per_model_analysis", {})
    if per_model:
        rows = []
        for name, r in per_model.items():
            if "error" in r:
                continue
            lc = r.get("learning_curve", {})
            rows.append([
                name,
                f"{r.get('train_score', 0):.3f}",
                f"{r.get('test_score', 0):.3f}",
                f"{r.get('train_test_gap', 0):.3f}",
                f"{r.get('cv_mean', 0):.3f} ± {r.get('cv_std', 0):.3f}",
                r.get("overfitting_risk", "-"),
                "Yes" if lc.get("is_converging") else "No"
            ])
        story.append(data_table(
            ["Model", "Train", "Test", "Gap", "CV (mean±std)", "Risk", "Converging"],
            rows, [35*mm, 18*mm, 18*mm, 15*mm, 32*mm, 20*mm, 22*mm]
        ))
    story.append(Spacer(1, 6*mm))


# ── MAIN GENERATOR ────────────────────────────────────────────────────────────

def generate_pdf_report(analysis_data: dict, filename: str = "dataset") -> bytes:
    """
    Generate a professional multi-page PDF report from the /upload analysis result.

    Sections:
    1. Cover page — grade, score, key stats, positive signals
    2. Dataset Quality Scorecard — 6 dimensions with visual bars
    3. Issues Found — all issues with severity, explanation, fix
    4. Action Plan — numbered prioritized steps
    5. Feature Analysis — column overview + importance ranking
    6. Model Training Results — all models with metrics
    7. Preprocessing Pipeline — complete DROP/IMPUTE/ENCODE/SCALE plan
    8. FAIR Assessment — Findable/Accessible/Interoperable/Reusable
    9. Anomaly Detection — Isolation Forest results
    10. Data Leakage Analysis
    11. Overfitting Analysis — learning curves, train/test gaps

    Returns bytes — ready to stream as PDF download.
    """

    buffer = io.BytesIO()
    styles = make_styles()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=15*mm,
        leftMargin=15*mm,
        topMargin=15*mm,
        bottomMargin=15*mm,
        title=f"ML Readiness Report — {filename}",
        author="ML Data Readiness Analyzer v3.0",
        subject="Dataset Analysis Report"
    )

    story = []

    # Build all sections
    build_cover_page(story, analysis_data, styles, filename)
    build_quality_section(story, analysis_data, styles)
    story.append(PageBreak())
    build_issues_section(story, analysis_data, styles)
    story.append(PageBreak())
    build_action_plan_section(story, analysis_data, styles)
    story.append(PageBreak())
    build_features_section(story, analysis_data, styles)
    story.append(PageBreak())
    build_models_section(story, analysis_data, styles)
    story.append(PageBreak())
    build_pipeline_section(story, analysis_data, styles)
    story.append(PageBreak())
    build_fair_section(story, analysis_data, styles)
    build_anomaly_section(story, analysis_data, styles)
    story.append(PageBreak())
    build_leakage_section(story, analysis_data, styles)
    build_overfitting_section(story, analysis_data, styles)

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()