import pandas as pd
import numpy as np
import re


def fair_assessment(df: pd.DataFrame, filename: str = "") -> dict:
    """
    FAIR principles assessment for uploaded datasets.
    
    F - Findable: meaningful filename, proper headers, named columns
    A - Accessible: open format, no empty columns, usable structure
    I - Interoperable: standard column names, consistent types
    R - Reusable: sufficient data, descriptive columns, no raw ID dumps
    """

    scores = {}
    details = {}
    n_rows, n_cols = df.shape
    cols = df.columns.tolist()

    # ── F: FINDABLE ──────────────────────────────────────────────────────
    f_score = 100

    # Unnamed columns (pandas default: Unnamed: 0, Unnamed: 1)
    unnamed = [c for c in cols if str(c).startswith('Unnamed')]
    if unnamed:
        f_score -= len(unnamed) * 15
        details['unnamed_columns'] = f"Found {len(unnamed)} unnamed columns — dataset lacks proper headers"

    # Generic column names (col1, column1, var1, x1 etc.)
    generic = [c for c in cols if re.match(r'^(col|column|var|x|field)\d*$', str(c).lower())]
    if generic:
        f_score -= len(generic) * 10
        details['generic_columns'] = f"{len(generic)} generic column names detected (e.g. col1, var2)"

    # Filename quality
    if filename:
        name = filename.lower().replace('.csv','').replace('.xlsx','')
        if name in ['data', 'dataset', 'file', 'upload', 'test', '123']:
            f_score -= 20
            details['filename'] = "Generic filename — use a descriptive name"

    scores['findable'] = max(0, min(100, f_score))

    # ── A: ACCESSIBLE ────────────────────────────────────────────────────
    a_score = 100

    # Completely empty columns
    empty_cols = [c for c in cols if df[c].isnull().all()]
    if empty_cols:
        a_score -= len(empty_cols) * 20
        details['empty_columns'] = f"{len(empty_cols)} completely empty columns"

    # Mostly empty (>90% missing)
    mostly_empty = [c for c in cols if df[c].isnull().mean() > 0.9]
    if mostly_empty:
        a_score -= len(mostly_empty) * 10
        details['mostly_empty'] = f"{len(mostly_empty)} columns over 90% missing"

    # Too few rows to be usable
    if n_rows < 10:
        a_score -= 50
        details['too_few_rows'] = f"Only {n_rows} rows — dataset not practically accessible"

    scores['accessible'] = max(0, min(100, a_score))

    # ── I: INTEROPERABLE ────────────────────────────────────────────────
    i_score = 100

    # Column names with spaces or special characters
    bad_names = [c for c in cols if re.search(r'[^a-zA-Z0-9_]', str(c))]
    if bad_names:
        i_score -= len(bad_names) * 8
        details['non_standard_names'] = f"{len(bad_names)} columns have spaces/special chars — hard to use in code"

    # Mixed type columns (object column that has both numbers and strings)
    mixed = []
    for c in df.select_dtypes(include='object').columns:
        sample = df[c].dropna().head(100)
        numeric_count = sum(1 for v in sample if str(v).replace('.','').replace('-','').isdigit())
        if 0 < numeric_count < len(sample) * 0.8 and numeric_count > len(sample) * 0.2:
            mixed.append(c)
    if mixed:
        i_score -= len(mixed) * 10
        details['mixed_types'] = f"{len(mixed)} columns appear to have mixed data types"

    scores['interoperable'] = max(0, min(100, i_score))

    # ── R: REUSABLE ──────────────────────────────────────────────────────
    r_score = 100

    # Not enough data to be reusable for ML
    if n_rows < 50:
        r_score -= 40
        details['insufficient_data'] = f"Only {n_rows} rows — too few to train any reliable ML model"
    elif n_rows < 200:
        r_score -= 20
        details['limited_data'] = f"{n_rows} rows — limited reusability for complex models"

    # Column names too short (single char or 2 chars) = not descriptive
    short_names = [c for c in cols if len(str(c)) <= 2]
    if len(short_names) > n_cols * 0.5:
        r_score -= 25
        details['short_names'] = f"{len(short_names)} columns have very short names — not self-documenting"

    # All-unique columns that aren't the target (raw ID dumps)
    id_dumps = [c for c in cols if df[c].nunique() == n_rows and df[c].dtype == object]
    if id_dumps:
        r_score -= len(id_dumps) * 10
        details['id_dumps'] = f"{len(id_dumps)} columns are 100% unique strings — likely raw identifiers"

    scores['reusable'] = max(0, min(100, r_score))

    # ── OVERALL FAIR SCORE ───────────────────────────────────────────────
    overall = round(sum(scores.values()) / 4, 1)

    if overall >= 85:
        fair_grade = "FAIR-compliant"
    elif overall >= 65:
        fair_grade = "Partially FAIR"
    elif overall >= 40:
        fair_grade = "Needs improvement"
    else:
        fair_grade = "Not FAIR"

    return {
        "overall_fair_score": overall,
        "fair_grade": fair_grade,
        "dimension_scores": {
            "findable":       round(scores['findable'], 1),
            "accessible":     round(scores['accessible'], 1),
            "interoperable":  round(scores['interoperable'], 1),
            "reusable":       round(scores['reusable'], 1)
        },
        "issues_found": details,
        "summary": (
            f"Dataset scores {overall}/100 on FAIR principles. "
            f"Grade: {fair_grade}. "
            f"{len(details)} FAIR issues detected."
        )
    }