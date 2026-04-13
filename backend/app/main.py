from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse
from typing import List
import pandas as pd
import numpy as np
import io

from backend.app.ml.feature_analysis import analyze_features
from backend.app.ml.correlation_analysis import correlation_analysis
from app.ml.preprocessing_suggestions import preprocessing_suggestions
from app.ml.leakage_detection import detect_data_leakage
from app.ml.model_recommendation import recommend_model
from app.ml.auto_training import auto_train
from app.ml.cross_validation_engine import cross_validation_stability
from app.ml.overfitting_detection import detect_overfitting
from app.ml.feature_importance_engine import feature_importance_analysis
from app.ml.cluster_intelligence import cluster_intelligence
from app.ml.data_quality_score import dataset_quality_score
from app.ml.feature_selection_engine import smart_feature_selection
from app.ml.pipeline_builder import build_preprocessing_pipeline
from app.ml.target_detection import detect_target_column
from app.ml.explainability_engine import generate_explainability_report
from app.ml.fair_assessment import fair_assessment
from app.ml.anomaly_detection import detect_anomalies
from app.ml.pipeline_executor import execute_pipeline
from app.ml.benchmark_engine import run_benchmark
from app.ml.generate_pdf_report import generate_pdf_report
from app.ml.dataset_comparison import compare_datasets
from app.ml.whatif_simulator import run_whatif_simulation
from app.ml.nl_report_generator import generate_nl_report
from app.ml.automl_optimizer import run_automl_optimization
from app.ml.shap_explainability import run_shap_analysis
from fastapi.middleware.cors import CORSMiddleware

from app.utils import prepare_ml_dataset



def clean_for_json(obj):
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    return obj


app = FastAPI(
    title="ML Data Readiness Analyzer",
    description="Complete ML pipeline: analysis, preprocessing, ensemble training, SHAP, AutoML, drift detection.",
    version="4.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_df(file_obj, filename: str):
    fname = filename.lower()
    if fname.endswith(".csv"):
        return pd.read_csv(file_obj)
    elif fname.endswith((".xlsx", ".xls")):
        return pd.read_excel(file_obj)
    return None


def _detect_task(df, target):
    y = df[target].dropna()
    return "classification" if y.nunique() <= 15 or y.dtype == object else "regression"


def _run_full_analysis(df, target_override, filename):
    rows, cols = df.shape
    td = detect_target_column(df)

    if target_override and target_override in df.columns:
        target = target_override
        target_source = "user_specified"
    elif target_override:
        return {"error": f"Column '{target_override}' not found.", "available_columns": df.columns.tolist()}
    else:
        target = td["predicted_target"]
        target_source = "auto_detected"

    summary = {
        "rows": int(rows), "columns": int(cols),
        "missing_values": int(df.isnull().sum().sum()),
        "missing_value_percent": round(float(df.isnull().mean().mean() * 100), 2),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_columns": df.select_dtypes(include="number").columns.tolist(),
        "categorical_columns": df.select_dtypes(exclude="number").columns.tolist(),
        "column_dtypes": df.dtypes.astype(str).to_dict(),
        "memory_usage_mb": round(float(df.memory_usage(deep=True).sum() / 1e6), 3)
    }

    qs = dataset_quality_score(df)
    pipeline = build_preprocessing_pipeline(df, target_column=target)
    drop_cols = [e["column"] for e in pipeline.get("drop_columns", []) if e["column"] != target]
    clean_df = prepare_ml_dataset(df.drop(columns=drop_cols, errors="ignore"))
    if target not in clean_df.columns:
        clean_df = prepare_ml_dataset(df.copy())

    resp = {
        "dataset_summary": summary,
        "target_detection": {**td, "final_target": target, "target_source": target_source},
        "feature_analysis":          analyze_features(df),
        "correlation_analysis":      correlation_analysis(df),
        "preprocessing_advice":      preprocessing_suggestions(df),
        "data_leakage_analysis":     detect_data_leakage(df),
        "model_recommendation":      recommend_model(df, target_column=target),
        "dataset_quality_score":     qs,
        "recommended_pipeline":      pipeline,
        "fair_assessment":           fair_assessment(df, filename=filename),
        "anomaly_detection":         detect_anomalies(df),
        "auto_training_results":     auto_train(clean_df, target_column=target),
        "cross_validation":          cross_validation_stability(clean_df, target_column=target),
        "overfitting_analysis":      detect_overfitting(clean_df, target_column=target),
        "feature_importance":        feature_importance_analysis(clean_df, target_column=target),
        "cluster_intelligence":      cluster_intelligence(clean_df),
        "smart_feature_selection":   smart_feature_selection(clean_df, target_column=target),
    }
    resp["explainability_report"] = generate_explainability_report(resp)
    return resp, target


@app.get("/")
def home():
    return {
        "name": "ML Data Readiness Analyzer",
        "version": "4.0.0",
        "total_modules": 25,
        "endpoints": {
            "POST /upload":   "Full 18-module analysis — JSON report",
            "POST /execute":  "Apply pipeline — download cleaned CSV",
            "POST /benchmark":"Compare 2-20 datasets — research table",
            "POST /report":   "Full analysis + PDF download",
            "POST /compare":  "Train vs test — drift detection",
            "POST /whatif":   "What-if simulator — fix impact measurement",
            "POST /nlreport": "Natural language report — plain English",
            "POST /automl":   "AutoML optimizer — ensemble + best pipeline search",
            "POST /shap":     "SHAP explainability — why the model predicts what it predicts",
            "GET  /health":   "Health check"
        }
    }


@app.get("/health")
def health():
    return {"status": "ok", "version": "4.0.0", "modules": 25}


# ── ENDPOINT 1: FULL ANALYSIS ─────────────────────────────────────────────────
@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...), target_column: str = Query(default=None)):
    """Full 18-module analysis. Returns complete JSON readiness report."""
    try:
        df = _load_df(file.file, file.filename)
        if df is None: return JSONResponse({"error": "Unsupported format."})
        if df.empty: return JSONResponse({"error": "Dataset is empty."})
        if df.shape[1] < 2: return JSONResponse({"error": "Need at least 2 columns."})
        result, _ = _run_full_analysis(df, target_column, file.filename)
        if "error" in result: return JSONResponse(result)
        return JSONResponse(content=clean_for_json(result))
    except Exception as e:
        import traceback
        return JSONResponse({"error": str(e), "traceback": traceback.format_exc()[-800:]})


# ── ENDPOINT 2: EXECUTE PIPELINE ─────────────────────────────────────────────
@app.post("/execute")
async def execute_dataset_pipeline(
    file: UploadFile = File(...),
    target_column: str = Query(default=None),
    download: bool = Query(default=True)
):
    """Apply full preprocessing pipeline. Download cleaned ML-ready CSV."""
    try:
        df = _load_df(file.file, file.filename)
        if df is None: return JSONResponse({"error": "Unsupported format."})
        if df.empty: return JSONResponse({"error": "Dataset is empty."})
        td = detect_target_column(df)
        target = target_column if (target_column and target_column in df.columns) else td["predicted_target"]
        pipeline = build_preprocessing_pipeline(df, target_column=target)
        result = execute_pipeline(df, pipeline)
        if not result["success"]: return JSONResponse({"error": "Pipeline failed."})
        cleaned_df = result["cleaned_dataframe"]
        stats = result["stats"]
        if download:
            out = io.StringIO()
            cleaned_df.to_csv(out, index=False)
            out.seek(0)
            base = file.filename.rsplit(".", 1)[0]
            return StreamingResponse(io.BytesIO(out.getvalue().encode()), media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename={base}_cleaned.csv",
                         "X-Original-Shape": f"{stats['original_shape']['rows']}x{stats['original_shape']['columns']}",
                         "X-Final-Shape": f"{stats['final_shape']['rows']}x{stats['final_shape']['columns']}",
                         "X-Steps-Applied": str(stats["steps_applied"])})
        return JSONResponse(content=clean_for_json({
            "success": True, "target": target, "stats": stats,
            "execution_log": result["execution_log"],
            "preview": cleaned_df.head(5).to_dict(orient="records"),
            "columns": cleaned_df.columns.tolist()
        }))
    except Exception as e:
        return JSONResponse({"error": str(e)})


# ── ENDPOINT 3: BENCHMARK ─────────────────────────────────────────────────────
@app.post("/benchmark")
async def benchmark_datasets(files: List[UploadFile] = File(...)):
    """Compare 2-20 datasets. Returns ranked table for research paper."""
    if len(files) < 2: return JSONResponse({"error": "Upload at least 2 files."})
    if len(files) > 20: return JSONResponse({"error": "Max 20 datasets."})
    datasets = []
    for f in files:
        try:
            df = _load_df(f.file, f.filename)
            datasets.append({"name": f.filename.rsplit(".", 1)[0], "dataframe": df})
        except Exception as e:
            datasets.append({"name": f.filename, "dataframe": None, "error": str(e)})
    result = run_benchmark(datasets)
    return JSONResponse(content=clean_for_json(result))


# ── ENDPOINT 4: PDF REPORT ────────────────────────────────────────────────────
@app.post("/report")
async def generate_report(file: UploadFile = File(...), target_column: str = Query(default=None)):
    """Full analysis + professional multi-page PDF report download."""
    try:
        df = _load_df(file.file, file.filename)
        if df is None: return JSONResponse({"error": "Unsupported format."})
        if df.empty: return JSONResponse({"error": "Dataset is empty."})
        if df.shape[1] < 2: return JSONResponse({"error": "Need at least 2 columns."})
        analysis, _ = _run_full_analysis(df, target_column, file.filename)
        if "error" in analysis: return JSONResponse(analysis)
        base = file.filename.rsplit(".", 1)[0]
        pdf_bytes = generate_pdf_report(analysis, filename=base)
        er = analysis.get("explainability_report", {})
        return StreamingResponse(io.BytesIO(pdf_bytes), media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={base}_readiness_report.pdf",
                     "X-Readiness-Score": str(er.get("readiness_score", 0)),
                     "X-Grade": er.get("grade", "?")})
    except Exception as e:
        return JSONResponse({"error": str(e)})


# ── ENDPOINT 5: COMPARE (DRIFT) ───────────────────────────────────────────────
@app.post("/compare")
async def compare_train_test(
    train_file: UploadFile = File(...),
    test_file: UploadFile = File(...)
):
    """
    Distribution drift detection between train and test datasets.
    KS test + PSI + Jensen-Shannon divergence for numeric columns.
    Chi-squared + new category detection for categorical columns.
    """
    try:
        train_df = _load_df(train_file.file, train_file.filename)
        test_df  = _load_df(test_file.file,  test_file.filename)
        if train_df is None: return JSONResponse({"error": f"Unsupported: {train_file.filename}"})
        if test_df is None:  return JSONResponse({"error": f"Unsupported: {test_file.filename}"})
        if train_df.empty: return JSONResponse({"error": "Train dataset is empty."})
        if test_df.empty:  return JSONResponse({"error": "Test dataset is empty."})
        common = set(train_df.columns) & set(test_df.columns)
        if not common: return JSONResponse({"error": "No common columns.", "train_cols": train_df.columns.tolist(), "test_cols": test_df.columns.tolist()})
        result = compare_datasets(train_df, test_df)
        result["file_info"] = {
            "train_file": train_file.filename, "test_file": test_file.filename,
            "train_shape": {"rows": int(train_df.shape[0]), "columns": int(train_df.shape[1])},
            "test_shape":  {"rows": int(test_df.shape[0]),  "columns": int(test_df.shape[1])},
            "common_columns": len(common)
        }
        return JSONResponse(content=clean_for_json(result))
    except Exception as e:
        import traceback
        return JSONResponse({"error": str(e), "traceback": traceback.format_exc()[-800:]})


# ── ENDPOINT 6: WHAT-IF SIMULATOR ────────────────────────────────────────────
@app.post("/whatif")
async def whatif_simulation(
    file: UploadFile = File(...),
    target_column: str = Query(default=None)
):
    """
    What-If Simulator: proves that fixing data issues improves ML performance.

    Applies each fix one by one (drop high-missing → remove duplicates →
    remove IDs → impute → winsorize outliers → log transform → encode)
    and measures model performance after each step.

    Returns before/after comparison and improvement curve.
    This is the experimental validation for the research paper:
    'Higher readiness score → better model performance.'
    """
    try:
        df = _load_df(file.file, file.filename)
        if df is None: return JSONResponse({"error": "Unsupported format."})
        if df.empty: return JSONResponse({"error": "Dataset is empty."})
        if df.shape[1] < 2: return JSONResponse({"error": "Need at least 2 columns."})
        if len(df) < 20: return JSONResponse({"error": "Need at least 20 rows."})

        td = detect_target_column(df)
        target = target_column if (target_column and target_column in df.columns) else td["predicted_target"]
        task = _detect_task(df, target)

        issues = []
        result = run_whatif_simulation(df, target, task, issues)
        return JSONResponse(content=clean_for_json(result))
    except Exception as e:
        import traceback
        return JSONResponse({"error": str(e), "traceback": traceback.format_exc()[-800:]})


# ── ENDPOINT 7: NATURAL LANGUAGE REPORT ──────────────────────────────────────
@app.post("/nlreport")
async def nl_report(
    file: UploadFile = File(...),
    target_column: str = Query(default=None),
    format: str = Query(default="json", description="json or text")
):
    """
    Natural Language Report Generator.

    Runs full analysis then converts every finding into plain English paragraphs.
    No JSON to interpret — just readable paragraphs explaining:
      - What each issue is
      - Why it causes ML failure (root cause)
      - What the expected impact is
      - What to do about it

    Returns either JSON (with sections dict) or plain text (full readable report).
    Use format=text for a copy-paste ready research memo.
    """
    try:
        df = _load_df(file.file, file.filename)
        if df is None: return JSONResponse({"error": "Unsupported format."})
        if df.empty: return JSONResponse({"error": "Dataset is empty."})
        if df.shape[1] < 2: return JSONResponse({"error": "Need at least 2 columns."})

        analysis, _ = _run_full_analysis(df, target_column, file.filename)
        if "error" in analysis: return JSONResponse(analysis)

        base = file.filename.rsplit(".", 1)[0]
        nl_result = generate_nl_report(analysis, filename=base)

        if format == "text":
            return PlainTextResponse(nl_result.get("full_report", ""))

        return JSONResponse(content=clean_for_json(nl_result))
    except Exception as e:
        import traceback
        return JSONResponse({"error": str(e), "traceback": traceback.format_exc()[-800:]})


# ── ENDPOINT 8: AUTOML OPTIMIZER ─────────────────────────────────────────────
@app.post("/automl")
async def automl_optimize(
    file: UploadFile = File(...),
    target_column: str = Query(default=None),
    max_configs: int = Query(default=12, description="Max preprocessing configs to test (6-24)")
):
    """
    AutoML Pipeline Optimizer with Ensemble Learning.

    Tests combinations of:
      Preprocessing: mean/median imputation, log transform, feature selection, scaling
      Individual models: Random Forest, Gradient Boosting, Logistic Regression,
                         SVM, Extra Trees, AdaBoost, KNN, Naive Bayes
      Ensemble models:
        - VotingClassifier (soft) — RF + GB + LR predict probabilities, vote on average
        - VotingClassifier (hard) — RF + GB + SVM + LR vote on majority class
        - BaggingClassifier — 30 Decision Trees trained on random subsets
        - AdaBoostClassifier — sequential boosting, focuses on hard examples
        - StackingClassifier — RF + GB + LR as base, LR as meta-learner

    Returns:
      best_config: winning model + preprocessing combination
      ensemble_comparison: how each ensemble method performed
      preprocessing_impact: which preprocessing decisions actually help
      research_insight: publication-ready summary paragraph
    """
    try:
        df = _load_df(file.file, file.filename)
        if df is None: return JSONResponse({"error": "Unsupported format."})
        if df.empty: return JSONResponse({"error": "Dataset is empty."})
        if df.shape[1] < 2: return JSONResponse({"error": "Need at least 2 columns."})
        if len(df) < 20: return JSONResponse({"error": "Need at least 20 rows for AutoML."})

        td = detect_target_column(df)
        target = target_column if (target_column and target_column in df.columns) else td["predicted_target"]
        task = _detect_task(df, target)
        max_configs = max(6, min(24, max_configs))

        result = run_automl_optimization(df, target=target, task=task, max_configs=max_configs)
        return JSONResponse(content=clean_for_json(result))
    except Exception as e:
        import traceback
        return JSONResponse({"error": str(e), "traceback": traceback.format_exc()[-800:]})


# ── ENDPOINT 9: SHAP EXPLAINABILITY ──────────────────────────────────────────
@app.post("/shap")
async def shap_explainability(
    file: UploadFile = File(...),
    target_column: str = Query(default=None),
    max_samples: int = Query(default=200, description="Max rows for SHAP (larger = slower)"),
    top_n_features: int = Query(default=10, description="Top N features to return")
):
    """
    SHAP (SHapley Additive exPlanations) Explainability Analysis.

    Explains WHY the model makes each prediction using game theory.

    Returns:
      global_feature_importance: mean |SHAP| per feature (true importance)
      feature_direction: does each feature push predictions up or down?
      top_feature_interactions: which features interact with each other?
      sample_explanations: for 3 sample rows, which features caused that prediction?
      rf_vs_shap_comparison: where RF importance and SHAP disagree (SHAP is more reliable)
      shap_summary: plain-English explanation of model behavior

    SHAP is grounded in cooperative game theory (Shapley values).
    It is the gold standard for ML explainability — used by Google,
    Microsoft, and all major ML teams in production.

    Cite: Lundberg & Lee (2017), NeurIPS.
    """
    try:
        df = _load_df(file.file, file.filename)
        if df is None: return JSONResponse({"error": "Unsupported format."})
        if df.empty: return JSONResponse({"error": "Dataset is empty."})
        if df.shape[1] < 2: return JSONResponse({"error": "Need at least 2 columns."})
        if len(df) < 20: return JSONResponse({"error": "Need at least 20 rows for SHAP."})

        td = detect_target_column(df)
        target = target_column if (target_column and target_column in df.columns) else td["predicted_target"]
        task = _detect_task(df, target)

        result = run_shap_analysis(
            df, target=target, task=task,
            max_samples_for_shap=max_samples,
            top_n_features=top_n_features
        )
        return JSONResponse(content=clean_for_json(result))
    except Exception as e:
        import traceback
        return JSONResponse({"error": str(e), "traceback": traceback.format_exc()[-800:]})