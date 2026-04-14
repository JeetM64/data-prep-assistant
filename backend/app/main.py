"""
main.py — ML Data Readiness Analyzer v5.0.0
Production-ready: Auth, Reports History, PDF, Charts data, all edge cases handled.
"""

from fastapi import FastAPI, UploadFile, File, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse
from typing import List, Optional
from datetime import datetime
from bson import ObjectId

import pandas as pd
import numpy as np
import io
import logging
import traceback

# ── ML Modules ───────────────────────────────────────────────────────────────
from app.ml.feature_analysis import analyze_features
from app.ml.correlation_analysis import correlation_analysis
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

# ── DB & Auth ─────────────────────────────────────────────────────────────────
from app.db import reports_collection, users_collection
from app.auth import (
    RegisterRequest, LoginRequest, TokenResponse,
    register_user, login_user,
    get_current_user, get_optional_user
)
from app.utils import prepare_ml_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── JSON Cleaner ──────────────────────────────────────────────────────────────
def clean_for_json(obj):
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else obj
    elif isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    return obj


# ── App Init ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ML Data Readiness Analyzer",
    description="Production ML pipeline: Auth, Analysis, AutoML, SHAP, Reports.",
    version="5.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Internal Helpers ──────────────────────────────────────────────────────────
def _load_df(file_obj, filename: str) -> Optional[pd.DataFrame]:
    fname = filename.lower()
    try:
        if fname.endswith(".csv"):
            return pd.read_csv(file_obj)
        elif fname.endswith((".xlsx", ".xls")):
            return pd.read_excel(file_obj)
    except Exception as e:
        logger.error(f"Failed to load file {filename}: {e}")
    return None


def _detect_task(df: pd.DataFrame, target: str) -> str:
    y = df[target].dropna()
    return "classification" if (y.nunique() <= 15 or y.dtype == object) else "regression"


def _validate_df(df, filename: str):
    """Returns (df, error_response). error_response is None if valid."""
    if df is None:
        return None, JSONResponse({"error": f"Unsupported file format: {filename}. Use CSV or Excel."}, status_code=400)
    if df.empty:
        return None, JSONResponse({"error": "Dataset is empty."}, status_code=400)
    if df.shape[1] < 2:
        return None, JSONResponse({"error": "Need at least 2 columns."}, status_code=400)
    return df, None


def _run_full_analysis(df: pd.DataFrame, target_override: Optional[str], filename: str):
    """
    Core analysis pipeline. Returns (result_dict, target_column).
    All target column edge cases handled here.
    """
    rows, cols = df.shape
    td = detect_target_column(df)

    # ── Target Resolution (fixed edge cases) ─────────────────────────────────
    if target_override:
        # User specified a target
        if target_override in df.columns:
            target = target_override
            target_source = "user_specified"
        else:
            # Fuzzy match: case-insensitive
            col_map = {c.lower(): c for c in df.columns}
            fuzzy = col_map.get(target_override.lower())
            if fuzzy:
                target = fuzzy
                target_source = "user_specified_fuzzy_match"
            else:
                return {
                    "error": f"Column '{target_override}' not found.",
                    "available_columns": df.columns.tolist()
                }, None
    else:
        target = td["predicted_target"]
        target_source = "auto_detected"

    logger.info(f"Target resolved: '{target}' ({target_source})")

    # ── Dataset Summary ───────────────────────────────────────────────────────
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

    # ── Prepare clean ML dataframe (target must survive) ─────────────────────
    qs = dataset_quality_score(df)
    pipeline = build_preprocessing_pipeline(df, target_column=target)
    drop_cols = [
        e["column"] for e in pipeline.get("drop_columns", [])
        if e["column"] != target  # NEVER drop target
    ]
    clean_df = prepare_ml_dataset(df.drop(columns=drop_cols, errors="ignore"))

    # If target got dropped during prepare_ml_dataset, restore from original
    if target not in clean_df.columns:
        logger.warning(f"Target '{target}' lost during cleaning — restoring from original.")
        clean_df[target] = df[target].values

    # Final check
    if target not in clean_df.columns:
        logger.error(f"Target '{target}' could not be restored. Using full original df.")
        clean_df = prepare_ml_dataset(df.copy())
        if target not in clean_df.columns:
            clean_df[target] = df[target].values

    # ── Run All Modules ───────────────────────────────────────────────────────
    resp = {
        "dataset_summary":          summary,
        "target_detection":         {**td, "final_target": target, "target_source": target_source},
        "feature_analysis":         analyze_features(df),
        "correlation_analysis":     correlation_analysis(df),
        "preprocessing_advice":     preprocessing_suggestions(df),
        "data_leakage_analysis":    detect_data_leakage(df),
        "model_recommendation":     recommend_model(df, target_column=target),
        "dataset_quality_score":    qs,
        "recommended_pipeline":     pipeline,
        "fair_assessment":          fair_assessment(df, filename=filename),
        "anomaly_detection":        detect_anomalies(df),
        "auto_training_results":    auto_train(clean_df, target_column=target),
        "cross_validation":         cross_validation_stability(clean_df, target_column=target),
        "overfitting_analysis":     detect_overfitting(clean_df, target_column=target),
        "feature_importance":       feature_importance_analysis(clean_df, target_column=target),
        "cluster_intelligence":     cluster_intelligence(clean_df),
        "smart_feature_selection":  smart_feature_selection(clean_df, target_column=target),
    }
    resp["explainability_report"] = generate_explainability_report(resp)
    return resp, target


# ═══════════════════════════════════════════════════════════════════════════════
# AUTH ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/auth/register", response_model=TokenResponse, tags=["Auth"])
def register(req: RegisterRequest):
    """Register a new user. Returns JWT token."""
    return register_user(req)


@app.post("/auth/login", response_model=TokenResponse, tags=["Auth"])
def login(req: LoginRequest):
    """Login. Returns JWT token."""
    return login_user(req)


@app.get("/auth/me", tags=["Auth"])
def me(user: dict = Depends(get_current_user)):
    """Get current user info from token."""
    return {"user": user}


# ═══════════════════════════════════════════════════════════════════════════════
# REPORTS ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/reports", tags=["Reports"])
def get_reports(
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=20, ge=1, le=100),
    user: Optional[dict] = Depends(get_optional_user)
):
    """
    Get paginated list of analysis reports.
    - If authenticated: returns only your reports.
    - If not authenticated: returns all public reports (no result payload, just metadata).
    """
    try:
        skip = (page - 1) * limit
        query = {}
        if user:
            query["user_id"] = user["_id"]

        # Metadata only — no full result payload (too large)
        projection = {
            "_id": 1,
            "filename": 1,
            "created_at": 1,
            "user_id": 1,
            "summary": 1,
        }

        cursor = reports_collection.find(query, projection) \
            .sort("created_at", -1) \
            .skip(skip) \
            .limit(limit)

        reports = []
        for r in cursor:
            reports.append({
                "id":          str(r["_id"]),
                "filename":    r.get("filename", "unknown"),
                "created_at":  r["created_at"].isoformat() if isinstance(r.get("created_at"), datetime) else str(r.get("created_at", "")),
                "summary":     r.get("summary", {}),
            })

        total = reports_collection.count_documents(query)

        return {
            "reports": reports,
            "pagination": {
                "page":        page,
                "limit":       limit,
                "total":       total,
                "total_pages": max(1, -(-total // limit))  # ceiling division
            }
        }
    except Exception as e:
        logger.error(f"GET /reports error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/reports/{report_id}", tags=["Reports"])
def get_report_by_id(
    report_id: str,
    user: Optional[dict] = Depends(get_optional_user)
):
    """Get full report by ID."""
    try:
        oid = ObjectId(report_id)
    except Exception:
        return JSONResponse({"error": "Invalid report ID format."}, status_code=400)

    query = {"_id": oid}
    if user:
        query["user_id"] = user["_id"]

    report = reports_collection.find_one(query, {"_id": 0})
    if not report:
        return JSONResponse({"error": "Report not found."}, status_code=404)

    return JSONResponse(content=clean_for_json(report))


@app.delete("/reports/{report_id}", tags=["Reports"])
def delete_report(
    report_id: str,
    user: dict = Depends(get_current_user)
):
    """Delete a report (must be owner)."""
    try:
        oid = ObjectId(report_id)
    except Exception:
        return JSONResponse({"error": "Invalid report ID."}, status_code=400)

    result = reports_collection.delete_one({"_id": oid, "user_id": user["_id"]})
    if result.deleted_count == 0:
        return JSONResponse({"error": "Report not found or not authorized."}, status_code=404)
    return {"message": "Report deleted."}


# ═══════════════════════════════════════════════════════════════════════════════
# CORE ML ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Meta"])
def home():
    return {
        "name": "ML Data Readiness Analyzer",
        "version": "5.0.0",
        "endpoints": {
            "POST /auth/register": "Register user",
            "POST /auth/login":    "Login user",
            "GET  /auth/me":       "Get current user",
            "POST /upload":        "Full 18-module analysis",
            "GET  /reports":       "List your reports",
            "GET  /reports/{id}":  "Get full report",
            "POST /execute":       "Download cleaned CSV",
            "POST /benchmark":     "Compare datasets",
            "POST /report":        "Download PDF report",
            "POST /compare":       "Drift detection",
            "POST /whatif":        "What-if simulator",
            "POST /nlreport":      "Natural language report",
            "POST /automl":        "AutoML optimizer",
            "POST /shap":          "SHAP explainability",
            "GET  /health":        "Health check",
        }
    }


@app.get("/health", tags=["Meta"])
def health():
    try:
        reports_collection.find_one({}, {"_id": 1})
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"
    return {"status": "ok", "version": "5.0.0", "database": db_status}


# ── ENDPOINT 1: FULL ANALYSIS ─────────────────────────────────────────────────
@app.post("/upload", tags=["Analysis"])
async def upload_dataset(
    file: UploadFile = File(...),
    target_column: str = Query(default=None),
    user: Optional[dict] = Depends(get_optional_user)
):
    """Full 18-module analysis. Saves report to DB. Returns complete JSON."""
    try:
        df = _load_df(file.file, file.filename)
        df, err = _validate_df(df, file.filename)
        if err: return err

        result, target = _run_full_analysis(df, target_column, file.filename)

        if isinstance(result, dict) and "error" in result:
            return JSONResponse(result, status_code=400)

        # ── Build summary snapshot for report list ───────────────────────────
        qs = result.get("dataset_quality_score", {})
        er = result.get("explainability_report", {})
        at = result.get("auto_training_results", {})
        ds = result.get("dataset_summary", {})

        summary = {
            "rows":              ds.get("rows"),
            "columns":           ds.get("columns"),
            "target":            target,
            "quality_score":     qs.get("overall_score"),
            "quality_status":    qs.get("status"),
            "readiness_score":   er.get("readiness_score"),
            "readiness_grade":   er.get("grade"),
            "best_model":        at.get("best_model"),
            "best_score":        at.get("best_score"),
            "metric":            at.get("primary_metric"),
            "task_type":         at.get("task_type"),
        }

        # ── Save to MongoDB ──────────────────────────────────────────────────
        doc = {
            "filename":   file.filename,
            "result":     clean_for_json(result),
            "summary":    summary,
            "created_at": datetime.utcnow(),
        }
        if user:
            doc["user_id"] = user["_id"]

        try:
            inserted = reports_collection.insert_one(doc)
            result["_report_id"] = str(inserted.inserted_id)
            # Update user's report count
            if user:
                users_collection.update_one(
                    {"_id": ObjectId(user["_id"])},
                    {"$inc": {"reports_count": 1}}
                )
        except Exception as db_err:
            logger.warning(f"MongoDB save failed (non-fatal): {db_err}")

        return JSONResponse(content=clean_for_json(result))

    except Exception as e:
        logger.error(f"POST /upload error: {traceback.format_exc()}")
        return JSONResponse({"error": str(e), "traceback": traceback.format_exc()[-1000:]}, status_code=500)


# ── ENDPOINT 2: EXECUTE PIPELINE ─────────────────────────────────────────────
@app.post("/execute", tags=["Analysis"])
async def execute_dataset_pipeline(
    file: UploadFile = File(...),
    target_column: str = Query(default=None),
    download: bool = Query(default=True)
):
    """Apply full preprocessing pipeline. Download cleaned ML-ready CSV."""
    try:
        df = _load_df(file.file, file.filename)
        df, err = _validate_df(df, file.filename)
        if err: return err

        td = detect_target_column(df)
        target = target_column if (target_column and target_column in df.columns) else td["predicted_target"]
        pipeline = build_preprocessing_pipeline(df, target_column=target)
        result = execute_pipeline(df, pipeline)

        if not result["success"]:
            return JSONResponse({"error": "Pipeline execution failed.", "details": result}, status_code=500)

        cleaned_df = result["cleaned_dataframe"]
        stats = result["stats"]

        if download:
            out = io.StringIO()
            cleaned_df.to_csv(out, index=False)
            out.seek(0)
            base = file.filename.rsplit(".", 1)[0]
            return StreamingResponse(
                io.BytesIO(out.getvalue().encode()),
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename={base}_cleaned.csv",
                    "X-Original-Shape": f"{stats['original_shape']['rows']}x{stats['original_shape']['columns']}",
                    "X-Final-Shape":    f"{stats['final_shape']['rows']}x{stats['final_shape']['columns']}",
                    "X-Steps-Applied":  str(stats["steps_applied"])
                }
            )

        return JSONResponse(content=clean_for_json({
            "success": True, "target": target, "stats": stats,
            "execution_log": result["execution_log"],
            "preview": cleaned_df.head(5).to_dict(orient="records"),
            "columns": cleaned_df.columns.tolist()
        }))
    except Exception as e:
        logger.error(f"POST /execute error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# ── ENDPOINT 3: BENCHMARK ─────────────────────────────────────────────────────
@app.post("/benchmark", tags=["Analysis"])
async def benchmark_datasets(files: List[UploadFile] = File(...)):
    """Compare 2-20 datasets. Returns ranked research table."""
    if len(files) < 2:  return JSONResponse({"error": "Upload at least 2 files."}, status_code=400)
    if len(files) > 20: return JSONResponse({"error": "Max 20 datasets."}, status_code=400)

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
@app.post("/report", tags=["Analysis"])
async def generate_report(
    file: UploadFile = File(...),
    target_column: str = Query(default=None)
):
    """Full analysis + professional PDF download."""
    try:
        df = _load_df(file.file, file.filename)
        df, err = _validate_df(df, file.filename)
        if err: return err

        analysis, _ = _run_full_analysis(df, target_column, file.filename)
        if isinstance(analysis, dict) and "error" in analysis:
            return JSONResponse(analysis, status_code=400)

        base = file.filename.rsplit(".", 1)[0]
        pdf_bytes = generate_pdf_report(analysis, filename=base)
        er = analysis.get("explainability_report", {})

        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={base}_readiness_report.pdf",
                "X-Readiness-Score":   str(er.get("readiness_score", 0)),
                "X-Grade":             er.get("grade", "?")
            }
        )
    except Exception as e:
        logger.error(f"POST /report error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# ── ENDPOINT 5: COMPARE (DRIFT) ───────────────────────────────────────────────
@app.post("/compare", tags=["Analysis"])
async def compare_train_test(
    train_file: UploadFile = File(...),
    test_file: UploadFile = File(...)
):
    """Distribution drift detection: KS test + PSI + Jensen-Shannon."""
    try:
        train_df = _load_df(train_file.file, train_file.filename)
        test_df  = _load_df(test_file.file,  test_file.filename)

        if train_df is None: return JSONResponse({"error": f"Unsupported: {train_file.filename}"}, status_code=400)
        if test_df  is None: return JSONResponse({"error": f"Unsupported: {test_file.filename}"},  status_code=400)
        if train_df.empty:   return JSONResponse({"error": "Train dataset is empty."},              status_code=400)
        if test_df.empty:    return JSONResponse({"error": "Test dataset is empty."},               status_code=400)

        common = set(train_df.columns) & set(test_df.columns)
        if not common:
            return JSONResponse({
                "error": "No common columns between files.",
                "train_columns": train_df.columns.tolist(),
                "test_columns":  test_df.columns.tolist()
            }, status_code=400)

        result = compare_datasets(train_df, test_df)
        result["file_info"] = {
            "train_file":    train_file.filename,
            "test_file":     test_file.filename,
            "train_shape":   {"rows": int(train_df.shape[0]), "columns": int(train_df.shape[1])},
            "test_shape":    {"rows": int(test_df.shape[0]),  "columns": int(test_df.shape[1])},
            "common_columns": len(common)
        }
        return JSONResponse(content=clean_for_json(result))
    except Exception as e:
        logger.error(f"POST /compare error: {traceback.format_exc()}")
        return JSONResponse({"error": str(e), "traceback": traceback.format_exc()[-800:]}, status_code=500)


# ── ENDPOINT 6: WHAT-IF ───────────────────────────────────────────────────────
@app.post("/whatif", tags=["Analysis"])
async def whatif_simulation(
    file: UploadFile = File(...),
    target_column: str = Query(default=None)
):
    """What-If Simulator: measure improvement from each data fix."""
    try:
        df = _load_df(file.file, file.filename)
        df, err = _validate_df(df, file.filename)
        if err: return err
        if len(df) < 20: return JSONResponse({"error": "Need at least 20 rows."}, status_code=400)

        td = detect_target_column(df)
        target = target_column if (target_column and target_column in df.columns) else td["predicted_target"]
        task = _detect_task(df, target)

        result = run_whatif_simulation(df, target, task, [])
        return JSONResponse(content=clean_for_json(result))
    except Exception as e:
        logger.error(f"POST /whatif error: {e}")
        return JSONResponse({"error": str(e), "traceback": traceback.format_exc()[-800:]}, status_code=500)


# ── ENDPOINT 7: NATURAL LANGUAGE REPORT ──────────────────────────────────────
@app.post("/nlreport", tags=["Analysis"])
async def nl_report(
    file: UploadFile = File(...),
    target_column: str = Query(default=None),
    format: str = Query(default="json", description="json or text")
):
    """Natural language explanation of every ML finding."""
    try:
        df = _load_df(file.file, file.filename)
        df, err = _validate_df(df, file.filename)
        if err: return err

        analysis, _ = _run_full_analysis(df, target_column, file.filename)
        if isinstance(analysis, dict) and "error" in analysis:
            return JSONResponse(analysis, status_code=400)

        base = file.filename.rsplit(".", 1)[0]
        nl_result = generate_nl_report(analysis, filename=base)

        if format == "text":
            return PlainTextResponse(nl_result.get("full_report", ""))
        return JSONResponse(content=clean_for_json(nl_result))
    except Exception as e:
        return JSONResponse({"error": str(e), "traceback": traceback.format_exc()[-800:]}, status_code=500)


# ── ENDPOINT 8: AUTOML ────────────────────────────────────────────────────────
@app.post("/automl", tags=["Analysis"])
async def automl_optimize(
    file: UploadFile = File(...),
    target_column: str = Query(default=None),
    max_configs: int = Query(default=12, ge=6, le=24)
):
    """AutoML: tests ensemble combos, returns best pipeline."""
    try:
        df = _load_df(file.file, file.filename)
        df, err = _validate_df(df, file.filename)
        if err: return err
        if len(df) < 20: return JSONResponse({"error": "Need at least 20 rows."}, status_code=400)

        td = detect_target_column(df)
        target = target_column if (target_column and target_column in df.columns) else td["predicted_target"]
        task = _detect_task(df, target)

        result = run_automl_optimization(df, target=target, task=task, max_configs=max_configs)
        return JSONResponse(content=clean_for_json(result))
    except Exception as e:
        return JSONResponse({"error": str(e), "traceback": traceback.format_exc()[-800:]}, status_code=500)


# ── ENDPOINT 9: SHAP ──────────────────────────────────────────────────────────
@app.post("/shap", tags=["Analysis"])
async def shap_explainability(
    file: UploadFile = File(...),
    target_column: str = Query(default=None),
    max_samples: int = Query(default=200, ge=50, le=1000),
    top_n_features: int = Query(default=10, ge=3, le=50)
):
    """SHAP explainability — why the model predicts what it predicts."""
    try:
        df = _load_df(file.file, file.filename)
        df, err = _validate_df(df, file.filename)
        if err: return err
        if len(df) < 20: return JSONResponse({"error": "Need at least 20 rows."}, status_code=400)

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
        return JSONResponse({"error": str(e), "traceback": traceback.format_exc()[-800:]}, status_code=500)


# ── LEGACY TEST ENDPOINTS (keep for backward compat) ─────────────────────────
@app.get("/test-db", tags=["Meta"])
def test_db():
    try:
        reports_collection.insert_one({"test": "working", "created_at": datetime.utcnow()})
        return {"status": "MongoDB connected ✅"}
    except Exception as e:
        return JSONResponse({"status": "MongoDB error ❌", "error": str(e)}, status_code=500)