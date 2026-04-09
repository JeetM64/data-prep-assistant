from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List
import pandas as pd
import numpy as np
import io

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
from app.ml.report_generator import generate_pdf_report

from app.utils import prepare_ml_dataset


def clean_for_json(obj):
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


app = FastAPI(
    title="ML Data Readiness Analyzer",
    version="3.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "ML Data Readiness Analyzer v3.1 running",
        "total_modules": 20,
        "endpoints": {
            "POST /upload":   "Full 18-module analysis — JSON report",
            "POST /execute":  "Apply pipeline — download cleaned CSV",
            "POST /benchmark":"Compare 2-20 datasets — research table",
            "POST /report":   "Full analysis + download PDF report",
            "GET /health":    "API health check"
        }
    }


@app.get("/health")
def health():
    return {"status": "ok", "version": "3.1.0", "modules": 20}


def _load_df(file_obj, filename: str):
    fname = filename.lower()
    if fname.endswith(".csv"):
        return pd.read_csv(file_obj)
    elif fname.endswith((".xlsx", ".xls")):
        return pd.read_excel(file_obj)
    return None


def _run_analysis(df, target_override, filename):
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
    return resp


@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...), target_column: str = Query(default=None)):
    try:
        df = _load_df(file.file, file.filename)
        if df is None: return JSONResponse({"error": "Unsupported format."})
        if df.empty: return JSONResponse({"error": "Dataset is empty."})
        if df.shape[1] < 2: return JSONResponse({"error": "Need at least 2 columns."})
        result = _run_analysis(df, target_column, file.filename)
        if "error" in result: return JSONResponse(result)
        return JSONResponse(content=clean_for_json(result))
    except Exception as e:
        return JSONResponse({"error": str(e)})


@app.post("/execute")
async def execute_pipeline_endpoint(
    file: UploadFile = File(...),
    target_column: str = Query(default=None),
    download: bool = Query(default=True)
):
    try:
        df = _load_df(file.file, file.filename)
        if df is None: return JSONResponse({"error": "Unsupported format."})
        if df.empty: return JSONResponse({"error": "Dataset is empty."})

        td = detect_target_column(df)
        target = target_column if (target_column and target_column in df.columns) else td["predicted_target"]

        pipeline = build_preprocessing_pipeline(df, target_column=target)
        result = execute_pipeline(df, pipeline)
        if not result["success"]: return JSONResponse({"error": "Pipeline failed."})

        cleaned = result["cleaned_dataframe"]
        stats = result["stats"]

        if download:
            out = io.StringIO()
            cleaned.to_csv(out, index=False)
            out.seek(0)
            base = file.filename.rsplit('.', 1)[0]
            return StreamingResponse(
                io.BytesIO(out.getvalue().encode()),
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename={base}_cleaned.csv",
                    "X-Original-Shape": f"{stats['original_shape']['rows']}x{stats['original_shape']['columns']}",
                    "X-Final-Shape": f"{stats['final_shape']['rows']}x{stats['final_shape']['columns']}",
                    "X-Steps-Applied": str(stats["steps_applied"]),
                }
            )
        return JSONResponse(content=clean_for_json({
            "success": True, "target": target, "stats": stats,
            "execution_log": result["execution_log"],
            "preview": cleaned.head(5).to_dict(orient="records"),
            "columns": cleaned.columns.tolist()
        }))
    except Exception as e:
        return JSONResponse({"error": str(e)})


@app.post("/benchmark")
async def benchmark(files: List[UploadFile] = File(...)):
    if len(files) < 2: return JSONResponse({"error": "Upload at least 2 files."})
    if len(files) > 20: return JSONResponse({"error": "Max 20 datasets."})
    datasets = []
    for f in files:
        try:
            df = _load_df(f.file, f.filename)
            datasets.append({"name": f.filename.rsplit('.', 1)[0], "dataframe": df})
        except Exception as e:
            datasets.append({"name": f.filename, "dataframe": None, "error": str(e)})
    result = run_benchmark(datasets)
    return JSONResponse(content=clean_for_json(result))


@app.post("/report")
async def generate_report(file: UploadFile = File(...), target_column: str = Query(default=None)):
    """
    Full analysis + professional multi-page PDF report download.
    Sections: Cover, Quality Scorecard, Issues, Action Plan,
    Features, Models, Pipeline, FAIR, Anomalies, Leakage, Overfitting.
    """
    try:
        df = _load_df(file.file, file.filename)
        if df is None: return JSONResponse({"error": "Unsupported format."})
        if df.empty: return JSONResponse({"error": "Dataset is empty."})
        if df.shape[1] < 2: return JSONResponse({"error": "Need at least 2 columns."})

        analysis = _run_analysis(df, target_column, file.filename)
        if "error" in analysis: return JSONResponse(analysis)

        base = file.filename.rsplit('.', 1)[0]
        pdf_bytes = generate_pdf_report(analysis, filename=base)
        er = analysis.get("explainability_report", {})

        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={base}_readiness_report.pdf",
                "X-Readiness-Score": str(er.get("readiness_score", 0)),
                "X-Grade": er.get("grade", "?"),
            }
        )
    except Exception as e:
        return JSONResponse({"error": str(e)})