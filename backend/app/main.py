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
    description=(
        "Automated dataset analysis, preprocessing pipeline generation, "
        "ML readiness scoring, FAIR assessment, anomaly detection, "
        "pipeline execution, and multi-dataset benchmarking."
    ),
    version="3.0.0"
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
        "message": "ML Data Readiness Analyzer v3.0 running",
        "total_modules": 19,
        "endpoints": {
            "POST /upload": "Full 18-module analysis — complete readiness report",
            "POST /upload?target_column=NAME": "Analyze with manually specified target",
            "POST /execute": "Apply pipeline — download cleaned ML-ready CSV",
            "POST /execute?download=false": "Execute pipeline — return JSON preview",
            "POST /benchmark": "Compare multiple datasets — research paper table",
            "GET /health": "API health check"
        }
    }


@app.get("/health")
def health():
    return {"status": "ok", "version": "3.0.0"}


@app.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    target_column: str = Query(default=None)
):
    """
    Full analysis — runs all 18 modules and returns complete readiness report.
    """
    try:
        filename = file.filename.lower()
        if filename.endswith(".csv"):
            df = pd.read_csv(file.file)
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file.file)
        else:
            return JSONResponse({"error": "Unsupported file format. Upload CSV or Excel."})

        if df.empty:
            return JSONResponse({"error": "Dataset is empty."})
        if df.shape[1] < 2:
            return JSONResponse({"error": "Need at least 2 columns."})

        rows, cols = df.shape

        target_detection_result = detect_target_column(df)
        auto_detected_target = target_detection_result["predicted_target"]

        if target_column and target_column in df.columns:
            target = target_column
            target_source = "user_specified"
        elif target_column and target_column not in df.columns:
            return JSONResponse({
                "error": f"Column '{target_column}' not found.",
                "available_columns": df.columns.tolist()
            })
        else:
            target = auto_detected_target
            target_source = "auto_detected"

        dataset_summary = {
            "rows": int(rows),
            "columns": int(cols),
            "missing_values": int(df.isnull().sum().sum()),
            "missing_value_percent": round(float(df.isnull().mean().mean() * 100), 2),
            "duplicate_rows": int(df.duplicated().sum()),
            "numeric_columns": df.select_dtypes(include="number").columns.tolist(),
            "categorical_columns": df.select_dtypes(exclude="number").columns.tolist(),
            "column_dtypes": df.dtypes.astype(str).to_dict(),
            "memory_usage_mb": round(float(df.memory_usage(deep=True).sum() / 1e6), 3)
        }

        quality_score = dataset_quality_score(df)
        pipeline = build_preprocessing_pipeline(df, target_column=target)

        cols_to_drop = [
            entry["column"]
            for entry in pipeline.get("drop_columns", [])
            if entry["column"] != target
        ]
        clean_df = prepare_ml_dataset(
            df.drop(columns=cols_to_drop, errors="ignore")
        )
        if target not in clean_df.columns:
            clean_df = prepare_ml_dataset(df.copy())

        response = {
            "dataset_summary": dataset_summary,
            "target_detection": {
                **target_detection_result,
                "final_target": target,
                "target_source": target_source,
                "override_hint": "POST /upload?target_column=<n> to override"
            },
            "feature_analysis":          analyze_features(df),
            "correlation_analysis":      correlation_analysis(df),
            "preprocessing_advice":      preprocessing_suggestions(df),
            "data_leakage_analysis":     detect_data_leakage(df),
            "model_recommendation":      recommend_model(df, target_column=target),
            "dataset_quality_score":     quality_score,
            "recommended_pipeline":      pipeline,
            "fair_assessment":           fair_assessment(df, filename=file.filename),
            "anomaly_detection":         detect_anomalies(df),
            "auto_training_results":     auto_train(clean_df, target_column=target),
            "cross_validation":          cross_validation_stability(clean_df, target_column=target),
            "overfitting_analysis":      detect_overfitting(clean_df, target_column=target),
            "feature_importance":        feature_importance_analysis(clean_df, target_column=target),
            "cluster_intelligence":      cluster_intelligence(clean_df),
            "smart_feature_selection":   smart_feature_selection(clean_df, target_column=target),
        }

        response["explainability_report"] = generate_explainability_report(response)

        return JSONResponse(content=clean_for_json(response))

    except Exception as e:
        return JSONResponse({
            "error": str(e),
            "hint": "Check your file has column headers in the first row."
        })


@app.post("/execute")
async def execute_dataset_pipeline(
    file: UploadFile = File(...),
    target_column: str = Query(default=None),
    download: bool = Query(default=True)
):
    """
    Pipeline execution — builds and applies preprocessing, returns cleaned CSV.
    """
    try:
        filename = file.filename.lower()
        if filename.endswith(".csv"):
            df = pd.read_csv(file.file)
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file.file)
        else:
            return JSONResponse({"error": "Unsupported file format."})

        if df.empty:
            return JSONResponse({"error": "Dataset is empty."})
        if df.shape[1] < 2:
            return JSONResponse({"error": "Need at least 2 columns."})

        target_detection_result = detect_target_column(df)

        if target_column and target_column in df.columns:
            target = target_column
        elif target_column and target_column not in df.columns:
            return JSONResponse({
                "error": f"Column '{target_column}' not found.",
                "available_columns": df.columns.tolist()
            })
        else:
            target = target_detection_result["predicted_target"]

        pipeline = build_preprocessing_pipeline(df, target_column=target)
        result = execute_pipeline(df, pipeline)

        if not result["success"]:
            return JSONResponse({"error": "Pipeline execution failed."})

        cleaned_df = result["cleaned_dataframe"]
        stats = result["stats"]
        execution_log = result["execution_log"]

        if download:
            output = io.StringIO()
            cleaned_df.to_csv(output, index=False)
            output.seek(0)
            original_name = file.filename.rsplit('.', 1)[0]
            download_filename = f"{original_name}_cleaned.csv"

            return StreamingResponse(
                io.BytesIO(output.getvalue().encode()),
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename={download_filename}",
                    "X-Original-Shape": f"{stats['original_shape']['rows']}x{stats['original_shape']['columns']}",
                    "X-Final-Shape": f"{stats['final_shape']['rows']}x{stats['final_shape']['columns']}",
                    "X-Missing-Reduced": str(stats["missing_reduction"]),
                    "X-Columns-Dropped": str(stats["columns_dropped"]),
                    "X-Steps-Applied": str(stats["steps_applied"]),
                }
            )
        else:
            return JSONResponse(content=clean_for_json({
                "success": True,
                "target_used": target,
                "stats": stats,
                "execution_log": execution_log,
                "preview": cleaned_df.head(5).to_dict(orient="records"),
                "column_names": cleaned_df.columns.tolist(),
                "hint": "Set ?download=true to get the cleaned CSV file"
            }))

    except Exception as e:
        return JSONResponse({
            "error": str(e),
            "hint": "Check your file has column headers in the first row."
        })


@app.post("/benchmark")
async def benchmark_datasets(
    files: List[UploadFile] = File(...)
):
    """
    Multi-dataset benchmark — upload 2-20 CSV/Excel files, get comparison table.

    Returns:
    - Readiness score per dataset (0-100, A-F grade)
    - Rank and percentile among all datasets
    - 6-dimension quality breakdown per dataset
    - Issue counts per dataset
    - Comparison table for research paper
    - Overall insights: average, grade distribution, weakest dimension
    """
    if len(files) < 2:
        return JSONResponse({
            "error": "Upload at least 2 files to compare.",
            "hint": "Select multiple files: Titanic.csv, Iris.csv, Housing.csv etc."
        })

    if len(files) > 20:
        return JSONResponse({"error": "Maximum 20 datasets per benchmark run."})

    datasets = []

    for f in files:
        fname = f.filename.lower()
        try:
            if fname.endswith(".csv"):
                df = pd.read_csv(f.file)
            elif fname.endswith((".xlsx", ".xls")):
                df = pd.read_excel(f.file)
            else:
                datasets.append({"name": f.filename, "dataframe": None})
                continue

            display_name = f.filename.rsplit('.', 1)[0]
            datasets.append({"name": display_name, "dataframe": df})

        except Exception as e:
            datasets.append({"name": f.filename, "dataframe": None, "error": str(e)})

    if not any(d.get("dataframe") is not None for d in datasets):
        return JSONResponse({"error": "No valid datasets could be loaded."})

    benchmark_result = run_benchmark(datasets)

    return JSONResponse(content=clean_for_json(benchmark_result))