import json
import os
from datetime import datetime
from typing import Dict, Any

TRACKING_FILE = os.path.join(os.path.dirname(__file__), "experiments.json")

def log_experiment(dataset_name: str, task_type: str, features: list, best_model: str, best_score: float, metrics: dict, params: dict = None) -> str:
    """
    Logs an ML experiment in a lightweight MLflow-like JSON format.
    Returns the generated run_id.
    """
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    experiment_data = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "dataset_name": dataset_name,
        "task_type": task_type,
        "features": {
            "count": len(features),
            "list": features
        },
        "model": {
            "best_model_name": best_model,
            "best_score": best_score,
            "hyperparameters": params or {}
        },
        "metrics": metrics
    }
    
    # Load existing tracking data
    history = []
    if os.path.exists(TRACKING_FILE):
        try:
            with open(TRACKING_FILE, "r") as f:
                history = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
            
    history.append(experiment_data)
    
    # Save back
    with open(TRACKING_FILE, "w") as f:
        json.dump(history, f, indent=4)
        
    return run_id

def get_experiment_history() -> list:
    """Returns the list of all tracked experiments."""
    if not os.path.exists(TRACKING_FILE):
        return []
    try:
        with open(TRACKING_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []

def compare_experiments(run_id_1: str, run_id_2: str) -> dict:
    """Compare two runs by run_id."""
    history = get_experiment_history()
    run1 = next((r for r in history if r["run_id"] == run_id_1), None)
    run2 = next((r for r in history if r["run_id"] == run_id_2), None)
    
    if not run1 or not run2:
        return {"error": "One or both run IDs not found."}
        
    return {
        "run_1": run1,
        "run_2": run2,
        "score_diff": run2["model"]["best_score"] - run1["model"]["best_score"]
    }
