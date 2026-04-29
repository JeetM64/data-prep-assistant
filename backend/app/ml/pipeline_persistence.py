import os
import joblib
from typing import Any

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "saved_models")
os.makedirs(MODELS_DIR, exist_ok=True)

def save_pipeline(pipeline: Any, filename: str) -> str:
    """
    Saves a trained sklearn Pipeline (or any object) to disk using joblib.
    Returns the absolute path to the saved file.
    """
    if not filename.endswith(".joblib"):
        filename += ".joblib"
        
    filepath = os.path.join(MODELS_DIR, filename)
    joblib.dump(pipeline, filepath)
    return filepath

def load_pipeline(filename: str) -> Any:
    """
    Loads a trained sklearn Pipeline from disk using joblib.
    """
    if not filename.endswith(".joblib"):
        filename += ".joblib"
        
    filepath = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Saved pipeline not found: {filepath}")
        
    return joblib.load(filepath)
