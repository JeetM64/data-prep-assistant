import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from typing import Dict, Any

def extract_schema(df: pd.DataFrame, target: str = None) -> dict:
    """
    Extracts the training schema including distributions for drift detection.
    """
    schema = {
        "columns": list(df.columns),
        "target": target,
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "stats": {}
    }
    
    for col in df.columns:
        if col == target:
            continue
            
        s = df[col].dropna()
        if pd.api.types.is_numeric_dtype(s):
            schema["stats"][col] = {
                "type": "numeric",
                "mean": float(s.mean()) if not s.empty else 0,
                "std": float(s.std()) if not s.empty else 0,
                # Store sample for KS test approximation
                "sample": s.sample(min(1000, len(s)), random_state=42).tolist() if not s.empty else []
            }
        else:
            schema["stats"][col] = {
                "type": "categorical",
                "distribution": s.value_counts(normalize=True).to_dict() if not s.empty else {}
            }
            
    return schema

def _calculate_psi(expected: dict, actual: dict, buckets: int = 10) -> float:
    """
    Calculate Population Stability Index (PSI) for categorical variables.
    """
    psi_value = 0.0
    all_keys = set(expected.keys()).union(set(actual.keys()))
    
    for key in all_keys:
        e_val = expected.get(key, 0.0001)  # small epsilon
        a_val = actual.get(key, 0.0001)
        # normalize epsilon addition
        e_val = max(e_val, 0.0001)
        a_val = max(a_val, 0.0001)
        
        psi_value += (a_val - e_val) * np.log(a_val / e_val)
        
    return psi_value

def detect_drift(schema: dict, new_df: pd.DataFrame) -> dict:
    """
    Compares a new dataframe against the stored training schema to detect drift.
    Returns flags: LOW, MEDIUM, HIGH drift severity.
    """
    report = {
        "missing_columns": [],
        "type_mismatches": [],
        "drift_analysis": {},
        "overall_severity": "LOW"
    }
    
    expected_cols = [c for c in schema["columns"] if c != schema.get("target")]
    
    # 1. Schema Validation
    for col in expected_cols:
        if col not in new_df.columns:
            report["missing_columns"].append(col)
            report["overall_severity"] = "HIGH"
            continue
            
        expected_type = schema["dtypes"][col]
        actual_type = str(new_df[col].dtype)
        
        # Simple numeric vs categorical check
        exp_is_num = "int" in expected_type or "float" in expected_type
        act_is_num = "int" in actual_type or "float" in actual_type
        
        if exp_is_num != act_is_num:
            report["type_mismatches"].append({"column": col, "expected": expected_type, "actual": actual_type})
            report["overall_severity"] = "HIGH"
            
    # 2. Data Drift Detection
    high_drift_count = 0
    med_drift_count = 0
    
    for col in expected_cols:
        if col not in new_df.columns:
            continue
            
        s = new_df[col].dropna()
        if s.empty:
            continue
            
        stats = schema["stats"].get(col)
        if not stats: continue
        
        if stats["type"] == "numeric" and pd.api.types.is_numeric_dtype(s):
            # Kolmogorov-Smirnov Test
            sample_train = stats["sample"]
            if not sample_train: continue
            
            stat, p_value = ks_2samp(sample_train, s.tolist())
            
            # Interpretation
            if p_value < 0.01:
                severity = "HIGH"
                high_drift_count += 1
            elif p_value < 0.05:
                severity = "MEDIUM"
                med_drift_count += 1
            else:
                severity = "LOW"
                
            report["drift_analysis"][col] = {
                "method": "KS-Test",
                "statistic": round(stat, 4),
                "p_value": round(p_value, 4),
                "severity": severity
            }
            
        elif stats["type"] == "categorical":
            # PSI Test
            actual_dist = s.value_counts(normalize=True).to_dict()
            psi = _calculate_psi(stats["distribution"], actual_dist)
            
            if psi > 0.2:
                severity = "HIGH"
                high_drift_count += 1
            elif psi > 0.1:
                severity = "MEDIUM"
                med_drift_count += 1
            else:
                severity = "LOW"
                
            report["drift_analysis"][col] = {
                "method": "PSI",
                "psi_value": round(psi, 4),
                "severity": severity
            }
            
    if report["overall_severity"] != "HIGH":
        total_features = len(expected_cols)
        if high_drift_count > 0 or (med_drift_count / max(total_features, 1)) > 0.3:
            report["overall_severity"] = "MEDIUM"
            if high_drift_count / max(total_features, 1) > 0.2:
                report["overall_severity"] = "HIGH"
                
    return report
