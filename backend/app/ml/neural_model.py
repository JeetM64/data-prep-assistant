"""
backend/app/ml/neural_model.py
================================
Deep Learning module for AutoEDA v4.0.

Trains a Multi-Layer Perceptron (MLP) neural network on tabular data.
- Uses PyTorch if installed (full DL: BatchNorm, Dropout, Adam, early stopping)
- Falls back to sklearn MLPClassifier/MLPRegressor (no extra install needed)

Public API:
    train_neural_model(X, y, task_type, epochs, hidden_layers) -> dict
    quick_neural_score(X, y, task_type) -> float
"""

from __future__ import annotations

import warnings
import time
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _safe(val: float) -> float:
    """Convert to plain Python float, replacing NaN/Inf with 0."""
    try:
        v = float(val)
        return round(v, 6) if not (v != v or v == float("inf") or v == float("-inf")) else 0.0
    except Exception:
        return 0.0


def _parse_layers(hidden_layers) -> list[int]:
    """Accept '128,64,32' string or [128, 64, 32] list."""
    if isinstance(hidden_layers, str):
        return [int(x.strip()) for x in hidden_layers.split(",") if x.strip().isdigit()]
    return list(hidden_layers) if hidden_layers else [128, 64, 32]


# ─────────────────────────────────────────────────────────────────────────────
# PYTORCH BACKEND
# ─────────────────────────────────────────────────────────────────────────────

def _train_pytorch(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    layer_sizes: list[int],
    n_features: int, n_classes: int,
    is_clf: bool, epochs: int,
    primary_metric: str,
) -> dict:
    """Train MLP with PyTorch. Returns metrics + history."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.metrics import f1_score, r2_score, accuracy_score, mean_squared_error

    # Tensors
    X_tr_t = torch.FloatTensor(X_tr)
    X_te_t = torch.FloatTensor(X_te)

    if is_clf:
        y_tr_t = torch.LongTensor(y_tr.astype(int))
        y_te_t = torch.LongTensor(y_te.astype(int))
    else:
        y_tr_t = torch.FloatTensor(y_tr.astype(float)).unsqueeze(1)
        y_te_t = torch.FloatTensor(y_te.astype(float)).unsqueeze(1)

    # Build MLP: Input → [Dense+BN+ReLU+Drop] × N → Output
    layers = []
    in_dim = n_features
    for h in layer_sizes:
        layers += [
            nn.Linear(in_dim, h),
            nn.BatchNorm1d(h),
            nn.ReLU(),
            nn.Dropout(0.3),
        ]
        in_dim = h

    out_dim = n_classes if is_clf else 1
    layers.append(nn.Linear(in_dim, out_dim))
    model = nn.Sequential(*layers)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.CrossEntropyLoss() if is_clf else nn.MSELoss()

    batch  = min(64, max(8, len(X_tr) // 4))
    ds     = TensorDataset(X_tr_t, y_tr_t)
    loader = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=False)

    history      = []
    best_val     = float("inf")
    patience_cnt = 0
    patience     = 12

    for epoch in range(epochs):
        # ── Train ──────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            out  = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        avg_train_loss = epoch_loss / max(len(loader), 1)

        # ── Validate ───────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_out  = model(X_te_t)
            val_loss = _safe(criterion(val_out, y_te_t).item())

        scheduler.step(val_loss)

        # Metric on validation
        if is_clf:
            preds     = val_out.argmax(dim=1).numpy()
            metric_v  = _safe(f1_score(y_te, preds, average="weighted", zero_division=0))
        else:
            preds     = val_out.squeeze().numpy()
            metric_v  = _safe(r2_score(y_te, preds))

        # Log every ~10% of epochs
        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch < 3:
            history.append({
                "epoch":      epoch + 1,
                "train_loss": _safe(avg_train_loss),
                "val_loss":   val_loss,
                "metric":     metric_v,
            })

        # Early stopping
        if val_loss < best_val - 1e-5:
            best_val     = val_loss
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    # ── Final evaluation ────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        final_out = model(X_te_t)
        if is_clf:
            y_pred = final_out.argmax(dim=1).numpy().astype(int)
            try:
                proba = torch.softmax(final_out, dim=1).numpy()
            except Exception:
                proba = None
        else:
            y_pred = final_out.squeeze().numpy()
            proba  = None

    if is_clf:
        metrics = {
            "accuracy":    _safe(accuracy_score(y_te, y_pred)),
            "f1_weighted": _safe(f1_score(y_te, y_pred, average="weighted", zero_division=0)),
            "f1_macro":    _safe(f1_score(y_te, y_pred, average="macro",    zero_division=0)),
        }
        if proba is not None and n_classes == 2:
            try:
                from sklearn.metrics import roc_auc_score
                metrics["roc_auc"] = _safe(roc_auc_score(y_te, proba[:, 1]))
            except Exception:
                pass
    else:
        metrics = {
            "r2_score": _safe(r2_score(y_te, y_pred)),
            "rmse":     _safe(float(np.sqrt(mean_squared_error(y_te, y_pred)))),
            "mae":      _safe(float(np.mean(np.abs(y_te - y_pred)))),
        }

    return {
        "backend":          "PyTorch",
        "metrics":          metrics,
        "training_history": history,
        "epochs_trained":   len(history) * max(1, epochs // 10),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SKLEARN MLP BACKEND (fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _train_sklearn_mlp(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    layer_sizes: list[int],
    is_clf: bool, epochs: int,
) -> dict:
    """Train sklearn MLPClassifier / MLPRegressor. No extra installs needed."""
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.metrics import f1_score, r2_score, accuracy_score, mean_squared_error

    mlp_args = dict(
        hidden_layer_sizes = tuple(layer_sizes),
        activation         = "relu",
        solver             = "adam",
        alpha              = 1e-4,
        learning_rate_init = 1e-3,
        max_iter           = epochs,
        early_stopping     = True,
        validation_fraction= 0.15,
        n_iter_no_change   = 12,
        random_state       = 42,
        verbose            = False,
    )

    mlp = MLPClassifier(**mlp_args) if is_clf else MLPRegressor(**mlp_args)
    mlp.fit(X_tr, y_tr)
    y_pred = mlp.predict(X_te)

    # Build history from sklearn's loss_curve_
    history = []
    if hasattr(mlp, "loss_curve_"):
        lc    = mlp.loss_curve_
        vs    = getattr(mlp, "validation_scores_", [None]*len(lc))
        step  = max(1, len(lc) // 10)
        for i in range(0, len(lc), step):
            history.append({
                "epoch":      i + 1,
                "train_loss": _safe(lc[i]),
                "val_loss":   _safe(vs[i]) if vs[i] is not None else None,
                "metric":     _safe(vs[i]) if vs[i] is not None else None,
            })

    if is_clf:
        y_pred_int = y_pred.astype(int)
        metrics = {
            "accuracy":    _safe(accuracy_score(y_te, y_pred_int)),
            "f1_weighted": _safe(f1_score(y_te, y_pred_int, average="weighted", zero_division=0)),
            "f1_macro":    _safe(f1_score(y_te, y_pred_int, average="macro",    zero_division=0)),
        }
        if hasattr(mlp, "predict_proba"):
            try:
                from sklearn.metrics import roc_auc_score
                proba = mlp.predict_proba(X_te)
                if proba.shape[1] == 2:
                    metrics["roc_auc"] = _safe(roc_auc_score(y_te, proba[:, 1]))
            except Exception:
                pass
    else:
        metrics = {
            "r2_score": _safe(r2_score(y_te, y_pred)),
            "rmse":     _safe(float(np.sqrt(mean_squared_error(y_te, y_pred)))),
            "mae":      _safe(float(np.mean(np.abs(y_te - y_pred)))),
        }

    return {
        "backend":          "sklearn MLPClassifier/Regressor (pip install torch for full PyTorch DL)",
        "metrics":          metrics,
        "training_history": history,
        "epochs_trained":   getattr(mlp, "n_iter_", epochs),
    }


# ─────────────────────────────────────────────────────────────────────────────
# BASELINE COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def _baseline_comparison(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    is_clf: bool,
) -> dict[str, float]:
    """Train Logistic Regression + Random Forest for comparison."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import f1_score, r2_score

    baselines = {}
    model_pairs = (
        [
            ("Logistic Regression", LogisticRegression(max_iter=500, random_state=42)),
            ("Random Forest",       RandomForestClassifier(n_estimators=80, max_depth=8,
                                                           random_state=42, n_jobs=-1)),
        ]
        if is_clf else
        [
            ("Ridge Regression", Ridge(alpha=1.0)),
            ("Random Forest",    RandomForestRegressor(n_estimators=80, max_depth=8,
                                                       random_state=42, n_jobs=-1)),
        ]
    )
    for name, model in model_pairs:
        try:
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            if is_clf:
                score = _safe(f1_score(y_te, y_pred, average="weighted", zero_division=0))
            else:
                score = _safe(r2_score(y_te, y_pred))
            baselines[name] = score
        except Exception:
            pass
    return baselines


# ─────────────────────────────────────────────────────────────────────────────
# PERMUTATION FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

def _permutation_importance(
    X_te: np.ndarray,
    y_te: np.ndarray,
    feature_names: list[str],
    predict_fn,          # callable: X_arr -> y_pred_arr
    base_score: float,
    is_clf: bool,
    top_n: int = 10,
) -> dict[str, float]:
    """Measure feature importance by random permutation of each column."""
    from sklearn.metrics import f1_score, r2_score

    results = {}
    for i, col in enumerate(feature_names[:min(len(feature_names), 20)]):
        X_perm = X_te.copy()
        np.random.shuffle(X_perm[:, i])
        try:
            y_perm = predict_fn(X_perm)
            if is_clf:
                perm_score = _safe(f1_score(y_te, y_perm.astype(int),
                                            average="weighted", zero_division=0))
            else:
                perm_score = _safe(r2_score(y_te, y_perm))
            results[col] = _safe(base_score - perm_score)
        except Exception:
            results[col] = 0.0

    # Sort descending, return top_n
    return dict(sorted(results.items(), key=lambda x: -x[1])[:top_n])


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API — train_neural_model
# ─────────────────────────────────────────────────────────────────────────────

def train_neural_model(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str  = "classification",
    epochs: int     = 50,
    hidden_layers   = "128,64,32",
) -> dict:
    """
    Train a Multi-Layer Perceptron on tabular data.

    Parameters
    ----------
    X             : feature DataFrame (fully numeric, no NaN)
    y             : target Series (encoded integers for classification)
    task_type     : "classification" or "regression" (or subtype)
    epochs        : max training epochs (10-200)
    hidden_layers : layer sizes as string "128,64,32" or list [128,64,32]

    Returns
    -------
    dict with keys:
        status, backend, task_type, primary_metric, nn_score, nn_metrics,
        architecture, training_history, baseline_comparison,
        nn_vs_best_baseline, feature_importance, research_note
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    is_clf      = "classification" in str(task_type)
    layer_sizes = _parse_layers(hidden_layers)
    n_features  = X.shape[1]
    n_classes   = int(y.nunique()) if is_clf else 1

    # ── Train / test split ────────────────────────────────────────────────
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X.values, y.values,
            test_size=0.2, random_state=42,
            stratify=y.values if is_clf else None,
        )
    except Exception:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X.values, y.values, test_size=0.2, random_state=42
        )

    # ── Scale features ────────────────────────────────────────────────────
    sc   = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)

    primary_metric = "F1-Weighted" if is_clf else "R² Score"

    # ── Train neural network ──────────────────────────────────────────────
    t0 = time.time()
    try:
        result = _train_pytorch(
            X_tr, y_tr, X_te, y_te,
            layer_sizes, n_features, n_classes,
            is_clf, epochs, primary_metric,
        )
    except ImportError:
        result = _train_sklearn_mlp(
            X_tr, y_tr, X_te, y_te,
            layer_sizes, is_clf, epochs,
        )
    elapsed = round(time.time() - t0, 2)

    backend  = result["backend"]
    metrics  = result["metrics"]
    history  = result["training_history"]
    n_epochs = result["epochs_trained"]

    nn_score = metrics.get("f1_weighted") or metrics.get("r2_score") or 0.0

    # ── Baseline comparison ───────────────────────────────────────────────
    baselines = _baseline_comparison(X_tr, y_tr, X_te, y_te, is_clf)
    best_bl_name  = max(baselines, key=baselines.get) if baselines else "N/A"
    best_bl_score = baselines.get(best_bl_name, 0.0)
    nn_wins       = bool(nn_score > best_bl_score)
    score_diff    = _safe(nn_score - best_bl_score)

    comparison = {**baselines, "Neural Network (MLP)": _safe(nn_score)}

    # ── Architecture description ──────────────────────────────────────────
    architecture = {
        "type":           "Multi-Layer Perceptron (MLP)",
        "backend":        backend,
        "input_dim":      n_features,
        "hidden_layers":  layer_sizes,
        "output_dim":     n_classes if is_clf else 1,
        "dropout":        0.3,
        "batch_norm":     "PyTorch" in backend,
        "activation":     "ReLU",
        "optimizer":      "Adam (lr=1e-3, weight_decay=1e-4)",
        "loss_fn":        "CrossEntropyLoss" if is_clf else "MSELoss",
        "epochs_trained": n_epochs,
        "early_stopping": True,
        "training_time_s": elapsed,
        "layer_string":   f"{n_features} → {' → '.join(str(x) for x in layer_sizes)} → {n_classes if is_clf else 1}",
    }

    verdict = (
        f"Neural Network OUTPERFORMS best sklearn model ({best_bl_name}) "
        f"by {abs(score_diff):.3f} on {primary_metric}."
        if nn_wins else
        f"Neural Network ({_safe(nn_score):.3f}) vs {best_bl_name} ({_safe(best_bl_score):.3f}). "
        f"Sklearn wins by {abs(score_diff):.3f}. "
        f"Try more epochs or add more training data."
    )

    research_note = (
        f"MLP trained for {n_epochs} epochs in {elapsed}s. "
        f"Architecture: {architecture['layer_string']}. "
        f"Regularization: Dropout(0.3)"
        + (", BatchNorm" if architecture["batch_norm"] else "") +
        f", L2(1e-4). "
        f"Backend: {backend}. "
        f"For tabular data, tree ensembles often match DL — combining both "
        f"(stacking/blending) typically yields the best results."
    )

    return {
        "status":         "completed",
        "backend":        backend,
        "task_type":      str(task_type),
        "n_samples":      len(X),
        "n_features":     n_features,
        "n_classes":      n_classes if is_clf else None,
        "primary_metric": primary_metric,
        "nn_score":       _safe(nn_score),
        "nn_metrics":     {k: _safe(v) for k, v in metrics.items()},
        "architecture":   architecture,
        "training_history": history,
        "baseline_comparison": {k: _safe(v) for k, v in comparison.items()},
        "nn_vs_best_baseline": {
            "nn_score":            _safe(nn_score),
            "best_baseline":       best_bl_name,
            "best_baseline_score": _safe(best_bl_score),
            "nn_wins":             nn_wins,
            "score_difference":    score_diff,
            "verdict":             verdict,
        },
        "research_note": research_note,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API — quick_neural_score
# ─────────────────────────────────────────────────────────────────────────────

def quick_neural_score(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str = "classification",
    epochs: int    = 30,
) -> float:
    """
    Fast single-number neural network score.
    Used internally for comparisons. Returns F1-weighted or R².
    """
    try:
        result = train_neural_model(X, y, task_type=task_type, epochs=epochs,
                                    hidden_layers="64,32")
        return result["nn_score"]
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer, load_diabetes
    import pandas as pd

    print("=== Classification (breast cancer) ===")
    bc   = load_breast_cancer(as_frame=True)
    X_bc = bc.data
    y_bc = bc.target
    res  = train_neural_model(X_bc, y_bc, task_type="classification", epochs=30)
    print(f"Backend  : {res['backend']}")
    print(f"NN Score : {res['nn_score']:.4f}  ({res['primary_metric']})")
    print(f"Metrics  : {res['nn_metrics']}")
    print(f"Verdict  : {res['nn_vs_best_baseline']['verdict']}")
    print()

    print("=== Regression (diabetes) ===")
    diab = load_diabetes(as_frame=True)
    X_d  = diab.data
    y_d  = diab.target
    res2 = train_neural_model(X_d, y_d, task_type="regression", epochs=30)
    print(f"Backend  : {res2['backend']}")
    print(f"NN Score : {res2['nn_score']:.4f}  ({res2['primary_metric']})")
    print(f"Metrics  : {res2['nn_metrics']}")
    print(f"Verdict  : {res2['nn_vs_best_baseline']['verdict']}")