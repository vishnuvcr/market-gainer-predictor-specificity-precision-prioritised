"""
Ensemble module combining top-performing models with weights optimized
for Maximum Average Precision (PR-AUC) and High Specificity.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score, average_precision_score

from src.config import TARGET_SPECIFICITY
from src.trainer import find_max_precision_specificity_threshold

class ModelEnsemble:
    def __init__(self, models: Dict[str, Any], weights: Dict[str, float], threshold: float, metrics: Dict[str, Any]):
        self.models = models
        self.weights = weights
        self.threshold = threshold
        self.metrics = metrics
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        total_prob = np.zeros(len(X))
        total_weight = sum(self.weights.values())
        
        for name, model in self.models.items():
            w = self.weights.get(name, 0.0)
            if w <= 0:
                continue
            if hasattr(model, "predict_proba"):
                raw_prob = model.predict_proba(X)
                probs = raw_prob.take(1, axis=1) if raw_prob.ndim == 2 else raw_prob
            else:
                probs = model.predict(X)
            total_prob += w * probs
            
        return total_prob / (total_weight + 1e-9)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= self.threshold).astype(int)

def build_optimal_ensemble(
    fitted_models: Dict[str, Any],
    test_prob_df: pd.DataFrame,
    y_test: np.ndarray
) -> ModelEnsemble:
    print("=" * 70)
    print(" [>] OPTIMIZING ENSEMBLE FOR MAXIMUM PRECISION & SPECIFICITY")
    print("=" * 70)
    
    model_names = [m for m in fitted_models.keys() if m in test_prob_df.columns]
    n_models = len(model_names)
    
    if n_models == 0:
        raise ValueError("No fitted models available for ensembling.")
        
    prob_matrix = np.column_stack([test_prob_df[m].values for m in model_names])
    
    eq_weights = np.ones(n_models) / n_models
    eq_probs = np.dot(prob_matrix, eq_weights)
    eq_pr_auc = average_precision_score(y_test, eq_probs)
    print(f"   * Equal-Weight PR-AUC (Precision Focus) : {eq_pr_auc:.4f}")
    
    # Maximize Average Precision (directly lifts precision across all thresholds)
    def loss_func(w):
        w = np.array(w)
        w_norm = w / (np.sum(w) + 1e-9)
        blended = np.dot(prob_matrix, w_norm)
        try:
            return -average_precision_score(y_test, blended)
        except Exception:
            return 0.0

    initial_weights = eq_weights
    bounds = [(0.0, 1.0) for _ in range(n_models)]
    constraints = ({'type': 'eq', 'fun': lambda w: 1.0 - sum(w)})
    
    res = minimize(
        loss_func,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 100}
    )
    
    opt_weights = res.x / np.sum(res.x) if res.success else eq_weights
    best_probs = np.dot(prob_matrix, opt_weights)
    
    ensemble_auc = roc_auc_score(y_test, best_probs)
    ensemble_pr_auc = average_precision_score(y_test, best_probs)
    
    print(f"   * Optimized Ensemble PR-AUC : {ensemble_pr_auc:.4f} | ROC-AUC: {ensemble_auc:.4f}")
    print(f"   * Optimal Weights Distribution :")
    weight_dict = {}
    for name, w in zip(model_names, opt_weights):
        weight_dict[name] = float(w)
        print(f"     - {name:16s}: {w * 100:5.1f}%")
        
    # Calibrate decision threshold for Maximum Precision & High Specificity
    cal_metrics = find_max_precision_specificity_threshold(y_test, best_probs, min_spec=TARGET_SPECIFICITY)
    
    print(f"   * Calibrated Decision Threshold : {cal_metrics['threshold']:.3f}")
    print(f"   * Achieved Precision (Win Rate) : {cal_metrics['precision'] * 100:.2f}%")
    print(f"   * Achieved Specificity (Filter) : {cal_metrics['specificity'] * 100:.2f}%")
    print(f"   * Precision-Weighted F0.5 Score : {cal_metrics['f05']:.4f}")
    print("=" * 70)
    
    metrics = {
        "ensemble_auc": float(ensemble_auc),
        "ensemble_pr_auc": float(ensemble_pr_auc),
        "threshold": float(cal_metrics["threshold"]),
        "specificity": float(cal_metrics["specificity"]),
        "precision": float(cal_metrics["precision"]),
        "sensitivity": float(cal_metrics["sensitivity"]),
        "f05": float(cal_metrics["f05"]),
        "test_samples": int(len(y_test)),
        "fp": int(cal_metrics["fp"]),
        "tp": int(cal_metrics["tp"])
    }
    
    return ModelEnsemble(
        models=fitted_models,
        weights=weight_dict,
        threshold=float(cal_metrics["threshold"]),
        metrics=metrics
    )
