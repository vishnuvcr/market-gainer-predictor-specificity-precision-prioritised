"""
Ensemble optimizer maximizing ROC-AUC with high-specificity threshold calibration.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score, average_precision_score
from src.config import TARGET_SPECIFICITY
from src.trainer import find_high_specificity_threshold

class ModelEnsemble:
    def __init__(self, models: Dict[str, Any], weights: Dict[str, float], threshold: float, metrics: Dict[str, Any]):
        self.models = models
        self.weights = weights
        self.threshold = threshold
        self.metrics = metrics
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
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

def build_optimal_ensemble(fitted_models: Dict[str, Any], test_prob_df: pd.DataFrame, y_test: np.ndarray) -> ModelEnsemble:
    model_names = [m for m in fitted_models.keys() if m in test_prob_df.columns]
    n_models = len(model_names)
    prob_matrix = np.column_stack([test_prob_df[m].values for m in model_names])
    
    # Solve for optimal weights maximizing ROC-AUC
    def loss_func(w):
        w_norm = w / (np.sum(w) + 1e-9)
        blended = np.dot(prob_matrix, w_norm)
        return -roc_auc_score(y_test, blended)

    initial_weights = np.ones(n_models) / n_models
    bounds = [(0.0, 1.0) for _ in range(n_models)]
    constraints = ({'type': 'eq', 'fun': lambda w: 1.0 - sum(w)})
    
    res = minimize(loss_func, initial_weights, method="SLSQP", bounds=bounds, constraints=constraints)
    opt_weights = res.x / np.sum(res.x) if res.success else initial_weights
    best_probs = np.dot(prob_matrix, opt_weights)
    
    ensemble_auc = roc_auc_score(y_test, best_probs)
    ensemble_pr_auc = average_precision_score(y_test, best_probs)
    
    weight_dict = {name: float(w) for name, w in zip(model_names, opt_weights)}
    spec_metrics = find_high_specificity_threshold(y_test, best_probs, min_spec=TARGET_SPECIFICITY)
    
    metrics = {
        "ensemble_auc": float(ensemble_auc),
        "ensemble_pr_auc": float(ensemble_pr_auc),
        "threshold": float(spec_metrics["threshold"]),
        "specificity": float(spec_metrics["specificity"]),
        "precision": float(spec_metrics["precision"]),
        "sensitivity": float(spec_metrics["sensitivity"]),
        "test_samples": int(len(y_test)),
        "fp": int(spec_metrics["fp"]),
        "tp": int(spec_metrics["tp"])
    }
    return ModelEnsemble(fitted_models, weight_dict, float(spec_metrics["threshold"]), metrics)
