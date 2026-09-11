import time
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, brier_score_loss
from src.config import (
    FEATURE_COLUMNS, K_FOLDS, TEST_SIZE_RATIO, TARGET_SPECIFICITY,
    HYPERPARAM_SEARCH_ITER, MAX_TUNING_SAMPLES, RANDOM_STATE
)
from src.models import get_base_models, get_hyperparameter_distributions

def compute_metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=).ravel()
    return {
        "threshold": threshold,
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)
    }

def find_high_specificity_threshold(y_true: np.ndarray, y_prob: np.ndarray, min_spec: float = TARGET_SPECIFICITY) -> Dict[str, float]:
    best_thresh_metrics = None
    best_score = -1.0
    for thresh in np.linspace(0.10, 0.95, 86):
        m = compute_metrics_at_threshold(y_true, y_prob, thresh)
        if m["specificity"] >= min_spec:
            score = m["precision"] * 0.7 + m["sensitivity"] * 0.3
            if score > best_score and (m["tp"] + m["fp"]) > 0:
                best_score = score
                best_thresh_metrics = m
    if best_thresh_metrics is None:
        best_thresh_metrics = compute_metrics_at_threshold(y_true, y_prob, float(np.percentile(y_prob, 92)))
    return best_thresh_metrics

def train_and_tune_pipeline(df: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    unique_dates = sorted(df.index.unique())
    split_idx = int(len(unique_dates) * (1 - TEST_SIZE_RATIO))
    train_mask = df.index.isin(unique_dates[:split_idx])
    test_mask = df.index.isin(unique_dates[split_idx:])
    
    X_train, y_train = df.loc[train_mask, FEATURE_COLUMNS].values, df.loc[train_mask, "target"].values.astype(int)
    X_test, y_test = df.loc[test_mask, FEATURE_COLUMNS].values, df.loc[test_mask, "target"].values.astype(int)
    
    if len(X_train) > MAX_TUNING_SAMPLES:
        rng = np.random.RandomState(RANDOM_STATE)
        sub_idx = rng.choice(len(X_train), size=MAX_TUNING_SAMPLES, replace=False)
        sub_idx.sort()
        X_tune, y_tune = X_train[sub_idx], y_train[sub_idx]
    else:
        X_tune, y_tune = X_train, y_train
        
    base_models = get_base_models(random_state=RANDOM_STATE)
    param_grids = get_hyperparameter_distributions()
    fitted_models, model_eval_results = {}, {}
    test_prob_df = pd.DataFrame(index=df.loc[test_mask].index)
    test_prob_df["target"] = y_test
    tscv = TimeSeriesSplit(n_splits=K_FOLDS)
    
    for model_name, model in base_models.items():
        param_dist = param_grids.get(model_name, {})
        if param_dist:
            search = RandomizedSearchCV(model, param_dist, n_iter=HYPERPARAM_SEARCH_ITER, cv=tscv, scoring="roc_auc", random_state=RANDOM_STATE, n_jobs=1)
            search.fit(X_tune, y_tune)
            best_model = model.set_params(**search.best_params_).fit(X_train, y_train)
            best_params = search.best_params_
            cv_auc = search.best_score_
        else:
            best_model = model.fit(X_train, y_train)
            best_params, cv_auc = "Default", 0.0
            
        fitted_models[model_name] = best_model
        y_prob_test = best_model.predict_proba(X_test) if hasattr(best_model, "predict_proba") else best_model.predict(X_test)
        test_prob_df[model_name] = y_prob_test
        
        spec_m = find_high_specificity_threshold(y_test, y_prob_test, min_spec=TARGET_SPECIFICITY)
        model_eval_results[model_name] = {
            "cv_auc": float(cv_auc),
            "test_auc": float(roc_auc_score(y_test, y_prob_test)),
            "test_pr_auc": float(average_precision_score(y_test, y_prob_test)),
            "brier_score": float(brier_score_loss(y_test, y_prob_test)),
            "opt_threshold": float(spec_m["threshold"]),
            "specificity": float(spec_m["specificity"]),
            "sensitivity": float(spec_m["sensitivity"]),
            "precision": float(spec_m["precision"]),
            "best_params": best_params
        }
        
    feat_imp_df = pd.DataFrame()
    for m_name in ["LightGBM", "XGBoost", "RandomForest"]:
        if m_name in fitted_models and hasattr(fitted_models[m_name], "feature_importances_"):
            imp = fitted_models[m_name].feature_importances_
            feat_imp_df[m_name] = imp / (imp.sum() + 1e-9)
    if not feat_imp_df.empty:
        feat_imp_df["mean_importance"] = feat_imp_df.mean(axis=1)
        feat_imp_df["feature"] = FEATURE_COLUMNS
        feat_imp_df = feat_imp_df.sort_values(by="mean_importance", ascending=False)
        
    return fitted_models, model_eval_results, test_prob_df, feat_imp_df
