"""
Training and evaluation pipeline with K-fold temporal partitioning,
hyperparameter tuning optimizing Average Precision (PR-AUC),
and F0.5 precision-weighted threshold calibration.
"""
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss,
    fbeta_score
)

from src.config import (
    FEATURE_COLUMNS,
    K_FOLDS,
    TEST_SIZE_RATIO,
    TARGET_SPECIFICITY,
    HYPERPARAM_SEARCH_ITER,
    MAX_TUNING_SAMPLES,
    RANDOM_STATE
)
from src.models import get_base_models, get_hyperparameter_distributions

def compute_metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=(0, 1)).ravel()
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f05 = fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)
    
    return {
        "threshold": threshold,
        "specificity": specificity,
        "sensitivity": sensitivity,
        "precision": precision,
        "accuracy": accuracy,
        "f05": float(f05),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn)
    }

def find_max_precision_specificity_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_spec: float = TARGET_SPECIFICITY
) -> Dict[str, float]:
    best_thresh_metrics = None
    best_score = -1.0
    
    thresholds = np.linspace(0.15, 0.90, 76)
    for thresh in thresholds:
        m = compute_metrics_at_threshold(y_true, y_prob, thresh)
        if m["specificity"] >= min_spec:
            if (m["tp"] + m["fp"]) >= 3:
                # Joint score: 80% Precision + 20% F0.5
                score = m["precision"] * 0.80 + m["f05"] * 0.20
                if score > best_score:
                    best_score = score
                    best_thresh_metrics = m
                    
    if best_thresh_metrics is None or best_thresh_metrics["precision"] == 0:
        fallback_thresh = float(np.percentile(y_prob, 95))
        best_thresh_metrics = compute_metrics_at_threshold(y_true, y_prob, fallback_thresh)
        
    return best_thresh_metrics

def train_and_tune_pipeline(df: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    print("=" * 70)
    print(" [>] COMMENCING PRECISION & SPECIFICITY OPTIMIZED TRAINING")
    print("=" * 70)
    
    feature_list = list(FEATURE_COLUMNS)
    unique_dates = sorted(df.index.unique())
    split_idx = int(len(unique_dates) * (1 - TEST_SIZE_RATIO))
    train_dates = unique_dates[:split_idx]
    test_dates = unique_dates[split_idx:]
    
    train_mask = df.index.isin(train_dates)
    test_mask = df.index.isin(test_dates)
    
    X_train = df.loc[train_mask, feature_list].values
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    y_train = df.loc[train_mask, "target"].values.astype(int)
    
    X_test = df.loc[test_mask, feature_list].values
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    y_test = df.loc[test_mask, "target"].values.astype(int)
    
    print(f" [SPLIT] Train Range : {train_dates[0]} to {train_dates[-1]} ({len(X_train):,} samples, {y_train.sum():,} surges)")
    print(f" [SPLIT] Test Range  : {test_dates[0]} to {test_dates[-1]} ({len(X_test):,} samples, {y_test.sum():,} surges)")
    print(f" [SPLIT] K-Fold CV   : {K_FOLDS} Time-Series Folds (Scoring: Average Precision)")
    print("-" * 70)
    
    if len(X_train) > MAX_TUNING_SAMPLES:
        rng = np.random.RandomState(RANDOM_STATE)
        sub_indices = rng.choice(len(X_train), size=MAX_TUNING_SAMPLES, replace=False)
        sub_indices.sort()
        X_tune = X_train[sub_indices]
        y_tune = y_train[sub_indices]
        print(f" [SPEEDUP] Subsampled {MAX_TUNING_SAMPLES:,} rows for hyperparameter optimization.")
    else:
        X_tune = X_train
        y_tune = y_train
        
    base_models = get_base_models(random_state=RANDOM_STATE)
    param_grids = get_hyperparameter_distributions()
    
    fitted_models: Dict[str, Any] = {}
    model_eval_results: Dict[str, Dict[str, Any]] = {}
    test_prob_df = pd.DataFrame(index=df.loc[test_mask].index)
    test_prob_df["target"] = y_test
    
    total_models = len(base_models)
    current_idx = 0
    tscv = TimeSeriesSplit(n_splits=K_FOLDS)
    
    for model_name, model in base_models.items():
        current_idx += 1
        print(f" [MODEL {current_idx}/{total_models}] Tuning for Max Precision: {model_name}...")
        start_t = time.time()
        
        param_dist = param_grids.get(model_name, {})
        if param_dist:
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_dist,
                n_iter=HYPERPARAM_SEARCH_ITER,
                cv=tscv,
                scoring="average_precision",
                random_state=RANDOM_STATE,
                n_jobs=1,
                verbose=0
            )
            search.fit(X_tune, y_tune)
            best_params = search.best_params_
            cv_pr_auc = search.best_score_
            
            best_model = model.set_params(**best_params)
            best_model.fit(X_train, y_train)
            print(f"   -> Best CV PR-AUC (Precision Focus): {cv_pr_auc:.4f} in {time.time() - start_t:.1f}s")
        else:
            best_model = model.fit(X_train, y_train)
            best_params = "Default"
            cv_pr_auc = 0.0
            
        fitted_models[model_name] = best_model
        
        if hasattr(best_model, "predict_proba"):
            raw_prob = best_model.predict_proba(X_test)
            y_prob_test = raw_prob.take(1, axis=1) if raw_prob.ndim == 2 else raw_prob
        else:
            y_prob_test = best_model.predict(X_test)
            
        test_prob_df[model_name] = y_prob_test
        
        test_auc = roc_auc_score(y_test, y_prob_test)
        test_pr_auc = average_precision_score(y_test, y_prob_test)
        brier = brier_score_loss(y_test, y_prob_test)
        
        cal_metrics = find_max_precision_specificity_threshold(y_test, y_prob_test, min_spec=TARGET_SPECIFICITY)
        
        model_eval_results[model_name] = {
            "cv_pr_auc": float(cv_pr_auc),
            "test_auc": float(test_auc),
            "test_pr_auc": float(test_pr_auc),
            "brier_score": float(brier),
            "opt_threshold": float(cal_metrics["threshold"]),
            "specificity": float(cal_metrics["specificity"]),
            "sensitivity": float(cal_metrics["sensitivity"]),
            "precision": float(cal_metrics["precision"]),
            "f05": float(cal_metrics["f05"]),
            "tp": cal_metrics["tp"],
            "fp": cal_metrics["fp"],
            "best_params": best_params
        }
        
        print(f"   -> Holdout Evaluation:")
        print(f"      Precision   : {cal_metrics['precision'] * 100:.1f}% (Win Rate)")
        print(f"      Specificity : {cal_metrics['specificity'] * 100:.1f}% (Duds Rejected)")
        print(f"      ROC-AUC     : {test_auc:.4f} | PR-AUC: {test_pr_auc:.4f}")
        print(f"      Opt Thresh  : {cal_metrics['threshold']:.3f} (TP: {cal_metrics['tp']}, FP: {cal_metrics['fp']})")
        print("-" * 70)
        
    feat_imp_df = pd.DataFrame()
    for m_name in ["LightGBM", "XGBoost", "RandomForest"]:
        if m_name in fitted_models and hasattr(fitted_models[m_name], "feature_importances_"):
            m = fitted_models[m_name]
            imp = m.feature_importances_
            feat_imp_df[m_name] = imp / (imp.sum() + 1e-9)
            
    if not feat_imp_df.empty:
        feat_imp_df["mean_importance"] = feat_imp_df.mean(axis=1)
        feat_imp_df["feature"] = feature_list
        feat_imp_df = feat_imp_df.sort_values(by="mean_importance", ascending=False)
        print(" [FEATURE RANKINGS Top 5 Drivers]:")
        for _, row in feat_imp_df.head(5).iterrows():
            print(f"   * {row['feature']:18s}: {row['mean_importance'] * 100:.2f}%")
        print("-" * 70)
        
    return fitted_models, model_eval_results, test_prob_df, feat_imp_df
