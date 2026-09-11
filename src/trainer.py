"""
Training and evaluation pipeline with K-fold temporal partitioning,
hyperparameter tuning, and threshold calibration for high specificity.
Optimized for high-volume ticker universes (2,500+ stocks).
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
    brier_score_loss
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
    
    return {
        "threshold": threshold,
        "specificity": specificity,
        "sensitivity": sensitivity,
        "precision": precision,
        "accuracy": accuracy,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn)
    }

def find_high_specificity_threshold(y_true: np.ndarray, y_prob: np.ndarray, min_spec: float = TARGET_SPECIFICITY) -> Dict[str, float]:
    best_thresh_metrics = None
    best_score = -1.0
    
    thresholds = np.linspace(0.10, 0.95, 86)
    for thresh in thresholds:
        m = compute_metrics_at_threshold(y_true, y_prob, thresh)
        if m["specificity"] >= min_spec:
            score = m["precision"] * 0.7 + m["sensitivity"] * 0.3
            if score > best_score and (m["tp"] + m["fp"]) > 0:
                best_score = score
                best_thresh_metrics = m
                
    if best_thresh_metrics is None:
        fallback_thresh = float(np.percentile(y_prob, 92))
        best_thresh_metrics = compute_metrics_at_threshold(y_true, y_prob, fallback_thresh)
        
    return best_thresh_metrics

def train_and_tune_pipeline(df: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    print("=" * 70)
    print(" [>] COMMENCING MODEL TRAINING & K-FOLD HYPERPARAMETER TUNING")
    print("=" * 70)
    
    unique_dates = sorted(df.index.unique())
    split_idx = int(len(unique_dates) * (1 - TEST_SIZE_RATIO))
    train_dates = unique_dates[:split_idx]
    test_dates = unique_dates[split_idx:]
    
    train_mask = df.index.isin(train_dates)
    test_mask = df.index.isin(test_dates)
    
    X_train = df.loc[train_mask, FEATURE_COLUMNS].values
    y_train = df.loc[train_mask, "target"].values.astype(int)
    
    X_test = df.loc[test_mask, FEATURE_COLUMNS].values
    y_test = df.loc[test_mask, "target"].values.astype(int)
    
    print(f" [SPLIT] Train Range : {train_dates[0]} to {train_dates[-1]} ({len(X_train):,} samples, {y_train.sum():,} surges)")
    print(f" [SPLIT] Test Range  : {test_dates[0]} to {test_dates[-1]} ({len(X_test):,} samples, {y_test.sum():,} surges)")
    print(f" [SPLIT] K-Fold CV   : {K_FOLDS} Time-Series Temporal Folds")
    print("-" * 70)
    
    if len(X_train) > MAX_TUNING_SAMPLES:
        rng = np.random.RandomState(RANDOM_STATE)
        sub_indices = rng.choice(len(X_train), size=MAX_TUNING_SAMPLES, replace=False)
        sub_indices.sort()
        X_tune = X_train[sub_indices]
        y_tune = y_train[sub_indices]
        print(f" [SPEEDUP] Subsampled {MAX_TUNING_SAMPLES:,} rows for rapid hyperparameter optimization.")
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
        print(f" [MODEL {current_idx}/{total_models}] Tuning & Training: {model_name}...")
        start_t = time.time()
        
        param_dist = param_grids.get(model_name, {})
        if param_dist:
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_dist,
                n_iter=HYPERPARAM_SEARCH_ITER,
                cv=tscv,
                scoring="roc_auc",
                random_state=RANDOM_STATE,
                n_jobs=1,
                verbose=0
            )
            search.fit(X_tune, y_tune)
            best_params = search.best_params_
            cv_auc = search.best_score_
            
            best_model = model.set_params(**best_params)
            best_model.fit(X_train, y_train)
            print(f"   -> Best CV ROC-AUC: {cv_auc:.4f} (tuned & fitted in {time.time() - start_t:.1f}s)")
        else:
            best_model = model.fit(X_train, y_train)
            best_params = "Default"
            cv_auc = 0.0
            
        fitted_models[model_name] = best_model
        
        # 1D positive class probability extraction
        if hasattr(best_model, "predict_proba"):
            raw_prob = best_model.predict_proba(X_test)
            y_prob_test = raw_prob.T if raw_prob.ndim == 2 else raw_prob
        else:
            y_prob_test = best_model.predict(X_test)
            
        test_prob_df[model_name] = y_prob_test
        
        test_auc = roc_auc_score(y_test, y_prob_test)
        test_pr_auc = average_precision_score(y_test, y_prob_test)
        brier = brier_score_loss(y_test, y_prob_test)
        
        spec_metrics = find_high_specificity_threshold(y_test, y_prob_test, min_spec=TARGET_SPECIFICITY)
        
        model_eval_results[model_name] = {
            "cv_auc": float(cv_auc),
            "test_auc": float(test_auc),
            "test_pr_auc": float(test_pr_auc),
            "brier_score": float(brier),
            "opt_threshold": float(spec_metrics["threshold"]),
            "specificity": float(spec_metrics["specificity"]),
            "sensitivity": float(spec_metrics["sensitivity"]),
            "precision": float(spec_metrics["precision"]),
            "tp": spec_metrics["tp"],
            "fp": spec_metrics["fp"],
            "best_params": best_params
        }
        
        print(f"   -> Holdout Test Performance:")
        print(f"      ROC-AUC     : {test_auc:.4f} | PR-AUC: {test_pr_auc:.4f}")
        print(f"      Opt Thresh  : {spec_metrics['threshold']:.3f}")
        print(f"      Specificity : {spec_metrics['specificity'] * 100:.1f}% (False Positives: {spec_metrics['fp']})")
        print(f"      Precision   : {spec_metrics['precision'] * 100:.1f}% | Recall: {spec_metrics['sensitivity'] * 100:.1f}%")
        print("-" * 70)
        
    feat_imp_df = pd.DataFrame()
    for m_name in ["LightGBM", "XGBoost", "RandomForest"]:
        if m_name in fitted_models and hasattr(fitted_models[m_name], "feature_importances_"):
            m = fitted_models[m_name]
            imp = m.feature_importances_
            feat_imp_df[m_name] = imp / (imp.sum() + 1e-9)
            
    if not feat_imp_df.empty:
        feat_imp_df["mean_importance"] = feat_imp_df.mean(axis=1)
        feat_imp_df["feature"] = FEATURE_COLUMNS
        feat_imp_df = feat_imp_df.sort_values(by="mean_importance", ascending=False)
        print(" [FEATURE RANKINGS Top 5 Drivers]:")
        for _, row in feat_imp_df.head(5).iterrows():
            print(f"   * {row['feature']:18s}: {row['mean_importance'] * 100:.2f}%")
        print("-" * 70)
        
    return fitted_models, model_eval_results, test_prob_df, feat_imp_df
