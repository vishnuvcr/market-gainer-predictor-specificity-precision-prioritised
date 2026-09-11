"""
Model definitions and hyperparameter search distributions.
"""
from typing import Dict, Any

def get_base_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Instantiates candidate base models with initial baseline configurations.
    """
    models = {}
    
    # 1. LightGBM
    try:
        import lightgbm as lgb
        models["LightGBM"] = lgb.LGBMClassifier(
            random_state=random_state,
            n_estimators=100,
            learning_rate=0.05,
            class_weight="balanced",
            verbose=-1,
            n_jobs=-1
        )
    except ImportError:
        pass

    # 2. XGBoost
    try:
        import xgboost as xgb
        models["XGBoost"] = xgb.XGBClassifier(
            random_state=random_state,
            n_estimators=100,
            learning_rate=0.05,
            eval_metric="auc",
            tree_method="hist",
            n_jobs=-1
        )
    except ImportError:
        pass

    # 3. Random Forest (Scikit-Learn)
    try:
        from sklearn.ensemble import RandomForestClassifier
        models["RandomForest"] = RandomForestClassifier(
            random_state=random_state,
            n_estimators=60,
            max_depth=10,
            class_weight="balanced_subsample",
            n_jobs=-1
        )
    except ImportError:
        pass

    # 4. Gradient Boosting (Scikit-Learn fallback)
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        models["GradientBoosting"] = GradientBoostingClassifier(
            random_state=random_state,
            n_estimators=60,
            learning_rate=0.05,
            max_depth=4
        )
    except ImportError:
        pass

    return models

def get_hyperparameter_distributions() -> Dict[str, Dict[str, Any]]:
    """
    Defines hyperparameter distributions for RandomizedSearchCV tuning.
    """
    param_grids = {}
    
    # LightGBM Hyperparameter Grid
    param_grids["LightGBM"] = {
        "n_estimators": [60, 100, 150],
        "learning_rate": [0.03, 0.05, 0.08],
        "num_leaves": [15, 31, 63],
        "max_depth": [4, 6, 8],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    }
    
    # XGBoost Hyperparameter Grid
    param_grids["XGBoost"] = {
        "n_estimators": [60, 100, 150],
        "learning_rate": [0.03, 0.05, 0.08],
        "max_depth":,
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    }
    
    # Random Forest Hyperparameter Grid
    param_grids["RandomForest"] = {
        "n_estimators": [50, 80],
        "max_depth": [6, 10, 14],
        "min_samples_split": [5, 10],
        "max_features": ["sqrt", 0.8]
    }
    
    # Gradient Boosting Hyperparameter Grid
    param_grids["GradientBoosting"] = {
        "n_estimators": [50, 80],
        "learning_rate": [0.03, 0.06],
        "max_depth":
    }
    
    return param_grids
