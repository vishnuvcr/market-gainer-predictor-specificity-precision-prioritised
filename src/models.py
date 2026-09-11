"""
Optimized model definitions for fast multi-thousand ticker universes.
"""
from typing import Dict, Any

def get_base_models(random_state: int = 42) -> Dict[str, Any]:
    models = {}
    
    # 1. LightGBM (Lightning fast, native NaN handling)
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

    # 2. XGBoost (Fast histogram tree method, native NaN handling)
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

    # 3. Random Forest (with 40% subsampling for 3x speedup)
    try:
        from sklearn.ensemble import RandomForestClassifier
        models["RandomForest"] = RandomForestClassifier(
            random_state=random_state,
            n_estimators=50,
            max_depth=8,
            max_samples=0.4,
            class_weight="balanced_subsample",
            n_jobs=-1
        )
    except ImportError:
        pass

    return models

def get_hyperparameter_distributions() -> Dict[str, Dict[str, Any]]:
    return {
        "LightGBM": {
            "n_estimators": (60, 100),
            "learning_rate": (0.03, 0.06),
            "num_leaves": (15, 31),
            "max_depth": (4, 6)
        },
        "XGBoost": {
            "n_estimators": (60, 100),
            "learning_rate": (0.03, 0.06),
            "max_depth": (3, 5)
        },
        "RandomForest": {
            "n_estimators": (40, 60),
            "max_depth": (6, 10)
        }
    }
