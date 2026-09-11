from typing import Dict, Any

def get_base_models(random_state: int = 42) -> Dict[str, Any]:
    models = {}
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

    return models

def get_hyperparameter_distributions() -> Dict[str, Dict[str, Any]]:
    return {
        "LightGBM": {
            "n_estimators": [60, 100, 150],
            "learning_rate": [0.03, 0.05, 0.08],
            "num_leaves": [15, 31, 63],
            "max_depth": [4, 6, 8],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0]
        },
        "XGBoost": {
            "n_estimators": [60, 100, 150],
            "learning_rate": [0.03, 0.05, 0.08],
            "max_depth":,
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0]
        },
        "RandomForest": {
            "n_estimators": [50, 80],
            "max_depth": [6, 10, 14],
            "min_samples_split": [5, 10],
            "max_features": ["sqrt", 0.8]
        }
    }
