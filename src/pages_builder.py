import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from src.config import DOCS_DIR, DATA_DIR, LATEST_JSON, HISTORY_JSON, PINE_FILE

def build_github_pages(candidates: List[Dict[str, Any]], ensemble_metrics: Dict[str, Any], model_eval_results: Dict[str, Dict[str, Any]], feat_imp_df, total_tickers: int) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    current_time_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    top_features = []
    if feat_imp_df is not None and not feat_imp_df.empty:
        for _, row in feat_imp_df.head(10).iterrows():
            top_features.append({"feature": str(row["feature"]), "importance": float(row["mean_importance"])})
            
    run_payload = {
        "timestamp": current_time_iso,
        "date": current_date,
        "total_tickers_scanned": total_tickers,
        "total_candidates": len(candidates),
        "high_conviction_count": sum(1 for c in candidates if "HIGH" in c.get("conviction", "")),
        "ensemble_metrics": ensemble_metrics,
        "models_summary": model_eval_results,
        "top_features": top_features,
        "candidates": candidates
    }
    
    with open(LATEST_JSON, "w", encoding="utf-8") as f:
        json.dump(run_payload, f, indent=2)
        
    history_records = []
    if HISTORY_JSON.exists():
        try:
            with open(HISTORY_JSON, "r", encoding="utf-8") as f:
                history_records = json.load(f)
                if not isinstance(history_records, list):
                    history_records = []
        except Exception:
            history_records = []
            
    history_records = [r for r in history_records if r.get("date") != current_date]
    history_records.insert(0, run_payload)
    history_records = history_records[:60]
    
    with open(HISTORY_JSON, "w", encoding="utf-8") as f:
        json.dump(history_records, f, indent=2)
