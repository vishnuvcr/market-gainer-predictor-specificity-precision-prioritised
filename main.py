"""
Main entry point for the Automated Stock Surge ML Pipeline.
"""
import sys
import time
import argparse
from datetime import datetime

from src.config import TICKERS_FILE
from src.downloader import download_all_shards, load_tickers_from_file
from src.feature_engineering import build_dataset
from src.trainer import train_and_tune_pipeline
from src.ensembler import build_optimal_ensemble
from src.screener import generate_tomorrow_picks
from src.pine_generator import generate_pine_v6_script
from src.pages_builder import build_github_pages

def run_pipeline(quick_mode: bool = False, limit_tickers: int = None):
    start_total_time = time.time()
    
    print("\n" + "#" * 80)
    print(f"  NSE BREAKOUT SURGE MACHINE LEARNING PIPELINE [DAILY SCANNER]")
    print(f"  Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#" * 80 + "\n")
    
    # 1. Load tickers
    tickers = load_tickers_from_file(TICKERS_FILE)
    if limit_tickers and limit_tickers > 0:
        tickers = tickers[:limit_tickers]
        print(f" [*] Running with limited subset of {limit_tickers} tickers.")
        
    # 2. Download data using concurrent shards
    data_dict = download_all_shards(tickers)
    if not data_dict or len(data_dict) < 3:
        print(" [!] Insufficient ticker data downloaded. Exiting.")
        sys.exit(1)
        
    # 3. Feature Engineering & Target Labeling
    hist_df, live_df = build_dataset(data_dict)
    if hist_df.empty or len(hist_df) < 100:
        print(" [!] Not enough historical records to train models reliably. Exiting.")
        sys.exit(1)
        
    # 4. K-Fold Temporal CV & Hyperparameter Tuning
    fitted_models, model_eval_results, test_prob_df, feat_imp_df = train_and_tune_pipeline(hist_df)
    
    # 5. Model Ensembling & High-Specificity Calibration
    y_test = test_prob_df["target"].values
    ensemble = build_optimal_ensemble(fitted_models, test_prob_df, y_test)
    
    # 6. Generate Tomorrow's Stock Trade Setups
    candidates = generate_tomorrow_picks(ensemble, live_df, top_n=20)
    
    # 7. Generate Actionable TradingView Pine Script v6 Strategy
    generate_pine_v6_script(
        feat_imp_df=feat_imp_df,
        opt_threshold=ensemble.threshold
    )
    
    # 8. Build GitHub Pages Dashboard & Update Historical JSON Archive
    build_github_pages(
        candidates=candidates,
        ensemble_metrics=ensemble.metrics,
        model_eval_results=model_eval_results,
        feat_imp_df=feat_imp_df,
        total_tickers=len(data_dict)
    )
    
    elapsed_total = time.time() - start_total_time
    print("\n" + "=" * 80)
    print(f" [SUCCESS] Pipeline completed successfully in {elapsed_total:.2f} seconds.")
    print(f"           - High Conviction Picks for Tomorrow : {sum(1 for c in candidates if 'HIGH' in c.get('conviction', ''))}")
    print(f"           - Test ROC-AUC                       : {ensemble.metrics['ensemble_auc'] * 100:.2f}%")
    print(f"           - Calibrated Specificity             : {ensemble.metrics['specificity'] * 100:.2f}%")
    print(f"           - Pine v6 Strategy Script            : strategies/breakout_surge_v6.pine")
    print(f"           - GitHub Pages Output                : docs/index.html & docs/data/history.json")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Daily NSE Surge ML Pipeline")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tickers to process")
    args = parser.parse_args()
    run_pipeline(limit_tickers=args.limit)
