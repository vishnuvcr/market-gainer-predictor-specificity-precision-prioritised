"""
Master Unified Engine for Automated Stock Surge ML Pipeline.
Integrates:
  1. Deep Multi-Year Data Ingestion (5 Years)
  2. K-Fold Temporal ML Ensemble Training (LightGBM, XGBoost, Random Forest)
  3. Tomorrow's High-Conviction Stock Surge Screening
  4. Adaptive Walk-Forward Backtesting with Penalty-Based Error Learning
  5. Actionable TradingView Pine Script v6 Strategy Generation
  6. Synchronized Interactive GitHub Pages Web Dashboard
"""
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path

from src.config import (
    TICKERS_FILE,
    INITIAL_CAPITAL,
    MAX_CAPITAL_PER_TRADE,
    MIN_CONFIDENCE_THRESHOLD,
    DATA_DIR
)
from src.downloader import download_all_shards, load_tickers_from_file
from src.feature_engineering import build_dataset
from src.trainer import train_and_tune_pipeline
from src.ensembler import build_optimal_ensemble
from src.screener import generate_tomorrow_picks
from src.pine_generator import generate_pine_v6_script
from src.pages_builder import build_github_pages
from src.backtester import WalkForwardBacktester


def run_master_pipeline(
    mode: str = "full",
    train_models: bool = True,
    screen_stocks: bool = True,
    run_backtest: bool = True,
    capital: float = INITIAL_CAPITAL,
    max_per_trade: float = MAX_CAPITAL_PER_TRADE,
    min_confidence: float = MIN_CONFIDENCE_THRESHOLD,
    top_k: int = 5,
    retrain_days: int = 15,
    limit_tickers: int = None,
    quick_mode: bool = False
):
    start_total_time = time.time()

    # Determine execution flags based on mode
    if mode == "train_only":
        train_models = True
        screen_stocks = False
        run_backtest = False
    elif mode == "screen_only":
        train_models = True
        screen_stocks = True
        run_backtest = False
    elif mode == "backtest_only":
        train_models = False
        screen_stocks = False
        run_backtest = True
    elif mode == "full":
        train_models = True
        screen_stocks = True
        run_backtest = True

    print("\n" + "#" * 80)
    print("  NSE BREAKOUT SURGE MACHINE LEARNING MASTER PIPELINE")
    print(f"  Execution Time (UTC/Local) : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Mode                       : {mode.upper()}")
    print(f"  Train & Model Generation   : {'ENABLED' if train_models else 'DISABLED'}")
    print(f"  Tomorrow Stock Suggestions : {'ENABLED' if screen_stocks else 'DISABLED'}")
    print(f"  Walk-Forward Backtest      : {'ENABLED' if run_backtest else 'DISABLED'}")
    print(f"  Initial Capital            : ₹{capital:,.2f} (1 Lakh)")
    print(f"  Max Outlay Per Trade       : ₹{max_per_trade:,.2f} (Including Paytm Charges)")
    print(f"  Quality Filter Gate        : >={min_confidence * 100:.1f}% Confidence")
    print("#" * 80 + "\n")

    # Step 1: Load ticker universe
    tickers = load_tickers_from_file(TICKERS_FILE)
    if limit_tickers and limit_tickers > 0:
        tickers = tickers[:limit_tickers]
        print(f" [*] Running with limited universe of {limit_tickers} tickers.")

    # Step 2: Download concurrent data shards (Up to 5 Years of history)
    data_dict = download_all_shards(tickers)
    if not data_dict or len(data_dict) < 3:
        print(" [!] Insufficient ticker data downloaded. Exiting.")
        sys.exit(1)

    # Step 3: Walk-Forward Backtesting (if enabled)
    if run_backtest:
        print("\n" + "=" * 80)
        print(" [STAGE 1/3] EXECUTING ADAPTIVE WALK-FORWARD BACKTEST WITH LOSS PENALTIES")
        print("=" * 80)
        backtester = WalkForwardBacktester(
            data_dict=data_dict,
            initial_capital=capital,
            max_capital_per_trade=max_per_trade,
            min_confidence_threshold=min_confidence,
            top_k=top_k,
            warmup_days=120,
            retrain_frequency_days=retrain_days,
            enable_penalty_learning=True
        )
        bt_summary = backtester.run()

    # Step 4: Model Training & Tomorrow's Trade Screening
    candidates = []
    ensemble_metrics = {}
    model_eval_results = {}
    feat_imp_df = None

    if train_models or screen_stocks:
        print("\n" + "=" * 80)
        print(" [STAGE 2/3] FEATURE ENGINEERING & PRODUCTION MODEL CALIBRATION")
        print("=" * 80)
        hist_df, live_df = build_dataset(data_dict)
        if hist_df.empty or len(hist_df) < 100:
            print(" [!] Not enough historical records to train. Exiting.")
            sys.exit(1)

        fitted_models, model_eval_results, test_prob_df, feat_imp_df = train_and_tune_pipeline(hist_df)
        y_test = test_prob_df["target"].values
        ensemble = build_optimal_ensemble(fitted_models, test_prob_df, y_test)
        ensemble_metrics = ensemble.metrics

        if screen_stocks:
            print("\n" + "=" * 80)
            print(" [STAGE 3/3] SCREENING TOMORROW'S HIGH-CONVICTION BREAKOUT CANDIDATES")
            print("=" * 80)
            candidates = generate_tomorrow_picks(ensemble, live_df, top_n=25)

            # Generate Pine Script
            generate_pine_v6_script(
                feat_imp_df=feat_imp_df,
                opt_threshold=ensemble.threshold
            )

    # Step 5: Always Publish Synchronized Web Dashboard into docs/
    print("\n [PAGES] Publishing synchronized dashboard into docs/...")
    build_github_pages(
        candidates=candidates,
        ensemble_metrics=ensemble_metrics,
        model_eval_results=model_eval_results,
        feat_imp_df=feat_imp_df,
        total_tickers=len(data_dict)
    )

    elapsed_total = time.time() - start_total_time
    print("\n" + "=" * 80)
    print(f" [SUCCESS] Master Pipeline completed successfully in {elapsed_total:.2f} seconds ({elapsed_total/60:.1f} minutes)!")
    print(f"           - Mode Executed                      : {mode.upper()}")
    print(f"           - High Conviction Picks for Tomorrow : {sum(1 for c in candidates if 'HIGH' in str(c.get('conviction', '')))}")
    print(f"           - Web Dashboard Synchronized         : docs/index.html & docs/data/latest.json")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NSE Breakout ML Master Engine")
    parser.add_argument("--mode", type=str, default="full", choices=["full", "train_only", "screen_only", "backtest_only"], help="Pipeline execution mode")
    parser.add_argument("--train", action="store_true", default=None, help="Explicitly enable training")
    parser.add_argument("--no-train", action="store_false", dest="train", help="Disable training")
    parser.add_argument("--screen", action="store_true", default=None, help="Explicitly enable stock screening")
    parser.add_argument("--no-screen", action="store_false", dest="screen", help="Disable stock screening")
    parser.add_argument("--backtest", action="store_true", default=None, help="Explicitly enable backtesting")
    parser.add_argument("--no-backtest", action="store_false", dest="backtest", help="Disable backtesting")
    parser.add_argument("--capital", type=float, default=INITIAL_CAPITAL, help="Initial capital in INR")
    parser.add_argument("--max-per-trade", type=float, default=MAX_CAPITAL_PER_TRADE, help="Max capital per trade w/ fees")
    parser.add_argument("--min-confidence", type=float, default=MIN_CONFIDENCE_THRESHOLD, help="Min probability to enter trade")
    parser.add_argument("--retrain-days", type=int, default=15, help="Walk-forward retraining frequency")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tickers")
    parser.add_argument("--quick", action="store_true", help="Quick mode")
    args = parser.parse_args()

    train_flag = args.train if args.train is not None else (args.mode in ["full", "train_only", "screen_only"])
    screen_flag = args.screen if args.screen is not None else (args.mode in ["full", "screen_only"])
    backtest_flag = args.backtest if args.backtest is not None else (args.mode in ["full", "backtest_only"])

    run_master_pipeline(
        mode=args.mode,
        train_models=train_flag,
        screen_stocks=screen_flag,
        run_backtest=backtest_flag,
        capital=args.capital,
        max_per_trade=args.max_per_trade,
        min_confidence=args.min_confidence,
        retrain_days=args.retrain_days,
        limit_tickers=args.limit,
        quick_mode=args.quick
    )
