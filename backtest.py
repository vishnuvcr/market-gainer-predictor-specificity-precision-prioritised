"""
Standalone Walk-Forward Backtesting runner for NSE Daily Breakout Strategy.
Accounts for:
  - Initial Capital: ₹1,00,000 (1 Lakh)
  - Max per Trade: ₹20,000 (including Paytm Money statutory charges)
  - Paytm Money intraday charges schedule
  - Dynamic ATR stop-loss and +5% profit target cap
"""
import argparse
import sys
import time
from src.config import TICKERS_FILE
from src.downloader import download_all_shards, load_tickers_from_file
from src.backtester import WalkForwardBacktester

def main():
    parser = argparse.ArgumentParser(description="Run Walk-Forward Backtest for Stock Surge ML")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tickers (e.g. 50 or 100 for fast backtest)")
    parser.add_argument("--capital", type=float, default=100000.0, help="Initial capital in INR (default: 100000 = 1 Lakh)")
    parser.add_argument("--max-per-trade", type=float, default=20000.0, help="Maximum capital per trade including charges (default: 20000)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top probability stocks to trade daily (default: 5)")
    parser.add_argument("--target-cap", type=float, default=0.05, help="Profit target cap (default: 0.05 for +5%)")
    parser.add_argument("--warmup-days", type=int, default=120, help="Number of trading days for initial warm-up training (default: 120)")
    parser.add_argument("--retrain-days", type=int, default=10, help="Walk-forward retraining frequency in trading days (default: 10)")
    args = parser.parse_args()

    start_time = time.time()
    tickers = load_tickers_from_file(TICKERS_FILE)
    if args.limit and args.limit > 0:
        tickers = tickers[:args.limit]
        print(f" [*] Running backtest with limited universe of {args.limit} tickers.")

    data_dict = download_all_shards(tickers)
    if len(data_dict) < 5:
        print(" [!] Insufficient ticker data downloaded. Exiting.")
        sys.exit(1)

    backtester = WalkForwardBacktester(
        data_dict=data_dict,
        initial_capital=args.capital,
        max_capital_per_trade=args.max_per_trade,
        top_k=args.top_k,
        target_cap_pct=args.target_cap,
        warmup_days=args.warmup_days,
        retrain_frequency_days=args.retrain_days
    )

    summary = backtester.run()
    elapsed = time.time() - start_time
    print(f" [SUCCESS] Backtest completed in {elapsed:.2f} seconds ({elapsed/60:.1f} minutes).")

if __name__ == "__main__":
    main()
