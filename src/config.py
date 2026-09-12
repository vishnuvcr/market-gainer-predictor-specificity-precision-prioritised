"""
Configuration module optimized for maximum Precision and Specificity
across large-scale NSE ticker universes (2,500+ stocks) with 5-year historical depth.
"""
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
TICKERS_FILE = BASE_DIR / "tickers.txt"
DOCS_DIR = BASE_DIR / "docs"
DATA_DIR = DOCS_DIR / "data"
STRATEGIES_DIR = BASE_DIR / "strategies"

LATEST_JSON = DATA_DIR / "latest.json"
HISTORY_JSON = DATA_DIR / "history.json"
BACKTEST_SUMMARY_JSON = DATA_DIR / "backtest_summary.json"
BACKTEST_TRADES_JSON = DATA_DIR / "backtest_trades.json"
BACKTEST_DAILY_JSON = DATA_DIR / "backtest_daily.json"
PINE_FILE = STRATEGIES_DIR / "breakout_surge_v6.pine"

# Historical Data Config (Deep 5-Year History)
LOOKBACK_DAYS = 1825         # ~5 years of daily trading data for full bull/bear/consolidation cycles
INTERVAL = "1d"              # Daily OHLCV data
SHARD_SIZE = 50              # 50 tickers per parallel download shard (optimal for yfinance bulk)
MAX_WORKERS = 10             # 10 simultaneous download threads

# Modeling & Target Definitions
SURGE_THRESHOLD = 0.05       # Target: Next-day High >= Next-day Open * (1 + 0.05)
MIN_TRAINING_SAMPLES = 1000  # Minimum total historical samples required
K_FOLDS = 5                  # K-fold temporal partitioning
TEST_SIZE_RATIO = 0.20       # Out-of-time test set ratio (last 20% of trading dates)
TARGET_SPECIFICITY = 0.95    # Enforce at least 95% specificity (rejecting false breakouts)

# Precision Optimization Budget
MAX_TUNING_SAMPLES = 40000   # Statistically optimal subsample for fast hyperparameter search
HYPERPARAM_SEARCH_ITER = 4   # Fast hyperparameter search iterations
RANDOM_STATE = 42

# Risk Management & Trade Capital Settings
INITIAL_CAPITAL = 100000.0   # 1 Lakh INR default capital
MAX_CAPITAL_PER_TRADE = 20000.0  # ₹20,000 strictly capped per trade including charges
MIN_CONFIDENCE_THRESHOLD = 0.65  # Never take a trade below 65% probability
TARGET_1_PCT = 0.05          # +5% first profit objective (capped exit)
TARGET_2_PCT = 0.10          # +10% runner objective
TARGET_3_PCT = 0.18          # +18% upper circuit objective
ATR_SL_MULTIPLIER = 1.5      # Stop loss distance = 1.5 * ATR(14)
MAX_SL_PCT = 0.04            # Hard stop cap (max 4% risk per trade)
BREAKEVEN_TRIGGER_PCT = 0.025 # Move stop loss to entry once price reaches +2.5%

# Feature Columns list used across feature engineering, training, screening, and backtesting
FEATURE_COLUMNS = list((
    "return_1d", "return_3d", "return_5d", "return_10d", "return_20d",
    "vol_surge_20", "vol_surge_5", "turnover_surge",
    "rsi_14", "rsi_7", "rsi_slope_3",
    "atr_pct_14", "candle_range_pct", "body_to_range", "clv",
    "upper_shadow_pct", "lower_shadow_pct",
    "bb_width_20", "bb_pos_20",
    "dist_sma_20", "dist_sma_50", "dist_sma_200",
    "dist_high_20d", "consecutive_up_days", "gap_pct"
))
