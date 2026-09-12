"""
Configuration module for Multi-Timeframe (Daily + 1-Hour Opening Range)
NSE Stock Surge ML Engine with dynamic ML targets and penalty learning.
"""
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

# Historical Lookback Settings
LOOKBACK_DAYS = 1825         # ~5 years of historical market depth
INTERVAL = "1d"
SHARD_SIZE = 50
MAX_WORKERS = 10

# Capital & Trade Sizing
INITIAL_CAPITAL = 100000.0   # ₹1,00,000 (1 Lakh)
MAX_CAPITAL_PER_TRADE = 20000.0  # ₹20,000 max outlay including charges
MAX_TRADES_PER_DAY = 5       # Max 5 positions per day
MIN_DAILY_TRADES = 1         # Guarantee daily trade discovery

# Dynamic Targets & Risk Bounds
MIN_TARGET_PCT = 0.045       # Base target 4.5%
BREAKEVEN_TRIGGER_PCT = 0.020 # Move stop loss to entry once price reaches +2.0%
MIN_SL_PCT = 0.015           # Minimum stop loss 1.5%
MAX_SL_PCT = 0.035           # Maximum hard stop 3.5%

# Feature Columns (Daily + First-Hour Opening Range Features)
FEATURE_COLUMNS = list((
    "return_1d", "return_3d", "return_5d", "return_10d", "return_20d",
    "vol_surge_20", "vol_surge_5", "turnover_surge",
    "rsi_14", "rsi_7", "rsi_slope_3",
    "atr_pct_14", "candle_range_pct", "body_to_range", "clv",
    "upper_shadow_pct", "lower_shadow_pct",
    "bb_width_20", "bb_pos_20",
    "dist_sma_20", "dist_sma_50", "dist_sma_200",
    "dist_high_20d", "gap_pct",
    # 1-Hour Opening Range Features
    "first_hour_return", "first_hour_range", "first_hour_clv", "first_hour_vol_ratio"
))
