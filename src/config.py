"""
Configuration module optimized for scanning 2,500+ NSE tickers.
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
PINE_FILE = STRATEGIES_DIR / "breakout_surge_v6.pine"

# Historical Data Config
LOOKBACK_DAYS = 380          # ~1.5 years of trading data for robust feature calculation
INTERVAL = "1d"              # Daily OHLCV data
SHARD_SIZE = 50              # 50 tickers per parallel download shard (optimal for yfinance bulk)
MAX_WORKERS = 10             # 10 simultaneous download threads

# Modeling & Target Definitions
SURGE_THRESHOLD = 0.05       # Target: Next-day High >= Next-day Open * (1 + 0.05)
MIN_TRAINING_SAMPLES = 1000  # Minimum total historical samples required
K_FOLDS = 5                  # K-fold temporal partitioning
TEST_SIZE_RATIO = 0.20       # Out-of-time test set ratio (last 20% of trading dates)
TARGET_SPECIFICITY = 0.95    # High specificity target (>=95% to suppress false breakouts)

# Scalability & Search Settings
MAX_TUNING_SAMPLES = 50000  # Subsample cap for fast hyperparameter search
HYPERPARAM_SEARCH_ITER = 4   # Search iterations per model
RANDOM_STATE = 42

# Risk Management Settings
TARGET_1_PCT = 0.05          # +5% first profit objective
TARGET_2_PCT = 0.10          # +10% runner objective
TARGET_3_PCT = 0.18          # +18% upper circuit objective
ATR_SL_MULTIPLIER = 1.5      # Stop loss distance = 1.5 * ATR(14)
MAX_SL_PCT = 0.04            # Hard stop cap (max 4% risk per trade)

# Feature Columns definition used across feature engineering, training, and screening
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
