"""
Configuration module optimized for maximum Precision and Specificity
across large-scale NSE ticker universes with 5-year historical depth.
"""
import os
from pathlib import Path

# ==============================================================================
# 1. Project Paths & File Artifacts
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
TICKERS_FILE = BASE_DIR / "tickers.txt"
DOCS_DIR = BASE_DIR / "docs"
DATA_DIR = DOCS_DIR / "data"
STRATEGIES_DIR = BASE_DIR / "strategies"

for directory in [DATA_DIR, STRATEGIES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

LATEST_JSON = DATA_DIR / "latest.json"
HISTORY_JSON = DATA_DIR / "history.json"
BACKTEST_SUMMARY_JSON = DATA_DIR / "backtest_summary.json"
BACKTEST_TRADES_JSON = DATA_DIR / "backtest_trades.json"
BACKTEST_DAILY_JSON = DATA_DIR / "backtest_daily.json"
PINE_FILE = STRATEGIES_DIR / "breakout_surge_v6.pine"

# ==============================================================================
# 2. Historical Ingestion Config
# ==============================================================================
LOOKBACK_DAYS = 1825         # ~5 years of daily trading data
INTERVAL = "1d"              # Daily OHLCV data
SHARD_SIZE = 50              # 50 tickers per parallel download shard
MAX_WORKERS = 10             # 10 simultaneous download threads

# ==============================================================================
# 3. Target Labeling & ML Training
# ==============================================================================
SURGE_THRESHOLD = 0.05       # Target: Next-day High >= Next-day Open * (1 + 0.05)
MIN_TRAINING_SAMPLES = 1000  # Minimum historical samples required
K_FOLDS = 5                  # K-fold temporal partitioning
TEST_SIZE_RATIO = 0.20       # Out-of-time test set ratio (last 20% dates)
TARGET_SPECIFICITY = 0.95    # Enforce >=95% specificity

# Hyperparameter search budget
MAX_TUNING_SAMPLES = 40000
HYPERPARAM_SEARCH_ITER = 4
RANDOM_STATE = 42

# ==============================================================================
# 4. Risk, Trade Capital & Backtest Parameters
# ==============================================================================
INITIAL_CAPITAL = 100000.0        # ₹1,00,000 default capital
MAX_CAPITAL_PER_TRADE = 20000.0   # ₹20,000 max outlay per trade
MIN_CONFIDENCE_THRESHOLD = 0.50   # 50% minimum probability gate

# Targets & Screener Objectives
MIN_TARGET_PCT = 0.045            # +4.5% minimum volatility target floor
TARGET_1_PCT = 0.05               # +5.0% target 1
TARGET_2_PCT = 0.10               # +10.0% target 2
TARGET_3_PCT = 0.18               # +18.0% upper circuit target
MAX_TARGET_PCT = 0.08             # +8.0% upper bound target
TARGET_CAP_PCT = 0.05             # Profit target cap

# Stop Loss Bounds & Rules
MIN_SL_PCT = 0.015                # 1.5% minimum stop loss bound
MAX_SL_PCT = 0.035                # 3.5% maximum stop loss bound
ATR_SL_MULTIPLIER = 1.5           # Stop loss = 1.5 * ATR(14)
BREAKEVEN_TRIGGER_PCT = 0.020     # Move SL to entry at +2.0% surge

# Daily Trade Quotas & Execution Cadence
MAX_TRADES_PER_DAY = 5            # Maximum top picks per day
MIN_DAILY_TRADES = 1              # Daily trade guarantee fallback
TOP_K = 5                         # Top K setups
WARMUP_DAYS = 120                 # Warmup days
RETRAIN_DAYS = 15                 # Walk-forward retrain frequency
RETRAIN_FREQUENCY_DAYS = 15
ENABLE_PENALTY_LEARNING = True    # Adaptive error weighting

# ==============================================================================
# 5. Model Feature Columns (Exactly 28 features computed in feature_engineering.py)
# ==============================================================================
FEATURE_COLUMNS = list((
    # 1. Price Momentum & Returns
    "return_1d",
    "return_3d",
    "return_5d",
    "return_10d",
    "return_20d",
    
    # 2. Volume & Turnover
    "vol_surge_20",
    "vol_surge_5",
    "turnover_surge",
    
    # 3. RSI
    "rsi_14",
    "rsi_7",
    "rsi_slope_3",
    
    # 4. Volatility & Candle Morphology
    "atr_pct_14",
    "candle_range_pct",
    "body_to_range",
    "clv",
    "upper_shadow_pct",
    "lower_shadow_pct",
    
    # 5. Bollinger Bands & Moving Averages
    "bb_width_20",
    "bb_pos_20",
    "dist_sma_20",
    "dist_sma_50",
    "dist_sma_200",
    "dist_high_20d",
    "gap_pct",
    
    # 6. First-Hour Opening Range Simulated Indicators
    "first_hour_return",
    "first_hour_range",
    "first_hour_clv",
    "first_hour_vol_ratio",
))
