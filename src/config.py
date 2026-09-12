"""
Configuration settings for Market Gainer Predictor (Specificity & Precision Prioritised)
"""

from pathlib import Path

# ==========================================
# 1. Project Paths
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Ensure runtime directories exist
for directory in [DATA_DIR, MODELS_DIR, REPORTS_DIR, OUTPUTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. Target & Threshold Definitions
# ==========================================
# Intraday surge threshold (5% gain from opening price to high next day)
SURGE_THRESHOLD = 0.05

# Minimum prediction confidence / probability cutoff for precision priority
MIN_CONFIDENCE = 0.65

# ==========================================
# 3. Capital & Risk Management Defaults
# ==========================================
DEFAULT_CAPITAL = 100000.0       # ₹1,00,000 total capital
MAX_PER_TRADE = 20000.0         # ₹20,000 max allocation per stock
MAX_OPEN_POSITIONS = 5          # Number of concurrent trades

# ==========================================
# 4. Pipeline & Model Parameters
# ==========================================
DEFAULT_LIMIT = 150             # Top N stock universe limit
RETRAIN_DAYS = 15               # Frequency to retrain the ML model (days)
RANDOM_STATE = 42

# ==========================================
# 5. Feature Columns
# ==========================================
# Must match the features generated in src/feature_engineering.py
FEATURE_COLUMNS = [
    # 1. Price Momentum & Returns
    "return_1d",
    "return_3d",
    "return_5d",
    
    # 2. Distance from Moving Averages & Highs
    "dist_sma_20",
    "dist_sma_50",
    "dist_sma_200",
    "dist_high_20d",
    "gap_pct",
    
    # 3. Bollinger Bands Dynamics
    "bb_width_20",
    "bb_pos_20",
    
    # 4. First-Hour Opening Range Indicators
    "first_hour_return",
    "first_hour_range",
    "first_hour_clv",
    "first_hour_vol_ratio",
]
