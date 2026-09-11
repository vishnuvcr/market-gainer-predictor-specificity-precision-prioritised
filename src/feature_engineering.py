"""
Feature engineering module calculating technical indicators, volume dynamics,
volatility metrics, and target labels for next-day breakout prediction.
Robust against NaNs, Inf, and newly listed stocks.
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from src.config import SURGE_THRESHOLD, FEATURE_COLUMNS

def _calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

def _calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

def compute_features_for_ticker(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = df.copy().sort_index()
    if len(df) < 30:
        return pd.DataFrame()
    
    close = df["Close"]
    open_p = df["Open"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]
    
    # 1. Price Momentum & Returns
    df["return_1d"] = close.pct_change(1)
    df["return_3d"] = close.pct_change(3)
    df["return_5d"] = close.pct_change(5)
    df["return_10d"] = close.pct_change(10)
    df["return_20d"] = close.pct_change(20)
    df["gap_pct"] = (open_p - close.shift(1)) / (close.shift(1) + 1e-9)
    
    # 2. Volume Dynamics & Breakout Velocity
    vol_sma_20 = volume.rolling(20, min_periods=5).mean()
    vol_sma_5 = volume.rolling(5, min_periods=2).mean()
    df["vol_surge_20"] = volume / (vol_sma_20 + 1e-9)
    df["vol_surge_5"] = volume / (vol_sma_5 + 1e-9)
    
    turnover = close * volume
    turnover_sma_20 = turnover.rolling(20, min_periods=5).mean()
    df["turnover_surge"] = turnover / (turnover_sma_20 + 1e-9)
    
    # 3. Oscillators (RSI)
    df["rsi_14"] = _calc_rsi(close, 14)
    df["rsi_7"] = _calc_rsi(close, 7)
    df["rsi_slope_3"] = df["rsi_14"] - df["rsi_14"].shift(3)
    
    # 4. Volatility & Candle Structure
    atr_14 = _calc_atr(high, low, close, 14)
    df["atr_pct_14"] = atr_14 / (close + 1e-9)
    total_range = (high - low).clip(lower=1e-6)
    df["candle_range_pct"] = total_range / (close + 1e-9)
    body_range = (close - open_p).abs()
    df["body_to_range"] = body_range / total_range
    df["clv"] = ((close - low) - (high - close)) / total_range
    df["upper_shadow_pct"] = (high - np.maximum(open_p, close)) / total_range
    df["lower_shadow_pct"] = (np.minimum(open_p, close) - low) / total_range
    
    # 5. Bollinger Bands
    bb_mid = close.rolling(20, min_periods=5).mean()
    bb_std = close.rolling(20, min_periods=5).std().fillna(0)
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    df["bb_width_20"] = (bb_upper - bb_lower) / (bb_mid + 1e-9)
    df["bb_pos_20"] = (close - bb_lower) / (bb_upper - bb_lower + 1e-9)
    
    # 6. Trend Moving Averages
    sma_20 = close.rolling(20, min_periods=5).mean()
    sma_50 = close.rolling(50, min_periods=10).mean()
    sma_200 = close.rolling(200, min_periods=15).mean()
    df["dist_sma_20"] = (close - sma_20) / (sma_20 + 1e-9)
    df["dist_sma_50"] = (close - sma_50) / (sma_50 + 1e-9)
    df["dist_sma_200"] = (close - sma_200) / (sma_200 + 1e-9)
    
    # 7. Resistance Breakout (20-day high proximity)
    high_20d = high.rolling(20, min_periods=5).max()
    df["dist_high_20d"] = (close - high_20d) / (high_20d + 1e-9)
    
    # 8. Consecutive Upward Closes
    is_up = (close > close.shift(1)).astype(int)
    consec = []
    c = 0
    for val in is_up:
        c = c + 1 if val == 1 else 0
        consec.append(c)
    df["consecutive_up_days"] = consec
    
    # Target (Next day surge >= 5%)
    next_open = open_p.shift(-1)
    next_high = high.shift(-1)
    next_surge_pct = (next_high - next_open) / (next_open + 1e-9)
    df["next_surge_pct"] = next_surge_pct
    df["target"] = (next_surge_pct >= SURGE_THRESHOLD).astype(int)
    
    df["ticker"] = ticker
    df["atr_val"] = atr_14
    
    # Clean all NaNs and Infs
    feature_list = list(FEATURE_COLUMNS)
    df[feature_list] = df[feature_list].replace([np.inf, -np.inf], np.nan)
    df[feature_list] = df[feature_list].bfill().ffill().fillna(0.0)
    
    return df

def build_dataset(data_dict: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    historical_dfs = []
    latest_rows = []
    feature_list = list(FEATURE_COLUMNS)
    
    for ticker, df in data_dict.items():
        feat_df = compute_features_for_ticker(df, ticker)
        if feat_df.empty or len(feat_df) < 15:
            continue
            
        live_row = feat_df.iloc[-1:].copy()
        live_row[feature_list] = live_row[feature_list].fillna(0.0)
        latest_rows.append(live_row)
        
        hist_df = feat_df.iloc[:-1].copy()
        hist_df = hist_df.dropna(subset=feature_list + ["target"])
        if not hist_df.empty:
            historical_dfs.append(hist_df)
            
    if not historical_dfs:
        raise ValueError("No historical features could be computed.")
        
    combined_hist = pd.concat(historical_dfs, axis=0).sort_index()
    combined_latest = pd.concat(latest_rows, axis=0) if latest_rows else pd.DataFrame()
    
    return combined_hist, combined_latest
