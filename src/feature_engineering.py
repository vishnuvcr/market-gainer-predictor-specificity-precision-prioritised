"""
Feature engineering module computing combined Daily and First-Hour
Opening Range (9:15 AM - 10:15 AM) candle indicators across all NSE stocks.
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple

from src.config import FEATURE_COLUMNS, SURGE_THRESHOLD

def compute_features_for_ticker(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Computes technical indicators and simulates first-hour opening range features."""
    if df is None or len(df) < 35:
        return pd.DataFrame()

    df = df.copy()
    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values
    open_p = df["Open"].values
    volume = df["Volume"].values

    # 1. Price Momentum & Returns
    df["return_1d"] = df["Close"].pct_change(1)
    df["return_3d"] = df["Close"].pct_change(3)
    df["return_5d"] = df["Close"].pct_change(5)
    df["return_10d"] = df["Close"].pct_change(10)
    df["return_20d"] = df["Close"].pct_change(20)

    # 2. Volume & Turnover
    vol_sma20 = df["Volume"].rolling(20).mean()
    vol_sma5 = df["Volume"].rolling(5).mean()
    df["vol_surge_20"] = df["Volume"] / (vol_sma20 + 1e-6)
    df["vol_surge_5"] = df["Volume"] / (vol_sma5 + 1e-6)
    turnover = df["Close"] * df["Volume"]
    df["turnover_surge"] = turnover / (turnover.rolling(20).mean() + 1e-6)

    # 3. RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    roll_gain14 = gain.rolling(14).mean()
    roll_loss14 = loss.rolling(14).mean()
    rs14 = roll_gain14 / (roll_loss14 + 1e-9)
    df["rsi_14"] = 100 - (100 / (1 + rs14))

    roll_gain7 = gain.rolling(7).mean()
    roll_loss7 = loss.rolling(7).mean()
    rs7 = roll_gain7 / (roll_loss7 + 1e-9)
    df["rsi_7"] = 100 - (100 / (1 + rs7))
    df["rsi_slope_3"] = df["rsi_14"].diff(3)

    # 4. Volatility & Candle Morphology
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - df["Close"].shift(1)).abs()
    tr3 = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr_val"] = tr.rolling(14).mean()
    df["atr_pct_14"] = df["atr_val"] / (df["Close"] + 1e-6)

    c_range = df["High"] - df["Low"]
    c_body = (df["Close"] - df["Open"]).abs()
    df["candle_range_pct"] = c_range / (df["Close"] + 1e-6)
    df["body_to_range"] = c_body / (c_range + 1e-6)
    df["clv"] = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (c_range + 1e-6)

    df["upper_shadow_pct"] = (df["High"] - df[["Open", "Close"]].max(axis=1)) / (df["Close"] + 1e-6)
    df["lower_shadow_pct"] = (df[["Open", "Close"]].min(axis=1) - df["Low"]) / (df["Close"] + 1e-6)

    # 5. Bollinger Bands & Moving Averages
    sma20 = df["Close"].rolling(20).mean()
    std20 = df["Close"].rolling(20).std()
    upper_bb = sma20 + (2 * std20)
    lower_bb = sma20 - (2 * std20)
    df["bb_width_20"] = (upper_bb - lower_bb) / (sma20 + 1e-6)
    df["bb_pos_20"] = (df["Close"] - lower_bb) / ((upper_bb - lower_bb) + 1e-6)

    df["dist_sma_20"] = (df["Close"] - sma20) / (sma20 + 1e-6)
    df["dist_sma_50"] = (df["Close"] - df["Close"].rolling(50).mean()) / (df["Close"].rolling(50).mean() + 1e-6)
    df["dist_sma_200"] = (df["Close"] - df["Close"].rolling(200).mean()) / (df["Close"].rolling(200).mean() + 1e-6)
    df["dist_high_20d"] = (df["Close"] - df["High"].rolling(20).max()) / (df["Close"] + 1e-6)
    df["gap_pct"] = (df["Open"] - df["Close"].shift(1)) / (df["Close"].shift(1) + 1e-6)

    # 6. First-Hour Opening Range Simulated Indicators
    # Estimates the 9:15 - 10:15 candle expansion from open, high, low dynamics
    first_hour_est_close = open_p + 0.40 * (close - open_p)
    first_hour_est_high = np.maximum(open_p, open_p + 0.65 * (high - open_p))
    first_hour_est_low = np.minimum(open_p, open_p - 0.50 * (open_p - low))
    
    df["first_hour_return"] = (first_hour_est_close - open_p) / (open_p + 1e-6)
    df["first_hour_range"] = (first_hour_est_high - first_hour_est_low) / (open_p + 1e-6)
    df["first_hour_clv"] = ((first_hour_est_close - first_hour_est_low) - (first_hour_est_high - first_hour_est_close)) / ((first_hour_est_high - first_hour_est_low) + 1e-6)
    df["first_hour_vol_ratio"] = (df["Volume"] * 0.35) / (vol_sma20 * 0.35 + 1e-6)

    # 7. Next-Day Target Label (>5% intraday surge from entry)
    next_high = df["High"].shift(-1)
    next_open = df["Open"].shift(-1)
    df["target"] = ((next_high >= next_open * (1.0 + SURGE_THRESHOLD))).astype(int)

    df["ticker"] = ticker
    return df.dropna()

def build_dataset(data_dict: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Builds multi-year historical training dataset and live day scoring set."""
    hist_rows = []
    live_rows = []

    for ticker, df in data_dict.items():
        fdf = compute_features_for_ticker(df, ticker)
        if fdf.empty:
            continue
        hist_rows.append(fdf.iloc[:-1])
        live_rows.append(fdf.iloc[-1:])

    hist_df = pd.concat(hist_rows, axis=0) if hist_rows else pd.DataFrame()
    live_df = pd.concat(live_rows, axis=0) if live_rows else pd.DataFrame()
    return hist_df, live_df
