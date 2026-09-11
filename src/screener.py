import numpy as np
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime
from src.config import (
    FEATURE_COLUMNS, TARGET_1_PCT, TARGET_2_PCT, TARGET_3_PCT,
    ATR_SL_MULTIPLIER, MAX_SL_PCT
)
from src.ensembler import ModelEnsemble

def generate_tomorrow_picks(ensemble: ModelEnsemble, latest_df: pd.DataFrame, top_n: int = 25, max_export: int = 200) -> List[Dict[str, Any]]:
    if latest_df.empty:
        return []
        
    X_live = latest_df[list(FEATURE_COLUMNS)].values
    probs = ensemble.predict_proba(X_live)
    candidates = []
    
    for idx, (_, row) in enumerate(latest_df.iterrows()):
        prob = float(probs[idx])
        ticker = str(row["ticker"])
        clean_ticker = ticker.replace(".NS", "").replace(".BO", "")
        close_p = float(row["Close"])
        atr_val = float(row.get("atr_val", close_p * 0.025))
        
        sl_distance = min(ATR_SL_MULTIPLIER * atr_val, MAX_SL_PCT * close_p)
        stop_loss = max(round(close_p - sl_distance, 2), 0.05)
        sl_pct = ((stop_loss - close_p) / close_p) * 100
        
        t1, t2, t3 = round(close_p * (1.0 + TARGET_1_PCT), 2), round(close_p * (1.0 + TARGET_2_PCT), 2), round(close_p * (1.0 + TARGET_3_PCT), 2)
        rr_ratio = abs(TARGET_1_PCT * 100 / (abs(sl_pct) + 1e-6))
        
        if prob >= ensemble.threshold:
            conviction, badge_class = "HIGH CONVICTION (SURGE READY)", "badge-high"
        elif prob >= ensemble.threshold * 0.80:
            conviction, badge_class = "MODERATE CONVICTION", "badge-med"
        else:
            conviction, badge_class = "WATCHLIST ONLY", "badge-low"
            
        catalysts = []
        vol_surge = float(row.get("vol_surge_20", 1.0))
        if vol_surge >= 2.0:
            catalysts.append(f"Volume Breakout ({vol_surge:.1f}x 20D SMA)")
        rsi = float(row.get("rsi_14", 50.0))
        if 55 <= rsi <= 72:
            catalysts.append(f"Bullish RSI ({rsi:.1f})")
        bb_width = float(row.get("bb_width_20", 0.05))
        if bb_width < 0.08:
            catalysts.append("Bollinger Volatility Squeeze")
        if not catalysts:
            catalysts.append("Multi-factor Momentum Alignment")
            
        candidates.append({
            "ticker": ticker, "symbol": clean_ticker, "close": close_p,
            "probability": prob, "conviction": conviction, "badge_class": badge_class,
            "entry_ref": close_p, "stop_loss": stop_loss, "stop_loss_pct": round(sl_pct, 2),
            "target_1": t1, "target_1_pct": round(TARGET_1_PCT * 100, 1),
            "target_2": t2, "target_2_pct": round(TARGET_2_PCT * 100, 1),
            "target_3": t3, "target_3_pct": round(TARGET_3_PCT * 100, 1),
            "risk_reward": round(rr_ratio, 2), "volume_surge": round(vol_surge, 2),
            "rsi_14": round(rsi, 1), "catalysts": catalysts,
            "date": datetime.now().strftime("%Y-%m-%d")
        })
        
    candidates.sort(key=lambda x: x["probability"], reverse=True)
    filtered = [c for c in candidates if "WATCHLIST" not in c["conviction"]]
    watchlist = [c for c in candidates if "WATCHLIST" in c["conviction"]]
    filtered.extend(watchlist[:max(0, max_export - len(filtered))])
    return filtered
