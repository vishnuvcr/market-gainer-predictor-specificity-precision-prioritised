"""
Downloader module fetching OHLCV data using concurrent sharding.
"""
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List
import pandas as pd

from src.config import (
    TICKERS_FILE,
    LOOKBACK_DAYS,
    INTERVAL,
    SHARD_SIZE,
    MAX_WORKERS
)

def load_tickers_from_file(filepath=TICKERS_FILE) -> List[str]:
    tickers = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip().upper()
            if not t or t.startswith("#"):
                continue
            if not t.endswith(".NS") and not t.endswith(".BO"):
                t = f"{t}.NS"
            tickers.append(t)
    return sorted(list(set(tickers)))

def _chunk_list(lst: List[str], chunk_size: int) -> List[List[str]]:
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def _fetch_shard(shard_id: int, tickers: List[str], start_date: str, end_date: str, max_retries: int = 3) -> Dict[str, pd.DataFrame]:
    import yfinance as yf
    ticker_str = " ".join(tickers)
    results = {}
    
    for attempt in range(1, max_retries + 1):
        try:
            df = yf.download(
                tickers=ticker_str,
                start=start_date,
                end=end_date,
                interval=INTERVAL,
                group_by='ticker',
                auto_adjust=False,
                threads=False,
                progress=False,
                timeout=25
            )
            if df.empty:
                time.sleep(1.5 * attempt)
                continue
            
            if len(tickers) == 1:
                t = tickers[0]
                cleaned = _clean_ohlcv(df)
                if cleaned is not None and len(cleaned) >= 40:
                    results[t] = cleaned
            else:
                for t in tickers:
                    if t in df.columns.levels[0]:
                        cleaned = _clean_ohlcv(df[t].copy())
                        if cleaned is not None and len(cleaned) >= 40:
                            results[t] = cleaned
            if results:
                return results
        except Exception:
            time.sleep(2 * attempt)
    return results

def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return None
    df.columns = [c.capitalize() if isinstance(c, str) else str(c) for c in df.columns]
    required = ["Open", "High", "Low", "Close", "Volume"]
    if not all(col in df.columns for col in required):
        return None
    sub = df[required].dropna()
    sub = sub[(sub["Volume"] > 0) & (sub["Close"] > 0) & (sub["Open"] > 0)]
    return sub if len(sub) > 0 else None

def download_all_shards(tickers: List[str] = None) -> Dict[str, pd.DataFrame]:
    if tickers is None:
        tickers = load_tickers_from_file()
    
    shards = _chunk_list(tickers, SHARD_SIZE)
    total_shards = len(shards)
    end_dt = datetime.now() + timedelta(days=1)
    start_dt = end_dt - timedelta(days=LOOKBACK_DAYS)
    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")
    
    print("=" * 70)
    print(f" [>] DOWNLOADING TICKER DATA VIA CONCURRENT SHARDS")
    print(f"     Total Tickers : {len(tickers)} | Shards: {total_shards} | Max Workers: {MAX_WORKERS}")
    print("=" * 70)
    
    all_data: Dict[str, pd.DataFrame] = {}
    completed_shards = 0
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_shard = {
            executor.submit(_fetch_shard, idx + 1, shard, start_str, end_str): (idx + 1, len(shard))
            for idx, shard in enumerate(shards)
        }
        for future in as_completed(future_to_shard):
            shard_id, shard_len = future_to_shard[future]
            completed_shards += 1
            shard_result = future.result()
            all_data.update(shard_result)
            pct = (completed_shards / total_shards) * 100
            elapsed = time.time() - start_time
            print(f" [PROGRESS] Shard {completed_shards:2d}/{total_shards:2d} ({pct:5.1f}%) | "
                  f"Shard #{shard_id} loaded {len(shard_result):2d}/{shard_len:2d} tickers | "
                  f"Total Valid: {len(all_data):3d} | Elapsed: {elapsed:4.1f}s")
            
    print(f" [DONE] Loaded {len(all_data)} tickers in {time.time() - start_time:.2f}s.\n")
    return all_data
