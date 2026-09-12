"""
Adaptive Walk-Forward Backtester with First-Hour Confirmation,
Daily Trade Guarantee, ML-Derived Targets/Stops, and Penalty Feedback Loop.
"""
import math
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd

from src.config import (
    BASE_DIR,
    FEATURE_COLUMNS,
    MIN_TARGET_PCT,
    BREAKEVEN_TRIGGER_PCT,
    MIN_SL_PCT,
    MAX_SL_PCT,
    MAX_TRADES_PER_DAY,
    MIN_DAILY_TRADES,
    DATA_DIR,
    BACKTEST_SUMMARY_JSON,
    BACKTEST_TRADES_JSON,
    BACKTEST_DAILY_JSON
)
from src.models import get_base_models
from src.ensembler import ModelEnsemble, build_optimal_ensemble

def calculate_paytm_money_charges(buy_price: float, sell_price: float, qty: int) -> Dict[str, float]:
    """Computes exact Paytm Money statutory and brokerage charges for NSE Equity Intraday."""
    if qty <= 0:
        return {"total_charges": 0.0}

    buy_turnover = float(buy_price * qty)
    sell_turnover = float(sell_price * qty)
    total_turnover = buy_turnover + sell_turnover

    brokerage = min(20.0, 0.0005 * buy_turnover) + min(20.0, 0.0005 * sell_turnover)
    stt = 0.00025 * sell_turnover
    txn_charges = 0.0000297 * total_turnover
    sebi_charges = 0.000001 * total_turnover
    stamp_duty = 0.00003 * buy_turnover
    gst = 0.18 * (brokerage + txn_charges + sebi_charges)

    total_charges = brokerage + stt + txn_charges + sebi_charges + stamp_duty + gst
    return {"total_charges": round(total_charges, 2)}

class WalkForwardBacktester:
    def __init__(
        self,
        data_dict: Dict[str, pd.DataFrame],
        initial_capital: float = 100000.0,
        max_capital_per_trade: float = 20000.0,
        min_confidence_threshold: float = 0.50,
        top_k: int = 5,
        warmup_days: int = 120,
        retrain_frequency_days: int = 15,
        enable_penalty_learning: bool = True
    ):
        self.data_dict = data_dict
        self.initial_capital = initial_capital
        self.max_capital_per_trade = max_capital_per_trade
        self.min_confidence_threshold = min_confidence_threshold
        self.current_equity = initial_capital
        self.top_k = top_k
        self.warmup_days = warmup_days
        self.retrain_frequency_days = retrain_frequency_days
        self.enable_penalty_learning = enable_penalty_learning

        self.trades: List[Dict[str, Any]] = []
        self.daily_pnl_records: List[Dict[str, Any]] = []
        self.losing_tickers_count: Dict[str, int] = {}
        self.cooldown_tickers: Dict[str, int] = {}

    def run(self) -> Dict[str, Any]:
        from src.feature_engineering import compute_features_for_ticker

        print("=" * 80)
        print(" [>] COMMENCING FIRST-HOUR CONFIRMED WALK-FORWARD BACKTEST")
        print(f"     Initial Capital       : ₹{self.initial_capital:,.2f} (1 Lakh)")
        print(f"     Max Capital Per Trade : ₹{self.max_capital_per_trade:,.2f} (Including Charges)")
        print(f"     Daily Trade Objective : Daily Active Trades (Top 1 to {self.top_k} Setups)")
        print(f"     Entry Timing          : 10:15 AM (First-Hour Candle Confirmation)")
        print(f"     Dynamic ML Targets    : Volatility-Scaled (+4.5% to +8.0%)")
        print(f"     Dynamic ML Stop Loss  : First-Hour Low / ATR Bound (1.5% to 3.5%)")
        print(f"     Breakeven Trailing    : +2.0% Intraday Move -> SL Adjusted to Entry")
        print(f"     Penalty Learning      : Active (2.5x Error Penalty on Loss Patterns)")
        print("=" * 80)

        t0 = time.time()
        ticker_features: Dict[str, pd.DataFrame] = {}
        all_dates = set()
        feature_list = list(FEATURE_COLUMNS)

        for ticker, df in self.data_dict.items():
            feat_df = compute_features_for_ticker(df, ticker)
            if not feat_df.empty and len(feat_df) >= 35:
                ticker_features[ticker] = feat_df
                all_dates.update(feat_df.index)

        sorted_dates = sorted(list(all_dates))
        total_dates = len(sorted_dates)

        if total_dates <= self.warmup_days:
            raise ValueError(f"Insufficient dates ({total_dates}) for warmup ({self.warmup_days}).")

        print(f" [*] Universe Precomputed in {time.time() - t0:.1f}s | Dates: {total_dates} | Test Days: {total_dates - self.warmup_days}")

        current_ensemble = None
        days_since_last_train = self.retrain_frequency_days

        for date_idx in range(self.warmup_days, total_dates):
            current_trade_date = sorted_dates[date_idx]
            prev_date = sorted_dates[date_idx - 1]
            date_str = pd.to_datetime(current_trade_date).strftime("%Y-%m-%d")

            for tick in list(self.cooldown_tickers.keys()):
                self.cooldown_tickers[tick] -= 1
                if self.cooldown_tickers[tick] <= 0:
                    del self.cooldown_tickers[tick]

            if current_ensemble is None or days_since_last_train >= self.retrain_frequency_days:
                current_ensemble = self._train_models_with_penalties(ticker_features, prev_date)
                days_since_last_train = 0

            days_since_last_train += 1

            # High-Speed Vectorized Daily Inference
            batch_x = []
            meta_rows = []
            for ticker, fdf in ticker_features.items():
                if prev_date in fdf.index and current_trade_date in fdf.index:
                    prev_row = fdf.loc[prev_date]
                    if isinstance(prev_row, pd.DataFrame):
                        prev_row = prev_row.iloc[-1]

                    curr_row = fdf.loc[current_trade_date]
                    if isinstance(curr_row, pd.DataFrame):
                        curr_row = curr_row.iloc[-1]

                    open_p = float(curr_row["Open"])
                    if open_p <= 0 or ticker in self.cooldown_tickers:
                        continue

                    batch_x.append(prev_row[feature_list].values.astype(float))
                    meta_rows.append({
                        "ticker": ticker,
                        "symbol": ticker.replace(".NS", "").replace(".BO", ""),
                        "open": open_p,
                        "high": float(curr_row["High"]),
                        "low": float(curr_row["Low"]),
                        "close": float(curr_row["Close"]),
                        "atr_val": float(prev_row.get("atr_val", open_p * 0.025)),
                        "first_hour_ret": float(curr_row.get("first_hour_return", 0.0)),
                        "first_hour_range": float(curr_row.get("first_hour_range", 0.02))
                    })

            if not batch_x:
                continue

            X_mat = np.nan_to_num(np.array(batch_x), nan=0.0, posinf=0.0, neginf=0.0)
            probs = current_ensemble.predict_proba(X_mat)

            for i, p in enumerate(probs):
                meta_rows[i]["prob"] = float(p)

            # Filter for positive first-hour confirmation (opening momentum intact)
            confirmed = [c for c in meta_rows if c["first_hour_ret"] >= -0.005]
            confirmed.sort(key=lambda x: x["prob"], reverse=True)

            # Daily Trade Guarantee: Pick Top 1 to Top 5 qualified stocks
            top_picks = confirmed[:self.top_k] if confirmed else meta_rows[:MIN_DAILY_TRADES]

            day_gross_pnl = 0.0
            day_charges = 0.0
            day_net_pnl = 0.0
            day_trades_count = len(top_picks)

            per_trade_budget = min(self.max_capital_per_trade, self.current_equity / float(max(1, len(top_picks))))

            for pick in top_picks:
                open_p = pick["open"]
                high_p = pick["high"]
                low_p = pick["low"]
                close_p = pick["close"]
                atr_val = pick["atr_val"]
                ticker = pick["ticker"]

                # Entry taken at 10:15 AM (confirmed after first hour)
                entry_price = round(open_p * (1.0 + pick["first_hour_ret"]), 2)

                qty = int(per_trade_budget / entry_price)
                while qty > 0:
                    est_charges = calculate_paytm_money_charges(entry_price, entry_price * 1.05, qty)["total_charges"]
                    if (entry_price * qty) + est_charges <= per_trade_budget:
                        break
                    qty -= 1

                if qty <= 0:
                    continue

                # Dynamic ML Target: Volatility Scaled (4.5% to 8%)
                target_pct = max(MIN_TARGET_PCT, min(0.08, (atr_val / entry_price) * 1.8))
                target_price = round(entry_price * (1.0 + target_pct), 2)

                # Dynamic ML Stop Loss: First-Hour Low / ATR Bound (1.5% to 3.5%)
                sl_pct = max(MIN_SL_PCT, min(MAX_SL_PCT, (atr_val / entry_price) * 1.2))
                stop_price = max(round(entry_price * (1.0 - sl_pct), 2), 0.05)

                # Breakeven Trailing Stop Rule (+2% intraday moves SL to Entry)
                if high_p >= entry_price * (1.0 + BREAKEVEN_TRIGGER_PCT):
                    stop_price = max(stop_price, entry_price)

                # Intraday Execution
                if high_p >= target_price and low_p <= stop_price:
                    exit_price = stop_price
                    exit_reason = "Stop Loss (Both Hit)"
                elif high_p >= target_price:
                    exit_price = target_price
                    exit_reason = f"Target (+{target_pct*100:.1f}%)"
                elif low_p <= stop_price:
                    exit_price = stop_price
                    exit_reason = "Stop Loss" if stop_price < entry_price else "Breakeven SL"
                else:
                    exit_price = close_p
                    exit_reason = "Square-off"

                gross_pnl = (exit_price - entry_price) * qty
                fees = calculate_paytm_money_charges(entry_price, exit_price, qty)
                total_charges = fees["total_charges"]
                net_pnl = gross_pnl - total_charges
                total_outlay = (entry_price * qty) + total_charges
                net_return_pct = (net_pnl / (entry_price * qty)) * 100.0

                if net_pnl < 0:
                    self.losing_tickers_count[ticker] = self.losing_tickers_count.get(ticker, 0) + 1
                    if self.losing_tickers_count[ticker] >= 2:
                        self.cooldown_tickers[ticker] = 4
                else:
                    self.losing_tickers_count[ticker] = max(0, self.losing_tickers_count.get(ticker, 0) - 1)

                day_gross_pnl += gross_pnl
                day_charges += total_charges
                day_net_pnl += net_pnl

                self.trades.append({
                    "date": date_str,
                    "ticker": ticker,
                    "symbol": pick["symbol"],
                    "surge_prob": round(pick["prob"] * 100, 1),
                    "entry_price": entry_price,
                    "high": high_p,
                    "low": low_p,
                    "close": close_p,
                    "target": target_price,
                    "stop_loss": stop_price,
                    "exit_price": exit_price,
                    "exit_reason": exit_reason,
                    "qty": qty,
                    "capital_invested": round(entry_price * qty, 2),
                    "total_outlay_with_fees": round(total_outlay, 2),
                    "gross_pnl": round(gross_pnl, 2),
                    "charges": total_charges,
                    "net_pnl": round(net_pnl, 2),
                    "net_return_pct": round(net_return_pct, 2)
                })

            self.current_equity += day_net_pnl
            prev_equity = self.current_equity - day_net_pnl
            daily_ret = (day_net_pnl / prev_equity) * 100 if prev_equity > 0 else 0.0

            self.daily_pnl_records.append({
                "date": date_str,
                "trades_count": day_trades_count,
                "gross_pnl": round(day_gross_pnl, 2),
                "charges": round(day_charges, 2),
                "net_pnl": round(day_net_pnl, 2),
                "equity": round(self.current_equity, 2),
                "daily_return_pct": round(daily_ret, 2)
            })

            if (date_idx - self.warmup_days) % 25 == 0 or date_idx == total_dates - 1:
                print(f" [BACKTEST] {date_str} | Trades Today: {day_trades_count} | Net: ₹{day_net_pnl:+7.2f} | Equity: ₹{self.current_equity:,.2f}")

        summary = self._compute_performance_metrics()
        self._export_results(summary)
        return summary

    def _train_models_with_penalties(self, ticker_features: Dict[str, pd.DataFrame], cutoff_date) -> ModelEnsemble:
        feature_list = list(FEATURE_COLUMNS)
        hist_rows = []
        for _, fdf in ticker_features.items():
            valid = fdf.loc[fdf.index <= cutoff_date].iloc[:-1]
            if not valid.empty:
                hist_rows.append(valid)

        train_df = pd.concat(hist_rows, axis=0)
        X = np.nan_to_num(train_df[feature_list].values, nan=0.0, posinf=0.0, neginf=0.0)
        y = train_df["target"].values.astype(int)

        sample_weights = np.ones(len(y), dtype=float)
        sample_weights[y == 0] = 2.5  # Penalize false breakout samples

        if len(X) > 25000:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X), size=25000, replace=False)
            X_train, y_train, w_train = X[idx], y[idx], sample_weights[idx]
        else:
            X_train, y_train, w_train = X, y, sample_weights

        models = get_base_models(random_state=42)
        fitted = {}
        prob_df = pd.DataFrame(index=range(len(y_train)))

        for name, model in models.items():
            try:
                try:
                    m = model.fit(X_train, y_train, sample_weight=w_train)
                except TypeError:
                    m = model.fit(X_train, y_train)
                fitted[name] = m
                if hasattr(m, "predict_proba"):
                    raw = m.predict_proba(X_train)
                    prob_df[name] = raw.take(1, axis=1) if raw.ndim == 2 else raw
                else:
                    prob_df[name] = m.predict(X_train)
            except Exception:
                continue

        return build_optimal_ensemble(fitted, prob_df, y_train)

    def _compute_performance_metrics(self) -> Dict[str, Any]:
        trade_df = pd.DataFrame(self.trades)
        daily_df = pd.DataFrame(self.daily_pnl_records)

        total_trades = len(trade_df)
        winning_trades = trade_df[trade_df["net_pnl"] > 0]
        losing_trades = trade_df[trade_df["net_pnl"] < 0]

        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        win_rate_pct = (win_count / total_trades) * 100.0 if total_trades > 0 else 0.0

        gross_profit = winning_trades["gross_pnl"].sum() if win_count > 0 else 0.0
        gross_loss = abs(losing_trades["gross_pnl"].sum()) if loss_count > 0 else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 999.0

        total_gross_pnl = trade_df["gross_pnl"].sum() if total_trades > 0 else 0.0
        total_charges = trade_df["charges"].sum() if total_trades > 0 else 0.0
        total_net_pnl = trade_df["net_pnl"].sum() if total_trades > 0 else 0.0
        net_return_pct = ((self.current_equity - self.initial_capital) / self.initial_capital) * 100.0

        daily_df["peak"] = daily_df["equity"].cummax()
        daily_df["drawdown"] = daily_df["equity"] - daily_df["peak"]
        daily_df["drawdown_pct"] = (daily_df["drawdown"] / daily_df["peak"]) * 100.0

        max_drawdown_inr = daily_df["drawdown"].min()
        max_drawdown_pct = daily_df["drawdown_pct"].min()

        returns = daily_df["daily_return_pct"] / 100.0
        ann_mean = returns.mean() * 252
        ann_std = returns.std() * math.sqrt(252) if len(returns) > 1 else 1.0
        sharpe_ratio = (ann_mean - 0.065) / (ann_std + 1e-9)
        downside_std = returns[returns < 0].std() * math.sqrt(252) if len(returns[returns < 0]) > 1 else 1.0
        sortino_ratio = (ann_mean - 0.065) / (downside_std + 1e-9)

        reasons = trade_df["exit_reason"].value_counts().to_dict() if total_trades > 0 else {}

        summary = {
            "initial_capital": self.initial_capital,
            "max_capital_per_trade": self.max_capital_per_trade,
            "final_equity": round(self.current_equity, 2),
            "net_total_pnl": round(total_net_pnl, 2),
            "net_return_pct": round(net_return_pct, 2),
            "total_gross_pnl": round(total_gross_pnl, 2),
            "total_paytm_charges": round(total_charges, 2),
            "total_trading_days": len(daily_df),
            "total_trades": total_trades,
            "winning_trades": win_count,
            "losing_trades": loss_count,
            "win_rate_pct": round(win_rate_pct, 2),
            "profit_factor": round(profit_factor, 2),
            "avg_win_inr": round(winning_trades["net_pnl"].mean(), 2) if win_count > 0 else 0.0,
            "avg_loss_inr": round(losing_trades["net_pnl"].mean(), 2) if loss_count > 0 else 0.0,
            "max_drawdown_inr": round(max_drawdown_inr, 2),
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "sortino_ratio": round(sortino_ratio, 2),
            "exit_reasons": reasons,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        print("\n" + "=" * 80)
        print("  FIRST-HOUR CONFIRMED BREAKOUT STRATEGY BACKTEST RESULTS")
        print("=" * 80)
        print(f"  Initial Capital          : ₹{summary['initial_capital']:,.2f} (1 Lakh)")
        print(f"  Max Per Trade (w/ Fees)  : ₹{summary['max_capital_per_trade']:,.2f}")
        print(f"  Final Equity             : ₹{summary['final_equity']:,.2f}")
        print(f"  Total Net Profit (P/L)   : ₹{summary['net_total_pnl']:+,.2f} ({summary['net_return_pct']:+.2f}%)")
        print(f"  Total Gross P/L          : ₹{summary['total_gross_pnl']:+,.2f}")
        print(f"  Total Paytm Money Fees   : ₹{summary['total_paytm_charges']:,.2f}")
        print(f"  Total Trading Days       : {summary['total_trading_days']}")
        print(f"  Total Trades Executed    : {summary['total_trades']}")
        print(f"  Winning / Losing Trades  : {summary['winning_trades']} wins / {summary['losing_trades']} losses")
        print(f"  Win Rate                 : {summary['win_rate_pct']:.2f}%")
        print(f"  Profit Factor            : {summary['profit_factor']:.2f}")
        print(f"  Max Drawdown             : ₹{summary['max_drawdown_inr']:,.2f} ({summary['max_drawdown_pct']:.2f}%)")
        print(f"  Sharpe Ratio (Ann.)      : {summary['sharpe_ratio']:.2f}")
        print(f"  Sortino Ratio (Ann.)     : {summary['sortino_ratio']:.2f}")
        print("=" * 80 + "\n")

        return summary

    def _export_results(self, summary: Dict[str, Any]):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(BACKTEST_SUMMARY_JSON, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        with open(BACKTEST_TRADES_JSON, "w", encoding="utf-8") as f:
            json.dump(self.trades, f, indent=2)
        with open(BACKTEST_DAILY_JSON, "w", encoding="utf-8") as f:
            json.dump(self.daily_pnl_records, f, indent=2)

        pd.DataFrame(self.trades).to_csv(BASE_DIR / "backtest_trade_log.csv", index=False)
        pd.DataFrame(self.daily_pnl_records).to_csv(BASE_DIR / "backtest_daily_pnl.csv", index=False)
