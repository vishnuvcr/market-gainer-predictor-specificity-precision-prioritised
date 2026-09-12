"""
Self-contained Walk-Forward Backtester for NSE Daily Breakout Strategy.
Accounts for:
  - Initial Capital: ₹1,00,000 (1 Lakh)
  - Max per Trade: ₹20,000 (strictly including Paytm Money statutory charges)
  - Capped Intraday Profit Target: +5.0%
  - Dynamic ATR Stop Loss: min(1.5 * ATR, 4% hard stop)
  - Top 5 stocks daily
"""
import argparse
import sys
import time
import math
import json
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd

from src.config import (
    BASE_DIR,
    FEATURE_COLUMNS,
    ATR_SL_MULTIPLIER,
    MAX_SL_PCT,
    DATA_DIR,
    TICKERS_FILE
)
from src.downloader import download_all_shards, load_tickers_from_file
from src.feature_engineering import compute_features_for_ticker
from src.models import get_base_models
from src.ensembler import ModelEnsemble, build_optimal_ensemble


def calculate_paytm_money_charges(buy_price: float, sell_price: float, qty: int) -> Dict[str, float]:
    """Computes exact Paytm Money statutory and brokerage charges for NSE Equity Intraday."""
    if qty <= 0:
        return {
            "buy_turnover": 0.0, "sell_turnover": 0.0, "total_turnover": 0.0,
            "brokerage": 0.0, "stt": 0.0, "txn_charges": 0.0, "sebi_charges": 0.0,
            "stamp_duty": 0.0, "gst": 0.0, "total_charges": 0.0
        }

    buy_turnover = float(buy_price * qty)
    sell_turnover = float(sell_price * qty)
    total_turnover = buy_turnover + sell_turnover

    # Brokerage: min(20, 0.05%) on buy + sell
    buy_brokerage = min(20.0, 0.0005 * buy_turnover)
    sell_brokerage = min(20.0, 0.0005 * sell_turnover)
    brokerage = buy_brokerage + sell_brokerage

    # STT: 0.025% on sell side
    stt = 0.00025 * sell_turnover

    # NSE Transaction Charges: 0.00297% on total turnover
    txn_charges = 0.0000297 * total_turnover

    # SEBI Turnover Charges: Rs 10 per crore
    sebi_charges = 0.000001 * total_turnover

    # Stamp Duty: 0.003% on buy side
    stamp_duty = 0.00003 * buy_turnover

    # GST: 18% on (Brokerage + Txn Charges + SEBI Fees)
    gst = 0.18 * (brokerage + txn_charges + sebi_charges)

    total_charges = brokerage + stt + txn_charges + sebi_charges + stamp_duty + gst

    return {
        "buy_turnover": round(buy_turnover, 2),
        "sell_turnover": round(sell_turnover, 2),
        "total_turnover": round(total_turnover, 2),
        "brokerage": round(brokerage, 2),
        "stt": round(stt, 2),
        "txn_charges": round(txn_charges, 2),
        "sebi_charges": round(sebi_charges, 2),
        "stamp_duty": round(stamp_duty, 2),
        "gst": round(gst, 2),
        "total_charges": round(total_charges, 2)
    }


class WalkForwardBacktester:
    def __init__(
        self,
        data_dict: Dict[str, pd.DataFrame],
        initial_capital: float = 100000.0,
        max_capital_per_trade: float = 20000.0,
        top_k: int = 5,
        target_cap_pct: float = 0.05,
        warmup_days: int = 120,
        retrain_frequency_days: int = 10
    ):
        self.data_dict = data_dict
        self.initial_capital = initial_capital
        self.max_capital_per_trade = max_capital_per_trade
        self.current_equity = initial_capital
        self.top_k = top_k
        self.target_cap_pct = target_cap_pct
        self.warmup_days = warmup_days
        self.retrain_frequency_days = retrain_frequency_days

        self.trades: List[Dict[str, Any]] = []
        self.daily_pnl_records: List[Dict[str, Any]] = []

    def run(self) -> Dict[str, Any]:
        print("=" * 80)
        print(" [>] COMMENCING WALK-FORWARD INTRADAY BACKTEST")
        print(f"     Initial Capital       : ₹{self.initial_capital:,.2f} (1 Lakh)")
        print(f"     Max Capital Per Trade : ₹{self.max_capital_per_trade:,.2f} (Including Charges)")
        print(f"     Max Stocks / Day      : {self.top_k} (Equal Capital Allocation)")
        print(f"     Profit Target Cap     : +{self.target_cap_pct * 100:.1f}% Intraday")
        print(f"     Retrain Cadence       : Every {self.retrain_frequency_days} Trading Days")
        print(f"     Fee Structure         : Paytm Money NSE Intraday Schedule")
        print("=" * 80)

        print(" [*] Precomputing indicators & technical features across universe...")
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
            raise ValueError(f"Insufficient dates ({total_dates}) for warm-up period ({self.warmup_days}).")

        print(f" [*] Total Dates: {total_dates} | Warm-up: {self.warmup_days} | Test Days: {total_dates - self.warmup_days}")

        current_ensemble = None
        days_since_last_train = self.retrain_frequency_days

        for date_idx in range(self.warmup_days, total_dates):
            current_trade_date = sorted_dates[date_idx]
            prev_date = sorted_dates[date_idx - 1]
            date_str = pd.to_datetime(current_trade_date).strftime("%Y-%m-%d")

            if current_ensemble is None or days_since_last_train >= self.retrain_frequency_days:
                print(f" [TRAINING] Walk-forward refit on historical data up to {prev_date.strftime('%Y-%m-%d')}...")
                current_ensemble = self._train_models_up_to(ticker_features, prev_date)
                days_since_last_train = 0

            days_since_last_train += 1

            scored_candidates = []
            for ticker, fdf in ticker_features.items():
                if prev_date in fdf.index and current_trade_date in fdf.index:
                    prev_row = fdf.loc[prev_date]
                    if isinstance(prev_row, pd.DataFrame):
                        prev_row = prev_row.iloc[-1]

                    x_vec = np.nan_to_num(prev_row[feature_list].values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
                    prob = float(current_ensemble.predict_proba(x_vec.reshape(1, -1))[0])

                    curr_row = fdf.loc[current_trade_date]
                    if isinstance(curr_row, pd.DataFrame):
                        curr_row = curr_row.iloc[-1]

                    open_p = float(curr_row["Open"])
                    high_p = float(curr_row["High"])
                    low_p = float(curr_row["Low"])
                    close_p = float(curr_row["Close"])
                    atr_val = float(prev_row.get("atr_val", open_p * 0.025))
                    if np.isnan(atr_val) or atr_val <= 0:
                        atr_val = open_p * 0.025

                    scored_candidates.append({
                        "ticker": ticker,
                        "symbol": ticker.replace(".NS", "").replace(".BO", ""),
                        "prob": prob,
                        "open": open_p,
                        "high": high_p,
                        "low": low_p,
                        "close": close_p,
                        "atr_val": atr_val
                    })

            if not scored_candidates:
                continue

            scored_candidates.sort(key=lambda x: x["prob"], reverse=True)
            top_picks = scored_candidates[:self.top_k]

            day_gross_pnl = 0.0
            day_charges = 0.0
            day_net_pnl = 0.0
            day_trades_count = 0

            per_trade_budget = min(self.max_capital_per_trade, self.current_equity / float(self.top_k))

            for pick in top_picks:
                open_p = pick["open"]
                high_p = pick["high"]
                low_p = pick["low"]
                close_p = pick["close"]
                atr_val = pick["atr_val"]

                if open_p <= 0 or per_trade_budget <= 0:
                    continue

                qty = int(per_trade_budget / open_p)
                while qty > 0:
                    est_charges = calculate_paytm_money_charges(
                        open_p, open_p * (1.0 + self.target_cap_pct), qty
                    )["total_charges"]
                    if (open_p * qty) + est_charges <= per_trade_budget:
                        break
                    qty -= 1

                if qty <= 0:
                    continue

                day_trades_count += 1
                target_price = round(open_p * (1.0 + self.target_cap_pct), 2)
                sl_distance = min(ATR_SL_MULTIPLIER * atr_val, MAX_SL_PCT * open_p)
                stop_price = max(round(open_p - sl_distance, 2), 0.05)

                if high_p >= target_price and low_p <= stop_price:
                    exit_price = stop_price
                    exit_reason = "Stop Loss (Both Hit)"
                elif high_p >= target_price:
                    exit_price = target_price
                    exit_reason = "Target (+5% Cap)"
                elif low_p <= stop_price:
                    exit_price = stop_price
                    exit_reason = "Stop Loss"
                else:
                    exit_price = close_p
                    exit_reason = "Market Close Square-off"

                gross_pnl = (exit_price - open_p) * qty
                fees = calculate_paytm_money_charges(open_p, exit_price, qty)
                total_charges = fees["total_charges"]
                net_pnl = gross_pnl - total_charges
                total_outlay = (open_p * qty) + total_charges
                net_return_pct = (net_pnl / (open_p * qty)) * 100.0

                day_gross_pnl += gross_pnl
                day_charges += total_charges
                day_net_pnl += net_pnl

                self.trades.append({
                    "date": date_str,
                    "ticker": pick["ticker"],
                    "symbol": pick["symbol"],
                    "surge_prob": round(pick["prob"] * 100, 1),
                    "open_entry": open_p,
                    "high": high_p,
                    "low": low_p,
                    "close": close_p,
                    "target": target_price,
                    "stop_loss": stop_price,
                    "exit_price": exit_price,
                    "exit_reason": exit_reason,
                    "qty": qty,
                    "capital_invested": round(open_p * qty, 2),
                    "total_outlay_with_fees": round(total_outlay, 2),
                    "gross_pnl": round(gross_pnl, 2),
                    "charges": total_charges,
                    "net_pnl": round(net_pnl, 2),
                    "net_return_pct": round(net_return_pct, 2)
                })

            self.current_equity += day_net_pnl
            daily_return_pct = (day_net_pnl / (self.current_equity - day_net_pnl)) * 100 if (self.current_equity - day_net_pnl) > 0 else 0.0

            self.daily_pnl_records.append({
                "date": date_str,
                "trades_count": day_trades_count,
                "gross_pnl": round(day_gross_pnl, 2),
                "charges": round(day_charges, 2),
                "net_pnl": round(day_net_pnl, 2),
                "equity": round(self.current_equity, 2),
                "daily_return_pct": round(daily_return_pct, 2)
            })

            if (date_idx - self.warmup_days) % 15 == 0 or date_idx == total_dates - 1:
                print(f" [BACKTEST] {date_str} | Trades: {day_trades_count} | Day Net: ₹{day_net_pnl:+7.2f} | Equity: ₹{self.current_equity:,.2f}")

        summary = self._compute_performance_metrics()
        self._export_results(summary)
        return summary

    def _train_models_up_to(self, ticker_features: Dict[str, pd.DataFrame], cutoff_date) -> ModelEnsemble:
        feature_list = list(FEATURE_COLUMNS)
        hist_rows = []
        for _, fdf in ticker_features.items():
            valid_hist = fdf.loc[fdf.index <= cutoff_date].iloc[:-1]
            if not valid_hist.empty:
                hist_rows.append(valid_hist)

        train_df = pd.concat(hist_rows, axis=0)
        X = np.nan_to_num(train_df[feature_list].values, nan=0.0, posinf=0.0, neginf=0.0)
        y = train_df["target"].values.astype(int)

        if len(X) > 35000:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X), size=35000, replace=False)
            X_train, y_train = X[idx], y[idx]
        else:
            X_train, y_train = X, y

        models = get_base_models(random_state=42)
        fitted = {}
        prob_df = pd.DataFrame(index=range(len(y_train)))

        for name, model in models.items():
            try:
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

        gross_profit = winning_trades["gross_pnl"].sum()
        gross_loss = abs(losing_trades["gross_pnl"].sum())
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 999.0

        total_gross_pnl = trade_df["gross_pnl"].sum()
        total_charges = trade_df["charges"].sum()
        total_net_pnl = trade_df["net_pnl"].sum()
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

        reasons = trade_df["exit_reason"].value_counts().to_dict()

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
        print("  NSE INTRADAY BREAKOUT STRATEGY BACKTEST RESULTS")
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
        with open(DATA_DIR / "backtest_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        with open(DATA_DIR / "backtest_trades.json", "w", encoding="utf-8") as f:
            json.dump(self.trades, f, indent=2)
        with open(DATA_DIR / "backtest_daily.json", "w", encoding="utf-8") as f:
            json.dump(self.daily_pnl_records, f, indent=2)

        pd.DataFrame(self.trades).to_csv(BASE_DIR / "backtest_trade_log.csv", index=False)
        pd.DataFrame(self.daily_pnl_records).to_csv(BASE_DIR / "backtest_daily_pnl.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Run Walk-Forward Backtest for Stock Surge ML")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tickers")
    parser.add_argument("--capital", type=float, default=100000.0, help="Initial capital in INR")
    parser.add_argument("--max-per-trade", type=float, default=20000.0, help="Max capital per trade")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top stocks daily")
    parser.add_argument("--target-cap", type=float, default=0.05, help="Profit target cap")
    parser.add_argument("--warmup-days", type=int, default=120, help="Warm-up training days")
    parser.add_argument("--retrain-days", type=int, default=10, help="Retrain frequency days")
    args = parser.parse_args()

    start_time = time.time()
    tickers = load_tickers_from_file(TICKERS_FILE)
    if args.limit and args.limit > 0:
        tickers = tickers[:args.limit]
        print(f" [*] Running backtest with limited universe of {args.limit} tickers.")

    data_dict = download_all_shards(tickers)
    if len(data_dict) < 5:
        print(" [!] Insufficient ticker data downloaded. Exiting.")
        sys.exit(1)

    backtester = WalkForwardBacktester(
        data_dict=data_dict,
        initial_capital=args.capital,
        max_capital_per_trade=args.max_per_trade,
        top_k=args.top_k,
        target_cap_pct=args.target_cap,
        warmup_days=args.warmup_days,
        retrain_frequency_days=args.retrain_days
    )

    summary = backtester.run()
    elapsed = time.time() - start_time
    print(f" [SUCCESS] Backtest completed in {elapsed:.2f} seconds ({elapsed/60:.1f} minutes).")


if __name__ == "__main__":
    main()
