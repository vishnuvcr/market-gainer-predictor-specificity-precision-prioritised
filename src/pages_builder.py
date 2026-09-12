"""
Pages builder module generating interactive GitHub Pages web dashboard,
updating latest.json, maintaining timestamped history.json archive,
and embedding the Walk-Forward Backtest & Paytm P/L analysis tab.
"""
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from src.config import (
    DOCS_DIR,
    DATA_DIR,
    LATEST_JSON,
    HISTORY_JSON,
    PINE_FILE
)

def build_github_pages(
    candidates: List[Dict[str, Any]],
    ensemble_metrics: Dict[str, Any],
    model_eval_results: Dict[str, Dict[str, Any]],
    feat_imp_df,
    total_tickers: int
) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    current_time_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    top_features = []
    if feat_imp_df is not None and not feat_imp_df.empty:
        for _, row in feat_imp_df.head(10).iterrows():
            top_features.append({
                "feature": str(row["feature"]),
                "importance": float(row["mean_importance"])
            })
            
    run_payload = {
        "timestamp": current_time_iso,
        "date": current_date,
        "total_tickers_scanned": total_tickers,
        "total_candidates": len(candidates),
        "high_conviction_count": sum(1 for c in candidates if "HIGH" in str(c.get("conviction", ""))),
        "ensemble_metrics": ensemble_metrics,
        "models_summary": model_eval_results,
        "top_features": top_features,
        "candidates": candidates
    }
    
    with open(LATEST_JSON, "w", encoding="utf-8") as f:
        json.dump(run_payload, f, indent=2)
    print(f" [PAGES] Written latest scan to: {LATEST_JSON}")
    
    history_records = []
    if HISTORY_JSON.exists():
        try:
            with open(HISTORY_JSON, "r", encoding="utf-8") as f:
                history_records = json.load(f)
                if not isinstance(history_records, list):
                    history_records = []
        except Exception:
            history_records = []
            
    history_records = [run_payload] + [h for h in history_records if h.get("date") != current_date]
    history_records = history_records[:60]
    
    with open(HISTORY_JSON, "w", encoding="utf-8") as f:
        json.dump(history_records, f, indent=2)
    print(f" [PAGES] Updated historical session archive ({len(history_records)} snapshots) at: {HISTORY_JSON}")
    
    if PINE_FILE.exists():
        docs_strat_dir = DOCS_DIR / "strategies"
        docs_strat_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(PINE_FILE, docs_strat_dir / PINE_FILE.name)
        
    _write_dashboard_files(DOCS_DIR)

def _write_dashboard_files(docs_dir: Path) -> None:
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>NSE Stock Surge ML Scanner | High Specificity 5-20% Breakout Engine</title>
  <style>
:root {
  --bg-main: #0b0f19;
  --bg-card: #151c2e;
  --bg-card-hover: #1c263d;
  --border-color: #232f48;
  --text-main: #f8fafc;
  --text-muted: #94a3b8;
  --accent-green: #10b981;
  --accent-blue: #3b82f6;
  --accent-purple: #8b5cf6;
  --accent-red: #ef4444;
  --accent-amber: #f59e0b;
}

* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
}

body {
  background-color: var(--bg-main);
  color: var(--text-main);
  line-height: 1.5;
  padding-bottom: 60px;
}

.header {
  background: linear-gradient(180deg, #131b2e 0%, #0b0f19 100%);
  border-bottom: 1px solid var(--border-color);
  padding: 24px 32px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 16px;
}

.title-area h1 {
  font-size: 26px;
  font-weight: 800;
  color: #38bdf8;
  display: flex;
  align-items: center;
  gap: 12px;
}

.title-area p {
  color: var(--text-muted);
  font-size: 14px;
  margin-top: 4px;
}

.header-actions {
  display: flex;
  gap: 12px;
  align-items: center;
}

.btn-trigger {
  background: linear-gradient(135deg, #059669 0%, #10b981 100%);
  color: #ffffff;
  border: none;
  padding: 10px 18px;
  border-radius: 6px;
  font-weight: 600;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 8px;
  transition: all 0.2s ease;
  text-decoration: none;
}

.btn-trigger:hover {
  filter: brightness(1.1);
  transform: translateY(-1px);
}

.status-badge {
  background: rgba(16, 185, 129, 0.1);
  color: var(--accent-green);
  border: 1px solid rgba(16, 185, 129, 0.2);
  padding: 6px 12px;
  border-radius: 20px;
  font-size: 12px;
  font-weight: 600;
  display: inline-flex;
  align-items: center;
  gap: 6px;
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background-color: var(--accent-green);
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
  70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
  100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}

.container {
  max-width: 1400px;
  margin: 0 auto;
  padding: 24px;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 16px;
  margin-bottom: 24px;
}

.stat-card {
  background-color: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: 8px;
  padding: 18px;
  transition: transform 0.2s ease;
}

.stat-card:hover {
  transform: translateY(-2px);
  background-color: var(--bg-card-hover);
}

.stat-card .label {
  color: var(--text-muted);
  font-size: 13px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.stat-card .value {
  font-size: 24px;
  font-weight: 700;
  margin-top: 6px;
  color: var(--text-main);
}

.nav-tabs {
  display: flex;
  border-bottom: 1px solid var(--border-color);
  margin-bottom: 20px;
  gap: 8px;
  overflow-x: auto;
}

.tab-btn {
  background: none;
  border: none;
  color: var(--text-muted);
  padding: 12px 20px;
  font-size: 15px;
  font-weight: 600;
  cursor: pointer;
  border-bottom: 2px solid transparent;
  transition: all 0.2s ease;
  white-space: nowrap;
}

.tab-btn:hover {
  color: var(--text-main);
}

.tab-btn.active {
  color: #38bdf8;
  border-bottom-color: #38bdf8;
}

.tab-content {
  display: none;
}

.tab-content.active {
  display: block;
}

.controls-bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 16px;
  margin-bottom: 16px;
}

.search-group {
  display: flex;
  gap: 12px;
  flex: 1;
  max-width: 500px;
}

.search-input {
  background-color: var(--bg-card);
  border: 1px solid var(--border-color);
  color: var(--text-main);
  padding: 10px 14px;
  border-radius: 6px;
  width: 100%;
  font-size: 14px;
}

.search-input:focus {
  outline: none;
  border-color: #38bdf8;
}

.filter-select {
  background-color: var(--bg-card);
  border: 1px solid var(--border-color);
  color: var(--text-main);
  padding: 10px 14px;
  border-radius: 6px;
  font-size: 14px;
  cursor: pointer;
}

.export-group {
  display: flex;
  gap: 8px;
}

.btn-export {
  background-color: var(--bg-card);
  border: 1px solid var(--border-color);
  color: var(--text-muted);
  padding: 8px 14px;
  border-radius: 6px;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 6px;
  transition: all 0.2s ease;
  text-decoration: none;
}

.btn-export:hover {
  color: var(--text-main);
  border-color: var(--text-muted);
}

.table-wrapper {
  background-color: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: 8px;
  overflow-x: auto;
}

table {
  width: 100%;
  border-collapse: collapse;
  text-align: left;
  font-size: 14px;
}

th {
  background-color: rgba(21, 28, 46, 0.7);
  color: var(--text-muted);
  font-weight: 600;
  padding: 14px 16px;
  border-bottom: 1px solid var(--border-color);
  white-space: nowrap;
}

td {
  padding: 14px 16px;
  border-bottom: 1px solid rgba(35, 47, 72, 0.4);
  white-space: nowrap;
}

tr:hover td {
  background-color: var(--bg-card-hover);
}

.tag {
  display: inline-block;
  padding: 3px 8px;
  border-radius: 4px;
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
}

.tag-high {
  background: rgba(16, 185, 129, 0.15);
  color: var(--accent-green);
  border: 1px solid rgba(16, 185, 129, 0.3);
}

.tag-moderate {
  background: rgba(59, 130, 246, 0.15);
  color: var(--accent-blue);
  border: 1px solid rgba(59, 130, 246, 0.3);
}

.code-box {
  background-color: #0d1117;
  border: 1px solid var(--border-color);
  border-radius: 8px;
  padding: 16px;
  font-family: monospace;
  color: #e6edf3;
  overflow-x: auto;
  max-height: 500px;
  font-size: 13px;
  line-height: 1.6;
}

.modal {
  display: none;
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.7);
  justify-content: center;
  align-items: center;
  z-index: 1000;
}

.modal.active {
  display: flex;
}

.modal-content {
  background-color: var(--bg-card);
  border: 1px solid var(--border-color);
  padding: 24px;
  border-radius: 8px;
  max-width: 500px;
  width: 90%;
}
  </style>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js"></script>
</head>
<body>

  <!-- Top Navigation Header -->
  <header class="header">
    <div class="title-area">
      <h1>NSE Breakout Surge ML Scanner</h1>
      <p>Automated Machine Learning Pipeline Targeting Tomorrow's &gt;5% to 20% Surges</p>
    </div>
    <div class="header-actions">
      <button class="btn-trigger" id="btn-open-modal">&#9654; Run Pipeline Directly</button>
      <span class="status-badge"><span class="status-dot"></span> ACTIVE & RUNNING</span>
    </div>
  </header>

  <div class="container">
    <!-- Summary Metrics Cards -->
    <div class="stats-grid">
      <div class="stat-card">
        <div class="label">Last Pipeline Execution</div>
        <div class="value" id="stat-last-run" style="font-size: 18px; color:#38bdf8;">--</div>
      </div>
      <div class="stat-card">
        <div class="label">Total Scanned Stocks</div>
        <div class="value" id="stat-scanned-count">--</div>
      </div>
      <div class="stat-card">
        <div class="label">High Conviction Surges</div>
        <div class="value" id="stat-conviction-count" style="color:var(--accent-green);">--</div>
      </div>
      <div class="stat-card">
        <div class="label">Model Test ROC-AUC</div>
        <div class="value" id="stat-auc" style="color:var(--accent-blue);">--%</div>
      </div>
      <div class="stat-card">
        <div class="label">Calibrated Specificity</div>
        <div class="value" id="stat-specificity" style="color:var(--accent-purple);">--%</div>
      </div>
    </div>

    <!-- Navigation Tabs -->
    <div class="nav-tabs">
      <button class="tab-btn active" data-tab="tab-picks">Tomorrow's Trades</button>
      <button class="tab-btn" data-tab="tab-history">Historical Archive</button>
      <button class="tab-btn" data-tab="tab-diagnostics">ML Diagnostics & Models</button>
      <button class="tab-btn" data-tab="tab-backtest">📊 Backtest & Paytm P/L</button>
      <button class="tab-btn" data-tab="tab-pine">TradingView Pine v6 Script</button>
    </div>

    <!-- TAB 1: Tomorrow's Picks -->
    <div id="tab-picks" class="tab-content active">
      <div class="controls-bar">
        <div class="search-group">
          <input type="text" class="search-input" id="search-picks" placeholder="Search Ticker (e.g. TATA, INFY)...">
          <select class="filter-select" id="filter-conviction">
            <option value="ALL">All Signals</option>
            <option value="HIGH">High Conviction (&ge;95% Specificity)</option>
            <option value="MODERATE">Moderate Conviction</option>
          </select>
        </div>
        <div class="export-group">
          <button class="btn-export" id="btn-export-csv">&#128229; Export CSV</button>
          <button class="btn-export" id="btn-export-jpg">&#128444; Export JPG</button>
          <button class="btn-export" id="btn-export-pdf">&#128196; Export PDF</button>
        </div>
      </div>

      <div class="table-wrapper" id="picks-table-card">
        <table>
          <thead>
            <tr>
              <th>Symbol</th>
              <th>Conviction</th>
              <th>Surge Prob</th>
              <th>Current Price</th>
              <th>Stop Loss (ATR)</th>
              <th>Target 1 (+5%)</th>
              <th>Target 2 (+10%)</th>
              <th>Target 3 (+18%)</th>
              <th>Risk:Reward</th>
              <th>Technical Catalysts</th>
            </tr>
          </thead>
          <tbody id="picks-table-body">
            <tr><td colspan="10" style="text-align:center; padding:20px; color:var(--text-muted);">Loading live scan data...</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- TAB 2: Historical Archive -->
    <div id="tab-history" class="tab-content">
      <div class="controls-bar">
        <div style="color:var(--text-muted); font-size:14px;">
          Select a historical scan date to inspect past trade setups and targets:
        </div>
        <select class="filter-select" id="history-date-select">
          <option value="">Loading archived dates...</option>
        </select>
      </div>

      <div class="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>Scan Date</th>
              <th>Symbol</th>
              <th>Surge Prob</th>
              <th>Entry Price</th>
              <th>Target 1 (+5%)</th>
              <th>Target 2 (+10%)</th>
              <th>Target 3 (+18%)</th>
              <th>Stop Loss</th>
              <th>Catalysts</th>
            </tr>
          </thead>
          <tbody id="history-table-body">
            <tr><td colspan="9" style="text-align:center; padding:20px; color:var(--text-muted);">Select a date to view past recommendations.</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- TAB 3: ML Diagnostics -->
    <div id="tab-diagnostics" class="tab-content">
      <div class="stats-grid">
        <div class="stat-card">
          <div class="label">Ensemble Precision</div>
          <div class="value" id="diag-precision">--%</div>
        </div>
        <div class="stat-card">
          <div class="label">Brier Loss Score</div>
          <div class="value" id="diag-brier">--</div>
        </div>
        <div class="stat-card">
          <div class="label">Decision Threshold</div>
          <div class="value" id="diag-threshold">--</div>
        </div>
      </div>

      <h3 style="margin: 20px 0 12px 0; font-size: 18px;">Ensemble Model Weights</h3>
      <div class="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>Model Name</th>
              <th>Individual Weight</th>
            </tr>
          </thead>
          <tbody id="models-table-body">
            <tr><td colspan="2" style="text-align:center; padding:20px; color:var(--text-muted);">Loading model weights...</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- TAB 4: Strategy Backtest & Paytm P/L -->
    <div id="tab-backtest" class="tab-content">
      <div class="controls-bar">
        <div style="color:var(--text-muted); font-size:14px;">
          <strong>Adaptive Walk-Forward Backtest</strong> | Capital: <strong>₹1,00,000</strong> | Max / Trade: <strong>₹20,000 (Incl. Fees)</strong> | 10:15 AM First-Hour Confirmation | Dynamic ML Targets
        </div>
        <div class="export-group">
          <a href="../backtest_trade_log.csv" download class="btn-export" id="btn-dl-trades">📥 Download Trade Log CSV</a>
          <a href="../backtest_daily_pnl.csv" download class="btn-export" id="btn-dl-daily">📈 Download Equity Curve CSV</a>
        </div>
      </div>

      <!-- Stat Cards -->
      <div class="stats-grid" style="margin-top:16px;">
        <div class="stat-card">
          <div class="label">Net Total P/L</div>
          <div class="value" id="bt-net-pnl" style="color:var(--accent-green);">--</div>
          <small id="bt-net-pct" style="color:var(--text-muted);">--% Return</small>
        </div>
        <div class="stat-card">
          <div class="label">Win Rate (Precision)</div>
          <div class="value" id="bt-win-rate">--%</div>
          <small id="bt-win-loss" style="color:var(--text-muted);">-- Wins / -- Losses</small>
        </div>
        <div class="stat-card">
          <div class="label">Profit Factor</div>
          <div class="value" id="bt-profit-factor" style="color:var(--accent-blue);">--</div>
          <small style="color:var(--text-muted);">Gross Gains / Losses</small>
        </div>
        <div class="stat-card">
          <div class="label">Max Drawdown</div>
          <div class="value" id="bt-max-dd" style="color:var(--accent-red);">--%</div>
          <small id="bt-max-dd-inr" style="color:var(--text-muted);">Peak to Trough</small>
        </div>
        <div class="stat-card">
          <div class="label">Total Paytm Charges</div>
          <div class="value" id="bt-charges" style="color:var(--accent-amber);">₹--</div>
          <small style="color:var(--text-muted);">Brokerage, STT, GST</small>
        </div>
      </div>

      <h3 style="margin: 20px 0 12px 0; font-size: 18px;">Executed Daily Trade Log (Confirmed at 10:15 AM)</h3>
      <div class="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>Date</th>
              <th>Symbol</th>
              <th>Surge Prob</th>
              <th>Entry (10:15 AM)</th>
              <th>Qty</th>
              <th>Total Outlay (w/ Fees)</th>
              <th>Exit Price</th>
              <th>Exit Reason</th>
              <th>Gross P/L</th>
              <th>Paytm Fees</th>
              <th>Net P/L</th>
              <th>Net Return</th>
            </tr>
          </thead>
          <tbody id="backtest-trades-body">
            <tr><td colspan="12" style="text-align:center; padding:20px; color:var(--text-muted);">Run the pipeline workflow with backtesting enabled to populate historical trade logs.</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- TAB 5: TradingView Pine Script v6 -->
    <div id="tab-pine" class="tab-content">
      <div class="controls-bar">
        <p style="color:var(--text-muted); font-size:14px;">
          Copy and paste this script directly into the <strong>TradingView Pine Editor</strong> for real-time chart overlays and circuit alerts:
        </p>
        <button class="btn-export" id="btn-copy-pine">&#128203; Copy Script to Clipboard</button>
      </div>
      <pre class="code-box" id="pine-code-container">// Loading Pine Script...</pre>
    </div>
  </div>

  <!-- Manual Trigger Modal -->
  <div class="modal" id="trigger-modal">
    <div class="modal-content">
      <h3 style="margin-bottom: 12px; font-size: 18px;">Unified Master Pipeline Control</h3>
      <p style="color:var(--text-muted); font-size: 14px; margin-bottom: 16px;">
        Trigger the master engine to train models, screen tomorrow's breakout candidates, and run adaptive backtests:
      </p>
      <ol style="color:var(--text-muted); font-size: 14px; margin-left: 20px; margin-bottom: 20px;">
        <li style="margin-bottom:8px;">Open your repository's <strong>Actions</strong> tab.</li>
        <li style="margin-bottom:8px;">Select <strong>NSE Breakout ML Master Engine</strong>.</li>
        <li style="margin-bottom:8px;">Choose what to execute (Training / Tomorrow Picks / Backtesting).</li>
        <li style="margin-bottom:8px;">Click <strong>Run workflow</strong>.</li>
      </ol>
      <div style="display: flex; gap: 12px; justify-content: flex-end;">
        <button class="btn-export" id="btn-close-modal">Close</button>
        <a href="https://github.com/vishnuvcr/market-gainer-predictor-specificity-precision-prioritised/actions" target="_blank" class="btn-trigger">Open GitHub Actions &rarr;</a>
      </div>
    </div>
  </div>

  <script src="app.js"></script>
</body>
</html>
"""
    with open(docs_dir / "index.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    print(" [PAGES] Regenerated docs/index.html with all 5 tabs and unified layouts.")
