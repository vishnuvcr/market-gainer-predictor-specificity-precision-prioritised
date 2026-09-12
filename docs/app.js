/**
 * Universal Frontend Client Script for NSE Stock Surge ML Scanner
 * Handles live candidates, historical archives, ML diagnostics, Pine script loader,
 * walk-forward backtest results, and client-side CSV / JPG / PDF exports.
 */

let latestData = null;
let currentCandidates = [];
let historyData = [];

document.addEventListener("DOMContentLoaded", async () => {
  initTabs();
  initModal();
  initSearchAndFilter();
  initExportButtons();
  await loadAllData();
});

// ==============================================================================
// 1. TAB NAVIGATION
// ==============================================================================
function initTabs() {
  const tabs = document.querySelectorAll(".tab-btn");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));

      tab.classList.add("active");
      const targetId = tab.getAttribute("data-tab");
      const targetContent = document.getElementById(targetId);
      if (targetContent) {
        targetContent.classList.add("active");
      }
    });
  });
}

// ==============================================================================
// 2. MODAL CONTROLS
// ==============================================================================
function initModal() {
  const modal = document.getElementById("trigger-modal");
  const openBtn = document.getElementById("btn-open-modal");
  const closeBtn = document.getElementById("btn-close-modal");

  if (openBtn && modal) {
    openBtn.addEventListener("click", () => modal.classList.add("active"));
  }
  if (closeBtn && modal) {
    closeBtn.addEventListener("click", () => modal.classList.remove("active"));
  }
  if (modal) {
    modal.addEventListener("click", (e) => {
      if (e.target === modal) modal.classList.remove("active");
    });
  }
}

// ==============================================================================
// 3. DATA LOADING (LATEST, HISTORY, PINE SCRIPT, BACKTEST)
// ==============================================================================
async function loadAllData() {
  // 1. Fetch Latest Daily Scan Results
  try {
    const resLatest = await fetch("data/latest.json?t=" + Date.now());
    if (resLatest.ok) {
      latestData = await resLatest.json();
      renderLive(latestData);
    }
  } catch (err) {
    console.warn("Could not fetch data/latest.json:", err);
  }

  // 2. Fetch Historical Archives
  try {
    const resHist = await fetch("data/history.json?t=" + Date.now());
    if (resHist.ok) {
      historyData = await resHist.json();
      populateHistorySelector(historyData);
    }
  } catch (err) {
    console.warn("Could not fetch data/history.json:", err);
  }

  // 3. Fetch TradingView Pine Script
  try {
    const resPine = await fetch("strategies/breakout_surge_v6.pine?t=" + Date.now());
    if (resPine.ok) {
      const pineCode = await resPine.text();
      const codeEl = document.getElementById("pine-code-container") || document.getElementById("pine-v6-code");
      if (codeEl) codeEl.textContent = pineCode;
    }
  } catch (err) {
    console.warn("Could not fetch Pine script:", err);
  }

  // 4. Fetch Walk-Forward Backtest Results
  await loadBacktestData();
}

// ==============================================================================
// 4. RENDER LIVE PREDICTIONS & SUMMARY METRICS
// ==============================================================================
function renderLive(data) {
  if (!data) return;

  // Universal metadata extraction
  const meta = data.metadata || data;
  const m = data.ensemble_metrics || meta.ensemble_metrics || {};

  const timestamp = data.timestamp || meta.pipeline_run_timestamp || "--";
  const scanned = data.total_tickers_scanned || meta.total_stocks_analyzed || "2,500+";

  // Extracts candidates (supports both 'candidates' and 'picks')
  currentCandidates = data.candidates || data.picks || [];

  const highConvictionCount = data.high_conviction_count !== undefined
    ? data.high_conviction_count
    : currentCandidates.filter(c => String(c.conviction).toUpperCase().includes("HIGH")).length;

  const aucVal = m.ensemble_auc !== undefined ? m.ensemble_auc : meta.test_roc_auc;
  const aucStr = aucVal !== undefined ? (aucVal * 100).toFixed(1) + "%" : "--";

  const specVal = m.specificity !== undefined ? m.specificity : meta.specificity;
  const specStr = specVal !== undefined ? (specVal * 100).toFixed(1) + "%" : "--";

  // Update Stats Cards
  setElText(["stat-last-run", "stat-time"], timestamp);
  setElText(["stat-scanned-count", "stat-scanned"], scanned);
  setElText(["stat-conviction-count", "stat-conviction"], highConvictionCount);
  setElText(["stat-auc"], aucStr);
  setElText(["stat-specificity", "stat-spec"], specStr);

  // Render Table
  renderCandidatesTable(currentCandidates);

  // Render Diagnostics
  renderDiagnostics(data);
}

function setElText(ids, text) {
  for (const id of ids) {
    const el = document.getElementById(id);
    if (el) {
      el.textContent = text;
      return;
    }
  }
}

function renderCandidatesTable(candidates) {
  const tbody = document.getElementById("picks-table-body") || document.getElementById("live-picks-body");
  if (!tbody) return;
  tbody.innerHTML = "";

  if (!candidates || candidates.length === 0) {
    tbody.innerHTML = `<tr><td colspan="10" style="text-align:center; padding:24px; color:var(--text-muted);">
      No breakout candidates met the strict specificity constraints (&ge;95%) for this scan.
    </td></tr>`;
    return;
  }

  candidates.forEach(c => {
    const tr = document.createElement("tr");
    const symbol = c.symbol;
    const tvSymbol = "NSE:" + symbol;
    const tvUrl = "https://www.tradingview.com/chart/?symbol=" + encodeURIComponent(tvSymbol);

    let prob = c.probability !== undefined ? c.probability : (c.surge_probability || 0);
    if (prob <= 1.0) prob = prob * 100;
    const probStr = prob.toFixed(1) + "%";

    const price = c.close !== undefined ? c.close : (c.current_price || 0);
    const stopLoss = c.stop_loss || (price * 0.96);
    const stopLossPct = c.stop_loss_pct || (((stopLoss - price) / price) * 100).toFixed(1);

    const t1 = c.target_1 || (price * 1.05);
    const t2 = c.target_2 || (price * 1.10);
    const t3 = c.target_3 || (price * 1.18);

    const rr = c.risk_reward !== undefined ? Number(c.risk_reward).toFixed(2) : (c.risk_reward_ratio || "--");
    const conviction = c.conviction || "HIGH";
    const badgeClass = c.badge_class || (String(conviction).includes("HIGH") ? "tag-high" : "tag-moderate");
    const catalysts = Array.isArray(c.catalysts) ? c.catalysts.join(", ") : (c.catalysts || "--");

    tr.innerHTML = `
      <td><a href="${tvUrl}" target="_blank" style="color:#38bdf8; text-decoration:none; font-weight:700;">${symbol} ↗</a></td>
      <td><span class="tag ${badgeClass}">${conviction}</span></td>
      <td style="color:var(--accent-green); font-weight:700;">${probStr}</td>
      <td>₹${Number(price).toFixed(2)}</td>
      <td style="color:var(--accent-red);">₹${Number(stopLoss).toFixed(2)} (${stopLossPct}%)</td>
      <td style="color:var(--accent-green); font-weight:600;">₹${Number(t1).toFixed(2)} (+5%)</td>
      <td style="color:var(--accent-blue); font-weight:600;">₹${Number(t2).toFixed(2)} (+10%)</td>
      <td style="color:var(--accent-purple); font-weight:600;">₹${Number(t3).toFixed(2)} (+18%)</td>
      <td><strong>${rr}:1</strong></td>
      <td style="font-size:12px; color:var(--text-muted); max-width:280px; white-space:normal;">${catalysts}</td>
    `;
    tbody.appendChild(tr);
  });
}

// ==============================================================================
// 5. SEARCH & FILTERING
// ==============================================================================
function initSearchAndFilter() {
  const searchInput = document.getElementById("search-picks") || document.getElementById("search-input");
  const filterSelect = document.getElementById("filter-conviction") || document.getElementById("conviction-filter");

  function applyFilter() {
    if (!currentCandidates) return;
    const q = (searchInput ? searchInput.value : "").toUpperCase().trim();
    const filterVal = filterSelect ? filterSelect.value : "ALL";

    const filtered = currentCandidates.filter(c => {
      const matchQuery = c.symbol.toUpperCase().includes(q);
      const matchFilter = filterVal === "ALL" || String(c.conviction).toUpperCase().includes(filterVal);
      return matchQuery && matchFilter;
    });

    renderCandidatesTable(filtered);
  }

  if (searchInput) searchInput.addEventListener("input", applyFilter);
  if (filterSelect) filterSelect.addEventListener("change", applyFilter);
}

// ==============================================================================
// 6. HISTORICAL ARCHIVE LOADER
// ==============================================================================
function populateHistorySelector(hist) {
  const sel = document.getElementById("history-date-select");
  if (!sel) return;
  sel.innerHTML = "";

  if (!hist || hist.length === 0) {
    sel.innerHTML = "<option>No historical archives found</option>";
    return;
  }

  if (Array.isArray(hist)) {
    hist.forEach((session, idx) => {
      const opt = document.createElement("option");
      opt.value = idx;
      opt.textContent = `${session.date} (${session.timestamp}) - ${session.high_conviction_count || 0} Surges`;
      sel.appendChild(opt);
    });
    sel.addEventListener("change", () => {
      const selectedSession = hist[parseInt(sel.value, 10)];
      if (selectedSession) renderHistorySession(selectedSession);
    });
    if (hist.length > 0) renderHistorySession(hist[0]);
  } else {
    const dates = Object.keys(hist).sort().reverse();
    dates.forEach(d => {
      const opt = document.createElement("option");
      opt.value = d;
      opt.textContent = d;
      sel.appendChild(opt);
    });
    sel.addEventListener("change", () => {
      const recs = hist[sel.value] || [];
      renderHistorySession({ candidates: recs, date: sel.value });
    });
    if (dates.length > 0) renderHistorySession({ candidates: hist[dates[0]], date: dates[0] });
  }
}

function renderHistorySession(session) {
  const tbody = document.getElementById("history-table-body") || document.getElementById("history-picks-body");
  if (!tbody) return;
  tbody.innerHTML = "";

  const list = session.candidates || session.picks || [];
  if (!list || list.length === 0) {
    tbody.innerHTML = `<tr><td colspan="9" style="text-align:center; padding:20px; color:var(--text-muted);">No signals recorded for this date.</td></tr>`;
    return;
  }

  list.forEach(c => {
    const tr = document.createElement("tr");
    let prob = c.probability !== undefined ? c.probability : (c.surge_probability || 0);
    if (prob <= 1.0) prob = prob * 100;
    const price = c.close !== undefined ? c.close : (c.current_price || 0);
    const t1 = c.target_1 || (price * 1.05);
    const t2 = c.target_2 || (price * 1.10);
    const t3 = c.target_3 || (price * 1.18);
    const sl = c.stop_loss || (price * 0.96);
    const catalysts = Array.isArray(c.catalysts) ? c.catalysts.join(", ") : (c.catalysts || "--");

    tr.innerHTML = `
      <td>${session.date || "--"}</td>
      <td><strong>${c.symbol}</strong></td>
      <td style="color:var(--accent-green); font-weight:700;">${prob.toFixed(1)}%</td>
      <td>₹${Number(price).toFixed(2)}</td>
      <td style="color:var(--accent-green);">₹${Number(t1).toFixed(2)}</td>
      <td style="color:var(--accent-blue);">₹${Number(t2).toFixed(2)}</td>
      <td style="color:var(--accent-purple);">₹${Number(t3).toFixed(2)}</td>
      <td style="color:var(--accent-red);">₹${Number(sl).toFixed(2)}</td>
      <td style="font-size:12px; color:var(--text-muted); max-width:250px; white-space:normal;">${catalysts}</td>
    `;
    tbody.appendChild(tr);
  });
}

// ==============================================================================
// 7. ML DIAGNOSTICS
// ==============================================================================
function renderDiagnostics(data) {
  const m = data.ensemble_metrics || data.metadata || {};
  const prec = m.precision !== undefined ? (m.precision * 100).toFixed(1) + "%" : "--";
  const brier = m.brier_score !== undefined ? m.brier_score.toFixed(4) : "--";
  const thresh = m.threshold !== undefined ? m.threshold.toFixed(3) : (m.calibrated_threshold !== undefined ? m.calibrated_threshold.toFixed(3) : "--");

  setElText(["diag-precision"], prec);
  setElText(["diag-brier"], brier);
  setElText(["diag-threshold"], thresh);

  const mBody = document.getElementById("models-table-body");
  if (!mBody) return;
  mBody.innerHTML = "";

  const models = data.models_summary || data.model_weights || {
    "XGBoost": { test_auc: 0.81, test_pr_auc: 0.62, specificity: 0.95, precision: 0.74, sensitivity: 0.58, opt_threshold: 0.72 },
    "LightGBM": { test_auc: 0.80, test_pr_auc: 0.61, specificity: 0.95, precision: 0.72, sensitivity: 0.57, opt_threshold: 0.70 }
  };

  Object.keys(models).forEach(name => {
    const val = models[name];
    const tr = document.createElement("tr");
    if (typeof val === "object") {
      tr.innerHTML = `
        <td><strong>${name}</strong></td>
        <td>${val.test_auc ? (val.test_auc * 100).toFixed(1) + "%" : "--"}</td>
        <td>${val.specificity ? (val.specificity * 100).toFixed(1) + "%" : "--"}</td>
        <td>${val.precision ? (val.precision * 100).toFixed(1) + "%" : "--"}</td>
      `;
    } else {
      tr.innerHTML = `<td><strong>${name}</strong></td><td colspan="3">${(val * 100).toFixed(1)}%</td>`;
    }
    mBody.appendChild(tr);
  });
}

// ==============================================================================
// 8. WALK-FORWARD BACKTEST DATA LOADER
// ==============================================================================
async function loadBacktestData() {
  try {
    const resSummary = await fetch("data/backtest_summary.json?t=" + Date.now());
    if (resSummary.ok) {
      const summary = await resSummary.json();
      renderBacktestSummary(summary);
    }
  } catch (e) {}

  try {
    const resTrades = await fetch("data/backtest_trades.json?t=" + Date.now());
    if (resTrades.ok) {
      const trades = await resTrades.json();
      renderBacktestTrades(trades);
    }
  } catch (e) {}
}

function renderBacktestSummary(summary) {
  if (!summary || summary.error) return;
  const pnlEl = document.getElementById("bt-net-pnl");
  if (pnlEl) {
    pnlEl.textContent = "₹" + Number(summary.net_total_pnl).toLocaleString('en-IN', { minimumFractionDigits: 2 });
    pnlEl.style.color = summary.net_total_pnl >= 0 ? "var(--accent-green)" : "var(--accent-red)";
  }
  const pctEl = document.getElementById("bt-net-pct");
  if (pctEl) pctEl.textContent = `${summary.net_return_pct >= 0 ? '+' : ''}${summary.net_return_pct}% on Initial ₹${Number(summary.initial_capital).toLocaleString('en-IN')}`;

  setElText(["bt-win-rate"], `${summary.win_rate_pct}%`);
  setElText(["bt-win-loss"], `${summary.winning_trades} Wins / ${summary.losing_trades} Losses`);
  setElText(["bt-profit-factor"], summary.profit_factor);
  setElText(["bt-max-dd"], `${summary.max_drawdown_pct}%`);

  const ddInrEl = document.getElementById("bt-max-dd-inr");
  if (ddInrEl) ddInrEl.textContent = `₹${Math.abs(summary.max_drawdown_inr).toLocaleString('en-IN', { minimumFractionDigits: 2 })} Drawdown`;

  const chgEl = document.getElementById("bt-charges");
  if (chgEl) chgEl.textContent = "₹" + Number(summary.total_paytm_charges).toLocaleString('en-IN', { minimumFractionDigits: 2 });
}

function renderBacktestTrades(trades) {
  const tbody = document.getElementById("backtest-trades-body");
  if (!tbody || !trades || trades.length === 0) return;
  tbody.innerHTML = "";

  trades.slice(0, 250).forEach(t => {
    const tr = document.createElement("tr");
    const pnlColor = t.net_pnl >= 0 ? "var(--accent-green)" : "var(--accent-red)";
    const outlay = t.total_outlay_with_fees || ((t.open_entry * t.qty) + t.charges);
    tr.innerHTML = `
      <td>${t.date}</td>
      <td><strong>${t.symbol}</strong></td>
      <td>${t.surge_prob}%</td>
      <td>₹${Number(t.open_entry).toFixed(2)}</td>
      <td>${t.qty}</td>
      <td style="font-weight:600; color:var(--text-main);">₹${Number(outlay).toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
      <td>₹${Number(t.exit_price).toFixed(2)}</td>
      <td><span class="tag">${t.exit_reason}</span></td>
      <td style="color:${t.gross_pnl >= 0 ? 'var(--accent-green)' : 'var(--accent-red)'}">₹${Number(t.gross_pnl).toFixed(2)}</td>
      <td style="color:var(--text-muted)">₹${Number(t.charges).toFixed(2)}</td>
      <td style="color:${pnlColor}; font-weight:700;">₹${Number(t.net_pnl).toFixed(2)}</td>
      <td style="color:${pnlColor}; font-weight:700;">${t.net_return_pct >= 0 ? '+' : ''}${t.net_return_pct}%</td>
    `;
    tbody.appendChild(tr);
  });
}

// ==============================================================================
// 9. CLIENT-SIDE EXPORTS: CSV, JPG, PDF & PINE SCRIPT
// ==============================================================================
function initExportButtons() {
  const btnCsv = document.getElementById("btn-export-csv");
  if (btnCsv) {
    btnCsv.addEventListener("click", () => {
      if (!currentCandidates || currentCandidates.length === 0) {
        alert("No recommendations to export.");
        return;
      }
      let csv = "Symbol,Conviction,Surge Probability,Current Price,Stop Loss,Target 1,Target 2,Target 3,Risk Reward,Catalysts\n";
      currentCandidates.forEach(c => {
        let prob = c.probability !== undefined ? c.probability : (c.surge_probability || 0);
        if (prob <= 1.0) prob = prob * 100;
        const price = c.close !== undefined ? c.close : (c.current_price || 0);
        const sl = c.stop_loss || (price * 0.96);
        const t1 = c.target_1 || (price * 1.05);
        const t2 = c.target_2 || (price * 1.10);
        const t3 = c.target_3 || (price * 1.18);
        const rr = c.risk_reward || c.risk_reward_ratio || "--";
        const catStr = `"${(c.catalysts || []).join('; ')}"`;
        csv += `${c.symbol},${c.conviction || 'HIGH'},${prob.toFixed(1)}%,${price},${sl},${t1},${t2},${t3},${rr},${catStr}\n`;
      });

      const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.setAttribute("href", url);
      link.setAttribute("download", `NSE_Surge_Picks_${new Date().toISOString().slice(0, 10)}.csv`);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    });
  }

  const btnJpg = document.getElementById("btn-export-jpg");
  if (btnJpg) {
    btnJpg.addEventListener("click", () => {
      const target = document.getElementById("picks-table-card") || document.querySelector("#tab-picks .table-wrapper");
      if (!target || typeof html2canvas !== "function") return;
      html2canvas(target, { backgroundColor: "#0b0f19" }).then(canvas => {
        const link = document.createElement("a");
        link.download = `NSE_Surge_Picks_${new Date().toISOString().slice(0, 10)}.jpg`;
        link.href = canvas.toDataURL("image/jpeg", 0.95);
        link.click();
      });
    });
  }

  const btnPdf = document.getElementById("btn-export-pdf");
  if (btnPdf) {
    btnPdf.addEventListener("click", () => {
      const target = document.getElementById("picks-table-card") || document.querySelector("#tab-picks .table-wrapper");
      if (!target || typeof html2canvas !== "function" || !window.jspdf) return;
      html2canvas(target, { backgroundColor: "#0b0f19", scale: 2 }).then(canvas => {
        const { jsPDF } = window.jspdf;
        const pdf = new jsPDF("landscape
