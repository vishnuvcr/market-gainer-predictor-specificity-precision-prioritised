/**
 * Frontend client script for NSE Stock Surge ML Scanner
 * Handles live predictions, historical archive, ML diagnostics, Pine script loader,
 * walk-forward backtest results, and client-side CSV / JPG / PDF exports.
 */

let currentPicks = [];
let allHistoryData = {};
let latestMetadata = {};

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initModal();
  initSearchAndFilter();
  initExportButtons();
  loadData();
});

// ==============================================================================
// TAB SWITCHING LOGIC
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
// MODAL CONTROLS
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
// DATA FETCHING AND RENDERING
// ==============================================================================
async function loadData() {
  try {
    const resLatest = await fetch("data/latest.json?t=" + Date.now());
    if (resLatest.ok) {
      const data = await resLatest.json();
      latestMetadata = data.metadata || {};
      currentPicks = data.picks || [];
      renderTopCards(latestMetadata, currentPicks);
      renderPicksTable(currentPicks);
      renderDiagnostics(latestMetadata);
    }
  } catch (err) {
    console.warn("Could not load data/latest.json:", err);
  }

  try {
    const resHistory = await fetch("data/history.json?t=" + Date.now());
    if (resHistory.ok) {
      allHistoryData = await resHistory.json();
      populateHistoryDropdown(allHistoryData);
    }
  } catch (err) {
    console.warn("Could not load data/history.json:", err);
  }

  try {
    const resPine = await fetch("strategies/breakout_surge_v6.pine?t=" + Date.now());
    if (resPine.ok) {
      const pineCode = await resPine.text();
      const codeEl = document.getElementById("pine-code-container");
      if (codeEl) codeEl.textContent = pineCode;
    }
  } catch (err) {
    console.warn("Could not load Pine script:", err);
  }

  // Load Walk-Forward Backtest Results
  await loadBacktestData();
}

function renderTopCards(meta, picks) {
  document.getElementById("stat-last-run").textContent = meta.pipeline_run_timestamp || "Just Now";
  document.getElementById("stat-scanned-count").textContent = meta.total_stocks_analyzed || "2,500+";
  
  const highCount = picks.filter(p => p.conviction === "HIGH").length;
  document.getElementById("stat-conviction-count").textContent = highCount;
  
  const auc = meta.test_roc_auc ? (meta.test_roc_auc * 100).toFixed(1) + "%" : "--";
  document.getElementById("stat-auc").textContent = auc;
  
  const spec = meta.specificity ? (meta.specificity * 100).toFixed(1) + "%" : "--";
  document.getElementById("stat-specificity").textContent = spec;
}

function renderPicksTable(picks) {
  const tbody = document.getElementById("picks-table-body");
  if (!tbody) return;
  tbody.innerHTML = "";

  if (!picks || picks.length === 0) {
    tbody.innerHTML = `<tr><td colspan="10" style="text-align:center; padding:24px; color:var(--text-muted);">
      No high-probability breakout candidates detected for tomorrow under current strict specificity constraints (&ge;95%).
    </td></tr>`;
    return;
  }

  picks.forEach(p => {
    const tr = document.createElement("tr");
    const tagClass = p.conviction === "HIGH" ? "tag-high" : "tag-moderate";
    const catalysts = Array.isArray(p.catalysts) ? p.catalysts.join(", ") : (p.catalysts || "--");

    tr.innerHTML = `
      <td><strong>${p.symbol}</strong></td>
      <td><span class="tag ${tagClass}">${p.conviction}</span></td>
      <td style="color:var(--accent-green); font-weight:700;">${p.surge_probability}%</td>
      <td>₹${Number(p.current_price).toFixed(2)}</td>
      <td style="color:var(--accent-red);">₹${Number(p.stop_loss).toFixed(2)}</td>
      <td style="color:var(--accent-green); font-weight:600;">₹${Number(p.target_1).toFixed(2)}</td>
      <td style="color:var(--accent-green); font-weight:600;">₹${Number(p.target_2).toFixed(2)}</td>
      <td style="color:var(--accent-green); font-weight:600;">₹${Number(p.target_3).toFixed(2)}</td>
      <td>${p.risk_reward_ratio}</td>
      <td style="font-size:12px; color:var(--text-muted); max-width:300px; white-space:normal;">${catalysts}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderDiagnostics(meta) {
  const prec = meta.precision ? (meta.precision * 100).toFixed(1) + "%" : "--";
  const brier = meta.brier_score !== undefined ? meta.brier_score.toFixed(4) : "--";
  const thresh = meta.calibrated_threshold !== undefined ? meta.calibrated_threshold.toFixed(4) : "--";

  const pEl = document.getElementById("diag-precision");
  if (pEl) pEl.textContent = prec;
  const bEl = document.getElementById("diag-brier");
  if (bEl) bEl.textContent = brier;
  const tEl = document.getElementById("diag-threshold");
  if (tEl) tEl.textContent = thresh;

  const mBody = document.getElementById("models-table-body");
  if (!mBody) return;
  mBody.innerHTML = "";

  const weights = meta.model_weights || {
    "XGBoost Classifier": 0.35,
    "LightGBM Classifier": 0.35,
    "CatBoost Classifier": 0.20,
    "Random Forest Classifier": 0.10
  };

  Object.entries(weights).forEach(([modelName, weight]) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>${modelName}</strong></td>
      <td>${(weight * 100).toFixed(1)}%</td>
    `;
    mBody.appendChild(tr);
  });
}

function populateHistoryDropdown(history) {
  const select = document.getElementById("history-date-select");
  if (!select) return;
  select.innerHTML = "";

  const dates = Object.keys(history).sort().reverse();
  if (dates.length === 0) {
    select.innerHTML = `<option value="">No historical archives found</option>`;
    return;
  }

  dates.forEach(d => {
    const opt = document.createElement("option");
    opt.value = d;
    opt.textContent = d;
    select.appendChild(opt);
  });

  select.addEventListener("change", () => {
    renderHistoryTable(select.value);
  });

  if (dates.length > 0) {
    renderHistoryTable(dates[0]);
  }
}

function renderHistoryTable(dateStr) {
  const tbody = document.getElementById("history-table-body");
  if (!tbody || !allHistoryData[dateStr]) return;

  tbody.innerHTML = "";
  const records = allHistoryData[dateStr];

  if (!records || records.length === 0) {
    tbody.innerHTML = `<tr><td colspan="9" style="text-align:center; padding:20px; color:var(--text-muted);">No signals recorded for ${dateStr}.</td></tr>`;
    return;
  }

  records.forEach(r => {
    const tr = document.createElement("tr");
    const catalysts = Array.isArray(r.catalysts) ? r.catalysts.join(", ") : (r.catalysts || "--");
    tr.innerHTML = `
      <td>${dateStr}</td>
      <td><strong>${r.symbol}</strong></td>
      <td style="color:var(--accent-green); font-weight:700;">${r.surge_probability}%</td>
      <td>₹${Number(r.current_price).toFixed(2)}</td>
      <td style="color:var(--accent-green);">₹${Number(r.target_1).toFixed(2)}</td>
      <td style="color:var(--accent-green);">₹${Number(r.target_2).toFixed(2)}</td>
      <td style="color:var(--accent-green);">₹${Number(r.target_3).toFixed(2)}</td>
      <td style="color:var(--accent-red);">₹${Number(r.stop_loss).toFixed(2)}</td>
      <td style="font-size:12px; color:var(--text-muted); max-width:250px; white-space:normal;">${catalysts}</td>
    `;
    tbody.appendChild(tr);
  });
}

// ==============================================================================
// SEARCH AND FILTER
// ==============================================================================
function initSearchAndFilter() {
  const searchInput = document.getElementById("search-picks");
  const filterSelect = document.getElementById("filter-conviction");

  function applyFilter() {
    const query = (searchInput.value || "").toUpperCase().trim();
    const filterVal = filterSelect.value;

    const filtered = currentPicks.filter(p => {
      const matchQuery = p.symbol.toUpperCase().includes(query);
      const matchFilter = filterVal === "ALL" || p.conviction === filterVal;
      return matchQuery && matchFilter;
    });

    renderPicksTable(filtered);
  }

  if (searchInput) searchInput.addEventListener("input", applyFilter);
  if (filterSelect) filterSelect.addEventListener("change", applyFilter);
}

// ==============================================================================
// CLIENT-SIDE EXPORTS: CSV, JPG, PDF
// ==============================================================================
function initExportButtons() {
  const btnCsv = document.getElementById("btn-export-csv");
  if (btnCsv) {
    btnCsv.addEventListener("click", () => {
      if (!currentPicks || currentPicks.length === 0) {
        alert("No recommendations to export.");
        return;
      }
      let csv = "Symbol,Conviction,Surge Probability,Current Price,Stop Loss,Target 1,Target 2,Target 3,Risk Reward,Catalysts\n";
      currentPicks.forEach(p => {
        const catStr = `"${(p.catalysts || []).join('; ')}"`;
        csv += `${p.symbol},${p.conviction},${p.surge_probability}%,${p.current_price},${p.stop_loss},${p.target_1},${p.target_2},${p.target_3},${p.risk_reward_ratio},${catStr}\n`;
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
      const target = document.getElementById("picks-table-card");
      if (!target) return;
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
      const target = document.getElementById("picks-table-card");
      if (!target) return;
      html2canvas(target, { backgroundColor: "#0b0f19", scale: 2 }).then(canvas => {
        const { jsPDF } = window.jspdf;
        const pdf = new jsPDF("landscape", "pt", "a4");
        const imgData = canvas.toDataURL("image/png");
        const imgProps = pdf.getImageProperties(imgData);
        const pdfWidth = pdf.internal.pageSize.getWidth();
        const pdfHeight = (imgProps.height * pdfWidth) / imgProps.width;
        pdf.addImage(imgData, "PNG", 20, 20, pdfWidth - 40, pdfHeight);
        pdf.save(`NSE_Surge_Picks_${new Date().toISOString().slice(0, 10)}.pdf`);
      });
    });
  }

  const btnCopyPine = document.getElementById("btn-copy-pine");
  if (btnCopyPine) {
    btnCopyPine.addEventListener("click", () => {
      const code = document.getElementById("pine-code-container").textContent;
      navigator.clipboard.writeText(code).then(() => {
        btnCopyPine.textContent = "✓ Copied to Clipboard!";
        setTimeout(() => {
          btnCopyPine.textContent = "📋 Copy Script to Clipboard";
        }, 2000);
      });
    });
  }
}

// ==============================================================================
// WALK-FORWARD BACKTEST DATA LOADER
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
    pnlEl.textContent = "₹" + Number(summary.net_total_pnl).toLocaleString('en-IN', {minimumFractionDigits: 2});
    pnlEl.style.color = summary.net_total_pnl >= 0 ? "var(--accent-green)" : "var(--accent-red)";
  }
  const pctEl = document.getElementById("bt-net-pct");
  if (pctEl) pctEl.textContent = `${summary.net_return_pct >= 0 ? '+' : ''}${summary.net_return_pct}% on Initial ₹${summary.initial_capital.toLocaleString('en-IN')}`;
  
  const wrEl = document.getElementById("bt-win-rate");
  if (wrEl) wrEl.textContent = `${summary.win_rate_pct}%`;
  
  const wlEl = document.getElementById("bt-win-loss");
  if (wlEl) wlEl.textContent = `${summary.winning_trades} Wins / ${summary.losing_trades} Losses`;
  
  const pfEl = document.getElementById("bt-profit-factor");
  if (pfEl) pfEl.textContent = summary.profit_factor;
  
  const ddEl = document.getElementById("bt-max-dd");
  if (ddEl) ddEl.textContent = `${summary.max_drawdown_pct}%`;
  
  const ddInrEl = document.getElementById("bt-max-dd-inr");
  if (ddInrEl) ddInrEl.textContent = `₹${Math.abs(summary.max_drawdown_inr).toLocaleString('en-IN', {minimumFractionDigits: 2})} Drawdown`;
  
  const chgEl = document.getElementById("bt-charges");
  if (chgEl) chgEl.textContent = "₹" + Number(summary.total_paytm_charges).toLocaleString('en-IN', {minimumFractionDigits: 2});
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
      <td>₹${t.open_entry.toFixed(2)}</td>
      <td>${t.qty}</td>
      <td style="font-weight:600; color:var(--text-main);">₹${outlay.toLocaleString('en-IN', {minimumFractionDigits:2, maximumFractionDigits:2})}</td>
      <td>₹${t.exit_price.toFixed(2)}</td>
      <td><span class="tag">${t.exit_reason}</span></td>
      <td style="color:${t.gross_pnl >= 0 ? 'var(--accent-green)' : 'var(--accent-red)'}">₹${t.gross_pnl.toFixed(2)}</td>
      <td style="color:var(--text-muted)">₹${t.charges.toFixed(2)}</td>
      <td style="color:${pnlColor}; font-weight:700;">₹${t.net_pnl.toFixed(2)}</td>
      <td style="color:${pnlColor}; font-weight:700;">${t.net_return_pct >= 0 ? '+' : ''}${t.net_return_pct}%</td>
    `;
    tbody.appendChild(tr);
  });
}
  
