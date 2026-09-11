let latestData = null;
let historyData = [];

document.addEventListener("DOMContentLoaded", async () => {
  setupTabs();
  await loadData();
});

function setupTabs() {
  const tabs = document.querySelectorAll(".tab-btn");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
      tab.classList.add("active");
      const target = document.getElementById(tab.dataset.tab);
      if (target) target.classList.add("active");
    });
  });
}

async function loadData() {
  try {
    const resLatest = await fetch("data/latest.json");
    latestData = await resLatest.json();
    renderLive(latestData);
  } catch (err) {
    console.warn("Could not fetch latest.json:", err);
  }

  try {
    const resHist = await fetch("data/history.json");
    historyData = await resHist.json();
    populateHistorySelector(historyData);
  } catch (err) {
    console.warn("Could not fetch history.json:", err);
  }
}

function renderLive(data) {
  if (!data) return;
  document.getElementById("stat-time").textContent = data.timestamp || "--";
  document.getElementById("stat-scanned").textContent = data.total_tickers_scanned || "--";
  document.getElementById("stat-conviction").textContent = data.high_conviction_count || "0";
  
  const m = data.ensemble_metrics || {};
  document.getElementById("stat-auc").textContent = m.ensemble_auc ? (m.ensemble_auc * 100).toFixed(1) + "%" : "--";
  document.getElementById("stat-spec").textContent = m.specificity ? (m.specificity * 100).toFixed(1) + "%" : "--";
  
  renderTable("live-picks-body", data.candidates || []);
  renderDiagnostics(data);
}

function renderTable(tbodyId, candidates) {
  const tbody = document.getElementById(tbodyId);
  tbody.innerHTML = "";
  
  if (!candidates || candidates.length === 0) {
    tbody.innerHTML = '<tr><td colspan="10" style="text-align:center; padding: 24px; color:#94a3b8;">No picks matching criteria found.</td></tr>';
    return;
  }
  
  candidates.forEach(c => {
    const tr = document.createElement("tr");
    const tvSymbol = "NSE:" + c.symbol;
    const tvUrl = "https://www.tradingview.com/chart/?symbol=" + encodeURIComponent(tvSymbol);
    
    tr.innerHTML = `
      <td><a href="${tvUrl}" target="_blank" class="tv-link">${c.symbol} ↗</a></td>
      <td><span class="badge ${c.badge_class}">${c.conviction}</span></td>
      <td><strong>${(c.probability * 100).toFixed(1)}%</strong></td>
      <td>₹${c.close.toFixed(2)}</td>
      <td style="color:#ef4444;">₹${c.stop_loss.toFixed(2)} (${c.stop_loss_pct}%)</td>
      <td style="color:#10b981; font-weight:600;">₹${c.target_1.toFixed(2)} (+5%)</td>
      <td style="color:#0ea5e9;">₹${c.target_2.toFixed(2)} (+10%)</td>
      <td style="color:#8b5cf6;">₹${c.target_3.toFixed(2)} (+18%)</td>
      <td><strong>${c.risk_reward.toFixed(2)}:1</strong></td>
      <td>${(c.catalysts || []).map(t => `<span class="tag">${t}</span>`).join("")}</td>
    `;
    tbody.appendChild(tr);
  });
}

function filterLivePicks() {
  if (!latestData) return;
  const q = document.getElementById("search-input").value.toUpperCase();
  const filterVal = document.getElementById("conviction-filter").value;
  
  let list = latestData.candidates || [];
  if (q) list = list.filter(c => c.symbol.includes(q));
  if (filterVal !== "ALL") list = list.filter(c => c.conviction.includes(filterVal));
  renderTable("live-picks-body", list);
}

function populateHistorySelector(hist) {
  const sel = document.getElementById("history-date-select");
  sel.innerHTML = "";
  if (!hist || hist.length === 0) {
    sel.innerHTML = "<option>No historical archives found</option>";
    return;
  }
  hist.forEach((session, idx) => {
    const opt = document.createElement("option");
    opt.value = idx;
    opt.textContent = `${session.date} (${session.timestamp}) - ${session.high_conviction_count} Surges Flagged`;
    sel.appendChild(opt);
  });
  loadHistoricalSession(0);
}

function onHistoryDateChange() {
  const sel = document.getElementById("history-date-select");
  loadHistoricalSession(parseInt(sel.value, 10));
}

function loadHistoricalSession(idx) {
  const session = historyData[idx];
  if (!session) return;
  renderTable("history-picks-body", session.candidates || []);
}

function renderDiagnostics(data) {
  const tbody = document.getElementById("models-table-body");
  if (!tbody) return;
  tbody.innerHTML = "";
  const models = data.models_summary || {};
  Object.keys(models).forEach(mName => {
    const m = models[mName];
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>${mName}</strong></td>
      <td>${(m.test_auc * 100).toFixed(1)}%</td>
      <td>${(m.test_pr_auc * 100).toFixed(1)}%</td>
      <td>${(m.specificity * 100).toFixed(1)}%</td>
      <td>${(m.precision * 100).toFixed(1)}%</td>
      <td>${(m.sensitivity * 100).toFixed(1)}%</td>
      <td>${m.opt_threshold.toFixed(3)}</td>
    `;
    tbody.appendChild(tr);
  });

  const featBody = document.getElementById("features-table-body");
  if (!featBody) return;
  featBody.innerHTML = "";
  (data.top_features || []).forEach(f => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><code>${f.feature}</code></td>
      <td>
        <div style="background:#1e293b; border-radius:4px; height:12px; width:100%; overflow:hidden;">
          <div style="background:#3b82f6; height:100%; width:${(f.importance * 100).toFixed(1)}%;"></div>
        </div>
      </td>
      <td>${(f.importance * 100).toFixed(2)}%</td>
    `;
    featBody.appendChild(tr);
  });
}

function copyPineScript() {
  const code = document.getElementById("pine-v6-code").innerText;
  navigator.clipboard.writeText(code).then(() => {
    const btn = document.getElementById("copy-btn");
    btn.textContent = "Copied to Clipboard!";
    setTimeout(() => { btn.textContent = "Copy Pine v6 Script"; }, 2000);
  });
}
