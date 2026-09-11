const REPO_OWNER = "vishnuvcr";
const REPO_NAME = "market-gainer-predictor-specificity-precision-prioritised";
const WORKFLOW_ID = "daily_scanner.yml";

let latestData = null;
let historyData = [];
let monitorInterval = null;
let currentActiveRunId = null;
let runStartTime = null;

document.addEventListener("DOMContentLoaded", async () => {
  setupTabs();
  initTokenInput();
  await loadData();
  checkActiveWorkflow();
  setInterval(checkActiveWorkflow, 6000);
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

function initTokenInput() {
  const saved = localStorage.getItem("gh_pat_token");
  if (saved) {
    const input = document.getElementById("gh-token-input");
    if (input) input.value = saved;
  }
}

async function loadData() {
  try {
    const resLatest = await fetch("data/latest.json?t=" + Date.now());
    if (resLatest.ok) {
      latestData = await resLatest.json();
      renderLive(latestData);
    }
  } catch (err) {
    console.warn("Could not fetch latest.json:", err);
  }

  try {
    const resHist = await fetch("data/history.json?t=" + Date.now());
    if (resHist.ok) {
      historyData = await resHist.json();
      populateHistorySelector(historyData);
    }
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

// ==============================================================================
// EXPORT FUNCTIONS (CSV, JPG, PDF)
// ==============================================================================
function exportToCSV() {
  if (!latestData || !latestData.candidates || latestData.candidates.length === 0) {
    alert("No scan data available to export.");
    return;
  }

  const headers = [
    "Symbol", "Conviction", "Surge_Probability_Pct", "Current_Price_INR",
    "Stop_Loss_INR", "Stop_Loss_Pct", "Target_1_INR", "Target_2_INR",
    "Target_3_INR", "Risk_Reward_Ratio", "Volume_Surge_x", "RSI_14", "Catalysts", "Date"
  ];

  const rows = latestData.candidates.map(c => [
    c.symbol,
    `"${c.conviction}"`,
    (c.probability * 100).toFixed(1),
    c.close.toFixed(2),
    c.stop_loss.toFixed(2),
    c.stop_loss_pct,
    c.target_1.toFixed(2),
    c.target_2.toFixed(2),
    c.target_3.toFixed(2),
    c.risk_reward.toFixed(2),
    c.volume_surge.toFixed(2),
    c.rsi_14,
    `"${(c.catalysts || []).join('; ')}"`,
    c.date
  ]);

  const csvContent = "data:text/csv;charset=utf-8," + [headers.join(","), ...rows.map(e => e.join(","))].join("\n");
  const encodedUri = encodeURI(csvContent);
  const link = document.createElement("a");
  link.setAttribute("href", encodedUri);
  link.setAttribute("download", `NSE_Surge_Trades_${latestData.date || 'latest'}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

async function exportToImage() {
  const captureEl = document.getElementById("capture-table-area");
  if (!captureEl) return;
  const btn = document.getElementById("btn-export-jpg");
  const orig = btn.textContent;
  btn.textContent = "Generating...";
  btn.disabled = true;

  try {
    if (typeof html2canvas === "undefined") {
      throw new Error("html2canvas library is loading, please try again in a moment.");
    }
    const canvas = await html2canvas(captureEl, {
      backgroundColor: "#0b0f19",
      scale: 2,
      useCORS: true
    });
    const imgData = canvas.toDataURL("image/jpeg", 0.95);
    const link = document.createElement("a");
    link.href = imgData;
    link.download = `NSE_Surge_Trades_${latestData ? latestData.date : 'latest'}.jpg`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  } catch (err) {
    console.error("Image export failed:", err);
    alert("Could not generate JPG image: " + err.message);
  } finally {
    btn.textContent = orig;
    btn.disabled = false;
  }
}

async function exportToPDF() {
  const captureEl = document.getElementById("capture-table-area");
  if (!captureEl) return;
  const btn = document.getElementById("btn-export-pdf");
  const orig = btn.textContent;
  btn.textContent = "Generating...";
  btn.disabled = true;

  try {
    if (typeof html2canvas === "undefined" || !window.jspdf) {
      throw new Error("PDF export libraries loading, please try again in a moment.");
    }
    const canvas = await html2canvas(captureEl, {
      backgroundColor: "#0b0f19",
      scale: 2,
      useCORS: true
    });
    const imgData = canvas.toDataURL("image/jpeg", 0.95);
    const { jsPDF } = window.jspdf;
    const pdf = new jsPDF({
      orientation: "landscape",
      unit: "px",
      format: [canvas.width, canvas.height]
    });
    pdf.addImage(imgData, "JPEG", 0, 0, canvas.width, canvas.height);
    pdf.save(`NSE_Surge_Trades_${latestData ? latestData.date : 'latest'}.pdf`);
  } catch (err) {
    console.error("PDF export failed:", err);
    alert("Could not generate PDF: " + err.message);
  } finally {
    btn.textContent = orig;
    btn.disabled = false;
  }
}

// ==============================================================================
// GITHUB ACTIONS LIVE TRIGGER & PROGRESS MONITOR
// ==============================================================================
function openRunModal() {
  document.getElementById("modal-error-msg").style.display = "none";
  document.getElementById("run-modal").style.display = "flex";
}

function closeRunModal() {
  document.getElementById("run-modal").style.display = "none";
}

async function triggerWorkflow() {
  const token = document.getElementById("gh-token-input").value.trim();
  const limitInput = document.getElementById("limit-tickers-input").value.trim();
  const errorDiv = document.getElementById("modal-error-msg");
  const btn = document.getElementById("btn-trigger-action");

  if (!token) {
    errorDiv.textContent = "Please enter your GitHub Personal Access Token.";
    errorDiv.style.display = "block";
    return;
  }

  localStorage.setItem("gh_pat_token", token);
  btn.disabled = true;
  btn.textContent = "Launching Scanner...";
  errorDiv.style.display = "none";

  const payload = { ref: "main", inputs: {} };
  if (limitInput) payload.inputs.limit_tickers = limitInput;

  try {
    const url = `https://api.github.com/repos/${REPO_OWNER}/${REPO_NAME}/actions/workflows/${WORKFLOW_ID}/dispatches`;
    const res = await fetch(url, {
      method: "POST",
      headers: {
        "Accept": "application/vnd.github+json",
        "Authorization": `Bearer ${token}`
      },
      body: JSON.stringify(payload)
    });

    if (res.status === 204) {
      closeRunModal();
      runStartTime = Date.now();
      showBanner("Workflow Dispatched", "QUEUED", 10);
      setTimeout(checkActiveWorkflow, 2500);
    } else {
      const errJson = await res.json().catch(() => ({}));
      errorDiv.textContent = errJson.message || `Error ${res.status}: Check token permissions.`;
      errorDiv.style.display = "block";
    }
  } catch (err) {
    errorDiv.textContent = "Network error: " + err.message;
    errorDiv.style.display = "block";
  } finally {
    btn.disabled = false;
    btn.textContent = "▶ Launch Scanner Now";
  }
}

async function checkActiveWorkflow() {
  const token = localStorage.getItem("gh_pat_token");
  const headers = { "Accept": "application/vnd.github+json" };
  if (token) headers["Authorization"] = `Bearer ${token}`;

  try {
    const url = `https://api.github.com/repos/${REPO_OWNER}/${REPO_NAME}/actions/runs?per_page=1`;
    const res = await fetch(url, { headers });
    if (!res.ok) return;
    const data = await res.json();
    if (!data.workflow_runs || data.workflow_runs.length === 0) return;

    const run = data.workflow_runs[0];
    const isRunning = run.status === "in_progress" || run.status === "queued";

    if (isRunning) {
      currentActiveRunId = run.id;
      if (!runStartTime) runStartTime = new Date(run.run_started_at || run.created_at).getTime();

      document.getElementById("live-run-banner").style.display = "block";
      document.getElementById("live-run-title").textContent = `Workflow #${run.run_number}: Live Scanner`;
      document.getElementById("live-run-badge").textContent = run.status.toUpperCase();
      document.getElementById("live-run-link").href = run.html_url;

      const elapsedSec = Math.floor((Date.now() - runStartTime) / 1000);
      const min = Math.floor(elapsedSec / 60);
      const sec = elapsedSec % 60;
      document.getElementById("live-run-timer").textContent = `Active: ${min}m ${sec}s`;

      if (!monitorInterval) {
        monitorInterval = setInterval(fetchRunJobs, 3000);
      }
      fetchRunJobs();
    } else {
      if (currentActiveRunId && currentActiveRunId === run.id && run.conclusion === "success") {
        document.getElementById("live-run-badge").textContent = "SUCCESS";
        document.getElementById("live-progress-fill").style.width = "100%";
        setTimeout(() => {
          document.getElementById("live-run-banner").style.display = "none";
          runStartTime = null;
          loadData();
        }, 3500);
      }
      if (monitorInterval) {
        clearInterval(monitorInterval);
        monitorInterval = null;
      }
    }
  } catch (err) {
    console.warn("Could not check workflow status:", err);
  }
}

async function fetchRunJobs() {
  if (!currentActiveRunId) return;
  const token = localStorage.getItem("gh_pat_token");
  const headers = { "Accept": "application/vnd.github+json" };
  if (token) headers["Authorization"] = `Bearer ${token}`;

  try {
    const url = `https://api.github.com/repos/${REPO_OWNER}/${REPO_NAME}/actions/runs/${currentActiveRunId}/jobs`;
    const res = await fetch(url, { headers });
    if (!res.ok) return;
    const data = await res.json();
    if (!data.jobs || data.jobs.length === 0) return;

    renderStepsList(data.jobs[0].steps || []);
  } catch (err) {
    console.warn("Could not fetch jobs:", err);
  }
}

function renderStepsList(steps) {
  const container = document.getElementById("live-steps-list");
  if (!container) return;
  container.innerHTML = "";

  let completedCount = 0;
  steps.forEach(s => {
    if (s.status === "completed") completedCount++;
    const div = document.createElement("div");
    div.className = "live-step-item";

    let icon = "○";
    let statusClass = "step-pending";
    if (s.status === "completed") {
      icon = s.conclusion === "success" ? "✓" : "✗";
      statusClass = s.conclusion === "success" ? "step-done" : "step-fail";
    } else if (s.status === "in_progress") {
      icon = "⟳";
      statusClass = "step-active";
    }

    div.innerHTML = `<span class="${statusClass}">${icon}</span> <span>${s.name}</span>`;
    container.appendChild(div);
  });

  const pct = Math.min(100, Math.round((completedCount / Math.max(1, steps.length)) * 100));
  document.getElementById("live-progress-fill").style.width = pct + "%";
}

function showBanner(title, badge, pct) {
  document.getElementById("live-run-banner").style.display = "block";
  document.getElementById("live-run-title").textContent = title;
  document.getElementById("live-run-badge").textContent = badge;
  document.getElementById("live-progress-fill").style.width = pct + "%";
}
