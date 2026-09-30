"use strict";

const form = document.getElementById("analysis-form");
const fileInput = document.getElementById("image");
const dropZone = document.getElementById("drop-zone");
const statusPanel = document.getElementById("upload-status");
const statusText = document.getElementById("status-text");
const analyzeButton = document.getElementById("analyze-button");
const errorBox = document.getElementById("dashboard-error");
let previewUrl = null;

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[char]));
}

function setFile(file) {
  if (!file) return;
  const allowed = /\.(jpe?g|png|bmp|webp)$/i.test(file.name);
  if (!allowed || file.size > 16 * 1024 * 1024) {
    errorBox.textContent = !allowed ? "Choose a JPG, PNG, BMP, or WEBP image." : "The image exceeds the 16 MB upload limit.";
    errorBox.hidden = false;
    fileInput.value = "";
    return;
  }
  errorBox.hidden = true;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  document.getElementById("image-preview").src = previewUrl;
  document.getElementById("preview-wrap").hidden = false;
  document.getElementById("drop-title").textContent = "MRI ready for analysis";
  document.getElementById("drop-subtitle").textContent = "Choose a different image by dropping or browsing again";
  document.getElementById("file-name").textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB`;
}

if (fileInput) fileInput.addEventListener("change", () => setFile(fileInput.files[0]));
const removeFile = document.getElementById("remove-file");
if (removeFile) removeFile.addEventListener("click", () => {
  fileInput.value = "";
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  document.getElementById("preview-wrap").hidden = true;
  document.getElementById("drop-title").textContent = "Drop your MRI image here";
  document.getElementById("drop-subtitle").innerHTML = 'or <span class="browse-link">browse files</span> from your device';
  document.getElementById("file-name").textContent = "Select a scan to begin analysis.";
});

if (dropZone) {
  ["dragenter", "dragover"].forEach(type => dropZone.addEventListener(type, event => {
    event.preventDefault(); dropZone.classList.add("is-dragging");
  }));
  ["dragleave", "drop"].forEach(type => dropZone.addEventListener(type, event => {
    event.preventDefault(); dropZone.classList.remove("is-dragging");
  }));
  dropZone.addEventListener("drop", event => {
    const file = event.dataTransfer.files[0];
    if (!file) return;
    const transfer = new DataTransfer(); transfer.items.add(file); fileInput.files = transfer.files; setFile(file);
  });
}

function renderResult(data) {
  const result = data.result;
  const prediction = result.prediction;
  const uncertainty = result.uncertainty;
  const probabilities = Object.entries(prediction.probabilities).map(([name, value]) => {
    const deviation = prediction.std_by_class ? ` <small>± ${(prediction.std_by_class[name] * 100).toFixed(1)}%</small>` : "";
    return `<div class="result-prob"><div><span>${escapeHtml(name)}</span><strong>${(value * 100).toFixed(1)}%${deviation}</strong></div><div class="result-track"><i style="width:${Math.max(0, Math.min(100, value * 100))}%"></i></div></div>`;
  }).join("");
  const reliability = uncertainty ? `<section class="result-panel"><div class="result-panel-heading"><div><p class="eyebrow">RELIABILITY CONTEXT</p><h3>Model uncertainty</h3></div><span class="reliability-badge reliability-${escapeHtml(uncertainty.level.toLowerCase())}">${escapeHtml(uncertainty.level)} uncertainty</span></div><div class="result-metrics"><div><span>Predictive entropy</span><strong>${uncertainty.predictive_entropy.toFixed(4)} <small>nats</small></strong></div><div><span>Notebook reliability</span><strong>${escapeHtml(uncertainty.reliability)}</strong></div></div><p class="muted small-note">Reference thresholds from notebook validation: low ≤ 0.1099, medium ≤ 0.3690, high &gt; 0.3690 nats.</p></section>` : "";
  const explanation = (key, title, alt) => result.explanations[key] ? `<article class="explanation-tile"><img src="${result.explanations[key]}" alt="${alt}"><div><strong>${title}</strong><span>Model attribution visualization</span></div></article>` : `<article class="explanation-tile explanation-missing"><strong>${title}</strong><span>Not available for this scan</span></article>`;
  document.getElementById("live-result").innerHTML = `<section class="live-result-card"><div class="result-title-row"><div><p class="eyebrow"><span class="status-dot"></span> ANALYSIS READY</p><h2>Scan insights</h2></div><span class="confidence-chip">${(prediction.confidence * 100).toFixed(1)}% model confidence</span></div><div class="prediction-banner"><img src="${result.image}" alt="Analyzed brain MRI"><div><span>Top predicted class</span><strong>${escapeHtml(prediction.class_name)}</strong><small>Mean probability across stochastic passes</small></div></div><section class="result-panel"><p class="eyebrow">CLASS DISTRIBUTION</p><h3>Probability by class</h3>${probabilities}</section>${reliability}<section class="result-panel"><div class="result-panel-heading"><div><p class="eyebrow">EXPLAINABLE AI</p><h3>What influenced this output</h3></div>${result.eas == null ? "" : `<span class="eas-chip">EAS ${Number(result.eas).toFixed(3)}</span>`}</div><div class="explanation-grid">${explanation("gradcam", "Grad-CAM++", "Grad-CAM++ attribution overlay")}${explanation("integrated_gradients", "Integrated Gradients", "Integrated Gradients attribution overlay")}</div><p class="muted small-note">Attribution maps describe model behavior; they do not establish clinical causality. EAS compares the two maps.</p></section></section>`;
  document.getElementById("stat-total").textContent = data.total_scans;
  document.getElementById("stat-confidence").textContent = `${(data.average_confidence * 100).toFixed(1)}%`;
  document.getElementById("chart-total").textContent = data.total_scans;
  const distribution = document.getElementById("distribution-chart");
  distribution.innerHTML = Object.entries(data.class_counts).map(([name, count]) => `<div class="distribution-row" data-class="${escapeHtml(name)}"><div class="distribution-label"><span>${escapeHtml(name)}</span><b>${count}</b></div><div class="chart-track"><i style="width:${(count / Math.max(1, data.total_scans) * 100).toFixed(1)}%"></i></div></div>`).join("");
  document.getElementById("confidence-chart").innerHTML = data.chart_records.map((record, index) => `<div class="confidence-point" title="${escapeHtml(record.class_name)} · ${(record.confidence * 100).toFixed(1)}%"><span class="confidence-bar" style="height:${Math.max(record.confidence * 100, 5)}%"></span><small>${index + 1}</small></div>`).join("");
  const recent = document.getElementById("recent-scans");
  const oldRows = recent.querySelector(".activity-list");
  const row = `<li><span class="activity-mark" aria-hidden="true"><svg viewBox="0 0 24 24"><path d="M4 5h16v14H4zM8 9h8m-8 4h5"/><path d="M7 19v2m10-2v2"/></svg></span><div class="activity-copy"><strong>${escapeHtml(data.record.class_name)}</strong><span>${escapeHtml(data.record.created)}</span></div><span class="activity-confidence">${(data.record.confidence * 100).toFixed(0)}%</span></li>`;
  if (oldRows) oldRows.insertAdjacentHTML("afterbegin", row);
  else recent.innerHTML = `<ul class="activity-list">${row}</ul>`;
  const list = recent.querySelectorAll(".activity-list li");
  list.forEach((item, index) => { if (index > 4) item.remove(); });
  document.getElementById("live-result").scrollIntoView({behavior:"smooth", block:"start"});
}

if (form) form.addEventListener("submit", async event => {
  event.preventDefault();
  if (!fileInput.files.length) { fileInput.click(); return; }
  errorBox.hidden = true;
  statusPanel.hidden = false;
  analyzeButton.disabled = true;
  analyzeButton.classList.add("is-loading");
  const stages = ["Validating image…", "Running model inference…", "Sampling MC Dropout uncertainty…", "Generating explanation maps…"];
  let stage = 0;
  statusText.textContent = stages[stage];
  const timer = setInterval(() => { stage = Math.min(stage + 1, stages.length - 1); statusText.textContent = stages[stage]; }, 2500);
  try {
    const response = await fetch(form.action, {method:"POST", body:new FormData(form), headers:{Accept:"application/json"}});
    let payload;
    try {
      payload = await response.json();
    } catch {
      throw new Error(`The server returned an unreadable response (HTTP ${response.status}). Check the deployment logs.`);
    }
    if (!response.ok) throw new Error(payload.error || "Analysis could not be completed.");
    statusText.textContent = "Preparing your dashboard…";
    renderResult(payload);
  } catch (error) {
    errorBox.textContent = error.message || "Analysis failed. Please try again.";
    errorBox.hidden = false;
  } finally {
    clearInterval(timer);
    statusPanel.hidden = true;
    analyzeButton.disabled = false;
    analyzeButton.classList.remove("is-loading");
  }
});
