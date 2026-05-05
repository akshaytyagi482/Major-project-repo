/* SENTINEL — Frontend JS */

const API = "";  // Same origin
let currentResult = null;
let currentTTS = null;
let analysisHistory = [];

// ── Tab Navigation ──────────────────────────────────────────────────────────
document.querySelectorAll(".pill").forEach(pill => {
  pill.addEventListener("click", () => {
    const tab = pill.dataset.tab;
    document.querySelectorAll(".pill").forEach(p => p.classList.remove("pill--active"));
    pill.classList.add("pill--active");
    document.querySelectorAll(".tab-content").forEach(t => t.classList.remove("active"));
    document.getElementById(`tab-${tab}`).classList.add("active");
  });
});

// ── Char counter ────────────────────────────────────────────────────────────
const textInput = document.getElementById("textInput");
if (textInput) {
  textInput.addEventListener("input", updateCharCount);
}
function updateCharCount() {
  const n = (document.getElementById("textInput").value || "").length;
  const el = document.getElementById("charCount");
  if (el) el.textContent = `${n.toLocaleString()} characters`;
}

// ── File Handlers ────────────────────────────────────────────────────────────
function handleImageFile(input) {
  const file = input.files[0];
  if (!file) return;
  const preview = document.getElementById("imagePreview");
  const drop = document.getElementById("imageDropContent");
  const url = URL.createObjectURL(file);
  preview.src = url;
  preview.style.display = "block";
  if (drop) drop.style.display = "none";
}

function handleVideoFile(input) {
  const file = input.files[0];
  if (!file) return;
  const preview = document.getElementById("videoPreview");
  const drop = document.getElementById("videoDropContent");
  const url = URL.createObjectURL(file);
  preview.src = url;
  preview.style.display = "block";
  if (drop) drop.style.display = "none";
}

// Drag & drop support
["imageDropzone", "videoDropzone"].forEach(id => {
  const el = document.getElementById(id);
  if (!el) return;
  el.addEventListener("dragover", e => { e.preventDefault(); el.classList.add("dragover"); });
  el.addEventListener("dragleave", () => el.classList.remove("dragover"));
  el.addEventListener("drop", e => {
    e.preventDefault(); el.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    if (!file) return;
    const inputId = id === "imageDropzone" ? "imageFile" : "videoFile";
    const input = document.getElementById(inputId);
    const dt = new DataTransfer();
    dt.items.add(file);
    input.files = dt.files;
    id === "imageDropzone" ? handleImageFile(input) : handleVideoFile(input);
  });
});

// ── URL helper ──────────────────────────────────────────────────────────────
function setUrl(url) {
  document.getElementById("urlInput").value = url.replace("https://", "");
}

// ── Loading States ──────────────────────────────────────────────────────────
const LOADING_STEPS = {
  text:  ["Tokenizing text...", "Running NLP analysis...", "Detecting fallacies...", "Computing fusion score..."],
  image: ["Preprocessing image...", "Running ELA forensics...", "Analyzing noise patterns...", "Computing fusion score..."],
  video: ["Extracting frames...", "Analyzing temporal consistency...", "Detecting face-swap artifacts...", "Computing fusion score..."],
  url:   ["Fetching URL...", "Extracting text content...", "Running NLP analysis...", "Computing fusion score..."],
};

function showLoading(type) {
  document.getElementById("idleState").style.display = "none";
  document.getElementById("resultsContainer").style.display = "none";
  document.getElementById("loadingState").style.display = "flex";

  const steps = LOADING_STEPS[type] || LOADING_STEPS.text;
  const stepsEl = document.getElementById("loadingSteps");
  stepsEl.innerHTML = "";
  let i = 0;
  const tick = () => {
    if (i < steps.length) {
      document.getElementById("loadingText").textContent = steps[i].toUpperCase();
      stepsEl.innerHTML += `<div>✓ ${steps[i]}</div>`;
      i++;
      setTimeout(tick, 700 + Math.random() * 400);
    }
  };
  tick();
}

function hideLoading() {
  document.getElementById("loadingState").style.display = "none";
}

// ── Render Results ──────────────────────────────────────────────────────────
function renderResults(data, mediaType) {
  hideLoading();
  const analysis = data.analysis;
  currentResult = data;

  // Verdict banner
  const banner = document.getElementById("verdictBanner");
  banner.className = `verdict-banner risk-${analysis.risk_level}`;
  document.getElementById("verdictMain").textContent = analysis.verdict;

  const badge = document.getElementById("riskBadge");
  badge.textContent = analysis.risk_level;
  badge.className = `verdict-badge ${analysis.risk_level}`;

  // Probability bars (animate)
  const fakeP = Math.round((analysis.fake_probability || 0) * 100);
  const authP = Math.round((analysis.authentic_probability || 0) * 100);
  setTimeout(() => {
    document.getElementById("fakeBar").style.width = fakeP + "%";
    document.getElementById("authBar").style.width = authP + "%";
  }, 100);
  document.getElementById("fakeVal").textContent = fakeP + "%";
  document.getElementById("authVal").textContent = authP + "%";
  document.getElementById("confVal").textContent = Math.round((analysis.confidence || 0) * 100) + "%";
  document.getElementById("crossVal").textContent = Math.round((analysis.cross_modal_inconsistency || 0) * 100) + "%";

  // NLP card
  renderNLPCard(analysis.nlp_result);
  // CV card
  renderCVCard(analysis.cv_result);

  // Reasoning
  const reasoningList = document.getElementById("reasoningList");
  reasoningList.innerHTML = "";
  (analysis.reasoning || []).forEach(r => {
    const li = document.createElement("li");
    li.textContent = r;
    reasoningList.appendChild(li);
  });

  // Actions
  const actionsList = document.getElementById("actionsList");
  actionsList.innerHTML = "";
  (analysis.recommended_actions || []).forEach(a => {
    const li = document.createElement("li");
    li.textContent = a;
    actionsList.appendChild(li);
  });

  // TTS
  if (data.tts && data.tts.success && data.tts.url) {
    currentTTS = data.tts.url;
    document.getElementById("ttsBtn").style.opacity = "1";
  } else {
    document.getElementById("ttsBtn").style.opacity = "0.4";
    currentTTS = null;
  }

  document.getElementById("resultsContainer").style.display = "block";

  // Add to history
  addToHistory(analysis, mediaType);
}

function renderNLPCard(nlp) {
  const score = document.getElementById("nlpScore");
  const body = document.getElementById("nlpBody");

  if (!nlp) {
    score.textContent = "—";
    body.innerHTML = '<p class="no-data">No text analyzed</p>';
    return;
  }

  score.textContent = Math.round((nlp.fake_probability || 0) * 100) + "%";
  const comps = nlp.component_scores || {};
  let html = "";

  // Component scores
  const labels = {
    emotional_score: "Emotional manipulation",
    stylometric_score: "Stylometric anomaly",
    credibility_score: "Low credibility signals",
    fallacy_score: "Logical fallacies",
    sensationalism_score: "Sensationalism",
  };

  Object.entries(comps).forEach(([k, v]) => {
    if (labels[k]) {
      const pct = Math.round(v * 100);
      const color = v > 0.6 ? "var(--accent2)" : v > 0.35 ? "var(--accent3)" : "var(--accent4)";
      html += `<div class="card-metric"><span class="card-metric-label">${labels[k]}</span><span style="color:${color}">${pct}%</span></div>`;
    }
  });

  (nlp.detected_issues || []).forEach(i => {
    html += `<div class="card-issue">${i}</div>`;
  });
  (nlp.detected_positives || []).forEach(p => {
    html += `<div class="card-positive">${p}</div>`;
  });

  if (nlp.word_count) {
    html += `<div class="card-metric" style="margin-top:0.5rem"><span class="card-metric-label">Word count</span><span class="card-metric-val">${nlp.word_count}</span></div>`;
  }

  body.innerHTML = html || '<p class="no-data">Analysis complete</p>';
}

function renderCVCard(cv) {
  const score = document.getElementById("cvScore");
  const body = document.getElementById("cvBody");

  if (!cv) {
    score.textContent = "—";
    body.innerHTML = '<p class="no-data">No visual content analyzed</p>';
    return;
  }

  score.textContent = Math.round((cv.fake_probability || 0) * 100) + "%";
  const comps = cv.component_scores || {};
  let html = "";

  const labels = {
    compression_artifacts: "Compression artifacts",
    noise_pattern: "Noise pattern anomaly",
    color_distribution: "Color distribution",
    edge_consistency: "Edge consistency",
    synthetic_patterns: "Synthetic patterns",
    ela_score: "ELA forensics",
    temporal_consistency: "Temporal consistency",
    face_swap_artifacts: "Face-swap artifacts",
    motion_naturalness: "Motion naturalness",
  };

  Object.entries(comps).forEach(([k, v]) => {
    if (labels[k]) {
      const pct = Math.round(v * 100);
      const color = v > 0.6 ? "var(--accent2)" : v > 0.35 ? "var(--accent3)" : "var(--accent4)";
      html += `<div class="card-metric"><span class="card-metric-label">${labels[k]}</span><span style="color:${color}">${pct}%</span></div>`;
    }
  });

  (cv.detected_anomalies || []).forEach(a => {
    html += `<div class="card-issue">${a}</div>`;
  });

  if (cv.image_metadata) {
    const m = cv.image_metadata;
    html += `<div class="card-metric" style="margin-top:0.5rem"><span class="card-metric-label">Dimensions</span><span class="card-metric-val">${m.dimensions}</span></div>`;
    html += `<div class="card-metric"><span class="card-metric-label">File size</span><span class="card-metric-val">${m.file_size_kb} KB</span></div>`;
  }
  if (cv.frames_analyzed !== undefined) {
    html += `<div class="card-metric" style="margin-top:0.5rem"><span class="card-metric-label">Frames analyzed</span><span class="card-metric-val">${cv.frames_analyzed}</span></div>`;
  }

  body.innerHTML = html || '<p class="no-data">Analysis complete</p>';
}

// ── Analysis Functions ──────────────────────────────────────────────────────
async function analyzeText() {
  const text = document.getElementById("textInput").value.trim();
  if (!text) { alert("Please enter some text to analyze."); return; }

  setBtnDisabled("analyzeText", true);
  showLoading("text");

  try {
    const res = await fetch(`${API}/api/analyze/text`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    renderResults(data, "TEXT");
  } catch (err) {
    showError(err.message);
  } finally {
    setBtnDisabled("analyzeText", false);
  }
}

async function analyzeImage() {
  const fileInput = document.getElementById("imageFile");
  const text = document.getElementById("imageText").value.trim();

  if (!fileInput.files[0] && !text) {
    alert("Please upload an image or provide text.");
    return;
  }

  const formData = new FormData();
  if (fileInput.files[0]) formData.append("file", fileInput.files[0]);
  if (text) formData.append("text", text);

  setBtnDisabled("analyzeImage", true);
  showLoading("image");

  try {
    const res = await fetch(`${API}/api/analyze/image`, { method: "POST", body: formData });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    renderResults(data, "IMAGE");
  } catch (err) {
    showError(err.message);
  } finally {
    setBtnDisabled("analyzeImage", false);
  }
}

async function analyzeVideo() {
  const fileInput = document.getElementById("videoFile");
  if (!fileInput.files[0]) { alert("Please upload a video file."); return; }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);
  const text = document.getElementById("videoText").value.trim();
  if (text) formData.append("text", text);

  setBtnDisabled("analyzeVideo", true);
  showLoading("video");

  try {
    const res = await fetch(`${API}/api/analyze/video`, { method: "POST", body: formData });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    renderResults(data, "VIDEO");
  } catch (err) {
    showError(err.message);
  } finally {
    setBtnDisabled("analyzeVideo", false);
  }
}

async function analyzeUrl() {
  let url = document.getElementById("urlInput").value.trim();
  if (!url) { alert("Please enter a URL."); return; }
  if (!url.startsWith("http")) url = "https://" + url;

  setBtnDisabled("analyzeUrl", true);
  showLoading("url");

  try {
    const res = await fetch(`${API}/api/analyze/url`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    renderResults(data, "URL");
  } catch (err) {
    showError(err.message);
  } finally {
    setBtnDisabled("analyzeUrl", false);
  }
}

// ── TTS ─────────────────────────────────────────────────────────────────────
function playTTS() {
  if (!currentTTS) return;
  const audio = document.getElementById("ttsAudio");
  audio.src = currentTTS;
  audio.style.display = "block";
  audio.play().catch(() => {
    // Fallback: open in new tab
    window.open(currentTTS, "_blank");
  });
  document.getElementById("ttsBtn").textContent = "🔊 PLAYING...";
  audio.onended = () => {
    document.getElementById("ttsBtn").textContent = "🔊 HEAR ANALYSIS";
  };
}

// ── Report Download ──────────────────────────────────────────────────────────
async function downloadReport() {
  if (!currentResult) return;
  try {
    const res = await fetch(`${API}/api/report/text`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ fusion_result: currentResult.analysis }),
    });
    const data = await res.json();
    const blob = new Blob([data.report], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = "sentinel_report.txt"; a.click();
    URL.revokeObjectURL(url);
  } catch (err) {
    // Fallback: download JSON
    const blob = new Blob([JSON.stringify(currentResult.report, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = "sentinel_report.json"; a.click();
    URL.revokeObjectURL(url);
  }
}

// ── History ──────────────────────────────────────────────────────────────────
function addToHistory(analysis, mediaType) {
  analysisHistory.unshift({
    mediaType,
    verdict: analysis.verdict,
    fakeProb: analysis.fake_probability,
    riskLevel: analysis.risk_level,
    time: new Date().toLocaleTimeString(),
  });
  if (analysisHistory.length > 12) analysisHistory.pop();
  renderHistory();
}

function renderHistory() {
  const grid = document.getElementById("historyGrid");
  if (!analysisHistory.length) {
    grid.innerHTML = '<div class="history-empty">No analyses yet</div>';
    return;
  }
  grid.innerHTML = analysisHistory.map(h => {
    const pct = Math.round((h.fakeProb || 0) * 100);
    const color = h.riskLevel === "CRITICAL" || h.riskLevel === "HIGH"
      ? "var(--accent2)" : h.riskLevel === "MODERATE"
      ? "var(--accent3)" : "var(--accent4)";
    return `
      <div class="history-card">
        <div class="history-card-type">${h.mediaType} ANALYSIS</div>
        <div class="history-card-verdict" style="color:${color}">${h.verdict}</div>
        <div class="history-card-prob" style="color:${color}">Fake probability: ${pct}%</div>
        <div class="history-card-time">${h.time}</div>
      </div>
    `;
  }).join("");
}

// ── Helpers ──────────────────────────────────────────────────────────────────
function setBtnDisabled(id, disabled) {
  const btn = document.getElementById(id);
  if (btn) btn.disabled = disabled;
}

function showError(msg) {
  hideLoading();
  document.getElementById("idleState").style.display = "none";
  document.getElementById("resultsContainer").style.display = "none";
  // Show error in results area
  const res = document.getElementById("resultsContainer");
  res.innerHTML = `
    <div style="padding:2rem;text-align:center;font-family:var(--mono);color:var(--accent2)">
      <div style="font-size:2rem;margin-bottom:1rem">⚠</div>
      <div style="font-size:0.9rem;margin-bottom:0.5rem">ANALYSIS FAILED</div>
      <div style="font-size:0.75rem;color:var(--text2)">${msg}</div>
      <button class="btn-secondary" style="margin-top:1.5rem" onclick="resetResults()">↺ TRY AGAIN</button>
    </div>
  `;
  res.style.display = "block";
}

function resetResults() {
  currentResult = null;
  currentTTS = null;
  document.getElementById("resultsContainer").style.display = "none";
  document.getElementById("loadingState").style.display = "none";
  document.getElementById("idleState").style.display = "flex";
  // Restore results container markup if it was replaced by error
  location.reload();
}
