// ---- tiny helpers ----------------------------------------------------------
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));
const api = async (url, body) => {
  const opts = body
    ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
    : {};
  const r = await fetch(url, opts);
  return r.json();
};

const state = {
  step: 0,
  selectedData: new Set(),
  vocabBuilt: false,
  arch: "gpt2-small",
  trained: false,
};

const NUM_STEPS = 8;

// ---- navigation ------------------------------------------------------------
function goTo(step) {
  state.step = Math.max(0, Math.min(NUM_STEPS - 1, step));
  $$(".step").forEach((s) => s.classList.toggle("active", +s.dataset.step === state.step));
  $$(".nav-item").forEach((n) => n.classList.toggle("active", +n.dataset.step === state.step));
  $("#progressPill").textContent = `Step ${state.step + 1} of ${NUM_STEPS}`;
  $("#prevBtn").disabled = state.step === 0;
  $("#nextBtn").style.visibility = state.step === NUM_STEPS - 1 ? "hidden" : "visible";
  onEnterStep(state.step);
}

$$(".nav-item").forEach((n) => n.addEventListener("click", () => goTo(+n.dataset.step)));
$("#nextBtn").addEventListener("click", () => goTo(state.step + 1));
$("#prevBtn").addEventListener("click", () => goTo(state.step - 1));

function markDone(step) {
  const n = $(`.nav-item[data-step="${step}"]`);
  if (n) n.classList.add("done");
}

// lazy-load content the first time each step is shown
const loaded = {};
function onEnterStep(step) {
  if (step === 2 && !loaded[2]) { loadEmbeddings(); loaded[2] = true; }
  if (step === 3 && !loaded[3]) { loadAttention(); loaded[3] = true; }
  if (step === 4 && !loaded[4]) { loadArchitectures(); loaded[4] = true; }
  if (step === 5) updateTotalSteps();
}

// ---- STEP 0: datasets ------------------------------------------------------
async function loadDatasets() {
  const { datasets } = await api("/api/datasets");
  const grid = $("#datasetList");
  grid.innerHTML = "";
  datasets.forEach((d) => {
    const card = document.createElement("div");
    card.className = "dataset-card";
    card.dataset.id = d.id;
    card.innerHTML = `
      <div class="dc-name">${d.name} <span class="check"></span></div>
      <div class="dc-stats">${d.words.toLocaleString()} words · ${d.chars.toLocaleString()} chars</div>
      <div class="dc-preview">${escapeHtml(d.preview)}</div>`;
    card.addEventListener("click", () => toggleData(d.id, card));
    grid.appendChild(card);
  });
}

function toggleData(id, card) {
  if (state.selectedData.has(id)) {
    state.selectedData.delete(id);
    card.classList.remove("selected");
    card.querySelector(".check").textContent = "";
  } else {
    state.selectedData.add(id);
    card.classList.add("selected");
    card.querySelector(".check").textContent = "✓";
  }
  $("#buildVocabBtn").disabled = state.selectedData.size === 0;
  $("#dataTotals").textContent = state.selectedData.size
    ? `${state.selectedData.size} corpus(es) selected.`
    : "";
}

$("#buildVocabBtn").addEventListener("click", async () => {
  const status = $("#dataStatus");
  status.textContent = "Building vocabulary…";
  status.className = "status";
  const res = await api("/api/select-data", { ids: [...state.selectedData] });
  if (!res.ok) { status.textContent = res.error; status.className = "status err"; return; }
  state.vocabBuilt = true;
  status.textContent = `✓ Vocab of ${res.vocab_size} unique characters from ${res.num_tokens.toLocaleString()} tokens.`;
  status.className = "status ok";
  markDone(0);
  tokenize(); // refresh step 2 if visited later
});

// ---- STEP 1: tokenization --------------------------------------------------
async function tokenize() {
  if (!state.vocabBuilt) { $("#tokenOutput").innerHTML = '<span class="status">Select & build data in Step 1 first.</span>'; return; }
  const res = await api("/api/tokenize", { text: $("#tokenInput").value });
  if (!res.ok) { $("#tokenOutput").innerHTML = `<span class="status err">${res.error}</span>`; return; }
  $("#tokenOutput").innerHTML = res.tokens.map((t) =>
    `<div class="token-chip"><span class="tc-char">${escapeHtml(t.char === " " ? "␣" : t.char)}</span><span class="tc-id">${t.id}</span></div>`
  ).join("");
  $("#vocabInfo").innerHTML = `Vocabulary size: <code>${res.vocab_size}</code>. Encoded: <code>[${res.encoded.join(", ")}]</code>`;
  markDone(1);
}
$("#tokenInput").addEventListener("input", debounce(tokenize, 200));

// ---- STEP 2: embeddings ----------------------------------------------------
async function loadEmbeddings() {
  const d = await api("/api/embeddings-demo");
  const words = d.words;
  let html = '<table><tr><th>word</th><th>embedding vector</th>';
  words.forEach((w) => (html += `<th>${w}</th>`));
  html += "</tr>";
  words.forEach((w, i) => {
    html += `<tr><td><b>${w}</b></td><td class="embvec">[${d.embeddings[w].join(", ")}]</td>`;
    d.similarity[i].forEach((sim) => {
      const g = Math.round(((sim + 1) / 2) * 180);
      html += `<td style="background:rgba(255,122,26,${Math.max(0, sim).toFixed(2)})">${sim.toFixed(3)}</td>`;
    });
    html += "</tr>";
  });
  html += "</table>";
  html += '<div class="vocab-info">Notice: cat/dog/kitten score ~0.99 with each other, but cat vs car is negative — different meaning, far apart.</div>';
  $("#embeddingDemo").innerHTML = html;
  markDone(2);
}

// ---- STEP 3: attention -----------------------------------------------------
async function loadAttention() {
  const d = await api("/api/attention-demo");
  let html = `<div class="vocab-info">Raw relevance scores for <code>"${d.focus}"</code>: `;
  html += Object.entries(d.scores).map(([k, v]) => `${k}=${v}`).join(", ");
  html += " → after softmax:</div>";
  Object.entries(d.weights).forEach(([word, w]) => {
    html += `<div class="attn-row"><div class="attn-word">${word}</div>
      <div class="attn-bar-wrap"><div class="attn-bar" style="width:${(w * 100).toFixed(0)}%">${w.toFixed(3)}</div></div></div>`;
  });
  html += '<div class="note">The model learns that <code>"sat"</code> should pay most attention to <code>"cat"</code> — that\'s who did the sitting.</div>';
  $("#attentionDemo").innerHTML = html;
  markDone(3);
}

// ---- STEP 4: architecture --------------------------------------------------
async function loadArchitectures() {
  const { presets } = await api("/api/architectures");
  const grid = $("#archList");
  grid.innerHTML = "";
  Object.entries(presets).forEach(([id, p]) => {
    const card = document.createElement("div");
    card.className = "arch-card" + (id === state.arch ? " selected" : "");
    card.dataset.id = id;
    card.innerHTML = `<div class="ac-name">${p.label}</div>
      <div class="ac-spec">${p.n_layer} layers · ${p.n_head} heads<br>${p.n_embd}-dim embeddings</div>`;
    card.addEventListener("click", () => {
      state.arch = id;
      $$(".arch-card").forEach((c) => c.classList.toggle("selected", c.dataset.id === id));
      $("#archStatus").textContent = `Selected: ${p.label}`;
      markDone(4);
    });
    grid.appendChild(card);
  });
  markDone(4);
}

// ---- STEP 5: hyperparameters ----------------------------------------------
function updateTotalSteps() {
  const epochs = +$("#hpEpochs").value;
  const steps = +$("#hpSteps").value;
  $("#totalStepsNote").innerHTML =
    `Total training steps = <code>${steps} steps/epoch × ${epochs} epochs = ${(steps * epochs).toLocaleString()}</code> weight updates.`;
}
["hpEpochs", "hpSteps"].forEach((id) => $("#" + id).addEventListener("input", updateTotalSteps));

// ---- STEP 6: training ------------------------------------------------------
let pollTimer = null;
const lossData = [];

$("#trainBtn").addEventListener("click", async () => {
  if (!state.vocabBuilt) { setTrainStatus("Select training data in Step 1 first.", "err"); return; }
  const body = {
    arch: state.arch,
    block_size: +$("#hpBlock").value,
    batch_size: +$("#hpBatch").value,
    epochs: +$("#hpEpochs").value,
    max_steps: +$("#hpSteps").value,
    learning_rate: +$("#hpLr").value,
    weight_decay: +$("#hpWd").value,
    eval_interval: 50,
  };
  const res = await api("/api/train", body);
  if (!res.ok) { setTrainStatus(res.error, "err"); return; }
  lossData.length = 0;
  $("#trainBtn").disabled = true;
  $("#stopBtn").disabled = false;
  setTrainStatus("Training…");
  pollTimer = setInterval(pollTrain, 500);
});

$("#stopBtn").addEventListener("click", async () => {
  await api("/api/train/stop", {});
  setTrainStatus("Stopping…");
});

async function pollTrain() {
  const s = await api("/api/train/status");
  if (s.history && s.history.length) {
    lossData.length = 0;
    s.history.forEach((h) => lossData.push(h));
    drawLoss();
  }
  $("#trainMetrics").innerHTML =
    `Step <b>${s.step}/${s.total_steps}</b> · Loss <b>${s.loss}</b> · ${s.elapsed}s · Params <b>${(s.num_params || 0).toLocaleString()}</b>`;
  if (s.sample) $("#liveSample").textContent = s.sample;

  if (s.status === "done" || s.status === "stopped" || s.status === "error") {
    clearInterval(pollTimer);
    $("#trainBtn").disabled = false;
    $("#stopBtn").disabled = true;
    if (s.status === "error") { setTrainStatus(s.message, "err"); return; }
    state.trained = true;
    markDone(6);
    setTrainStatus(s.status === "done" ? `✓ Done in ${s.elapsed}s — head to Step 8!` : "Stopped.", s.status === "done" ? "ok" : "");
  } else {
    setTrainStatus(`Training… ${s.step}/${s.total_steps}`);
  }
}

function setTrainStatus(msg, cls) { const e = $("#trainStatus"); e.textContent = msg; e.className = "status " + (cls || ""); }

function drawLoss() {
  const c = $("#lossChart");
  const ctx = c.getContext("2d");
  const W = c.width, H = c.height, pad = 34;
  ctx.clearRect(0, 0, W, H);
  if (lossData.length < 2) return;
  const losses = lossData.map((d) => d.loss);
  const maxL = Math.max(...losses), minL = Math.min(...losses);
  const maxStep = lossData[lossData.length - 1].step;
  const x = (step) => pad + (step / maxStep) * (W - pad - 10);
  const y = (loss) => H - pad - ((loss - minL) / (maxL - minL || 1)) * (H - pad - 10);

  // axes
  ctx.strokeStyle = "#2a3340"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad, 10); ctx.lineTo(pad, H - pad); ctx.lineTo(W - 10, H - pad); ctx.stroke();
  ctx.fillStyle = "#8b98a8"; ctx.font = "11px monospace";
  ctx.fillText(maxL.toFixed(2), 2, 16);
  ctx.fillText(minL.toFixed(2), 2, H - pad);
  ctx.fillText("steps", W / 2, H - 8);

  // line
  ctx.strokeStyle = "#ff7a1a"; ctx.lineWidth = 2; ctx.beginPath();
  lossData.forEach((d, i) => {
    const px = x(d.step), py = y(d.loss);
    i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
  });
  ctx.stroke();
}

// ---- STEP 7: generation ----------------------------------------------------
$("#genTemp").addEventListener("input", () => ($("#genTempVal").textContent = $("#genTemp").value));

$("#genBtn").addEventListener("click", async () => {
  if (!state.trained) { setGenStatus("Train a model first (Step 7).", "err"); return; }
  setGenStatus("Generating…");
  $("#genBtn").disabled = true;
  const res = await api("/api/generate", {
    prompt: $("#genPrompt").value,
    max_new_tokens: +$("#genMax").value,
    temperature: +$("#genTemp").value,
    top_k: +$("#genTopK").value,
    top_p: +$("#genTopP").value,
  });
  $("#genBtn").disabled = false;
  if (!res.ok) { setGenStatus(res.error, "err"); return; }
  $("#genOutput").textContent = res.text;
  setGenStatus("✓ Done", "ok");
});
function setGenStatus(msg, cls) { const e = $("#genStatus"); e.textContent = msg; e.className = "status " + (cls || ""); }

// ---- utils -----------------------------------------------------------------
function escapeHtml(s) { return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }
function debounce(fn, ms) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }

// ---- init ------------------------------------------------------------------
loadDatasets();
goTo(0);
