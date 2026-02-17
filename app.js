/* ─────────────────────────────────────────────────────────────────────────────
   XAI Platform — app.js
   ───────────────────────────────────────────────────────────────────────────── */

'use strict';

let SESSION_DATA = {};

// ── Drag & Drop ────────────────────────────────────────────────────────────────
const dz = document.getElementById('drop-zone');
dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('drag'); });
dz.addEventListener('dragleave', () => dz.classList.remove('drag'));
dz.addEventListener('drop', e => {
  e.preventDefault(); dz.classList.remove('drag');
  const f = e.dataTransfer.files[0];
  if (f) handleFile(f);
});
document.getElementById('file-input').addEventListener('change', e => {
  if (e.target.files[0]) handleFile(e.target.files[0]);
});

// ── File upload ────────────────────────────────────────────────────────────────
function handleFile(file) {
  if (!file.name.endsWith('.csv')) {
    showStatus('upload-status', 'danger', '❌ Only CSV files are supported.');
    return;
  }
  const fd = new FormData();
  fd.append('file', file);
  showStatus('upload-status', 'info', '⏳ Uploading and analyzing dataset…');
  fetch('/upload', { method: 'POST', body: fd })
    .then(r => r.json())
    .then(data => {
      if (data.error) { showStatus('upload-status', 'danger', '❌ ' + data.error); return; }
      SESSION_DATA = data;
      showStatus('upload-status', 'success', `✅ Loaded ${data.rows} rows × ${data.cols} columns`);
      renderAnalysis(data);
      setStep(2);
    })
    .catch(err => showStatus('upload-status', 'danger', '❌ ' + err));
}

// ── Render analysis ────────────────────────────────────────────────────────────
function renderAnalysis(d) {
  const stats = [
    { label: 'Rows',         val: d.rows.toLocaleString() },
    { label: 'Columns',      val: d.cols },
    { label: 'Numeric',      val: d.numeric_cols.length },
    { label: 'Categorical',  val: d.categorical_cols.length },
    { label: 'Missing Vals', val: d.total_missing },
    { label: 'Duplicates',   val: d.duplicates },
  ];
  document.getElementById('overview-stats').innerHTML = stats.map(s =>
    `<div class="stat-box"><div class="stat-label">${s.label}</div><div class="stat-value">${s.val}</div></div>`
  ).join('');

  document.getElementById('col-tbody').innerHTML = d.columns.map(c => {
    const typePill = c.kind === 'numeric'
      ? `<span class="pill pill-num">numeric</span>`
      : `<span class="pill pill-cat">categorical</span>`;
    const missPill = c.missing > 0
      ? `<span class="pill pill-miss">${c.missing} (${c.miss_pct}%)</span>`
      : `<span class="pill pill-ok">none</span>`;
    return `<tr>
      <td><strong>${c.name}</strong></td>
      <td>${typePill}</td>
      <td style="color:var(--muted)">${c.dtype}</td>
      <td>${missPill}</td>
      <td style="color:var(--accent4)">${c.unique}</td>
      <td style="color:var(--muted);font-size:.75rem">${(c.samples||[]).slice(0,3).join(', ')}</td>
    </tr>`;
  }).join('');

  const cols = d.preview_cols;
  const rows = d.preview_rows;
  let tbl = `<table><thead><tr>${cols.map(c=>`<th>${c}</th>`).join('')}</tr></thead><tbody>`;
  rows.forEach(r => { tbl += `<tr>${cols.map(c=>`<td>${r[c]??''}</td>`).join('')}</tr>`; });
  tbl += '</tbody></table>';
  document.getElementById('preview-wrap').innerHTML = tbl;

  if (d.dist_charts) {
    document.getElementById('dist-charts').innerHTML = d.dist_charts.map(ch =>
      `<div><img src="data:image/png;base64,${ch.img}" class="chart-img" style="margin-bottom:.5rem"/>
      <p style="font-size:.75rem;color:var(--muted);text-align:center;font-family:var(--font-mono)">${ch.col}</p></div>`
    ).join('');
  }

  const qItems = [];
  if (d.total_missing > 0) qItems.push(`<div style="color:var(--accent4)">⚠️ ${d.total_missing} missing values detected</div>`);
  if (d.duplicates    > 0) qItems.push(`<div style="color:var(--danger)">❌ ${d.duplicates} duplicate rows</div>`);
  if (qItems.length === 0)  qItems.push(`<div style="color:var(--success)">✅ No major quality issues</div>`);
  document.getElementById('quality-content').innerHTML = qItems.join('');
  document.getElementById('quality-box').style.display = '';

  const sel = document.getElementById('target-select');
  sel.innerHTML = d.columns.map(c => `<option value="${c.name}">${c.name}</option>`).join('');
  sel.addEventListener('change', updateTaskBadge);
  updateTaskBadge();

  show('analysis-card');
  show('target-card');
}

function updateTaskBadge() {
  const col  = document.getElementById('target-select').value;
  const meta = (SESSION_DATA.columns||[]).find(c => c.name === col);
  if (!meta) return;
  const isClass = meta.kind === 'categorical' || meta.unique <= 10;
  document.getElementById('task-badge').innerHTML =
    isClass ? `🏷️ Classification <span style="color:var(--muted)">(${meta.unique} classes)</span>`
            : `📈 Regression`;
}

// ── Train ──────────────────────────────────────────────────────────────────────
function trainModels() {
  const target = document.getElementById('target-select').value;
  show('training-card');
  setStep(3);
  animateProgress([
    'Preprocessing data…',
    'Training Random Forest…',
    'Training Gradient Boosting…',
    'Training Linear Model…',
    'Evaluating models…',
  ]);

  fetch('/train', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ target }),
  })
  .then(r => r.json())
  .then(data => {
    if (data.error) { showAlert('training-card', 'danger', data.error); return; }
    SESSION_DATA.models       = data.models;
    SESSION_DATA.task         = data.task;
    SESSION_DATA.features     = data.features;
    SESSION_DATA.feature_info = data.feature_info;
    SESSION_DATA.best_model   = data.best_model;

    finishProgress();
    renderModels(data);
    renderModelSummary(data.model_summary, data.best_model);
    buildPredForm(data);
    renderGlobalImportance(data);

    show('predict-card');
    show('xai-card');
    setStep(4);
  })
  .catch(err => showAlert('training-card', 'danger', String(err)));
}

let progressTimer = null;
let progressPhase = 0;

function animateProgress(phases) {
  progressPhase = 0;
  let pct = 0;
  const bar    = document.getElementById('prog-bar');
  const status = document.getElementById('train-status');
  const pctEl  = document.getElementById('train-pct');
  bar.style.width = '0%';
  clearInterval(progressTimer);
  progressTimer = setInterval(() => {
    pct = Math.min(pct + 1.2, 90);
    bar.style.width = pct + '%';
    pctEl.textContent = Math.round(pct) + '%';
    const idx = Math.min(Math.floor(pct / (90 / phases.length)), phases.length - 1);
    status.textContent = phases[idx];
  }, 120);
}

function finishProgress() {
  clearInterval(progressTimer);
  document.getElementById('prog-bar').style.width  = '100%';
  document.getElementById('train-pct').textContent = '100%';
  document.getElementById('train-status').textContent = '✅ Training complete!';
}

function renderModels(data) {
  const grid = document.getElementById('models-grid');
  grid.innerHTML = data.models.map(m => {
    const isBest     = m.name === data.best_model;
    const metricName = data.task === 'classification' ? 'Accuracy' : 'R²';
    const metricVal  = data.task === 'classification'
      ? (m.accuracy * 100).toFixed(1) + '%'
      : m.r2.toFixed(4);
    return `<div class="model-card ${isBest ? 'best' : ''}">
      <div class="model-name">
        ${m.name} ${isBest ? '<span class="best-badge">BEST</span>' : ''}
      </div>
      <div style="font-size:.75rem;color:var(--muted);margin-bottom:.5rem">${data.task}</div>
      <div class="metric-row">
        <span class="metric-label">${metricName}</span>
        <span class="metric-val">${metricVal}</span>
      </div>
      ${data.task === 'classification' ? `
      <div class="metric-row">
        <span class="metric-label">CV Score</span>
        <span class="metric-val">${(m.cv_score*100).toFixed(1)}%</span>
      </div>` : `
      <div class="metric-row">
        <span class="metric-label">RMSE</span>
        <span class="metric-val">${m.rmse.toFixed(4)}</span>
      </div>`}
    </div>`;
  }).join('');
}

// ── Model summary narrative (after training) ───────────────────────────────────
function renderModelSummary(summary, bestModelName) {
  if (!summary) return;
  const wrap = document.getElementById('model-summary-wrap');
  const body = document.getElementById('model-summary-body');

  const comparisonHTML = (summary.comparison || []).map(item => {
    const isBest = item.includes('✦ best');
    return `<li class="${isBest ? 'is-best' : ''}">${item}</li>`;
  }).join('');

  body.innerHTML = `
    <div class="narr-body-row">
      <div class="narr-body-label">📋 Overview</div>
      <div>${summary.overview}</div>
    </div>
    <div class="narr-body-row">
      <div class="narr-body-label">🏆 Winner</div>
      <div>${summary.best}</div>
    </div>
    <div class="narr-body-row">
      <div class="narr-body-label">📊 Model Comparison</div>
      <ul class="narr-comparison-list">${comparisonHTML}</ul>
    </div>
    ${summary.features_note ? `
    <div class="narr-body-row">
      <div class="narr-body-label">🔑 Top Features</div>
      <div>${summary.features_note}</div>
    </div>` : ''}
    <div class="narr-advice">💡 ${summary.advice}</div>
  `;

  wrap.classList.remove('hidden');
}

// ── Prediction form ────────────────────────────────────────────────────────────
function buildPredForm(data) {
  const wrap = document.getElementById('pred-inputs');
  wrap.innerHTML = data.features.map(f => {
    const info = data.feature_info[f];
    if (info.kind === 'categorical') {
      return `<div class="field-group" data-feat="${f}">
        <label>${f}</label>
        <select name="${f}">
          ${info.values.map(v => `<option value="${v}">${v}</option>`).join('')}
        </select>
      </div>`;
    } else {
      return `<div class="field-group" data-feat="${f}">
        <label>${f} <span style="color:var(--muted);font-size:.7rem">[${info.min?.toFixed(2)} – ${info.max?.toFixed(2)}]</span></label>
        <input type="number" name="${f}" step="any"
               value="${((info.min||0)+(info.max||1))/2|0}"
               placeholder="${f}"/>
      </div>`;
    }
  }).join('');
}

function randomFill() {
  (SESSION_DATA.features || []).forEach(f => {
    const info = SESSION_DATA.feature_info[f];
    const el   = document.querySelector(`[data-feat="${f}"] input, [data-feat="${f}"] select`);
    if (!el) return;
    if (info.kind === 'categorical') {
      el.value = info.values[Math.floor(Math.random() * info.values.length)];
    } else {
      el.value = (info.min + Math.random() * (info.max - info.min)).toFixed(3);
    }
  });
}

function predict() {
  const row = {};
  (SESSION_DATA.features || []).forEach(f => {
    const el = document.querySelector(`[data-feat="${f}"] input, [data-feat="${f}"] select`);
    row[f] = el ? el.value : '';
  });

  fetch('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ row }),
  })
  .then(r => r.json())
  .then(data => {
    if (data.error) { alert(data.error); return; }
    renderPrediction(data);
    renderLocalExplanation(data);
    renderNarrative(data.narrative);

    if (data.shap_chart) {
      const img = document.getElementById('shap-chart-img');
      img.src = 'data:image/png;base64,' + data.shap_chart;
      img.classList.remove('hidden');
      document.getElementById('shap-placeholder').classList.add('hidden');
    }
    if (data.lime_chart) {
      const img = document.getElementById('lime-chart-img');
      img.src = 'data:image/png;base64,' + data.lime_chart;
      img.classList.remove('hidden');
      document.getElementById('lime-placeholder').classList.add('hidden');
    }
  })
  .catch(err => alert(String(err)));
}

function renderPrediction(data) {
  const wrap = document.getElementById('pred-result-wrap');
  const confHtml = data.confidence !== null
    ? `<div><div class="pred-label">CONFIDENCE</div>
        <div class="pred-confidence">${(data.confidence*100).toFixed(1)}%</div></div>`
    : '';
  const probsHtml = data.class_probs
    ? `<div style="margin-top:1rem;width:100%">
        <div class="pred-label" style="margin-bottom:.5rem">CLASS PROBABILITIES</div>
        <div style="display:flex;gap:.5rem;flex-wrap:wrap">
          ${Object.entries(data.class_probs).map(([cls, prob]) =>
            `<div style="background:var(--bg3);border:1px solid var(--border);border-radius:6px;padding:.4rem .8rem">
              <div style="font-family:var(--font-mono);font-size:.7rem;color:var(--muted)">${cls}</div>
              <div style="font-family:var(--font-mono);font-size:.95rem;color:var(--accent2)">${(prob*100).toFixed(1)}%</div>
            </div>`
          ).join('')}
        </div>
      </div>` : '';

  wrap.innerHTML = `<div class="pred-result">
    <div>
      <div class="pred-label">PREDICTION</div>
      <div class="pred-value">${data.prediction}</div>
    </div>
    ${confHtml}
    ${probsHtml}
  </div>`;
  wrap.classList.remove('hidden');
}

// ── Narrative tab rendering ────────────────────────────────────────────────────
function renderNarrative(n) {
  if (!n) return;

  document.getElementById('narrative-placeholder').classList.add('hidden');
  document.getElementById('narrative-content').classList.remove('hidden');

  // Verdict
  document.getElementById('narr-verdict').innerHTML = n.verdict || '';

  // Confidence
  if (n.confidence_note) {
    document.getElementById('narr-confidence').innerHTML = n.confidence_note;
    document.getElementById('narr-confidence-wrap').style.display = '';
  }

  // Drivers
  const driversList = document.getElementById('narr-drivers');
  driversList.innerHTML = (n.drivers || []).map(d => {
    const isPos = d.includes('higher');
    return `<li class="${isPos ? 'positive' : 'negative'}">${d}</li>`;
  }).join('') || '<li style="color:var(--muted)">No dominant drivers detected.</li>';

  // Neutral features
  if (n.neutrals && n.neutrals.length > 0) {
    document.getElementById('narr-neutrals').textContent =
      `These features had negligible impact: ${n.neutrals.join(', ')}.`;
    document.getElementById('narr-neutrals-wrap').style.display = '';
  }

  // Global context
  if (n.global_note) {
    document.getElementById('narr-global').innerHTML = n.global_note;
    document.getElementById('narr-global-wrap').style.display = '';
  }

  // Method note
  document.getElementById('narr-method').innerHTML = n.method_note || '';
}

// ── XAI bar charts ─────────────────────────────────────────────────────────────
function renderGlobalImportance(data) {
  if (!data.global_importance) return;
  const bars   = document.getElementById('global-bars');
  const items  = data.global_importance;
  const maxVal = Math.max(...items.map(i => Math.abs(i.importance)));
  bars.innerHTML = items.map((item, idx) => {
    const pct = Math.abs(item.importance) / maxVal * 100;
    const hue = 160 + idx * 15;
    return `<div class="feat-bar-row">
      <div class="feat-name" title="${item.feature}">${item.feature}</div>
      <div class="feat-bar-bg">
        <div class="feat-bar-fill" style="width:${pct}%;background:hsl(${hue},70%,55%)"></div>
      </div>
      <div class="feat-val">${item.importance.toFixed(4)}</div>
    </div>`;
  }).join('');

  if (data.global_chart) {
    const img = document.getElementById('global-chart');
    img.src = 'data:image/png;base64,' + data.global_chart;
    img.classList.remove('hidden');
  }
}

function renderLocalExplanation(data) {
  if (!data.local_explanation) return;
  const bars   = document.getElementById('local-bars');
  const items  = data.local_explanation;
  const maxVal = Math.max(...items.map(i => Math.abs(i.contribution)));
  bars.innerHTML = items.map(item => {
    const pct   = maxVal > 0 ? Math.abs(item.contribution) / maxVal * 100 : 0;
    const color = item.contribution >= 0 ? 'var(--accent)' : 'var(--danger)';
    const dir   = item.contribution >= 0 ? '▲' : '▼';
    return `<div class="feat-bar-row">
      <div class="feat-name" title="${item.feature}: ${item.value}">
        ${item.feature}<span style="color:var(--muted);font-size:.65rem"> (${item.value})</span>
      </div>
      <div class="feat-bar-bg">
        <div class="feat-bar-fill" style="width:${pct}%;background:${color}"></div>
      </div>
      <div class="feat-val" style="color:${color}">${dir} ${Math.abs(item.contribution).toFixed(4)}</div>
    </div>`;
  }).join('');
}

// ── Tabs ───────────────────────────────────────────────────────────────────────
function switchTab(name) {
  document.querySelectorAll('#analysis-card .tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('#analysis-card .tab-content').forEach(t => t.classList.remove('active'));
  event.currentTarget.classList.add('active');
  document.getElementById('tab-' + name).classList.add('active');
}

function switchXaiTab(name) {
  document.querySelectorAll('#xai-card .tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.xai-pane').forEach(p => p.classList.add('hidden'));
  event.currentTarget.classList.add('active');
  document.getElementById('xai-' + name).classList.remove('hidden');
}

// ── Helpers ────────────────────────────────────────────────────────────────────
function show(id) {
  document.getElementById(id).classList.remove('hidden');
}

function showStatus(id, type, msg) {
  document.getElementById(id).innerHTML = `<div class="alert alert-${type}">${msg}</div>`;
  document.getElementById(id).classList.remove('hidden');
}

function showAlert(containerId, type, msg) {
  const el = document.createElement('div');
  el.className = `alert alert-${type}`;
  el.style.marginTop = '1rem';
  el.textContent = msg;
  document.getElementById(containerId).appendChild(el);
}

function setStep(n) {
  for (let i = 1; i <= 4; i++) {
    document.getElementById(`step${i}-card`).classList.toggle('active', i === n);
  }
}
