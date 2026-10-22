/**
 * VerifAI — main.js
 * Handles: tab switching, drag-and-drop, form submission, loading states, results rendering
 */

/* ============================================================
   State
   ============================================================ */
const state = {
  imageFile: null,
  videoFile: null,
  charts: {},        // Chart.js instances
  currentPanel: 'text',
};

/* ============================================================
   DOM References
   ============================================================ */
const $ = id => document.getElementById(id);
const $$ = sel => document.querySelectorAll(sel);

/* ============================================================
   Helpers
   ============================================================ */
function fmt(n, dec = 1) { return (n * 100).toFixed(dec) + '%'; }
function fmtBytes(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}
function clamp(v, lo = 0, hi = 1) { return Math.max(lo, Math.min(hi, v)); }

/* ============================================================
   Tab Management
   ============================================================ */
function initTabs() {
  $$('.mod-tab').forEach(tab => {
    tab.addEventListener('click', () => switchTab(tab.dataset.panel));
  });
}

function switchTab(panelId) {
  $$('.mod-tab').forEach(t => {
    const active = t.dataset.panel === panelId;
    t.classList.toggle('active', active);
    t.setAttribute('aria-selected', String(active));
  });
  $$('.panel').forEach(p => {
    const active = p.id === panelId;
    p.classList.toggle('active', active);
    p.hidden = !active;
  });
  state.currentPanel = panelId.replace('panel-', '');
}

/* ============================================================
   Text input — char counter
   ============================================================ */
function initTextInput() {
  const ta = $('newsText');
  ta.addEventListener('input', () => {
    const n = ta.value.length;
    $('charCount').textContent = n.toLocaleString() + ' character' + (n !== 1 ? 's' : '');
    updateActiveChips();
  });

  $('clearText').addEventListener('click', () => {
    ta.value = '';
    $('charCount').textContent = '0 characters';
    updateActiveChips();
  });
}

/* ============================================================
   Image Upload — drag & drop + file input
   ============================================================ */
function initImageUpload() {
  const dz = $('imageDropzone');
  const fi = $('imageFile');

  $('browseImage').addEventListener('click', e => { e.stopPropagation(); fi.click(); });
  dz.addEventListener('click', () => fi.click());
  dz.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fi.click(); } });

  fi.addEventListener('change', () => {
    if (fi.files[0]) setImageFile(fi.files[0]);
  });

  dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('dragover'); });
  dz.addEventListener('dragleave', () => dz.classList.remove('dragover'));
  dz.addEventListener('drop', e => {
    e.preventDefault();
    dz.classList.remove('dragover');
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith('image/')) setImageFile(f);
    else showToast('Please drop an image file (JPG, PNG, WebP…)', 'warn');
  });

  $('clearImage').addEventListener('click', clearImage);
}

function setImageFile(file) {
  state.imageFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    $('imgPreview').style.display = 'block';
    $('imgDzContent').style.display = 'none';
    $('previewImg').src = e.target.result;
    $('imgName').textContent = file.name;
    $('imgSize').textContent = fmtBytes(file.size);
  };
  reader.readAsDataURL(file);
  setBadge('image', true);
  updateActiveChips();
}

function clearImage() {
  state.imageFile = null;
  $('imageFile').value = '';
  $('imgPreview').style.display = 'none';
  $('imgDzContent').style.display = 'block';
  $('previewImg').src = '';
  setBadge('image', false);
  updateActiveChips();
}

/* ============================================================
   Video Upload
   ============================================================ */
function initVideoUpload() {
  const dz = $('videoDropzone');
  const fi = $('videoFile');

  $('browseVideo').addEventListener('click', e => { e.stopPropagation(); fi.click(); });
  dz.addEventListener('click', () => fi.click());
  dz.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fi.click(); } });

  fi.addEventListener('change', () => {
    if (fi.files[0]) setVideoFile(fi.files[0]);
  });

  dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('dragover'); });
  dz.addEventListener('dragleave', () => dz.classList.remove('dragover'));
  dz.addEventListener('drop', e => {
    e.preventDefault();
    dz.classList.remove('dragover');
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith('video/')) setVideoFile(f);
    else showToast('Please drop a video file (MP4, AVI, MOV…)', 'warn');
  });

  $('clearVideo').addEventListener('click', clearVideo);
}

function setVideoFile(file) {
  state.videoFile = file;
  $('vidPreview').style.display = 'block';
  $('vidDzContent').style.display = 'none';
  $('vidName').textContent = file.name;
  $('vidSize').textContent = fmtBytes(file.size);
  setBadge('video', true);
  updateActiveChips();
}

function clearVideo() {
  state.videoFile = null;
  $('videoFile').value = '';
  $('vidPreview').style.display = 'none';
  $('vidDzContent').style.display = 'block';
  setBadge('video', false);
  updateActiveChips();
}

/* ============================================================
   Badge & Active Chips
   ============================================================ */
function setBadge(modality, on) {
  const b = $(`badge-${modality}`);
  if (b) b.classList.toggle('visible', on);
}

function updateActiveChips() {
  const chips = $('aiChips');
  const active = [];
  if ($('newsText').value.trim()) active.push({ label: '📝 Text', key: 'text' });
  if (state.imageFile) active.push({ label: '🖼️ Image', key: 'image' });
  if (state.videoFile) active.push({ label: '🎥 Video', key: 'video' });

  if (active.length === 0) {
    chips.innerHTML = '<span class="ai-empty">No input yet — add text, image or video above</span>';
  } else {
    chips.innerHTML = active.map(a =>
      `<span class="ai-chip" aria-label="${a.label} input active">${a.label}</span>`
    ).join('');
  }
}

/* ============================================================
   Form Submission
   ============================================================ */
function initForm() {
  $('analysisForm').addEventListener('submit', async e => {
    e.preventDefault();
    const text = $('newsText').value.trim();
    if (!text && !state.imageFile && !state.videoFile) {
      showToast('Please provide at least one input (text, image, or video)', 'warn');
      return;
    }
    await runAnalysis(text);
  });

  $('reAnalyzeBtn').addEventListener('click', resetUI);
}

async function runAnalysis(text) {
  // Hide results, show loader
  $('resultsWrap').style.display = 'none';
  $('loadingWrap').style.display = 'flex';
  $('analyzeBtn').disabled = true;
  $('loadingBar').style.width = '0%';

  const steps = [
    text        ? 'Processing text with BERT model…'       : null,
    state.imageFile ? 'Running ELA & forensic image analysis…' : null,
    state.videoFile ? 'Extracting and analysing video frames…' : null,
    'Fusing multimodal predictions…',
    'Generating report…',
  ].filter(Boolean);

  let stepEl = null;
  function addStep(label) {
    if (stepEl) stepEl.classList.remove('active'), stepEl.classList.add('done');
    stepEl = document.createElement('div');
    stepEl.className = 'ls-step active';
    stepEl.innerHTML = `<span>${label}</span>`;
    $('loadingSteps').appendChild(stepEl);
  }

  // Simulate step progression
  let stepIdx = 0;
  addStep(steps[stepIdx]);
  const stepTimer = setInterval(() => {
    stepIdx++;
    if (stepIdx < steps.length) {
      addStep(steps[stepIdx]);
      $('loadingBar').style.width = `${Math.round((stepIdx / steps.length) * 80)}%`;
    } else {
      clearInterval(stepTimer);
    }
  }, 1200);

  // Build FormData
  const fd = new FormData();
  if (text) fd.append('text', text);
  if (state.imageFile) fd.append('image', state.imageFile, state.imageFile.name);
  if (state.videoFile) fd.append('video', state.videoFile, state.videoFile.name);

  try {
    const resp = await fetch('/analyze', { method: 'POST', body: fd });
    clearInterval(stepTimer);
    $('loadingBar').style.width = '100%';

    const data = await resp.json();
    if (!resp.ok || data.error) throw new Error(data.error || `HTTP ${resp.status}`);

    await new Promise(r => setTimeout(r, 400)); // brief pause to show 100%
    showResults(data);
  } catch (err) {
    clearInterval(stepTimer);
    $('loadingWrap').style.display = 'none';
    $('analyzeBtn').disabled = false;
    showToast(`Analysis failed: ${err.message}`, 'error');
  }
}

/* ============================================================
   Results Rendering
   ============================================================ */
function showResults(data) {
  $('loadingWrap').style.display = 'none';
  $('analyzeBtn').disabled = false;

  const isFake = data.verdict === 'FAKE';

  // Verdict card
  const vc = $('verdictCard');
  vc.className = `verdict-card ${isFake ? 'fake-card' : 'real-card'}`;
  $('vcEmoji').textContent = isFake ? '🚨' : '✅';
  const vv = $('vcVerdict');
  vv.textContent = data.verdict;
  vv.className = `vc-verdict ${isFake ? 'fake-text' : 'real-text'}`;
  $('vcSub').textContent = `${data.modalities_used?.length || 0} modalities analysed · ${data.fusion_method || ''}`;
  const vb = $('vcBadge');
  vb.textContent = data.verdict;
  vb.className = `vc-badge ${isFake ? 'fake-badge' : 'real-badge'}`;

  // Prob bars
  const fp = clamp(data.fake_probability || 0);
  const rp = clamp(data.real_probability || 0);
  setTimeout(() => {
    $('fakeBar').style.width = fmt(fp, 1);
    $('fakeBar').setAttribute('aria-valuenow', Math.round(fp * 100));
    $('realBar').style.width = fmt(rp, 1);
    $('realBar').setAttribute('aria-valuenow', Math.round(rp * 100));
  }, 100);
  $('fakeProbVal').textContent = fmt(fp);
  $('realProbVal').textContent = fmt(rp);

  // Meta row
  const riskBadge = $('riskBadge');
  riskBadge.textContent = data.risk_level || '—';
  const riskColor = data.risk_color || '#94a3b8';
  riskBadge.style.cssText = `background:${riskColor}22;color:${riskColor};border:1px solid ${riskColor}44;`;
  $('confidenceVal').textContent = fmt(clamp(data.confidence || 0));
  $('modalitiesVal').textContent = (data.modalities_used || []).join(' + ') || '—';
  $('fusionVal').textContent = data.fusion_method || '—';

  // Charts
  renderVerdictChart(fp, rp);
  renderModalityChart(data.modality_results || {});

  // Breakdown
  renderBreakdown(data.modality_results || {});

  // Indicators
  const indicators = data.manipulation_indicators || [];
  if (indicators.length) {
    $('indicatorsSection').style.display = '';
    $('indicatorsList').innerHTML = indicators.map(ind =>
      `<li class="ind-item" role="listitem"><span class="ind-icon" aria-hidden="true">⚠️</span>${escHtml(ind)}</li>`
    ).join('');
  } else {
    $('indicatorsSection').style.display = 'none';
  }

  // Per-modality detail sections
  const mr = data.modality_results || {};
  renderTextFeatures(mr.text);
  renderImageDetails(mr.image);
  renderVideoDetails(mr.video);

  // Show results
  $('resultsWrap').style.display = '';
  $('resultsWrap').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/* -------- Verdict donut chart -------- */
function renderVerdictChart(fp, rp) {
  const ctx = $('verdictChart').getContext('2d');
  if (state.charts.verdict) state.charts.verdict.destroy();
  state.charts.verdict = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Fake', 'Real'],
      datasets: [{
        data: [fp * 100, rp * 100],
        backgroundColor: ['rgba(239,68,68,0.7)', 'rgba(16,185,129,0.7)'],
        borderColor:     ['rgba(239,68,68,0.9)', 'rgba(16,185,129,0.9)'],
        borderWidth: 2, hoverOffset: 6,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      cutout: '72%',
      plugins: {
        legend: { position: 'bottom', labels: { color: '#94a3b8', font: { family: 'Outfit', size: 12 }, padding: 16 } },
        tooltip: { callbacks: { label: ctx => ` ${ctx.label}: ${ctx.parsed.toFixed(1)}%` } }
      },
      animation: { animateRotate: true, duration: 900 }
    }
  });
}

/* -------- Modality bar chart -------- */
function renderModalityChart(modalityResults) {
  const labels = Object.keys(modalityResults);
  const fakeData = labels.map(k => (modalityResults[k].fake_probability || 0) * 100);
  const realData = labels.map(k => (modalityResults[k].real_probability || 0) * 100);

  const ctx = $('modalityChart').getContext('2d');
  if (state.charts.modality) state.charts.modality.destroy();
  state.charts.modality = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels.map(l => l.charAt(0).toUpperCase() + l.slice(1)),
      datasets: [
        { label: 'Fake %', data: fakeData, backgroundColor: 'rgba(239,68,68,0.65)', borderColor: 'rgba(239,68,68,0.9)', borderWidth: 2, borderRadius: 6 },
        { label: 'Real %', data: realData, backgroundColor: 'rgba(16,185,129,0.65)', borderColor: 'rgba(16,185,129,0.9)', borderWidth: 2, borderRadius: 6 },
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { position: 'bottom', labels: { color: '#94a3b8', font: { family: 'Outfit', size: 12 }, padding: 16 } } },
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8', font: { family: 'Outfit' } } },
        y: { min: 0, max: 100, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8', font: { family: 'Outfit' }, callback: v => v + '%' } }
      },
      animation: { duration: 900 }
    }
  });
}

/* -------- Modality breakdown cards -------- */
function renderBreakdown(modalityResults) {
  const icons = { text: '📝', image: '🖼️', video: '🎥' };
  const container = $('breakdownCards');
  container.innerHTML = '';

  Object.entries(modalityResults).forEach(([mod, result]) => {
    const v = result.verdict || 'UNCERTAIN';
    const fp = clamp(result.fake_probability || 0.5);
    const conf = clamp(result.confidence || 0.5);
    const vClass = v === 'FAKE' ? 'fake' : v === 'REAL' ? 'real' : 'uncertain';
    const details = result.key_details || {};

    const detailTags = Object.entries(details)
      .slice(0, 5)
      .map(([k, v2]) => {
        const label = k.replace(/_/g, ' ');
        const val = typeof v2 === 'number' ? (v2 < 1 ? (v2 * 100).toFixed(1) + '%' : v2) : String(v2);
        return `<span class="bc-tag" title="${label}: ${val}">${label}: ${val}</span>`;
      }).join('');

    const card = document.createElement('div');
    card.className = 'bc';
    card.setAttribute('role', 'listitem');
    card.innerHTML = `
      <div class="bc-header">
        <div class="bc-title">
          <span class="bc-icon" aria-hidden="true">${icons[mod] || '🔍'}</span>
          ${mod.charAt(0).toUpperCase() + mod.slice(1)} Analysis
        </div>
        <span class="bc-verdict ${vClass}">${v}</span>
      </div>
      <p class="bc-prob">Fake: <strong>${fmt(fp)}</strong> · Confidence: <strong>${fmt(conf)}</strong></p>
      <div class="bc-mini-bar"><div class="bc-mini-fill" style="width:${fp * 100}%;background:${v === 'FAKE' ? 'linear-gradient(90deg,#b91c1c,#ef4444)' : 'linear-gradient(90deg,#059669,#10b981)'}"></div></div>
      <div class="bc-details">${detailTags}</div>
    `;
    container.appendChild(card);
  });
}

/* -------- Text features -------- */
function renderTextFeatures(textResult) {
  if (!textResult) { $('textFeaturesSection').style.display = 'none'; return; }
  const f = textResult.features || {};
  const items = [
    { label: 'Word Count',       val: f.word_count || 0,                 cls: '' },
    { label: 'Sentences',        val: f.sentence_count || 0,             cls: '' },
    { label: 'Clickbait Score',  val: pct(f.clickbait_score),            cls: score2class(f.clickbait_score) },
    { label: 'Emotional Score',  val: pct(f.emotional_score),            cls: score2class(f.emotional_score) },
    { label: 'CAPS Ratio',       val: pct(f.caps_ratio),                 cls: score2class(f.caps_ratio) },
    { label: 'Vague Sources',    val: pct(f.vague_source_score),         cls: score2class(f.vague_source_score) },
    { label: 'Credibility',      val: pct(f.credibility_score),          cls: f.credibility_score > 0.2 ? 'good' : '' },
    { label: 'Exclamations',     val: f.exclamation_count || 0,          cls: '' },
    { label: 'Avg Word Length',  val: (f.avg_word_length || 0) + ' chr', cls: '' },
    { label: 'Model Used',       val: textResult.model_used || '—',      cls: 'highlight' },
  ];
  renderFeatureGrid($('textFeaturesGrid'), items);
  $('textFeaturesSection').style.display = '';
}

/* -------- Image details -------- */
function renderImageDetails(imgResult) {
  if (!imgResult) { $('imageDetailsSection').style.display = 'none'; return; }
  const ela = imgResult.ela_analysis || {};
  const meta = imgResult.metadata || {};
  const visual = imgResult.visual_analysis || {};
  const faces = imgResult.face_detection || {};

  const items = [
    { label: 'ELA Score',          val: pct(ela.ela_score),         cls: score2class(ela.ela_score) },
    { label: 'ELA Mean',           val: ela.ela_mean?.toFixed(4) || '—',  cls: '' },
    { label: 'Has EXIF',           val: meta.has_exif ? 'Yes' : 'No',     cls: meta.has_exif ? 'good' : 'danger' },
    { label: 'Edit Software',      val: meta.software_edit ? meta.software || 'Detected' : 'None', cls: meta.software_edit ? 'danger' : 'good' },
    { label: 'Faces Detected',     val: faces.face_count || 0,            cls: '' },
    { label: 'Color Inconsistency',val: pct(visual.color_inconsistency),  cls: score2class(visual.color_inconsistency) },
    { label: 'Image Size',         val: imgResult.image_size || '—',      cls: 'highlight' },
    { label: 'File Size',          val: (meta.file_size_kb || 0) + ' KB', cls: '' },
  ];
  renderFeatureGrid($('imageDetailsGrid'), items);
  $('imageDetailsSection').style.display = '';
}

/* -------- Video details -------- */
function renderVideoDetails(vidResult) {
  if (!vidResult) { $('videoDetailsSection').style.display = 'none'; return; }
  const vi = vidResult.video_info || {};
  const ta = vidResult.temporal_analysis || {};

  const items = [
    { label: 'Duration',         val: (vi.duration_sec || 0) + 's',       cls: '' },
    { label: 'FPS',              val: vi.fps || '—',                       cls: '' },
    { label: 'Resolution',       val: vi.resolution || '—',                cls: 'highlight' },
    { label: 'Total Frames',     val: vi.total_frames || 0,                cls: '' },
    { label: 'Analysed Frames',  val: vi.analyzed_frames || 0,            cls: '' },
    { label: 'Consistency',      val: pct(ta.consistency_score),           cls: ta.consistency_score > 0.6 ? 'good' : 'danger' },
    { label: 'Anomalous Frames', val: ta.anomalous_frames || 0,           cls: ta.anomalous_frames > 0 ? 'danger' : 'good' },
    { label: 'Fake Prob Std',    val: pct(ta.fake_prob_std),               cls: score2class(ta.fake_prob_std) },
  ];
  renderFeatureGrid($('videoDetailsGrid'), items);
  $('videoDetailsSection').style.display = '';

  // Frame chart
  const frames = vidResult.frame_analysis || [];
  if (frames.length > 0) {
    renderFrameChart(frames);
    $('frameChartWrap').style.display = '';
  }
}

function renderFrameChart(frames) {
  const ctx = $('frameChart').getContext('2d');
  if (state.charts.frame) state.charts.frame.destroy();
  state.charts.frame = new Chart(ctx, {
    type: 'line',
    data: {
      labels: frames.map(f => f.timestamp + 's'),
      datasets: [{
        label: 'Fake Probability',
        data: frames.map(f => (f.fake_probability * 100).toFixed(1)),
        borderColor: 'rgba(239,68,68,0.8)',
        backgroundColor: 'rgba(239,68,68,0.1)',
        borderWidth: 2, pointRadius: 5,
        pointBackgroundColor: 'rgba(239,68,68,0.9)',
        fill: true, tension: 0.35,
      }, {
        label: 'Real Probability',
        data: frames.map(f => ((1 - f.fake_probability) * 100).toFixed(1)),
        borderColor: 'rgba(16,185,129,0.8)',
        backgroundColor: 'rgba(16,185,129,0.05)',
        borderWidth: 2, pointRadius: 5,
        pointBackgroundColor: 'rgba(16,185,129,0.9)',
        fill: false, tension: 0.35,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: true, aspectRatio: 2.5,
      plugins: { legend: { labels: { color: '#94a3b8', font: { family: 'Outfit', size: 12 } } } },
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8', font: { family: 'Outfit', size: 11 } } },
        y: { min: 0, max: 100, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8', font: { family: 'Outfit', size: 11 }, callback: v => v + '%' } }
      },
      animation: { duration: 900 }
    }
  });
}

/* -------- Generic feature grid -------- */
function renderFeatureGrid(container, items) {
  container.innerHTML = items.map(item => `
    <div class="fg-item" role="listitem">
      <div class="fg-label">${escHtml(String(item.label))}</div>
      <div class="fg-val ${item.cls || ''}">${escHtml(String(item.val))}</div>
    </div>
  `).join('');
}

/* ============================================================
   UI Reset
   ============================================================ */
function resetUI() {
  $('resultsWrap').style.display = 'none';
  $('loadingWrap').style.display = 'none';
  $('loadingSteps').innerHTML = '';
  $('loadingBar').style.width = '0%';
  $('analysisForm').scrollIntoView({ behavior: 'smooth', block: 'start' });
  $('analyzeBtn').disabled = false;
}

/* ============================================================
   Toast Notification
   ============================================================ */
function showToast(msg, type = 'info') {
  let toast = document.getElementById('verifai-toast');
  if (!toast) {
    toast = document.createElement('div');
    toast.id = 'verifai-toast';
    Object.assign(toast.style, {
      position: 'fixed', bottom: '24px', right: '24px', zIndex: '9999',
      padding: '14px 20px', borderRadius: '12px', maxWidth: '380px',
      fontSize: '0.88rem', fontFamily: 'Outfit, sans-serif', fontWeight: '600',
      boxShadow: '0 4px 24px rgba(0,0,0,0.5)',
      transition: 'all 0.3s ease', transform: 'translateY(20px)', opacity: '0',
    });
    document.body.appendChild(toast);
  }
  const colors = {
    info:  ['rgba(6,182,212,0.15)', 'rgba(6,182,212,0.4)', '#22d3ee'],
    warn:  ['rgba(245,158,11,0.15)', 'rgba(245,158,11,0.4)', '#fbbf24'],
    error: ['rgba(239,68,68,0.15)', 'rgba(239,68,68,0.4)', '#f87171'],
  };
  const [bg, border, col] = colors[type] || colors.info;
  Object.assign(toast.style, { background: bg, border: `1px solid ${border}`, color: col });
  toast.textContent = msg;
  requestAnimationFrame(() => { toast.style.transform = 'translateY(0)'; toast.style.opacity = '1'; });
  setTimeout(() => { toast.style.transform = 'translateY(20px)'; toast.style.opacity = '0'; }, 4000);
}

/* ============================================================
   Utility Functions
   ============================================================ */
function pct(v) {
  if (v == null || isNaN(v)) return '—';
  return (clamp(v) * 100).toFixed(1) + '%';
}
function score2class(v) {
  if (v == null) return '';
  if (v > 0.6) return 'danger';
  if (v > 0.3) return 'highlight';
  return 'good';
}
function escHtml(str) {
  return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

/* ============================================================
   Smooth Scroll for Nav Links
   ============================================================ */
function initNavLinks() {
  $$('.nav-link, .hero-cta, #hero-cta-btn').forEach(a => {
    a.addEventListener('click', e => {
      const href = a.getAttribute('href');
      if (href && href.startsWith('#')) {
        e.preventDefault();
        const target = document.querySelector(href);
        if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });
}

/* ============================================================
   Intersection Observer — animate elements on scroll
   ============================================================ */
function initScrollAnimations() {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.style.animation = 'fadeUp 0.6s cubic-bezier(0.4,0,0.2,1) forwards';
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.12 });

  $$('.pipe-step, .ac, .chart-card').forEach(el => {
    el.style.opacity = '0';
    observer.observe(el);
  });
}

/* ============================================================
   Init
   ============================================================ */
document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initTextInput();
  initImageUpload();
  initVideoUpload();
  initForm();
  initNavLinks();
  initScrollAnimations();
  updateActiveChips();
  console.log('%cVerifAI 🔍', 'color:#7c3aed;font-size:20px;font-weight:900;');
  console.log('%cMultimodal Fake News Detection System', 'color:#06b6d4;font-size:12px;');
});
