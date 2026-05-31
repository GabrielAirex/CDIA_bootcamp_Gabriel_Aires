/* ══════════════════════════════════════════════════════════════
   CDIA — App Unificado (D1 + D2)
   ══════════════════════════════════════════════════════════════ */

let HW_D1 = {}, HW_D2 = {};
let currentChallenge = 'd1';

/* ── Bootstrap: hardware profiles ─────────────────────────── */
Promise.all([
  fetch('/d1/hardware').then(r => r.json()),
  fetch('/d2/hardware').then(r => r.json()),
  fetch('/d2/model_status').then(r => r.json()),
]).then(([hw1, hw2, model]) => {
  HW_D1 = hw1;
  HW_D2 = hw2;
  updateHW_D1();
  updateHW_D2();
  updateModelBanner(model.trained);
});

/* ── Challenge switcher ────────────────────────────────────── */
document.querySelectorAll('.ch-btn').forEach(btn => {
  btn.addEventListener('click', () => switchChallenge(btn.dataset.challenge));
});

function switchChallenge(ch) {
  currentChallenge = ch;
  document.body.classList.toggle('d2', ch === 'd2');

  document.querySelectorAll('.ch-btn').forEach(b =>
    b.classList.toggle('active', b.dataset.challenge === ch));

  document.querySelectorAll('.challenge-section').forEach(s =>
    s.classList.toggle('active', s.dataset.challenge === ch));

  const logoTitle = document.querySelector('.logo-title');
  logoTitle.textContent = ch === 'd1' ? 'SCREW COUNTER' : 'CRACK DETECTOR';

  const modelBanner = document.getElementById('modelBanner');
  modelBanner.classList.toggle('hidden', ch === 'd1');
}

function updateModelBanner(trained) {
  const b = document.getElementById('modelBanner');
  b.className = 'model-banner' + (currentChallenge === 'd1' ? ' hidden' : trained ? ' ready' : ' missing');
  b.innerHTML = trained
    ? '✓ Modelo YOLOv8n-seg pronto &nbsp;·&nbsp; mAP50 0.672 &nbsp;·&nbsp; Segmentação 0.510 &nbsp;·&nbsp; 41 ms/img na CPU'
    : '⚠ Modelo não treinado — rode <code>python3 train.py</code> para habilitar YOLOv8n-seg';
}

/* ── Generic drop zone ─────────────────────────────────────── */
function setupDropZone({ zone, fileInput, preview, placeholder, onFile }) {
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => {
    e.preventDefault(); zone.classList.remove('drag-over');
    if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener('change', () => { if (fileInput.files[0]) handleFile(fileInput.files[0]); });

  function handleFile(file) {
    if (!file.type.startsWith('image/')) return;
    const reader = new FileReader();
    reader.onload = e => {
      if (placeholder) placeholder.classList.add('hidden');
      preview.src = e.target.result;
      preview.classList.remove('hidden');
    };
    reader.readAsDataURL(file);
    onFile(file);
  }
}

/* ── Hardware info helpers ─────────────────────────────────── */
function applyHW(hw, { icon, label, ram, disk, gpu, offline }) {
  if (icon)    icon.textContent    = hw.icon        || '❓';
  if (label)   label.textContent   = hw.tier_label  || '—';
  if (ram)     ram.textContent     = hw.ram          || '—';
  if (disk)    disk.textContent    = hw.disk         || '—';
  if (gpu)     gpu.textContent     = hw.gpu          ? '✓ Sim' : '✗ Não';
  if (offline) offline.textContent = hw.offline      ? '✓' : '✗ Internet';
}

function g(id) { return document.getElementById(id); }

function animateCount(el, target) {
  const start = parseInt(el.textContent) || 0;
  const dur = 500, t0 = performance.now();
  const step = ts => {
    const p = Math.min((ts - t0) / dur, 1);
    el.textContent = Math.round(start + (target - start) * (1 - Math.pow(1 - p, 3)));
    if (p < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

/* ══════════════════════════════════════════════════════════════
   DESAFIO 1 — Parafusos
   ══════════════════════════════════════════════════════════════ */

let d1File = null;
const d1CountBtn = g('d1CountBtn');

setupDropZone({
  zone: g('d1DropZone'), fileInput: g('d1FileInput'),
  preview: g('d1PreviewImg'), placeholder: g('d1DropPlaceholder'),
  onFile: f => { d1File = f; d1CountBtn.disabled = false; },
});

g('d1MethodSelect').addEventListener('change', updateHW_D1);

function updateHW_D1() {
  const hw = HW_D1[g('d1MethodSelect').value] || {};
  applyHW(hw, {
    icon: g('d1HwIcon'), label: g('d1HwLabel'),
    ram: g('d1HwRam'), disk: g('d1HwDisk'),
    gpu: g('d1HwGpu'), offline: g('d1HwOffline'),
  });
}

d1CountBtn.addEventListener('click', async () => {
  if (!d1File) return;
  g('d1LoadingOverlay').classList.remove('hidden');
  const fd = new FormData();
  fd.append('file', d1File);
  fd.append('method', g('d1MethodSelect').value);
  try {
    const res = await fetch('/d1/count', { method: 'POST', body: fd });
    const data = await res.json();
    if (data.error) { alert(data.error); return; }
    g('d1ResultEmpty').classList.add('hidden');
    g('d1ResultContent').classList.remove('hidden');
    animateCount(g('d1CountNumber'), data.count ?? 0);
    g('d1MetaMethod').textContent = data.method;
    g('d1MetaTime').textContent = data.elapsed_ms + ' ms';
    g('d1AnnotatedImg').src = data.annotated_image;
    g('d1DownloadBtn').onclick = () => {
      const a = document.createElement('a');
      a.href = data.annotated_image;
      a.download = `d1_${data.method}_${Date.now()}.jpg`;
      a.click();
    };
  } catch (e) { alert('Erro: ' + e.message); }
  finally { g('d1LoadingOverlay').classList.add('hidden'); }
});

/* D1 compare */
let d1CompareFile = null;
const d1CompareBtn = g('d1CompareBtn');
const d1CompareGpuBtn = g('d1CompareGpuBtn');

setupDropZone({
  zone: g('d1DropZoneCompare'), fileInput: g('d1FileInputCompare'),
  preview: g('d1PreviewCompare'), placeholder: g('d1DropPlaceholderCompare'),
  onFile: f => {
    d1CompareFile = f;
    d1CompareBtn.disabled = false;
    d1CompareGpuBtn.disabled = false;
  },
});

function makeCompareCard(r) {
  const hw = HW_D1[r.method] || {};
  const tierClass = hw.tier === 1 ? 'tier-badge-1' : hw.tier === 2 ? 'tier-badge-2' : 'tier-badge-3';
  const card = document.createElement('div');
  card.className = 'compare-card';
  const countDisplay = r.count != null ? r.count : (r.error ? '✗' : '—');
  card.innerHTML = `
    <img src="${r.annotated_image || ''}" loading="lazy" ${!r.annotated_image ? 'style="display:none"' : ''} />
    <div class="compare-card-body">
      <div class="compare-card-title">${hw.icon || ''} ${r.method}</div>
      <div class="compare-count">${countDisplay}</div>
      <div class="compare-meta">
        <span class="tier-tag ${tierClass}">T${hw.tier || '?'}</span>
        ${r.elapsed_ms ? r.elapsed_ms + ' ms' : ''} · ${hw.accuracy || '?'}
      </div>
      ${r.error ? `<div class="compare-error">${r.error}</div>` : ''}
    </div>`;
  return card;
}

async function runCompare(endpoint, loadingMsg) {
  if (!d1CompareFile) return;
  g('d1CompareLoadingMsg').textContent = loadingMsg;
  g('d1CompareLoading').classList.remove('hidden');
  const fd = new FormData();
  fd.append('file', d1CompareFile);
  try {
    const res = await fetch(endpoint, { method: 'POST', body: fd });
    const data = await res.json();
    const grid = g('d1CompareGrid');
    grid.classList.remove('hidden');
    data.results.forEach(r => grid.appendChild(makeCompareCard(r)));
  } catch (e) { alert('Erro: ' + e.message); }
  finally { g('d1CompareLoading').classList.add('hidden'); }
}

d1CompareBtn.addEventListener('click', () => {
  g('d1CompareGrid').innerHTML = '';
  g('d1CompareGrid').classList.add('hidden');
  runCompare('/d1/benchmark', 'Rodando T1 + T2 (~2s)…');
});

d1CompareGpuBtn.addEventListener('click', () => {
  runCompare('/d1/benchmark_gpu', 'Rodando modelos GPU (pode demorar 5–15s)…');
});

/* D1 tabs */
setupTabs('d1');

/* ══════════════════════════════════════════════════════════════
   DESAFIO 2 — Trincas
   ══════════════════════════════════════════════════════════════ */

let d2File = null;
const d2DetectBtn = g('d2DetectBtn');

setupDropZone({
  zone: g('d2DropZone'), fileInput: g('d2FileInput'),
  preview: g('d2PreviewImg'), placeholder: g('d2DropPlaceholder'),
  onFile: f => { d2File = f; d2DetectBtn.disabled = false; },
});

g('d2MethodSelect').addEventListener('change', updateHW_D2);

function updateHW_D2() {
  const hw = HW_D2[g('d2MethodSelect').value] || {};
  applyHW(hw, {
    icon: g('d2HwIcon'), label: g('d2HwLabel'),
    ram: g('d2HwRam'), disk: g('d2HwDisk'),
    gpu: g('d2HwGpu'), offline: g('d2HwOffline'),
  });
}

d2DetectBtn.addEventListener('click', async () => {
  if (!d2File) return;
  g('d2LoadingOverlay').classList.remove('hidden');
  const fd = new FormData();
  fd.append('file', d2File);
  fd.append('method', g('d2MethodSelect').value);
  try {
    const res = await fetch('/d2/detect', { method: 'POST', body: fd });
    const data = await res.json();
    if (data.error) { alert(data.error); return; }
    g('d2ResultEmpty').classList.add('hidden');
    g('d2ResultContent').classList.remove('hidden');
    animateCount(g('d2CountNumber'), data.cracks_found ?? 0);
    g('d2MetaMethod').textContent = data.method;
    g('d2MetaTime').textContent = data.elapsed_ms + ' ms';
    const hint = g('d2ResultHint');
    if (data.method === 'YOLOv8n-seg') {
      hint.textContent = 'Overlay azul = máscara de segmentação · Caixas verdes = bounding boxes das fissuras';
    } else {
      hint.textContent = 'Overlay colorido = regiões detectadas pelo método tradicional (sem segmentação de instância)';
    }
    g('d2AnnotatedImg').src = data.annotated_image;
    g('d2DownloadBtn').onclick = () => {
      const a = document.createElement('a');
      a.href = data.annotated_image;
      a.download = `d2_${data.method}_${Date.now()}.jpg`;
      a.click();
    };
  } catch (e) { alert('Erro: ' + e.message); }
  finally { g('d2LoadingOverlay').classList.add('hidden'); }
});

/* D2 compare */
let d2CompareFile = null;
const d2CompareBtn = g('d2CompareBtn');

setupDropZone({
  zone: g('d2DropZoneCompare'), fileInput: g('d2FileInputCompare'),
  preview: g('d2PreviewCompare'), placeholder: g('d2DropPlaceholderCompare'),
  onFile: f => { d2CompareFile = f; d2CompareBtn.disabled = false; },
});

d2CompareBtn.addEventListener('click', async () => {
  if (!d2CompareFile) return;
  g('d2CompareLoading').classList.remove('hidden');
  const grid = g('d2CompareGrid');
  grid.classList.add('hidden'); grid.innerHTML = '';
  const fd = new FormData();
  fd.append('file', d2CompareFile);
  try {
    const res = await fetch('/d2/benchmark', { method: 'POST', body: fd });
    const data = await res.json();
    data.results.forEach(r => {
      const hw = HW_D2[r.method] || {};
      const card = document.createElement('div');
      card.className = 'compare-card';
      card.innerHTML = `
        <img src="${r.annotated_image}" loading="lazy" />
        <div class="compare-card-body">
          <div class="compare-card-title">${hw.icon || ''} ${r.method}</div>
          <div class="compare-count">${r.cracks_found ?? '—'}</div>
          <div class="compare-meta">${hw.tier_label || ''} · ${r.elapsed_ms} ms</div>
        </div>`;
      grid.appendChild(card);
    });
    grid.classList.remove('hidden');
  } catch (e) { alert('Erro: ' + e.message); }
  finally { g('d2CompareLoading').classList.add('hidden'); }
});

/* D2 tabs */
setupTabs('d2');

/* ── Tab helper ────────────────────────────────────────────── */
function setupTabs(prefix) {
  const section = document.querySelector(`.challenge-section[data-challenge="${prefix}"]`);
  section.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
      section.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      section.querySelectorAll('.tab-content').forEach(c => c.classList.add('hidden'));
      tab.classList.add('active');
      section.querySelector('#' + prefix + '-tab-' + tab.dataset.tab).classList.remove('hidden');
    });
  });
}

/* ── Init ──────────────────────────────────────────────────── */
switchChallenge('d1');
