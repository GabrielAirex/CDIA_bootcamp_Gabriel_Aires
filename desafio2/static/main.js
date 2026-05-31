/* ── Bootstrap ──────────────────────────────────────────────── */
let HARDWARE = {};
fetch('/api/hardware').then(r => r.json()).then(d => {
  HARDWARE = d;
  updateHardwareInfo();
});

fetch('/api/model_status').then(r => r.json()).then(d => {
  const banner = document.getElementById('modelBanner');
  if (d.trained) {
    banner.className = 'model-banner ready';
    banner.innerHTML = '✓ YOLOv8n-seg carregado — detecção com modelo fine-tuned disponível';
  } else {
    banner.className = 'model-banner missing';
    banner.innerHTML = '⚠ Modelo não treinado — rode <code>python3 train.py</code> para habilitar YOLOv8n-seg';
  }
});

/* ── Tabs ──────────────────────────────────────────────────── */
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.add('hidden'));
    tab.classList.add('active');
    document.getElementById('tab-' + tab.dataset.tab).classList.remove('hidden');
  });
});

/* ── Hardware info ─────────────────────────────────────────── */
const methodSelect = document.getElementById('methodSelect');

function updateHardwareInfo() {
  const m = methodSelect.value;
  const hw = HARDWARE[m] || {};
  const icon = document.getElementById('hwIcon');
  const label = document.getElementById('hwLabel');
  if (icon)  icon.textContent  = hw.icon       || '❓';
  if (label) label.textContent = hw.tier_label  || '—';
  const ram = document.getElementById('hwRam');
  const disk = document.getElementById('hwDisk');
  const gpu  = document.getElementById('hwGpu');
  const off  = document.getElementById('hwOffline');
  if (ram)  ram.textContent  = hw.ram     || '—';
  if (disk) disk.textContent = hw.disk    || '—';
  if (gpu)  gpu.textContent  = hw.gpu     ? '✓ Sim' : '✗ Não';
  if (off)  off.textContent  = hw.offline ? '✓' : '✗ Internet';
}
methodSelect.addEventListener('change', updateHardwareInfo);
updateHardwareInfo();

/* ── Generic drop zone ─────────────────────────────────────── */
function setupDropZone({ zone, fileInput, preview, placeholder, onFile }) {
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => {
    e.preventDefault(); zone.classList.remove('drag-over');
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
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

/* ── TAB: ANÁLISE INDIVIDUAL ───────────────────────────────── */
let singleFile = null;
const detectBtn = document.getElementById('detectBtn');

setupDropZone({
  zone: document.getElementById('dropZone'),
  fileInput: document.getElementById('fileInput'),
  preview: document.getElementById('previewImg'),
  placeholder: document.getElementById('dropPlaceholder'),
  onFile: f => { singleFile = f; detectBtn.disabled = false; },
});

detectBtn.addEventListener('click', async () => {
  if (!singleFile) return;
  showLoading(true);
  const fd = new FormData();
  fd.append('file', singleFile);
  fd.append('method', methodSelect.value);
  try {
    const res = await fetch('/detect', { method: 'POST', body: fd });
    const data = await res.json();
    if (data.error) { alert(data.error); return; }
    displayResult(data);
  } catch (e) {
    alert('Erro: ' + e.message);
  } finally {
    showLoading(false);
  }
});

function showLoading(show) {
  document.getElementById('loadingOverlay').classList.toggle('hidden', !show);
}

function displayResult(data) {
  document.getElementById('resultEmpty').classList.add('hidden');
  const content = document.getElementById('resultContent');
  content.classList.remove('hidden');
  animateCount(data.cracks_found ?? 0);
  document.getElementById('metaMethod').textContent = data.method;
  document.getElementById('metaTime').textContent = data.elapsed_ms + ' ms';
  document.getElementById('annotatedImg').src = data.annotated_image;
  document.getElementById('downloadBtn').onclick = () => {
    const a = document.createElement('a');
    a.href = data.annotated_image;
    a.download = `crack_${data.method}_${Date.now()}.jpg`;
    a.click();
  };
}

function animateCount(target) {
  const el = document.getElementById('countNumber');
  const start = parseInt(el.textContent) || 0;
  const dur = 600, t0 = performance.now();
  const step = ts => {
    const p = Math.min((ts - t0) / dur, 1);
    const ease = 1 - Math.pow(1 - p, 3);
    el.textContent = Math.round(start + (target - start) * ease);
    if (p < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

/* ── TAB: COMPARAR MÉTODOS ─────────────────────────────────── */
let compareFile = null;
const compareBtn = document.getElementById('compareBtn');

setupDropZone({
  zone: document.getElementById('dropZoneCompare'),
  fileInput: document.getElementById('fileInputCompare'),
  preview: document.getElementById('previewCompare'),
  placeholder: document.getElementById('dropPlaceholderCompare'),
  onFile: f => { compareFile = f; compareBtn.disabled = false; },
});

compareBtn.addEventListener('click', async () => {
  if (!compareFile) return;
  const loading = document.getElementById('compareLoading');
  const grid = document.getElementById('compareGrid');
  loading.classList.remove('hidden');
  grid.classList.add('hidden');
  grid.innerHTML = '';
  const fd = new FormData();
  fd.append('file', compareFile);
  try {
    const res = await fetch('/benchmark', { method: 'POST', body: fd });
    const data = await res.json();
    if (data.error) { alert(data.error); return; }
    data.results.forEach(r => {
      const hw = HARDWARE[r.method] || {};
      const card = document.createElement('div');
      card.className = 'compare-card';
      card.innerHTML = `
        <img src="${r.annotated_image}" alt="${r.method}" loading="lazy" />
        <div class="compare-card-body">
          <div class="compare-card-title">${hw.icon || ''} ${r.method}</div>
          <div class="compare-count">${r.cracks_found ?? '—'}</div>
          <div class="compare-meta">${hw.tier_label || ''} · ${r.elapsed_ms} ms</div>
        </div>`;
      grid.appendChild(card);
    });
    grid.classList.remove('hidden');
  } catch (e) {
    alert('Erro: ' + e.message);
  } finally {
    loading.classList.add('hidden');
  }
});
