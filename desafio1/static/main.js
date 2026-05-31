/* ── Bootstrap: load hardware profiles from API ────────────── */
let HARDWARE = {};
fetch('/api/hardware').then(r => r.json()).then(d => {
  HARDWARE = d;
  updateHardwareInfo();
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

/* ── Hardware info update ──────────────────────────────────── */
const methodSelect = document.getElementById('methodSelect');
const hwPill = document.getElementById('hwPill');
const hwIcon = document.getElementById('hwIcon');
const hwLabel = document.getElementById('hwLabel');
const hwRam = document.getElementById('hwRam');
const hwDisk = document.getElementById('hwDisk');
const hwGpu = document.getElementById('hwGpu');
const hwOffline = document.getElementById('hwOffline');

function updateHardwareInfo() {
  const m = methodSelect.value;
  const hw = HARDWARE[m] || {};
  hwIcon.textContent = hw.icon || '❓';
  hwLabel.textContent = hw.tier_label || '—';
  if (hwRam)     hwRam.textContent     = hw.ram || '—';
  if (hwDisk)    hwDisk.textContent    = hw.disk || '—';
  if (hwGpu)     hwGpu.textContent     = hw.gpu ? '✓ Sim' : '✗ Não';
  if (hwOffline) hwOffline.textContent = hw.offline ? '✓' : '✗ Internet';
}
methodSelect.addEventListener('change', updateHardwareInfo);
updateHardwareInfo();

/* ── Generic drop zone setup ───────────────────────────────── */
function setupDropZone({ zone, fileInput, preview, placeholder, onFile }) {
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  });
  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) handleFile(fileInput.files[0]);
  });

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
const countBtn = document.getElementById('countBtn');

setupDropZone({
  zone: document.getElementById('dropZone'),
  fileInput: document.getElementById('fileInput'),
  preview: document.getElementById('previewImg'),
  placeholder: document.getElementById('dropPlaceholder'),
  onFile: f => { singleFile = f; countBtn.disabled = false; },
});

countBtn.addEventListener('click', async () => {
  if (!singleFile) return;
  showLoading(true);

  const fd = new FormData();
  fd.append('file', singleFile);
  fd.append('method', methodSelect.value);

  try {
    const res = await fetch('/count', { method: 'POST', body: fd });
    const data = await res.json();
    if (data.error) { alert(data.error); return; }
    displayResult(data);
  } catch (e) {
    alert('Erro de comunicação: ' + e.message);
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

  // Animated counter
  animateCount(data.count ?? 0);

  document.getElementById('metaMethod').textContent = data.method;
  document.getElementById('metaTime').textContent = data.elapsed_ms + ' ms';

  const img = document.getElementById('annotatedImg');
  img.src = data.annotated_image;

  const dl = document.getElementById('downloadBtn');
  dl.onclick = () => {
    const a = document.createElement('a');
    a.href = data.annotated_image;
    a.download = `contagem_${data.method}_${Date.now()}.jpg`;
    a.click();
  };
}

function animateCount(target) {
  const el = document.getElementById('countNumber');
  const start = parseInt(el.textContent) || 0;
  const duration = 600;
  const startTime = performance.now();
  const step = ts => {
    const p = Math.min((ts - startTime) / duration, 1);
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

    // Sort by count desc then by method name
    const results = data.results.sort((a, b) => (b.count ?? -1) - (a.count ?? -1));

    results.forEach(r => {
      const hw = HARDWARE[r.method] || {};
      const card = document.createElement('div');
      card.className = 'compare-card';
      card.innerHTML = `
        <img src="${r.annotated_image}" alt="${r.method}" loading="lazy" />
        <div class="compare-card-body">
          <div class="compare-card-title">${hw.icon || ''} ${r.method}</div>
          <div class="compare-count">${r.count ?? '—'}</div>
          <div class="compare-meta">
            ${hw.tier_label || ''} · ${r.elapsed_ms} ms
            ${hw.stars !== undefined && hw.stars !== null ? ' · ' + '★'.repeat(hw.stars) + '☆'.repeat(5 - hw.stars) : ''}
          </div>
        </div>
      `;
      grid.appendChild(card);
    });

    grid.classList.remove('hidden');
  } catch (e) {
    alert('Erro: ' + e.message);
  } finally {
    loading.classList.add('hidden');
  }
});
