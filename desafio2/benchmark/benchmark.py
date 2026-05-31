"""
Benchmark — Detecção de Trincas em Paredes
Compara métodos CV tradicionais vs YOLOv8n-seg fine-tuned.
Métricas: detection rate, IoU médio, tempo de inferência.
"""

import time
import random
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

BENCH_DIR = Path(__file__).parent
D2_DIR = Path(__file__).parent.parent  # dataset/weights ficam em desafio2/
DATASET_DIR = D2_DIR / "dataset"
WEIGHTS = D2_DIR / "weights" / "best.pt"
OUTPUT_DIR = BENCH_DIR / "output_benchmark"
OUTPUT_DIR.mkdir(exist_ok=True)

# Usa imagens de val se split existe, senão amostra do flat
def _get_eval_images(n: int = 30, seed: int = 42) -> list[tuple[Path, Path]]:
    """Retorna pares (image_path, label_path) para avaliação."""
    val_dir = DATASET_DIR / "val" / "images"
    if val_dir.exists():
        imgs = sorted(val_dir.glob("*.jpg"))
        lbl_dir = DATASET_DIR / "val" / "labels"
    else:
        imgs = sorted((DATASET_DIR / "images").glob("*.jpg"))
        lbl_dir = DATASET_DIR / "labels"

    random.seed(seed)
    sample = random.sample(imgs, min(n, len(imgs)))
    return [(p, lbl_dir / p.with_suffix(".txt").name) for p in sample]


def _load_gt_mask(label_path: Path, h: int, w: int) -> np.ndarray:
    """Converte labels YOLO segmentação em máscara binária."""
    mask = np.zeros((h, w), dtype=np.uint8)
    if not label_path.exists():
        return mask
    with open(label_path) as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            if len(parts) < 7:
                continue
            coords = parts[1:]  # remove class_id
            pts = np.array([(coords[i] * w, coords[i + 1] * h)
                            for i in range(0, len(coords) - 1, 2)], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
    return mask


def _compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    inter = cv2.bitwise_and(pred_mask, gt_mask)
    union = cv2.bitwise_or(pred_mask, gt_mask)
    i = np.count_nonzero(inter)
    u = np.count_nonzero(union)
    return i / u if u > 0 else 0.0


@dataclass
class BenchResult:
    method: str
    image: str
    iou: float
    detected: bool
    time_ms: float
    memory_kb: float
    annotated: np.ndarray = field(repr=False)


# ─────────────────────────────────────────────────────────────
# Método 1 — Canny Edge + Morphological
# ─────────────────────────────────────────────────────────────
def method_canny_morph(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Canny adaptativo + dilatação para conectar bordas fragmentadas.
    Tier 1: sem GPU, ~10 MB RAM, offline.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    median = int(np.median(blurred))
    lo, hi = max(0, int(0.4 * median)), min(255, int(1.2 * median))
    edges = cv2.Canny(blurred, lo, hi)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, k, iterations=2)
    # Filtrar regiões pequenas (ruído)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(dilated)
    mask = np.zeros_like(dilated)
    h, w = image.shape[:2]
    min_area = h * w * 0.0002
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            mask[lbl == i] = 255

    annotated = image.copy()
    annotated[mask > 0] = [0, 0, 255]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        cv2.rectangle(annotated, (x, y), (x + cw, y + ch), (0, 255, 0), 2)

    return mask, annotated


# ─────────────────────────────────────────────────────────────
# Método 2 — Gabor Filter Bank
# ─────────────────────────────────────────────────────────────
def method_gabor(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Bank de filtros Gabor em 4 orientações para detectar texturas lineares (trincas).
    Tier 1: sem GPU, ~30 MB RAM, offline.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    combined = np.zeros_like(gray)
    for theta in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
        kernel = cv2.getGaborKernel((21, 21), sigma=3.0, theta=theta,
                                    lambd=8.0, gamma=0.5, psi=0)
        filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
        combined = np.maximum(combined, np.abs(filtered))

    combined = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=2)

    annotated = image.copy()
    annotated[mask > 0] = [0, 100, 255]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        cv2.rectangle(annotated, (x, y), (x + cw, y + ch), (0, 255, 0), 2)

    return mask, annotated


# ─────────────────────────────────────────────────────────────
# Método 3 — Adaptive Threshold + Morphological
# ─────────────────────────────────────────────────────────────
def method_adaptive_thresh(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Threshold adaptativo gaussiano para realçar padrões escuros (trincas em paredes claras).
    Tier 1: sem GPU, ~15 MB RAM, offline.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 51, 8)
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k_close, iterations=1)
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=2)

    # Remove blobs muito grandes (falso positivo: sombra/parede escura)
    h, w = image.shape[:2]
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask)
    clean = np.zeros_like(mask)
    max_area = h * w * 0.10
    min_area = h * w * 0.0001
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            clean[lbl == i] = 255
    mask = clean

    annotated = image.copy()
    annotated[mask > 0] = [255, 80, 0]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        cv2.rectangle(annotated, (x, y), (x + cw, y + ch), (0, 255, 0), 2)

    return mask, annotated


# ─────────────────────────────────────────────────────────────
# Método 4 — Sobel Gradient Magnitude
# ─────────────────────────────────────────────────────────────
def method_sobel(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Gradiente Sobel combinado + Otsu para detectar bordas abruptas.
    Tier 1: sem GPU, ~15 MB RAM, offline.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    sx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sx ** 2 + sy ** 2)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

    annotated = image.copy()
    annotated[mask > 0] = [0, 200, 200]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        cv2.rectangle(annotated, (x, y), (x + cw, y + ch), (0, 255, 0), 2)

    return mask, annotated


# ─────────────────────────────────────────────────────────────
# Método 5 — YOLOv8n-seg (fine-tuned)
# ─────────────────────────────────────────────────────────────
def method_yolov8(image: np.ndarray, model=None) -> tuple[np.ndarray, np.ndarray]:
    """
    YOLOv8n-seg fine-tuned em 1241 imagens de trincas.
    Tier 2: CPU ~300 MB RAM, 6 MB disco, offline.
    """
    if model is None:
        return np.zeros(image.shape[:2], dtype=np.uint8), image.copy()

    results = model(image, conf=0.25, iou=0.45, verbose=False)
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    annotated = image.copy()
    for r in results:
        if r.masks is not None:
            for seg in r.masks.xy:
                pts = np.array(seg, dtype=np.int32)
                cv2.fillPoly(mask, [pts], 255)
        if r.boxes is not None:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, f"crack {conf:.2f}", (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    annotated[mask > 0] = (annotated[mask > 0] * 0.6 + np.array([0, 0, 200]) * 0.4).astype(np.uint8)
    return mask, annotated


# ─────────────────────────────────────────────────────────────
# Método 6 — VLM Claude (cloud fallback)
# ─────────────────────────────────────────────────────────────
def method_vlm_claude(image_path: str) -> tuple[bool, str]:
    """Usa Claude para detectar presença de trinca. Tier 4: Cloud API."""
    try:
        import anthropic
        import base64
        with open(image_path, "rb") as f:
            data = base64.standard_b64encode(f.read()).decode()
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=64,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64",
                                                  "media_type": "image/jpeg", "data": data}},
                    {"type": "text",
                     "text": ("Does this image show cracks or fissures in a wall or surface? "
                              "Reply with 'yes' or 'no' and a one-line reason.")},
                ],
            }],
        )
        text = resp.content[0].text.strip().lower()
        return text.startswith("yes"), text
    except Exception as e:
        return False, f"error: {e}"


# ─────────────────────────────────────────────────────────────
# Runner principal do benchmark
# ─────────────────────────────────────────────────────────────
def run_benchmark(n_images: int = 30):
    pairs = _get_eval_images(n=n_images)
    print(f"\nBenchmark em {len(pairs)} imagens de validação\n")

    yolo_model = None
    if WEIGHTS.exists():
        from ultralytics import YOLO
        yolo_model = YOLO(str(WEIGHTS))
        print("✓ YOLOv8n-seg carregado\n")
    else:
        print("⚠ YOLOv8n-seg não encontrado — rode train.py primeiro\n")

    METHODS = {
        "Canny+Morph": (method_canny_morph, "1"),
        "Gabor Filter": (method_gabor, "1"),
        "Adaptive Thresh": (method_adaptive_thresh, "1"),
        "Sobel+Otsu": (method_sobel, "1"),
        "YOLOv8n-seg": (None, "2"),  # handled separately
    }

    totals: dict[str, dict] = {
        name: {"iou_sum": 0.0, "detected": 0, "time_ms": 0.0, "count": 0}
        for name in METHODS
    }

    for img_path, lbl_path in pairs:
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        gt_mask = _load_gt_mask(lbl_path, h, w)
        has_crack = np.any(gt_mask > 0)

        for name, (fn, tier) in METHODS.items():
            tracemalloc.start()
            t0 = time.perf_counter()
            try:
                if name == "YOLOv8n-seg":
                    pred_mask, annotated = method_yolov8(image.copy(), yolo_model)
                else:
                    pred_mask, annotated = fn(image.copy())
            except Exception as e:
                print(f"  ✗ {name} falhou em {img_path.name}: {e}")
                tracemalloc.stop()
                continue
            elapsed = (time.perf_counter() - t0) * 1000
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            iou = _compute_iou(pred_mask, gt_mask) if has_crack else 0.0
            detected = np.any(pred_mask > 0)

            totals[name]["iou_sum"] += iou
            totals[name]["detected"] += int(detected)
            totals[name]["time_ms"] += elapsed
            totals[name]["count"] += 1

            # Salva primeiras 5 imagens
            if totals[name]["count"] <= 5:
                out_name = f"{name.replace(' ', '_').replace('+', '_')}_{img_path.name}"
                cv2.imwrite(str(OUTPUT_DIR / out_name), annotated)

    print(f"\n{'Método':<20} {'Tier':>4}  {'Det%':>6}  {'IoU':>6}  {'ms/img':>8}")
    print("─" * 55)
    TIER_LABEL = {"1": "📱", "2": "💻"}
    for name, (fn, tier) in METHODS.items():
        t = totals[name]
        n = t["count"] or 1
        det_pct = t["detected"] / n * 100
        mean_iou = t["iou_sum"] / n
        mean_ms = t["time_ms"] / n
        print(f"{name:<20} {TIER_LABEL[tier]:>4}  {det_pct:>5.1f}%  {mean_iou:>6.3f}  {mean_ms:>8.1f}")

    print("\nImagens anotadas salvas em:", OUTPUT_DIR)

    if WEIGHTS.exists():
        print("\n─── mAP (validação completa YOLOv8n-seg) ───")
        _print_yolo_val()


def _print_yolo_val():
    """Roda validação oficial YOLO e exibe mAP."""
    try:
        from ultralytics import YOLO
        data_yaml = D2_DIR / "data.yaml"
        if not data_yaml.exists():
            print("  data.yaml não encontrado — rode train.py primeiro")
            return
        model = YOLO(str(WEIGHTS))
        metrics = model.val(data=str(data_yaml), imgsz=640, verbose=False)
        print(f"  mAP50 (box):    {metrics.box.map50:.4f}")
        print(f"  mAP50-95 (box): {metrics.box.map:.4f}")
        if hasattr(metrics, "seg") and metrics.seg is not None:
            print(f"  mAP50 (seg):    {metrics.seg.map50:.4f}")
            print(f"  mAP50-95 (seg): {metrics.seg.map:.4f}")
    except Exception as e:
        print(f"  Erro na validação YOLO: {e}")


if __name__ == "__main__":
    run_benchmark(n_images=30)
