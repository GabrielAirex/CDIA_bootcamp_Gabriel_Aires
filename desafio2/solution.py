"""
Desafio 2 — Detecção de Trincas em Paredes
Pipeline principal: YOLOv8n-seg (fine-tuned) → Canny fallback → VLM fallback.
"""

import base64
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).parent
WEIGHTS = BASE / "weights" / "best.pt"

_yolo_cache = None


def _load_yolo():
    global _yolo_cache
    if _yolo_cache is None and WEIGHTS.exists():
        from ultralytics import YOLO
        _yolo_cache = YOLO(str(WEIGHTS))
    return _yolo_cache


@dataclass
class CrackResult:
    image_path: str
    method: str
    cracks_found: int
    confidence: float
    annotated: np.ndarray = field(repr=False)
    mask: np.ndarray = field(repr=False)
    boxes: list[tuple[int, int, int, int]] = field(default_factory=list)


def _canny_fallback(image: np.ndarray) -> tuple[int, float, np.ndarray, np.ndarray, list]:
    """Detecção por Canny + morfologia — fallback sem modelo."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    median = int(np.median(blurred))
    lo, hi = max(0, int(0.4 * median)), min(255, int(1.2 * median))
    edges = cv2.Canny(blurred, lo, hi)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.dilate(edges, k, iterations=2)

    h, w = image.shape[:2]
    min_area = h * w * 0.0002
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask)
    clean = np.zeros_like(mask)
    boxes = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[lbl == i] = 255
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            boxes.append((x, y, x + bw, y + bh))
    mask = clean

    annotated = image.copy()
    annotated[mask > 0] = (annotated[mask > 0] * 0.5 + np.array([0, 0, 180]) * 0.5).astype(np.uint8)
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = "crack (Canny)"
    cv2.putText(annotated, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    n_cracks = len(boxes)
    confidence = 0.45 if n_cracks > 0 else 0.2
    return n_cracks, confidence, annotated, mask, boxes


def _yolo_detect(image: np.ndarray) -> tuple[int, float, np.ndarray, np.ndarray, list]:
    """Detecção YOLOv8n-seg fine-tuned."""
    model = _load_yolo()
    results = model(image, conf=0.25, iou=0.45, verbose=False)
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    boxes = []
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
                boxes.append((x1, y1, x2, y2))
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, f"crack {conf:.2f}", (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    annotated[mask > 0] = (annotated[mask > 0] * 0.6 + np.array([0, 0, 220]) * 0.4).astype(np.uint8)
    confidence = float(np.mean([float(r.boxes.conf[0]) for r in results if r.boxes is not None and len(r.boxes)])) if boxes else 0.0
    return len(boxes), confidence, annotated, mask, boxes


def _vlm_detect(image_path: str) -> tuple[int, float]:
    """Claude VLM como fallback cloud."""
    try:
        import anthropic
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
                     "text": ("Count the number of distinct cracks or fissures visible in this wall image. "
                              "Reply with a single integer only.")},
                ],
            }],
        )
        n = int(resp.content[0].text.strip())
        return n, 0.88
    except Exception as e:
        print(f"VLM fallback error: {e}")
        return 0, 0.0


def detect_cracks(image_path: str, vlm_threshold: float = 0.4) -> CrackResult:
    """
    Pipeline principal.
    1. Tenta YOLOv8n-seg (se modelo disponível).
    2. Se confiança < vlm_threshold, tenta VLM Claude.
    3. Fallback: Canny.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

    if WEIGHTS.exists():
        n, conf, annotated, mask, boxes = _yolo_detect(image)
        if conf >= vlm_threshold or n > 0:
            return CrackResult(image_path, "yolov8n-seg", n, conf, annotated, mask, boxes)
        # Baixa confiança → tenta VLM
        vlm_n, vlm_conf = _vlm_detect(image_path)
        if vlm_conf > 0:
            return CrackResult(image_path, "vlm+yolo", vlm_n, vlm_conf, annotated, mask, boxes)
        return CrackResult(image_path, "yolov8n-seg", n, conf, annotated, mask, boxes)

    # Sem modelo: tenta VLM primeiro
    vlm_n, vlm_conf = _vlm_detect(image_path)
    if vlm_conf >= vlm_threshold:
        n_canny, _, annotated, mask, boxes = _canny_fallback(image)
        return CrackResult(image_path, "vlm+canny", vlm_n, vlm_conf, annotated, mask, boxes)

    n, conf, annotated, mask, boxes = _canny_fallback(image)
    return CrackResult(image_path, "canny", n, conf, annotated, mask, boxes)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    imgs_dir = Path(__file__).parent / "dataset" / "val" / "images"
    if not imgs_dir.exists():
        imgs_dir = Path(__file__).parent / "dataset" / "images"

    paths = sorted(imgs_dir.glob("*.jpg"))[:5]
    if len(sys.argv) > 1:
        paths = [Path(sys.argv[1])]

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    print(f"{'Imagem':<45} {'Trincas':>8} {'Método':<18} {'Confiança':>10}")
    print("─" * 85)
    for p in paths:
        r = detect_cracks(str(p))
        print(f"{p.name:<45} {r.cracks_found:>8} {r.method:<18} {r.confidence:>9.0%}")
        cv2.imwrite(str(output_dir / f"annotated_{p.name}"), r.annotated)
    print(f"\nImagens anotadas em: {output_dir}")
