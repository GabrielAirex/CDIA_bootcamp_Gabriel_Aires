"""
Desafio 1 — Contagem de Parafusos
Abordagem: Watershed com Regional Maxima como pipeline principal.
           VLM (Claude) como fallback quando confiança é baixa.
"""

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from scipy import ndimage as ndi


@dataclass
class CountResult:
    image_path: str
    count: int
    method: str
    confidence: float
    annotated: np.ndarray


def _binary_mask(image: np.ndarray) -> np.ndarray:
    """
    Segmenta foreground (parafusos) do background uniforme.
    Usa blur 11x11 para apagar textura de rosca + threshold adaptativo
    com blockSize=35 (ignora detalhes finos) + closing para fechar a silhueta.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=35,
        C=4,
    )
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k5, iterations=2)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k3, iterations=1)
    return thresh


def _watershed_in_blob(blob_mask: np.ndarray, image: np.ndarray, min_region: int) -> list[tuple[int, int]]:
    """Aplica watershed dentro de um blob para separar parafusos sobrepostos."""
    dist = cv2.distanceTransform(blob_mask, cv2.DIST_L2, 5)
    if dist.max() == 0:
        return []
    # Threshold 60%: equilibrio entre separar sobrepostos e não dividir parafusos únicos
    _, sure_fg = cv2.threshold(dist, 0.60 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sure_bg = cv2.dilate(blob_mask, k, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers_ws = markers.copy()
    cv2.watershed(image, markers_ws)

    centers = []
    for label in np.unique(markers_ws):
        if label <= 1:
            continue
        region = np.uint8(markers_ws == label) * 255
        if cv2.countNonZero(region) < min_region:
            continue
        M = cv2.moments(region)
        if M["m00"] > 0:
            centers.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))
    return centers


def count_with_opencv(image: np.ndarray) -> tuple[int, float, np.ndarray]:
    """
    Watershed com Regional Maxima como seeds.
    - Blobs isolados (estimated=1): centroide direto.
    - Blobs com sobreposição (estimated>1): máximos regionais do distance transform
      substituem o limiar global fixo (0.60*max), detectando cada pico independente
      de sua altura absoluta — crucial para screws sobrepostos em pile.
    Footprint 5% da dimensão mínima: equilibrio entre não dividir screws alongados
    isolados e separar clusters compactos.
    """
    h, w = image.shape[:2]
    image_area = h * w
    min_area = image_area * 0.004
    max_area = image_area * 0.35

    binary = _binary_mask(image)

    n_labels, label_map, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    valid = [(i, int(stats[i, cv2.CC_STAT_AREA]))
             for i in range(1, n_labels)
             if min_area <= stats[i, cv2.CC_STAT_AREA] <= max_area]

    if not valid:
        return 0, 0.3, image.copy()

    ref_area = float(np.median([a for _, a in valid]))

    fp = max(7, int(min(h, w) * 0.05))
    if fp % 2 == 0:
        fp += 1

    centers: list[tuple[int, int]] = []
    for label, area in valid:
        blob_mask = np.uint8(label_map == label) * 255
        estimated = max(1, round(area / ref_area))

        if estimated <= 1:
            M = cv2.moments(blob_mask)
            if M["m00"] > 0:
                centers.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))
        else:
            dist = cv2.distanceTransform(blob_mask, cv2.DIST_L2, 5)
            max_filt = ndi.maximum_filter(dist, size=fp)
            sure_fg = ((dist == max_filt) & (dist > dist.max() * 0.12)).astype(np.uint8) * 255

            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            sure_bg = cv2.dilate(blob_mask, k, iterations=3)
            unknown = cv2.subtract(sure_bg, sure_fg)

            _, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown == 255] = 0

            markers_ws = markers.copy()
            cv2.watershed(image, markers_ws)

            for lbl in np.unique(markers_ws):
                if lbl <= 1:
                    continue
                region = np.uint8(markers_ws == lbl) * 255
                if cv2.countNonZero(region) < min_area / 5:
                    continue
                M = cv2.moments(region)
                if M["m00"] > 0:
                    centers.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))

    annotated = image.copy()
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(annotated, contours, -1, (0, 255, 0), 2)
    for i, (cx, cy) in enumerate(centers):
        cv2.circle(annotated, (cx, cy), 10, (0, 0, 255), -1)
        cv2.putText(annotated, str(i + 1), (cx + 12, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    confidence = 0.90 if centers else 0.3
    return len(centers), confidence, annotated


def count_with_vlm(image_path: str) -> tuple[int, float]:
    """Usa Claude como fallback para contagem zero-shot."""
    try:
        import anthropic
        import base64

        with open(image_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")

        suffix = Path(image_path).suffix.lower()
        media_type = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"

        client = anthropic.Anthropic()
        message = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=64,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "Count the number of screws (parafusos) in this image. "
                                "Reply with a single integer only. No explanation."
                            ),
                        },
                    ],
                }
            ],
        )
        count = int(message.content[0].text.strip())
        return count, 0.9
    except Exception as e:
        print(f"VLM fallback failed: {e}")
        return -1, 0.0


def count_screws(image_path: str, vlm_threshold: float = 0.5) -> CountResult:
    """
    Pipeline principal: tenta OpenCV; se confiança < vlm_threshold usa VLM.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

    cv_count, confidence, annotated = count_with_opencv(image)

    if confidence >= vlm_threshold:
        return CountResult(image_path, cv_count, "opencv", confidence, annotated)

    vlm_count, vlm_conf = count_with_vlm(image_path)
    if vlm_count >= 0:
        return CountResult(image_path, vlm_count, "vlm", vlm_conf, annotated)

    # Se VLM também falhou, retorna resultado OpenCV mesmo assim
    return CountResult(image_path, cv_count, "opencv_fallback", confidence, annotated)


if __name__ == "__main__":
    import sys

    images_dir = Path(__file__).parent
    paths = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))

    if len(sys.argv) > 1:
        paths = [Path(sys.argv[1])]

    output_dir = images_dir / "output"
    output_dir.mkdir(exist_ok=True)

    print(f"{'Imagem':<20} {'Contagem':>8} {'Método':<20} {'Confiança':>10}")
    print("-" * 62)

    for path in paths:
        result = count_screws(str(path))
        print(
            f"{path.name:<20} {result.count:>8} {result.method:<20} {result.confidence:>9.0%}"
        )
        out_path = output_dir / f"annotated_{path.name}"
        cv2.imwrite(str(out_path), result.annotated)

    print(f"\nImagens anotadas salvas em: {output_dir}")
