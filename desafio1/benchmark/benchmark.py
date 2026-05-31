"""
Benchmark — Contagem de Parafusos
Compara 5 métodos de visão computacional em tempo, memória e acurácia.
"""

import time
import tracemalloc
import cv2
import numpy as np
import psutil
import os
from pathlib import Path
from dataclasses import dataclass, field
from scipy import ndimage as ndi

# Ground truth (contagem visual)
GROUND_TRUTH = {
    "img1.jpg": 8,
    "img2.jpg": 1,
    "img3.jpg": 4,
    "img4.jpg": 2,
    "img5.jpg": 9,   # contagem conservadora
}

IMAGES_DIR = Path(__file__).parent.parent  # imagens ficam em desafio1/
BENCHMARK_DIR = Path(__file__).parent


def _binary_mask(image: np.ndarray) -> np.ndarray:
    """Segmenta foreground (parafusos) do background: blur → adaptive threshold → morph."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 4
    )
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k5, iterations=2)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k3, iterations=1)
    return thresh


@dataclass
class BenchResult:
    method: str
    image: str
    predicted: int
    ground_truth: int
    time_ms: float
    memory_kb: float
    annotated: np.ndarray = field(repr=False)

    @property
    def error(self):
        return abs(self.predicted - self.ground_truth)

    @property
    def correct(self):
        return self.predicted == self.ground_truth


# ─────────────────────────────────────────────
# Método 1 — Watershed + Connected Components
# ─────────────────────────────────────────────
def method_watershed(image: np.ndarray) -> tuple[int, np.ndarray]:
    """
    Pipeline: Adaptive Threshold → Connected Components →
    Watershed por blob (para parafusos sobrepostos).
    """
    h, w = image.shape[:2]
    image_area = h * w
    min_area = image_area * 0.004
    max_area = image_area * 0.35

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 4)
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k5, iterations=2)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k3, iterations=1)

    n, label_map, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    valid = [(i, int(stats[i, cv2.CC_STAT_AREA]))
             for i in range(1, n)
             if min_area <= stats[i, cv2.CC_STAT_AREA] <= max_area]

    if not valid:
        return 0, image.copy()

    ref_area = float(np.median([a for _, a in valid]))
    centers = []

    for label, area in valid:
        blob_mask = np.uint8(label_map == label) * 255
        estimated = max(1, round(area / ref_area))

        if estimated <= 1:
            M = cv2.moments(blob_mask)
            if M["m00"] > 0:
                centers.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))
        else:
            dist = cv2.distanceTransform(blob_mask, cv2.DIST_L2, 5)
            if dist.max() == 0:
                continue
            _, fg = cv2.threshold(dist, 0.60 * dist.max(), 255, 0)
            fg = np.uint8(fg)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            bg = cv2.dilate(blob_mask, k, iterations=3)
            unk = cv2.subtract(bg, fg)
            _, markers = cv2.connectedComponents(fg)
            markers += 1
            markers[unk == 255] = 0
            ws = markers.copy()
            cv2.watershed(image, ws)
            for lbl in np.unique(ws):
                if lbl <= 1:
                    continue
                region = np.uint8(ws == lbl) * 255
                if cv2.countNonZero(region) < min_area / 4:
                    continue
                M = cv2.moments(region)
                if M["m00"] > 0:
                    centers.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))

    ann = image.copy()
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ann, cnts, -1, (0, 255, 0), 2)
    for i, (cx, cy) in enumerate(centers):
        cv2.circle(ann, (cx, cy), 10, (0, 0, 255), -1)
        cv2.putText(ann, str(i + 1), (cx + 12, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return len(centers), ann


# ─────────────────────────────────────────────
# Método 2 — SimpleBlobDetector
# ─────────────────────────────────────────────
def method_blob_detector(image: np.ndarray) -> tuple[int, np.ndarray]:
    """
    SimpleBlobDetector: detecta as cabeças dos parafusos
    filtrando por área, circularidade e inércia.
    """
    h, w = image.shape[:2]
    image_area = h * w

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    # Inverte para que parafusos sejam regiões escuras
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = image_area * 0.002
    params.maxArea = image_area * 0.25
    params.filterByCircularity = False
    params.filterByConvexity = True
    params.minConvexity = 0.4
    params.filterByInertia = True
    params.minInertiaRatio = 0.05   # aceita formas alongadas (roscas)
    params.filterByColor = True
    params.blobColor = 255          # blobs claros na imagem invertida

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(binary)

    ann = image.copy()
    ann = cv2.drawKeypoints(ann, keypoints, np.array([]),
                             (0, 0, 255),
                             cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    for i, kp in enumerate(keypoints):
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.putText(ann, str(i + 1), (x + 12, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return len(keypoints), ann


# ─────────────────────────────────────────────
# Método 3 — SIFT Template Matching
# ─────────────────────────────────────────────
def _get_screw_template() -> np.ndarray:
    """Usa img2 (1 parafuso isolado) como template para SIFT."""
    img = cv2.imread(str(IMAGES_DIR / "img2.jpg"))
    # Recorta a região do parafuso (centro da imagem)
    h, w = img.shape[:2]
    margin = 0.15
    return img[int(h * margin):int(h * (1 - margin * 2)),
               int(w * 0.3):int(w * 0.7)]


_TEMPLATE_CACHE = None
_SIFT = cv2.SIFT_create()

def method_sift(image: np.ndarray) -> tuple[int, np.ndarray]:
    """
    SIFT Template Matching: extrai keypoints de um parafuso
    isolado (img2) e busca correspondências na imagem-alvo.
    Agrupa matches próximos com clustering espacial.
    """
    global _TEMPLATE_CACHE
    if _TEMPLATE_CACHE is None:
        _TEMPLATE_CACHE = _get_screw_template()

    template = _TEMPLATE_CACHE
    kp_t, des_t = _SIFT.detectAndCompute(template, None)
    kp_i, des_i = _SIFT.detectAndCompute(image, None)

    if des_t is None or des_i is None or len(des_t) < 2 or len(des_i) < 2:
        return 0, image.copy()

    # Lowe's ratio test
    flann = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 50})
    matches = flann.knnMatch(des_t, des_i, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good) < 4:
        return 0, image.copy()

    # Pontos matched na imagem-alvo
    pts = np.float32([kp_i[m.trainIdx].pt for m in good])

    # Agrupa por proximidade espacial — cada cluster = 1 parafuso
    h, w = image.shape[:2]
    min_dist = max(h, w) * 0.08   # 8% da dimensão da imagem

    clusters = []
    used = np.zeros(len(pts), dtype=bool)
    for i, p in enumerate(pts):
        if used[i]:
            continue
        cluster = [p]
        used[i] = True
        for j, q in enumerate(pts):
            if not used[j] and np.linalg.norm(p - q) < min_dist:
                cluster.append(q)
                used[j] = True
        clusters.append(np.mean(cluster, axis=0).astype(int))

    ann = image.copy()
    for i, (cx, cy) in enumerate(clusters):
        cv2.circle(ann, (cx, cy), 15, (255, 0, 0), 2)
        cv2.circle(ann, (cx, cy), 4, (255, 0, 0), -1)
        cv2.putText(ann, str(i + 1), (cx + 15, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    # Desenha matches
    ann = cv2.drawMatches(template, kp_t, ann, kp_i,
                           good[:30], None,
                           matchColor=(0, 255, 0),
                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return len(clusters), ann


# ─────────────────────────────────────────────
# Método 4 — Canny + Contornos
# ─────────────────────────────────────────────
def method_canny(image: np.ndarray) -> tuple[int, np.ndarray]:
    """
    Canny edge detection: detecta bordas dos parafusos,
    dilata para fechar silhueta e conta regiões fechadas.
    """
    h, w = image.shape[:2]
    image_area = h * w
    min_area = image_area * 0.004
    max_area = image_area * 0.35

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)

    # Auto-threshold via mediana (método de Otsu adaptado)
    med = np.median(blurred)
    lo, hi = int(max(0, 0.67 * med)), int(min(255, 1.33 * med))
    edges = cv2.Canny(blurred, lo, hi)

    # Fecha as bordas para formar regiões sólidas
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=2)
    filled = cv2.morphologyEx(closed, cv2.MORPH_DILATE, k, iterations=1)

    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (min_area <= area <= max_area):
            continue
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        if area / (hull_area + 1e-6) < 0.25:
            continue
        valid.append(cnt)

    ann = image.copy()
    cv2.drawContours(ann, valid, -1, (255, 165, 0), 2)
    for i, cnt in enumerate(valid):
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            cv2.circle(ann, (cx, cy), 8, (255, 165, 0), -1)
            cv2.putText(ann, str(i + 1), (cx + 10, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
    return len(valid), ann


# ─────────────────────────────────────────────
# Método 5 — Subtração de Background por Cor
# ─────────────────────────────────────────────
def method_color_bg(image: np.ndarray) -> tuple[int, np.ndarray]:
    """
    Background subtraction por cor: estima a cor dominante
    do fundo amostrando bordas e segmenta por distância de cor.
    Funciona para fundos uniformes (azul, branco, cinza).
    """
    h, w = image.shape[:2]
    image_area = h * w
    min_area = image_area * 0.004
    max_area = image_area * 0.35

    border = 30
    border_pixels = np.concatenate([
        image[:border, :].reshape(-1, 3),
        image[-border:, :].reshape(-1, 3),
        image[:, :border].reshape(-1, 3),
        image[:, -border:].reshape(-1, 3),
    ])
    bg_color = np.median(border_pixels, axis=0).astype(np.float32)

    img_f = image.astype(np.float32)
    diff = np.linalg.norm(img_f - bg_color, axis=2)

    # Normaliza e thresholda
    diff_norm = (diff / (diff.max() + 1e-6) * 255).astype(np.uint8)
    _, thresh = cv2.threshold(diff_norm, 40, 255, cv2.THRESH_BINARY)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k, iterations=1)

    n, label_map, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    valid = [(i, int(stats[i, cv2.CC_STAT_AREA]))
             for i in range(1, n)
             if min_area <= stats[i, cv2.CC_STAT_AREA] <= max_area]

    if not valid:
        return 0, image.copy()

    ref_area = float(np.median([a for _, a in valid]))
    centers = []
    for label, area in valid:
        blob_mask = np.uint8(label_map == label) * 255
        estimated = max(1, round(area / ref_area))
        for _ in range(estimated):
            M = cv2.moments(blob_mask)
            if M["m00"] > 0:
                centers.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))

    ann = image.copy()
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ann, cnts, -1, (0, 200, 200), 2)
    for i, (cx, cy) in enumerate(centers):
        cv2.circle(ann, (cx, cy), 10, (0, 200, 200), -1)
        cv2.putText(ann, str(i + 1), (cx + 12, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)
    return len(centers), ann


# ─────────────────────────────────────────────
# Método 6 — Hough Circles
# ─────────────────────────────────────────────
def method_hough_circles(image: np.ndarray) -> tuple[int, np.ndarray]:
    """
    Transformada de Hough para círculos: detecta as cabeças redondas dos
    parafusos diretamente. dp=1.2, param2=25 (sensível), raios relativos
    ao tamanho da imagem para funcionar em diferentes resoluções.
    """
    h, w = image.shape[:2]
    min_dim = min(h, w)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=int(min_dim * 0.07),
        param1=60,
        param2=25,
        minRadius=int(min_dim * 0.02),
        maxRadius=int(min_dim * 0.22),
    )

    ann = image.copy()
    if circles is None:
        return 0, ann

    circles = np.round(circles[0]).astype(int)
    for i, (x, y, r) in enumerate(circles):
        cv2.circle(ann, (x, y), r, (255, 0, 255), 2)
        cv2.circle(ann, (x, y), 4, (255, 0, 255), -1)
        cv2.putText(ann, str(i + 1), (x + r + 5, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    return len(circles), ann


# ─────────────────────────────────────────────
# Método 7 — GrabCut + Watershed
# ─────────────────────────────────────────────
def method_grabcut(image: np.ndarray) -> tuple[int, np.ndarray]:
    """
    GrabCut ajusta um GMM iterativo (5 iterações) para separar foreground
    de background — mais robusto a texturas do que limiarização simples.
    Após a máscara, aplica o mesmo watershed+componentes do Método 1.
    """
    h, w = image.shape[:2]
    margin = int(min(h, w) * 0.05)
    rect = (margin, margin, w - 2 * margin, h - 2 * margin)

    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)

    fg_mask = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)

    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, k5, iterations=2)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, k3, iterations=1)

    image_area = h * w
    min_area = image_area * 0.004
    max_area = image_area * 0.35

    n, label_map, stats, _ = cv2.connectedComponentsWithStats(fg_mask, connectivity=8)
    valid = [(i, int(stats[i, cv2.CC_STAT_AREA]))
             for i in range(1, n)
             if min_area <= stats[i, cv2.CC_STAT_AREA] <= max_area]

    if not valid:
        return 0, image.copy()

    ref_area = float(np.median([a for _, a in valid]))
    centers = []

    for label, area in valid:
        blob_mask = np.uint8(label_map == label) * 255
        estimated = max(1, round(area / ref_area))

        if estimated <= 1:
            M = cv2.moments(blob_mask)
            if M["m00"] > 0:
                centers.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))
        else:
            dist = cv2.distanceTransform(blob_mask, cv2.DIST_L2, 5)
            if dist.max() == 0:
                continue
            _, fg = cv2.threshold(dist, 0.60 * dist.max(), 255, 0)
            fg = np.uint8(fg)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            bg = cv2.dilate(blob_mask, k, iterations=3)
            unk = cv2.subtract(bg, fg)
            _, markers = cv2.connectedComponents(fg)
            markers += 1
            markers[unk == 255] = 0
            ws = markers.copy()
            cv2.watershed(image, ws)
            for lbl in np.unique(ws):
                if lbl <= 1:
                    continue
                region = np.uint8(ws == lbl) * 255
                if cv2.countNonZero(region) < min_area / 4:
                    continue
                M = cv2.moments(region)
                if M["m00"] > 0:
                    centers.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))

    ann = image.copy()
    cnts, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ann, cnts, -1, (0, 165, 255), 2)
    for i, (cx, cy) in enumerate(centers):
        cv2.circle(ann, (cx, cy), 10, (0, 165, 255), -1)
        cv2.putText(ann, str(i + 1), (cx + 12, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    return len(centers), ann


# ─────────────────────────────────────────────
# Método 8 — MobileNetV2 Feature Salience (CNN)
# ─────────────────────────────────────────────
_MOBILENET_MODEL = None


def _get_mobilenet():
    global _MOBILENET_MODEL
    if _MOBILENET_MODEL is None:
        import torch
        import torch.nn as nn
        import torchvision.models as tvm
        model = tvm.mobilenet_v2(weights=tvm.MobileNet_V2_Weights.IMAGENET1K_V1)
        # features[:8] → stride 16, resolução 1/16 da original
        _MOBILENET_MODEL = nn.Sequential(*list(model.features.children())[:8])
        _MOBILENET_MODEL.eval()
    return _MOBILENET_MODEL


def method_mobilenet_salience(image: np.ndarray) -> tuple[int, np.ndarray]:
    """
    MobileNetV2 pré-treinado (ImageNet) como extrator zero-shot.
    A norma L2 das ativações da camada 8 (stride 16) produz um mapa de
    saliência que destaca regiões visualmente distintas do fundo. Inspirado
    em density-map counting (CSRNet et al.) sem nenhum treino específico.
    Threshold Otsu + watershed sobre o mapa → contagem de peaks.
    """
    import torch
    import torchvision.transforms as T

    h, w = image.shape[:2]
    image_area = h * w

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((320, 320)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    inp = transform(img_rgb).unsqueeze(0)

    net = _get_mobilenet()
    with torch.no_grad():
        feats = net(inp)  # [1, C, H', W']

    saliency = feats[0].norm(dim=0).numpy()
    saliency_resized = cv2.resize(saliency, (w, h), interpolation=cv2.INTER_LINEAR)

    sal_min, sal_max = saliency_resized.min(), saliency_resized.max()
    sal_norm = ((saliency_resized - sal_min) / (sal_max - sal_min + 1e-6) * 255).astype(np.uint8)

    _, thresh = cv2.threshold(sal_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k5, iterations=2)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k3, iterations=1)

    min_area = image_area * 0.004
    max_area = image_area * 0.35
    n, label_map, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    valid = [(i, int(stats[i, cv2.CC_STAT_AREA]))
             for i in range(1, n)
             if min_area <= stats[i, cv2.CC_STAT_AREA] <= max_area]

    if not valid:
        return 0, image.copy()

    ref_area = float(np.median([a for _, a in valid]))
    centers = []
    for label, area in valid:
        blob_mask = np.uint8(label_map == label) * 255
        estimated = max(1, round(area / ref_area))
        M = cv2.moments(blob_mask)
        if M["m00"] > 0:
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            for _ in range(estimated):
                centers.append((cx, cy))

    heatmap = cv2.applyColorMap(sal_norm, cv2.COLORMAP_JET)
    ann = cv2.addWeighted(image.copy(), 0.55, heatmap, 0.45, 0)
    for i, (cx, cy) in enumerate(centers):
        cv2.circle(ann, (cx, cy), 10, (255, 255, 0), -1)
        cv2.putText(ann, str(i + 1), (cx + 12, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    return len(centers), ann


# ─────────────────────────────────────────────
# Método 9 — VLM Zero-Shot (Claude API)
# ─────────────────────────────────────────────
def method_vlm_claude(image: np.ndarray) -> tuple[int, np.ndarray]:
    """
    Claude claude-sonnet-4-6 via API para contagem zero-shot.
    Prompt explícito para incluir sobrepostos e parcialmente ocluídos —
    exatamente o caso onde CV clássico falha (img5).
    Requer ANTHROPIC_API_KEY no ambiente.
    """
    try:
        import anthropic
        import base64

        _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        img_b64 = base64.standard_b64encode(buf.tobytes()).decode("utf-8")

        client = anthropic.Anthropic()
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=32,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64},
                    },
                    {
                        "type": "text",
                        "text": (
                            "Count every screw and bolt visible in the image, "
                            "including overlapping or partially hidden ones. "
                            "Reply with a single integer only."
                        ),
                    },
                ],
            }],
        )
        count = int(msg.content[0].text.strip().split()[0])
        ann = image.copy()
        cv2.putText(ann, f"VLM: {count} screws", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 0), 3)
        return count, ann
    except Exception as e:
        print(f"  [VLM] {e}")
        return -1, image.copy()


# ─────────────────────────────────────────────
# Método 10 — Watershed com Regional Maxima
# ─────────────────────────────────────────────
def method_ws_regional_max(image: np.ndarray) -> tuple[int, np.ndarray]:
    """
    Melhoria direta do Watershed: substitui o limiar global (0.60 * max)
    por máximos regionais do distance transform (scipy.maximum_filter).
    Cada máximo local = 1 semente, independente da altura absoluta.
    Captura picos de screws sobrepostos que ficam abaixo de 60% do global max.
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
        return 0, image.copy()

    ref_area = float(np.median([a for _, a in valid]))
    centers = []

    # Footprint: ~5% da menor dimensão. Menor → mais picos (bom p/ sobrepostos),
    # maior → menos picos (bom p/ screws alongados isolados). 5% é o equilíbrio.
    fp = max(7, int(min(h, w) * 0.05))
    if fp % 2 == 0:
        fp += 1

    for label, area in valid:
        blob_mask = np.uint8(label_map == label) * 255
        estimated = max(1, round(area / ref_area))

        if estimated <= 1:
            # blob simples: centroide direto (evita over-segmentação)
            M = cv2.moments(blob_mask)
            if M["m00"] > 0:
                centers.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))
        else:
            # blob com sobreposição: regional maxima como seeds do watershed
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

    ann = image.copy()
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ann, cnts, -1, (80, 255, 80), 2)
    for i, (cx, cy) in enumerate(centers):
        cv2.circle(ann, (cx, cy), 10, (80, 255, 80), -1)
        cv2.putText(ann, str(i + 1), (cx + 12, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 255, 80), 2)
    return len(centers), ann


# ─────────────────────────────────────────────
# Método 11 — LoG Multi-Escala (scale-space)
# ─────────────────────────────────────────────
def method_log_blobs(image: np.ndarray) -> tuple[int, np.ndarray]:
    """
    Laplacian of Gaussian (LoG) aplicado ao distance transform em múltiplas
    escalas σ. A resposta normalizada por σ² é invariante à escala — cada
    parafuso produz um pico no espaço escala independente de seu tamanho.
    NMS 3D (escala × H × W) extrai centros sem duplicatas.
    Referência: Lindeberg (1998), feature detection with automatic scale selection.
    """
    h, w = image.shape[:2]
    binary = _binary_mask(image)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5).astype(np.float32)

    min_sigma = max(3.0, min(h, w) * 0.015)
    max_sigma = max(min_sigma + 5, min(h, w) * 0.11)
    n_sigmas = 14
    sigmas = np.linspace(min_sigma, max_sigma, n_sigmas)

    cube = np.zeros((n_sigmas, h, w), dtype=np.float32)
    for i, sigma in enumerate(sigmas):
        response = -ndi.gaussian_laplace(dist, sigma=sigma) * float(sigma ** 2)
        cube[i] = response

    fp_h = max(5, int(h * 0.07))
    fp_w = max(5, int(w * 0.07))
    max_cube = ndi.maximum_filter(cube, size=(3, fp_h, fp_w))
    threshold = max(cube.max() * 0.20, 1.0)
    peaks_3d = (cube == max_cube) & (cube >= threshold) & (dist[np.newaxis] > 2.0)

    coords = np.array(np.where(peaks_3d)).T  # [N, 3]: (scale, y, x)
    if len(coords) == 0:
        return 0, image.copy()

    ys, xs = coords[:, 1], coords[:, 2]
    scores = cube[coords[:, 0], ys, xs]
    min_dist_px = int(min(h, w) * 0.07)

    order = np.argsort(-scores)
    kept: list[tuple[int, int]] = []
    suppressed = np.zeros(len(order), dtype=bool)
    for idx in order:
        if suppressed[idx]:
            continue
        cy, cx = int(ys[idx]), int(xs[idx])
        kept.append((cx, cy))
        for jdx in order:
            if suppressed[jdx] or jdx == idx:
                continue
            d = np.hypot(cx - xs[jdx], cy - ys[jdx])
            if d < min_dist_px:
                suppressed[jdx] = True

    ann = image.copy()
    for i, (cx, cy) in enumerate(kept):
        cv2.circle(ann, (cx, cy), 12, (200, 80, 255), 2)
        cv2.circle(ann, (cx, cy), 4, (200, 80, 255), -1)
        cv2.putText(ann, str(i + 1), (cx + 14, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 80, 255), 2)
    return len(kept), ann


# ─────────────────────────────────────────────
# Método 12 — NCC Template Multi-Escala
# ─────────────────────────────────────────────
_TEMPLATE_NCC_CACHE = None


def _get_ncc_template() -> np.ndarray | None:
    global _TEMPLATE_NCC_CACHE
    if _TEMPLATE_NCC_CACHE is not None:
        return _TEMPLATE_NCC_CACHE
    img2 = cv2.imread(str(IMAGES_DIR / "img2.jpg"))
    if img2 is None:
        return None
    binary = _binary_mask(img2)
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        _TEMPLATE_NCC_CACHE = img2
        return img2
    cnt = max(cnts, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(cnt)
    margin = int(max(bw, bh) * 0.25)
    x1, y1 = max(0, x - margin), max(0, y - margin)
    x2, y2 = min(img2.shape[1], x + bw + margin), min(img2.shape[0], y + bh + margin)
    _TEMPLATE_NCC_CACHE = img2[y1:y2, x1:x2]
    return _TEMPLATE_NCC_CACHE


def method_ncc_template(image: np.ndarray) -> tuple[int, np.ndarray]:
    """
    Normalized Cross-Correlation (NCC) multi-escala usando img2 como template.
    Img2 contém exatamente 1 parafuso isolado — template de referência ideal.
    14 escalas (0.3×→2.5×). NMS greedy por score elimina duplicatas.
    NCC é invariante a offset de brilho mas não a rotação — compensado
    pelo limiar mais baixo (0.38) que captura matches em orientações diferentes.
    """
    template = _get_ncc_template()
    if template is None:
        return 0, image.copy()

    target_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    tmpl_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    h, w = image.shape[:2]
    th, tw = tmpl_gray.shape[:2]

    scales = np.linspace(0.3, 2.5, 14)
    threshold = 0.38
    all_det: list[tuple[float, int, int, int]] = []  # (score, cx, cy, r)

    for scale in scales:
        sw, sh = int(tw * scale), int(th * scale)
        if sw < 5 or sh < 5 or sw >= w or sh >= h:
            continue
        resized = cv2.resize(tmpl_gray, (sw, sh))
        result = cv2.matchTemplate(target_gray, resized, cv2.TM_CCOEFF_NORMED)
        locs = np.where(result >= threshold)
        for yy, xx in zip(*locs):
            score = float(result[yy, xx])
            cx, cy = xx + sw // 2, yy + sh // 2
            r = int(np.sqrt(sw * sh) / 2)
            all_det.append((score, cx, cy, r))

    if not all_det:
        return 0, image.copy()

    all_det.sort(reverse=True)
    kept: list[tuple[int, int, int]] = []
    used = [False] * len(all_det)
    for i, (_, cx, cy, r) in enumerate(all_det):
        if used[i]:
            continue
        kept.append((cx, cy, max(r, 10)))
        for j, (_, cx2, cy2, r2) in enumerate(all_det):
            if not used[j] and j != i:
                if np.hypot(cx - cx2, cy - cy2) < max(r, r2) * 1.3:
                    used[j] = True

    ann = image.copy()
    for i, (cx, cy, r) in enumerate(kept):
        cv2.circle(ann, (cx, cy), r, (0, 180, 80), 2)
        cv2.putText(ann, str(i + 1), (cx + r + 4, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 80), 2)
    return len(kept), ann


# ─────────────────────────────────────────────
# Método 13 — Convex Hull Defects
# ─────────────────────────────────────────────
def method_convex_defects(image: np.ndarray) -> tuple[int, np.ndarray]:
    """
    Conta concavidades profundas no hull convexo de cada blob.
    Parafusos tocando criam 'estrangulamentos' — cada estrangulamento
    gera 2 defects simétricas. Heurística: screws = 1 + deep_defects // 2.
    Profundidade mínima = 2% da menor dimensão da imagem.
    """
    h, w = image.shape[:2]
    image_area = h * w
    min_area = image_area * 0.004
    max_area = image_area * 0.35
    min_depth = int(min(h, w) * 0.02 * 256)  # depth em fixedpoint (÷256 = pixels)

    binary = _binary_mask(image)
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    ann = image.copy()

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if not (min_area <= area <= max_area):
            continue

        hull_idx = cv2.convexHull(cnt, returnPoints=False)
        M = cv2.moments(cnt)
        cx = int(M["m10"] / M["m00"]) if M["m00"] > 0 else 0
        cy = int(M["m01"] / M["m00"]) if M["m00"] > 0 else 0

        try:
            defects = cv2.convexityDefects(cnt, hull_idx)
        except cv2.error:
            defects = None

        if defects is None:
            centers.append((cx, cy))
            continue

        deep = int(np.sum(defects[:, 0, 3] > min_depth))
        screw_count = max(1, 1 + deep // 2)

        for _ in range(screw_count):
            centers.append((cx, cy))

        cv2.drawContours(ann, [cnt], -1, (0, 120, 255), 2)
        cv2.putText(ann, str(screw_count), (cx + 5, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 120, 255), 2)

    return len(centers), ann


# ─────────────────────────────────────────────
# Método 14 — MSER (Tier 1 — Embedded)
# ─────────────────────────────────────────────
def method_mser(image: np.ndarray) -> tuple[int, np.ndarray]:
    """
    MSER (Maximally Stable Extremal Regions).
    Detecta regiões que permanecem estáveis sob variação de threshold.
    Extremamente leve (< 5ms), sem dependências além do OpenCV.
    Fraco em objetos sobrepostos — MSER funde blobs conectados num único.
    """
    h, w = image.shape[:2]
    image_area = h * w
    min_area = int(image_area * 0.003)
    max_area = int(image_area * 0.28)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mser = cv2.MSER_create(5, min_area, max_area, max_variation=0.25)
    _, bboxes = mser.detectRegions(gray)

    if len(bboxes) == 0:
        return 0, image.copy()

    # Filtrar por compacidade: parafusos têm aspect ratio > 0.35
    candidates = []
    for x, y, bw, bh in bboxes:
        aspect = min(bw, bh) / (max(bw, bh) + 1e-6)
        if aspect > 0.35:
            candidates.append((x + bw // 2, y + bh // 2))

    # NMS por distância mínima
    min_dist = int(min(h, w) * 0.06)
    kept: list[tuple[int, int]] = []
    used = [False] * len(candidates)
    for i, (cx, cy) in enumerate(candidates):
        if used[i]:
            continue
        kept.append((cx, cy))
        for j, (cx2, cy2) in enumerate(candidates):
            if not used[j] and j != i and np.hypot(cx - cx2, cy - cy2) < min_dist:
                used[j] = True

    ann = image.copy()
    for i, (cx, cy) in enumerate(kept):
        cv2.circle(ann, (cx, cy), 14, (0, 180, 100), 2)
        cv2.putText(ann, str(i + 1), (cx + 15, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 100), 2)
    return len(kept), ann


# ─────────────────────────────────────────────
# Método 15 — Shi-Tomasi + Clustering (Tier 1)
# ─────────────────────────────────────────────
def method_shi_tomasi(image: np.ndarray) -> tuple[int, np.ndarray]:
    """
    Shi-Tomasi goodFeaturesToTrack + agrupamento por distância.
    Detecta cantos da cabeça dos parafusos; cantos próximos = 1 parafuso.
    Mascarado pelo foreground (só pontos sobre parafusos).
    Ultra-leve: < 5ms. Falha quando screws têm poucas arestas (ex.: pilha).
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    corners = cv2.goodFeaturesToTrack(
        blurred, maxCorners=600, qualityLevel=0.015,
        minDistance=int(min(h, w) * 0.015), blockSize=7
    )
    if corners is None:
        return 0, image.copy()

    corners = corners.reshape(-1, 2).astype(int)
    binary = _binary_mask(image)
    valid = [(x, y) for x, y in corners
             if 0 <= y < h and 0 <= x < w and binary[y, x] > 0]

    if not valid:
        return 0, image.copy()

    min_dist = max(h, w) * 0.065
    clusters: list[tuple[int, int]] = []
    used = [False] * len(valid)
    for i, (cx, cy) in enumerate(valid):
        if used[i]:
            continue
        group = [(cx, cy)]
        used[i] = True
        for j, (cx2, cy2) in enumerate(valid):
            if not used[j] and np.hypot(cx - cx2, cy - cy2) < min_dist:
                group.append((cx2, cy2))
                used[j] = True
        clusters.append((int(np.mean([p[0] for p in group])),
                         int(np.mean([p[1] for p in group]))))

    ann = image.copy()
    for x, y in corners:
        cv2.circle(ann, (x, y), 2, (200, 200, 0), -1)
    for i, (cx, cy) in enumerate(clusters):
        cv2.circle(ann, (cx, cy), 12, (200, 200, 0), 2)
        cv2.putText(ann, str(i + 1), (cx + 13, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
    return len(clusters), ann


# ─────────────────────────────────────────────
# Método 16 — Faster R-CNN COCO (Tier 3)
# ─────────────────────────────────────────────
_fasterrcnn_model = None


def _get_fasterrcnn():
    global _fasterrcnn_model
    if _fasterrcnn_model is None:
        import torchvision.models.detection as det
        _fasterrcnn_model = det.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        _fasterrcnn_model.eval()
    return _fasterrcnn_model


def method_fasterrcnn(image: np.ndarray) -> tuple[int, np.ndarray]:
    """
    Faster R-CNN ResNet50-FPN treinado no COCO (80 classes).
    COCO não tem classe 'screw' — filtra por tamanho de bbox (0.3%–15% da imagem)
    e score > 0.3 para capturar qualquer objeto pequeno detectado.
    Requer ~330 MB disco, ~1.5 GB RAM; GPU acelera 20×.
    Esperado: zero detecções na maioria das imagens (domínio fora do COCO).
    """
    import torch
    import torchvision.transforms.functional as TF

    h, w = image.shape[:2]
    model = _get_fasterrcnn()

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = TF.to_tensor(img_rgb)
    with torch.no_grad():
        preds = model([tensor])[0]

    boxes = preds["boxes"].numpy()
    scores = preds["scores"].numpy()

    min_area = h * w * 0.003
    max_area = h * w * 0.20
    threshold = 0.30

    valid = []
    for box, score in zip(boxes, scores):
        if score < threshold:
            continue
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        if min_area <= area <= max_area:
            valid.append((int(x1), int(y1), int(x2), int(y2), float(score)))

    ann = image.copy()
    for i, (x1, y1, x2, y2, score) in enumerate(valid):
        cv2.rectangle(ann, (x1, y1), (x2, y2), (255, 100, 0), 2)
        cv2.putText(ann, f"{i + 1}({score:.2f})", (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 100, 0), 1)
    return len(valid), ann


# ─────────────────────────────────────────────
# Métodos 17–18 — OwlViT + CLIP (Tier 3)
# ─────────────────────────────────────────────
def _import_advanced():
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location(
        "adv_models", BENCHMARK_DIR / "advanced_models.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["adv_models"] = mod
    spec.loader.exec_module(mod)
    return mod


_adv = None


def method_owlvit(image: np.ndarray) -> tuple[int, np.ndarray]:
    """
    OwlViT (Google, 2022) — detecção zero-shot por texto.
    Prompt: 'a screw', 'a bolt', 'a metal fastener', 'a metallic cylinder'.
    ~340 MB disco, ~2 GB RAM; sem GPU é lento (~8s/img).
    """
    global _adv
    if _adv is None:
        _adv = _import_advanced()
    return _adv.method_owlvit(image)


def method_clip(image: np.ndarray) -> tuple[int, np.ndarray]:
    """
    CLIP (OpenAI, 2021) — contagem semântica texto→imagem.
    Escolhe N (1–25) com maior similaridade para 'N screws'.
    ~300 MB disco, ~1 GB RAM. Sem bounding boxes: abordagem holística.
    """
    global _adv
    if _adv is None:
        _adv = _import_advanced()
    return _adv.method_clip_count(image)


# ─────────────────────────────────────────────────────────────
# MÉTODOS MELHORADOS — Fine-tuning com dataset Cross-Recessed-Screw
# ─────────────────────────────────────────────────────────────
MODELS_DIR = BENCHMARK_DIR / "models"

_ncc_templates = None
_log_config    = None
_owlvit_config = None
_clip_reg      = None
_mobilenet_reg = None
_fasterrcnn_ft = None


# ─────────────────────────────────────────────
# NCC_MultiTemplate (Tier 2 melhorado)
# ─────────────────────────────────────────────
def _make_synthetic_screw_templates(sizes=(48, 64, 80)) -> list[np.ndarray]:
    """
    Gera templates sintéticos de parafusos (cabeça circular + cruz Phillips).
    Domínio-específico: fundo uniforme, parafuso metálico centralizado.
    Evita domain mismatch com dataset de hardware (laptops/keyboards).
    """
    templates = []
    for sz in sizes:
        for bg_val in (40, 80, 120):  # diferentes tons de fundo
            tmpl = np.full((sz, sz), bg_val, dtype=np.uint8)
            cx, cy, r = sz//2, sz//2, sz//2 - 4
            # Cabeça circular com gradiente radial (metal)
            for y in range(sz):
                for x in range(sz):
                    d = np.hypot(x - cx, y - cy)
                    if d < r:
                        bright = int(180 - 60 * (d / r))
                        tmpl[y, x] = bright
            # Cruz Phillips
            cross_w = max(2, sz // 10)
            for i in range(-cross_w, cross_w + 1):
                tmpl[cy + i, max(0, cx - r//2):min(sz, cx + r//2)] = 60
                tmpl[max(0, cy - r//2):min(sz, cy + r//2), cx + i] = 60
            # Borda escura
            cv2.circle(tmpl, (cx, cy), r, 30, 2)
            templates.append(tmpl)
            # Variação com hexágono (parafuso hex)
            tmpl2 = tmpl.copy()
            pts = np.array([
                [cx + int(r * np.cos(np.pi/3 * i)) for i in range(6)],
                [cy + int(r * np.sin(np.pi/3 * i)) for i in range(6)],
            ]).T
            cv2.polylines(tmpl2, [pts], True, 30, 2)
            templates.append(tmpl2)
    return templates


_synthetic_templates = None


def method_ncc_multi(image: np.ndarray) -> tuple[int, np.ndarray]:
    """
    NCC Híbrido — 2 etapas:
    1. Segmentação morphológica (igual WS_RegionalMax) identifica blobs de screws.
    2. Para cada blob, NCC com templates sintéticos em múltiplas escalas.
       Conta quantos templates encaixam dentro do blob → screws por blob.
    Vantagem: NCC restrito ao ROI de cada blob; evita falsos positivos no fundo.
    Limitação: ainda pode errar em pilhas densas onde blobs se fundem.
    """
    global _synthetic_templates
    if _synthetic_templates is None:
        _synthetic_templates = _make_synthetic_screw_templates()

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
        return 0, image.copy()

    ref_area = float(np.median([a for _, a in valid]))
    gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    centers: list[tuple[int, int]] = []

    for label, area in valid:
        blob_mask = np.uint8(label_map == label) * 255
        estimated = max(1, round(area / ref_area))

        if estimated <= 1:
            M = cv2.moments(blob_mask)
            if M["m00"] > 0:
                centers.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))
            continue

        # ROI do blob para NCC focado
        ys, xs = np.where(blob_mask > 0)
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        roi_gray = gray_full[y1:y2+1, x1:x2+1]
        roi_h, roi_w = roi_gray.shape[:2]

        # Tamanho esperado por parafuso dentro do blob
        expected_screw_sz = int(np.sqrt(area / estimated))
        scales = np.linspace(0.5, 1.5, 6)
        all_det: list[tuple[float, int, int]] = []

        for tmpl in _synthetic_templates[::3]:  # amostra 1/3 dos templates
            th, tw = tmpl.shape[:2]
            for scale in scales:
                sw = max(8, int(expected_screw_sz * scale))
                sh = sw
                if sw >= roi_w or sh >= roi_h:
                    continue
                resized = cv2.resize(tmpl, (sw, sh))
                result  = cv2.matchTemplate(roi_gray, resized, cv2.TM_CCOEFF_NORMED)
                locs    = np.where(result >= 0.38)
                for yy, xx in zip(*locs):
                    cx_roi = xx + sw // 2
                    cy_roi = yy + sh // 2
                    # Só pontos dentro do blob
                    cx_abs = x1 + cx_roi
                    cy_abs = y1 + cy_roi
                    if cy_abs < blob_mask.shape[0] and cx_abs < blob_mask.shape[1]:
                        if blob_mask[min(cy_abs, h-1), min(cx_abs, w-1)] > 0:
                            all_det.append((float(result[yy, xx]), cx_abs, cy_abs))

        if not all_det:
            # Fallback: regional maxima (como WS_RegionalMax)
            dist = cv2.distanceTransform(blob_mask, cv2.DIST_L2, 5)
            fp = max(7, int(min(h, w) * 0.05))
            if fp % 2 == 0:
                fp += 1
            max_filt = ndi.maximum_filter(dist, size=fp)
            peaks    = ((dist == max_filt) & (dist > dist.max() * 0.12)).astype(np.uint8) * 255
            _, m = cv2.connectedComponents(peaks)
            for lbl in np.unique(m):
                if lbl == 0:
                    continue
                ys_p, xs_p = np.where(m == lbl)
                cx_p = int(np.mean(xs_p))
                cy_p = int(np.mean(ys_p))
                centers.append((cx_p, cy_p))
            continue

        # NMS dentro do blob
        all_det.sort(reverse=True)
        used = [False] * len(all_det)
        min_d = max(10, expected_screw_sz // 2)
        for i, (_, cx, cy_) in enumerate(all_det):
            if used[i]:
                continue
            centers.append((cx, cy_))
            for j, (_, cx2, cy2) in enumerate(all_det):
                if not used[j] and j != i:
                    if np.hypot(cx - cx2, cy_ - cy2) < min_d:
                        used[j] = True

    ann = image.copy()
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ann, cnts, -1, (50, 200, 50), 2)
    for i, (cx, cy_) in enumerate(centers):
        cv2.circle(ann, (cx, cy_), 10, (50, 200, 50), -1)
        cv2.putText(ann, str(i + 1), (cx + 12, cy_ + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 200, 50), 2)
    return len(centers), ann


# ─────────────────────────────────────────────
# LoG_Calibrado (Tier 2 melhorado)
# ─────────────────────────────────────────────
def method_log_calibrated(image: np.ndarray) -> tuple[int, np.ndarray]:
    """
    LoG Scale-Space com sigma range calibrado a partir dos tamanhos reais
    de parafusos do dataset Cross-Recessed-Screw (100 imagens de treino).
    min_sigma e max_sigma adaptam-se à resolução da imagem de entrada.
    """
    import json

    global _log_config
    cfg_path = MODELS_DIR / "log_config.json"
    if _log_config is None and cfg_path.exists():
        with open(cfg_path) as f:
            _log_config = json.load(f)

    h, w = image.shape[:2]
    min_dim = min(h, w)

    if _log_config:
        min_sigma = max(3.0, _log_config["min_sigma_rel"] * min_dim)
        max_sigma = max(min_sigma + 5, _log_config["max_sigma_rel"] * min_dim)
        n_sigmas  = _log_config["n_sigmas"]
    else:
        min_sigma = max(3.0, min_dim * 0.015)
        max_sigma = max(min_sigma + 5, min_dim * 0.11)
        n_sigmas  = 14

    binary = _binary_mask(image)
    dist   = cv2.distanceTransform(binary, cv2.DIST_L2, 5).astype(np.float32)

    sigmas = np.linspace(min_sigma, max_sigma, n_sigmas)
    cube   = np.zeros((n_sigmas, h, w), dtype=np.float32)
    for i, sigma in enumerate(sigmas):
        resp = -ndi.gaussian_laplace(dist, sigma=sigma) * float(sigma ** 2)
        cube[i] = resp

    fp_h   = max(5, int(h * 0.06))
    fp_w   = max(5, int(w * 0.06))
    max_c  = ndi.maximum_filter(cube, size=(3, fp_h, fp_w))
    thr    = max(cube.max() * 0.18, 0.8)
    peaks  = (cube == max_c) & (cube >= thr) & (dist[np.newaxis] > 1.5)

    coords = np.array(np.where(peaks)).T
    if len(coords) == 0:
        return 0, image.copy()

    ys, xs   = coords[:, 1], coords[:, 2]
    scores   = cube[coords[:, 0], ys, xs]
    min_dist = int(min_dim * 0.06)

    order = np.argsort(-scores)
    kept: list[tuple[int, int]] = []
    supp = np.zeros(len(order), dtype=bool)
    for idx in order:
        if supp[idx]:
            continue
        cy, cx = int(ys[idx]), int(xs[idx])
        kept.append((cx, cy))
        for jdx in order:
            if not supp[jdx] and jdx != idx:
                if np.hypot(cx - xs[jdx], cy - ys[jdx]) < min_dist:
                    supp[jdx] = True

    ann = image.copy()
    for i, (cx, cy) in enumerate(kept):
        cv2.circle(ann, (cx, cy), 12, (100, 80, 255), 2)
        cv2.putText(ann, str(i + 1), (cx + 14, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 80, 255), 2)
    return len(kept), ann


# ─────────────────────────────────────────────
# OwlViT_Adapted (Tier 3 melhorado)
# ─────────────────────────────────────────────
def method_owlvit_adapted(image: np.ndarray) -> tuple[int, np.ndarray]:
    """
    OwlViT com threshold e prompts calibrados no dataset real.
    Threshold ótimo encontrado: 0.15 (em vez de 0.08 original).
    Prompts focados em "cross recessed screw" e "Phillips head screw".
    Melhoria direta sem re-treinamento: calibração zero-shot.
    """
    import json
    import torch
    from PIL import Image as PILImage

    global _adv, _owlvit_config
    if _adv is None:
        _adv = _import_advanced()

    cfg_path = MODELS_DIR / "owlvit_config.json"
    if _owlvit_config is None and cfg_path.exists():
        with open(cfg_path) as f:
            _owlvit_config = json.load(f)

    threshold = _owlvit_config["threshold"] if _owlvit_config else 0.12
    prompts   = _owlvit_config["prompts"]   if _owlvit_config else ["a screw", "a bolt"]

    from transformers import OwlViTForObjectDetection, OwlViTProcessor

    proc  = _adv._load_owlvit()[0]
    model = _adv._load_owlvit()[1]

    h, w = image.shape[:2]
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = PILImage.fromarray(img_rgb)

    inputs  = proc(text=[prompts], images=pil_img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    logits    = outputs.logits[0]
    pred_boxes = outputs.pred_boxes[0]
    scores    = logits.sigmoid().max(dim=-1).values
    mask      = scores > threshold
    filt_scores = scores[mask].numpy()
    filt_boxes  = pred_boxes[mask].numpy()

    if len(filt_boxes) == 0:
        return 0, image.copy()

    cx, cy = filt_boxes[:, 0], filt_boxes[:, 1]
    bw_, bh_ = filt_boxes[:, 2], filt_boxes[:, 3]
    boxes = np.stack([
        (cx - bw_/2)*w, (cy - bh_/2)*h,
        (cx + bw_/2)*w, (cy + bh_/2)*h,
    ], axis=1)

    order = np.argsort(-filt_scores)
    kept_boxes = []
    used = np.zeros(len(order), dtype=bool)
    for i in order:
        if used[i]:
            continue
        kept_boxes.append(boxes[i])
        for j in order:
            if not used[j] and j != i and _adv._iou(boxes[i], boxes[j]) > 0.35:
                used[j] = True

    ann = image.copy()
    for idx, (x1, y1, x2, y2) in enumerate(kept_boxes):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(ann, (x1, y1), (x2, y2), (50, 160, 255), 2)
        cv2.putText(ann, str(idx + 1), (x1 + 4, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 160, 255), 2)
    return len(kept_boxes), ann


# ─────────────────────────────────────────────
# CLIP_Regressor (Tier 3 melhorado)
# ─────────────────────────────────────────────
def method_clip_regressor(image: np.ndarray) -> tuple[int, np.ndarray]:
    """
    CLIP + Regressão Ridge treinada em 300 imagens sintéticas de parafusos.
    Feature: vision_model pooler_output (768-dim).
    Treino sintético: crops reais colados em fundos uniformes, 1-12 parafusos.
    MAE treino sintético: 0.08 — avalia generalização para domínio real.
    """
    import pickle
    import torch
    from PIL import Image as PILImage
    from transformers import CLIPModel, CLIPProcessor

    global _clip_reg
    reg_path = MODELS_DIR / "clip_regressor.pkl"
    if _clip_reg is None and reg_path.exists():
        with open(reg_path, "rb") as f:
            _clip_reg = pickle.load(f)

    if _clip_reg is None:
        return method_clip(image)  # fallback

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = PILImage.fromarray(img_rgb)

    proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    inp = proc(images=[pil_img], return_tensors="pt")
    with torch.no_grad():
        feat = model.vision_model(pixel_values=inp["pixel_values"]).pooler_output
    feat_np = feat.detach().cpu().numpy()

    count_f = float(_clip_reg.predict(feat_np)[0])
    count   = max(0, int(round(count_f)))

    ann = image.copy()
    h, w = ann.shape[:2]
    cv2.putText(ann, f"CLIP-Reg: {count} screws (raw={count_f:.1f})",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 200, 50), 2)
    return count, ann


# ─────────────────────────────────────────────
# MobileNet_FT (Tier 2/3 melhorado)
# ─────────────────────────────────────────────
def method_mobilenet_ft(image: np.ndarray) -> tuple[int, np.ndarray]:
    """
    MobileNetV2 + cabeça de regressão treinada em imagens sintéticas
    de parafusos (500 imagens, crops colados em fundo azul).
    Congela backbone ImageNet; treina apenas camadas lineares.
    Diferencial: alinha a distribuição de features para o domínio de contagem.
    """
    import torch
    import torch.nn as nn
    import torchvision.models as tvm
    import torchvision.transforms as T

    global _mobilenet_reg
    reg_path = MODELS_DIR / "mobilenet_regressor.pth"

    if _mobilenet_reg is None and reg_path.exists():
        # Tenta arquitetura nova (256→64→1) primeiro, depois fallback para antiga (128→1)
        for head_arch in [
            nn.Sequential(nn.Linear(1000,256),nn.ReLU(),nn.Dropout(0.3),
                           nn.Linear(256,64),nn.ReLU(),nn.Dropout(0.2),nn.Linear(64,1)),
            nn.Sequential(nn.Linear(1000,128),nn.ReLU(),nn.Dropout(0.3),nn.Linear(128,1)),
        ]:
            try:
                head_arch.load_state_dict(torch.load(reg_path, map_location="cpu"))
                head_arch.eval()
                _mobilenet_reg = head_arch
                break
            except Exception:
                continue

    if _mobilenet_reg is None:
        return method_mobilenet_salience(image)  # fallback

    backbone = tvm.mobilenet_v2(weights=tvm.MobileNet_V2_Weights.IMAGENET1K_V1)
    backbone.eval()

    transform = T.Compose([
        T.ToPILImage(), T.Resize((224, 224)), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    inp     = transform(img_rgb).unsqueeze(0)

    with torch.no_grad():
        feats  = backbone(inp)
        pred   = _mobilenet_reg(feats)

    count_f = float(pred[0, 0])
    count   = max(0, int(round(count_f)))

    heatmap = cv2.applyColorMap(
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET
    )
    ann = cv2.addWeighted(image.copy(), 0.6, heatmap, 0.4, 0)
    cv2.putText(ann, f"MobileNet-FT: {count} (raw={count_f:.1f})",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 0), 2)
    return count, ann


# ─────────────────────────────────────────────
# FasterRCNN_FT (Tier 3 — pós fine-tune)
# ─────────────────────────────────────────────
def method_fasterrcnn_ft(image: np.ndarray) -> tuple[int, np.ndarray]:
    """
    Faster R-CNN fine-tuned no dataset Cross-Recessed-Screw
    (600 imagens, 15 épocas, backbone layer4+FPN descongelado).
    Classe 2 = 'screw'. Filtra por score > 0.4 e área relativa.
    Disponível após rodar: python3 benchmark/finetune.py --step fasterrcnn
    """
    import torch
    import torchvision.models.detection as det
    import torchvision.transforms.functional as TF

    global _fasterrcnn_ft
    ft_path = MODELS_DIR / "fasterrcnn_screw.pth"

    if _fasterrcnn_ft is None and ft_path.exists():
        model = det.fasterrcnn_resnet50_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = det.faster_rcnn.FastRCNNPredictor(in_features, 2)
        model.load_state_dict(torch.load(ft_path, map_location="cpu"))
        model.eval()
        _fasterrcnn_ft = model

    if _fasterrcnn_ft is None:
        # Modelo ainda treinando
        ann = image.copy()
        cv2.putText(ann, "FasterRCNN-FT: treinando...", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        return 0, ann

    h, w = image.shape[:2]
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor  = TF.to_tensor(img_rgb)

    with torch.no_grad():
        preds = _fasterrcnn_ft([tensor])[0]

    boxes  = preds["boxes"].numpy()
    scores = preds["scores"].numpy()
    labels = preds["labels"].numpy()

    min_area = h * w * 0.002
    max_area = h * w * 0.25

    valid = []
    for box, score, label in zip(boxes, scores, labels):
        if score < 0.40 or label != 1:
            continue
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        if min_area <= area <= max_area:
            valid.append((int(x1), int(y1), int(x2), int(y2), float(score)))

    ann = image.copy()
    for i, (x1, y1, x2, y2, sc) in enumerate(valid):
        cv2.rectangle(ann, (x1, y1), (x2, y2), (0, 200, 80), 2)
        cv2.putText(ann, f"{i+1} ({sc:.2f})", (x1, max(0, y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 80), 2)
    return len(valid), ann


# ─────────────────────────────────────────────
# Runner + Benchmark
# ─────────────────────────────────────────────

# tier: 1=Embedded, 2=CPU/Notebook, 3=GPU/Server, 4=Cloud
METHODS: dict[str, tuple] = {
    # Tier 1 — Embedded / Smartphone
    "MSER":           (method_mser,              1),
    "Shi-Tomasi":     (method_shi_tomasi,         1),
    "ConvexDefects":  (method_convex_defects,     1),
    "SimpleBlobDet":  (method_blob_detector,      1),
    "HoughCircles":   (method_hough_circles,      1),
    "ColorBG_Sub":    (method_color_bg,           1),
    "Canny_Contour":  (method_canny,              1),
    "Watershed":      (method_watershed,          1),
    "WS_RegionalMax": (method_ws_regional_max,    1),
    # Tier 2 — CPU / Notebook
    "NCC_Template":   (method_ncc_template,       2),
    "SIFT_Template":  (method_sift,               2),
    "LoG_Blobs":      (method_log_blobs,          2),
    "GrabCut":        (method_grabcut,            2),
    "MobileNet_CNN":  (method_mobilenet_salience, 2),
    # Tier 3 — GPU / Server (versões originais zero-shot)
    "FasterRCNN":          (method_fasterrcnn,          3),
    "OwlViT":              (method_owlvit,              3),
    "CLIP":                (method_clip,                3),
    # Tier 2 melhorado — CPU com fine-tuning
    "NCC_Multi":           (method_ncc_multi,           2),
    "LoG_Calibrado":       (method_log_calibrated,      2),
    "MobileNet_FT":        (method_mobilenet_ft,        2),
    # Tier 3 melhorado — GPU com fine-tuning / calibração
    "OwlViT_Adapted":      (method_owlvit_adapted,      3),
    "CLIP_Regressor":      (method_clip_regressor,      3),
    "FasterRCNN_FT":       (method_fasterrcnn_ft,       3),
    # Tier 4 — Cloud API
    "VLM_Claude":          (method_vlm_claude,          4),
}

ENSEMBLE_MEMBERS = ["Watershed", "WS_RegionalMax", "ConvexDefects", "ColorBG_Sub"]

TIER_LABELS = {1: "Embedded/Smartphone", 2: "CPU/Notebook", 3: "GPU/Server", 4: "Cloud API"}


def run_benchmark(skip_tiers: set[int] | None = None):
    skip_tiers = skip_tiers or set()
    images = sorted(IMAGES_DIR.glob("img*.jpg"))
    output_dir = BENCHMARK_DIR / "output_benchmark"
    output_dir.mkdir(exist_ok=True)

    results: list[BenchResult] = []

    for method_name, (method_fn, tier) in METHODS.items():
        if tier in skip_tiers:
            continue
        print(f"  [T{tier}] → {method_name}...", flush=True)
        for img_path in images:
            image = cv2.imread(str(img_path))
            gt = GROUND_TRUTH.get(img_path.name, -1)

            proc = psutil.Process(os.getpid())
            mem_before = proc.memory_info().rss / 1024
            t0 = time.perf_counter()
            try:
                count, annotated = method_fn(image.copy())
            except Exception as e:
                count, annotated = -1, image.copy()
                print(f"    [ERRO] {img_path.name}: {e}")
            elapsed_ms = (time.perf_counter() - t0) * 1000
            mem_kb = max(0, proc.memory_info().rss / 1024 - mem_before)

            results.append(BenchResult(
                method=method_name,
                image=img_path.name,
                predicted=count,
                ground_truth=gt,
                time_ms=elapsed_ms,
                memory_kb=mem_kb,
                annotated=annotated,
            ))

            out_name = f"{method_name.replace(' ', '_')}_{img_path.name}"
            ann_save = annotated
            if ann_save.shape[1] > 1300:
                scale = 1300 / ann_save.shape[1]
                ann_save = cv2.resize(ann_save, None, fx=scale, fy=scale)
            cv2.imwrite(str(output_dir / out_name), ann_save)

    # Ensemble Tier 1 (sem Cloud)
    print("  [T1] → Ensemble...", flush=True)
    for img_path in images:
        gt = GROUND_TRUTH.get(img_path.name, -1)
        image = cv2.imread(str(img_path))
        member_counts = []
        for m in ENSEMBLE_MEMBERS:
            r = next((r for r in results if r.method == m and r.image == img_path.name), None)
            if r and r.predicted >= 0:
                member_counts.append(r.predicted)
        count = int(round(float(np.median(member_counts)))) if member_counts else 0
        ann = image.copy()
        label = " | ".join(f"{m.split('_')[0]}={c}" for m, c in zip(ENSEMBLE_MEMBERS, member_counts))
        cv2.putText(ann, f"Ensemble={count} ({label})", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        results.append(BenchResult(
            method="Ensemble", image=img_path.name, predicted=count,
            ground_truth=gt, time_ms=0, memory_kb=0, annotated=ann,
        ))
        cv2.imwrite(str(output_dir / f"Ensemble_{img_path.name}"), ann)

    _print_report(results)
    return results


def _print_report(results: list[BenchResult]):
    print("\n" + "=" * 80)
    print("BENCHMARK — CONTAGEM DE PARAFUSOS (por tier)")
    print("=" * 80)

    method_to_tier = {name: tier for name, (_, tier) in METHODS.items()}
    method_to_tier["Ensemble"] = 1

    for tier in sorted(TIER_LABELS):
        tier_methods = [name for name, t in method_to_tier.items() if t == tier]
        tier_results = [r for r in results if r.method in tier_methods]
        if not tier_results:
            continue
        print(f"\n── Tier {tier}: {TIER_LABELS[tier]} ──")
        print(f"{'Método':<18} {'Acertos':>8} {'MAE':>6} {'ms/img':>8} {'MB RAM':>8}")
        print("-" * 54)
        for method_name in tier_methods:
            m_r = [r for r in tier_results if r.method == method_name]
            valid = [r for r in m_r if r.predicted >= 0]
            if not valid:
                print(f"{method_name:<18} {'n/a':>8} {'—':>6} {'—':>8} {'—':>8}")
                continue
            acertos = sum(r.correct for r in valid)
            mae = float(np.mean([r.error for r in valid]))
            avg_ms = float(np.mean([r.time_ms for r in valid]))
            avg_mb = float(np.mean([r.memory_kb for r in valid])) / 1024
            star = " ★" if acertos == len(valid) else ""
            print(f"{method_name:<18} {acertos:>5}/{len(valid)} {mae:>6.2f} {avg_ms:>8.1f} {avg_mb:>8.1f}{star}")

    print(f"\nImagens anotadas: {BENCHMARK_DIR / 'output_benchmark'}")


if __name__ == "__main__":
    run_benchmark()
