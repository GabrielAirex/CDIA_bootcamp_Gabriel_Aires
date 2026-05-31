"""
CDIA Bootcamp — App Unificado
Desafio 1 (Parafusos) + Desafio 2 (Trincas) em uma única interface.
Seletor de desafio no frontend; endpoints separados por prefixo /d1 /d2.
"""

from __future__ import annotations

import base64
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import importlib.util

ROOT = Path(__file__).parent
D1 = ROOT / "desafio1"
D2 = ROOT / "desafio2"

app = FastAPI(title="CDIA — Desafios 1 & 2")


def _load_module(file: Path, name: str):
    """Carrega um módulo Python de um path arbitrário com nome único."""
    old_cwd = Path.cwd()
    import os
    os.chdir(file.parent)
    try:
        spec = importlib.util.spec_from_file_location(name, file)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    finally:
        os.chdir(old_cwd)
    return mod

app.mount("/d1/static", StaticFiles(directory=str(D1 / "static")), name="d1_static")
app.mount("/d2/static", StaticFiles(directory=str(D2 / "static")), name="d2_static")
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")

_HTML = (ROOT / "templates" / "index.html").read_text()

# ── Hardware profiles ──────────────────────────────────────────────────────────
HW_D1 = {
    "WS_RegionalMax": {
        "tier": 1, "tier_label": "Smartphone", "icon": "📱",
        "ram": "~50 MB", "disk": "0 MB", "gpu": False, "offline": True,
        "accuracy": "5/5", "time_ms": 8,
        "description": "Watershed + Regional Maxima (scipy). Melhor método geral.",
    },
    "Watershed": {
        "tier": 1, "tier_label": "Smartphone", "icon": "📱",
        "ram": "~50 MB", "disk": "0 MB", "gpu": False, "offline": True,
        "accuracy": "4/5", "time_ms": 12,
        "description": "Watershed clássico com threshold global. Robusto e rápido.",
    },
    "ConvexDefects": {
        "tier": 1, "tier_label": "Smartphone", "icon": "📱",
        "ram": "~10 MB", "disk": "0 MB", "gpu": False, "offline": True,
        "accuracy": "3/5", "time_ms": 9, "mae": 0.60,
        "description": "Concavidades do hull convexo. Ultraleve, sem dependências.",
    },
    "MSER": {
        "tier": 1, "tier_label": "Smartphone", "icon": "📱",
        "ram": "~30 MB", "disk": "0 MB", "gpu": False, "offline": True,
        "accuracy": "2/5", "time_ms": 19, "mae": 0.80,
        "description": "Maximally Stable Extremal Regions. Funde parafusos sobrepostos.",
    },
    "ColorBG_Sub": {
        "tier": 1, "tier_label": "Smartphone", "icon": "📱",
        "ram": "~50 MB", "disk": "0 MB", "gpu": False, "offline": True,
        "accuracy": "2/5", "time_ms": 11, "mae": 1.60,
        "description": "Subtração de background por cor.",
    },
    "HoughCircles": {
        "tier": 1, "tier_label": "Smartphone", "icon": "📱",
        "ram": "~20 MB", "disk": "0 MB", "gpu": False, "offline": True,
        "accuracy": "1/5", "time_ms": 7, "mae": 8.60,
        "description": "Transformada de Hough para círculos. Super-detecta.",
    },
    "SimpleBlobDet": {
        "tier": 1, "tier_label": "Smartphone", "icon": "📱",
        "ram": "~20 MB", "disk": "0 MB", "gpu": False, "offline": True,
        "accuracy": "1/5", "time_ms": 11, "mae": 2.60,
        "description": "Detector de blobs por área/convexidade.",
    },
    "Canny_Contour": {
        "tier": 1, "tier_label": "Smartphone", "icon": "📱",
        "ram": "~5 MB", "disk": "0 MB", "gpu": False, "offline": True,
        "accuracy": "0/5", "time_ms": 2, "mae": 4.00,
        "description": "Canny + contornos. Fragmenta bordas, sem contagem coerente.",
    },
    "Shi-Tomasi": {
        "tier": 1, "tier_label": "Smartphone", "icon": "📱",
        "ram": "~10 MB", "disk": "0 MB", "gpu": False, "offline": True,
        "accuracy": "0/5", "time_ms": 15, "mae": 8.80,
        "description": "Shi-Tomasi corners + clustering. Cantos se multiplicam em pilha.",
    },
    "NCC_Template": {
        "tier": 2, "tier_label": "CPU (Notebook)", "icon": "💻",
        "ram": "~80 MB", "disk": "0 MB", "gpu": False, "offline": True,
        "accuracy": "1/5", "time_ms": 200, "mae": 3.40,
        "description": "NCC multi-escala com img2 como template.",
    },
    "SIFT_Template": {
        "tier": 2, "tier_label": "CPU (Notebook)", "icon": "💻",
        "ram": "~150 MB", "disk": "0 MB", "gpu": False, "offline": True,
        "accuracy": "0/5", "time_ms": 76, "mae": 4.80,
        "description": "SIFT template matching. Keypoints não correspondem cross-context.",
    },
    "LoG_Blobs": {
        "tier": 2, "tier_label": "CPU (Notebook)", "icon": "💻",
        "ram": "~100 MB", "disk": "0 MB", "gpu": False, "offline": True,
        "accuracy": "0/5", "time_ms": 1928, "mae": 5.00,
        "description": "LoG scale-space (scipy). Lento, sobre-detecta.",
    },
    "GrabCut": {
        "tier": 2, "tier_label": "CPU (Notebook)", "icon": "💻",
        "ram": "~50 MB", "disk": "0 MB", "gpu": False, "offline": True,
        "accuracy": "0/5", "time_ms": 5272, "mae": 4.60,
        "description": "GrabCut GMM iterativo. Segmenta blob único, não conta.",
    },
    "MobileNet_CNN": {
        "tier": 2, "tier_label": "CPU (Notebook)", "icon": "💻",
        "ram": "~200 MB", "disk": "14 MB", "gpu": False, "offline": True,
        "accuracy": "0/5", "time_ms": 302, "mae": 6.20,
        "description": "MobileNetV2 saliency (ImageNet). Ativa em texturas erradas.",
    },
    "FasterRCNN": {
        "tier": 3, "tier_label": "GPU / Servidor", "icon": "🖥️",
        "ram": "~1500 MB", "disk": "160 MB", "gpu": True, "offline": True,
        "accuracy": "0/5", "time_ms": 1187, "mae": 4.80,
        "description": "Faster R-CNN COCO. Sem classe 'screw' — zero detecções.",
    },
    "OwlViT": {
        "tier": 3, "tier_label": "GPU / Servidor", "icon": "🖥️",
        "ram": "~2000 MB", "disk": "340 MB", "gpu": True, "offline": True,
        "accuracy": "0/5", "time_ms": 1996, "mae": 2.40,
        "description": "OwlViT zero-shot por prompt textual. Detecta, conta impreciso.",
    },
    "CLIP": {
        "tier": 3, "tier_label": "GPU / Servidor", "icon": "🖥️",
        "ram": "~1000 MB", "disk": "300 MB", "gpu": True, "offline": True,
        "accuracy": "2/5", "time_ms": 1110, "mae": 3.20,
        "description": "CLIP semântico holístico. Sem bboxes, funciona até 5 screws.",
    },
    "VLM_Claude": {
        "tier": 4, "tier_label": "Cloud API", "icon": "☁️",
        "ram": "0 MB", "disk": "0 MB", "gpu": False, "offline": False,
        "accuracy": "5/5", "time_ms": 1800, "mae": 0.00,
        "description": "Claude Opus via API. Empata com WS_RegionalMax. Requer internet.",
    },
}

HW_D2 = {
    "Canny+Morph": {
        "tier": 1, "tier_label": "CV Tradicional", "icon": "📱",
        "ram": "~15 MB", "disk": "0 MB", "gpu": False, "offline": True,
        "iou": "0.161", "det_rate": "96.7%", "time_ms": 309,
        "description": "Canny adaptativo + morfologia. Melhor método tradicional (IoU 0.161).",
    },
    "Gabor Filter": {
        "tier": 1, "tier_label": "CV Tradicional", "icon": "📱",
        "ram": "~30 MB", "disk": "0 MB", "gpu": False, "offline": True,
        "iou": "0.009", "det_rate": "100%", "time_ms": 468,
        "description": "Filtros Gabor em 4 orientações. Detecta texturas lineares.",
    },
    "Adaptive Thresh": {
        "tier": 1, "tier_label": "CV Tradicional", "icon": "📱",
        "ram": "~15 MB", "disk": "0 MB", "gpu": False, "offline": True,
        "iou": "0.117", "det_rate": "96.7%", "time_ms": 830,
        "description": "Threshold adaptativo gaussiano.",
    },
    "Sobel+Otsu": {
        "tier": 1, "tier_label": "CV Tradicional", "icon": "📱",
        "ram": "~15 MB", "disk": "0 MB", "gpu": False, "offline": True,
        "iou": "0.107", "det_rate": "100%", "time_ms": 192,
        "description": "Gradiente Sobel + threshold Otsu.",
    },
    "YOLOv8n-seg": {
        "tier": 2, "tier_label": "Modelo Fine-tuned", "icon": "💻",
        "ram": "~300 MB", "disk": "26 MB", "gpu": False, "offline": True,
        "iou": "0.510", "det_rate": "—", "time_ms": 41,
        "description": "YOLOv8n-seg fine-tuned. mAP50(seg)=0.510 · 3.2× melhor que Canny. Mostra onde está a fissura (bbox + máscara).",
    },
    "VLM_Claude": {
        "tier": 4, "tier_label": "Cloud API", "icon": "☁️",
        "ram": "0 MB", "disk": "0 MB", "gpu": False, "offline": False,
        "iou": "N/A", "det_rate": "~95%", "time_ms": 1500,
        "description": "Claude via API. Detecção zero-shot, sem bbox preciso.",
    },
}

# Compare rápido: Tier 1 + Tier 2 sem os lentos (GrabCut ~5s, LoG ~1.9s)
_D1_SLOW = {"GrabCut", "LoG_Blobs"}
D1_RUNNABLE = {k for k, v in HW_D1.items() if v["tier"] in {1, 2} and k not in _D1_SLOW}
D2_RUNNABLE = {k for k, v in HW_D2.items() if v["tier"] in {1, 2}}

# ── Lazy method loaders ────────────────────────────────────────────────────────
_d1_methods: dict | None = None
_d2_methods: dict | None = None


def get_d1_methods() -> dict:
    global _d1_methods
    if _d1_methods is None:
        b = _load_module(D1 / "benchmark" / "benchmark.py", "d1_benchmark")
        _d1_methods = {
            "WS_RegionalMax": b.method_ws_regional_max,
            "Watershed":      b.method_watershed,
            "ConvexDefects":  b.method_convex_defects,
            "MSER":           b.method_mser,
            "Shi-Tomasi":     b.method_shi_tomasi,
            "ColorBG_Sub":    b.method_color_bg,
            "HoughCircles":   b.method_hough_circles,
            "SimpleBlobDet":  b.method_blob_detector,
            "Canny_Contour":  b.method_canny,
            "NCC_Template":   b.method_ncc_template,
            "SIFT_Template":  b.method_sift,
            "LoG_Blobs":      b.method_log_blobs,
            "GrabCut":        b.method_grabcut,
            "MobileNet_CNN":  b.method_mobilenet_salience,
            "FasterRCNN":     b.method_fasterrcnn,
            "OwlViT":         b.method_owlvit,
            "CLIP":           b.method_clip,
            "VLM_Claude":     b.method_vlm_claude,
        }
    return _d1_methods


def get_d2_methods() -> dict:
    global _d2_methods
    if _d2_methods is None:
        b = _load_module(D2 / "benchmark" / "benchmark.py", "d2_benchmark")
        sol = _load_module(D2 / "solution.py", "d2_solution")
        yolo = sol._load_yolo()

        def _yolo_wrapper(image):
            return b.method_yolov8(image, yolo)

        _d2_methods = {
            "Canny+Morph":     b.method_canny_morph,
            "Gabor Filter":    b.method_gabor,
            "Adaptive Thresh": b.method_adaptive_thresh,
            "Sobel+Otsu":      b.method_sobel,
            "YOLOv8n-seg":     _yolo_wrapper,
        }
    return _d2_methods


# ── Shared helpers ─────────────────────────────────────────────────────────────
def _decode_image(contents: bytes) -> np.ndarray | None:
    np_arr = np.frombuffer(contents, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def _encode_image(image: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(_HTML)


@app.get("/d1/hardware")
async def d1_hardware():
    return JSONResponse(HW_D1)


@app.get("/d2/hardware")
async def d2_hardware():
    return JSONResponse(HW_D2)


@app.get("/d2/model_status")
async def d2_model_status():
    weights = D2 / "weights" / "best.pt"
    return JSONResponse({"trained": weights.exists()})


# ── D1: Contagem de Parafusos ──────────────────────────────────────────────────
@app.post("/d1/count")
async def d1_count(
    file: UploadFile = File(...),
    method: str = Form("WS_RegionalMax"),
):
    image = _decode_image(await file.read())
    if image is None:
        return JSONResponse({"error": "Imagem inválida"}, status_code=400)

    methods = get_d1_methods()
    fn = methods.get(method)
    if fn is None:
        return JSONResponse({"error": f"Método '{method}' não disponível"}, status_code=400)

    t0 = time.perf_counter()
    try:
        count, annotated = fn(image.copy())
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return JSONResponse({
        "count": int(count) if count >= 0 else None,
        "method": method,
        "elapsed_ms": round(elapsed_ms, 1),
        "annotated_image": _encode_image(annotated),
        "hardware": HW_D1.get(method, {}),
    })


@app.post("/d1/benchmark")
async def d1_benchmark(file: UploadFile = File(...)):
    image = _decode_image(await file.read())
    if image is None:
        return JSONResponse({"error": "Imagem inválida"}, status_code=400)

    methods = get_d1_methods()
    results = []
    for name in sorted(D1_RUNNABLE, key=lambda k: (HW_D1[k]["tier"], HW_D1[k]["time_ms"])):
        fn = methods.get(name)
        if fn is None:
            continue
        t0 = time.perf_counter()
        try:
            count, annotated = fn(image.copy())
        except Exception:
            continue
        elapsed_ms = (time.perf_counter() - t0) * 1000
        results.append({
            "method": name,
            "count": int(count) if count >= 0 else None,
            "elapsed_ms": round(elapsed_ms, 1),
            "annotated_image": _encode_image(annotated),
            "hardware": HW_D1[name],
        })
    return JSONResponse({"results": results})


@app.post("/d1/benchmark_gpu")
async def d1_benchmark_gpu(file: UploadFile = File(...)):
    """Roda os métodos Tier 3 (GPU) individualmente — pode levar 5–15s."""
    image = _decode_image(await file.read())
    if image is None:
        return JSONResponse({"error": "Imagem inválida"}, status_code=400)

    methods = get_d1_methods()
    gpu_methods = [k for k, v in HW_D1.items() if v["tier"] == 3]
    results = []
    for name in gpu_methods:
        fn = methods.get(name)
        if fn is None:
            continue
        t0 = time.perf_counter()
        try:
            count, annotated = fn(image.copy())
        except Exception as e:
            results.append({"method": name, "count": None, "elapsed_ms": 0,
                            "error": str(e), "hardware": HW_D1[name]})
            continue
        elapsed_ms = (time.perf_counter() - t0) * 1000
        results.append({
            "method": name,
            "count": int(count) if count >= 0 else None,
            "elapsed_ms": round(elapsed_ms, 1),
            "annotated_image": _encode_image(annotated),
            "hardware": HW_D1[name],
        })
    return JSONResponse({"results": results})


# ── D2: Detecção de Trincas ────────────────────────────────────────────────────
@app.post("/d2/detect")
async def d2_detect(
    file: UploadFile = File(...),
    method: str = Form("YOLOv8n-seg"),
):
    image = _decode_image(await file.read())
    if image is None:
        return JSONResponse({"error": "Imagem inválida"}, status_code=400)

    methods = get_d2_methods()
    fn = methods.get(method)
    if fn is None:
        return JSONResponse({"error": f"Método '{method}' não disponível"}, status_code=400)

    t0 = time.perf_counter()
    try:
        pred_mask, annotated = fn(image.copy())
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    h, w = image.shape[:2]
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(pred_mask, connectivity=8)
    min_area = h * w * 0.0001
    n_cracks = sum(1 for i in range(1, n_labels) if stats[i, cv2.CC_STAT_AREA] >= min_area)

    return JSONResponse({
        "cracks_found": n_cracks,
        "method": method,
        "elapsed_ms": round(elapsed_ms, 1),
        "annotated_image": _encode_image(annotated),
        "hardware": HW_D2.get(method, {}),
    })


@app.post("/d2/benchmark")
async def d2_benchmark(file: UploadFile = File(...)):
    image = _decode_image(await file.read())
    if image is None:
        return JSONResponse({"error": "Imagem inválida"}, status_code=400)

    methods = get_d2_methods()
    results = []
    for name in D2_RUNNABLE:
        fn = methods.get(name)
        if fn is None:
            continue
        t0 = time.perf_counter()
        try:
            pred_mask, annotated = fn(image.copy())
        except Exception:
            continue
        elapsed_ms = (time.perf_counter() - t0) * 1000

        h, w = image.shape[:2]
        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(pred_mask, connectivity=8)
        min_area = h * w * 0.0001
        n_cracks = sum(1 for i in range(1, n_labels) if stats[i, cv2.CC_STAT_AREA] >= min_area)

        results.append({
            "method": name,
            "cracks_found": n_cracks,
            "elapsed_ms": round(elapsed_ms, 1),
            "annotated_image": _encode_image(annotated),
            "hardware": HW_D2[name],
        })
    return JSONResponse({"results": results})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
