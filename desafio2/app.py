"""
Desafio 2 — Crack Detector Web App
FastAPI servindo detecção de trincas em paredes.
Mesmo padrão do Desafio 1: 3 abas — análise, comparação, catálogo.
"""

from __future__ import annotations

import base64
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

BASE = Path(__file__).parent

app = FastAPI(title="Crack Detector — CDIA")
app.mount("/static", StaticFiles(directory=str(BASE / "static")), name="static")
_HTML = (BASE / "templates" / "index.html").read_text()

HARDWARE = {
    "Canny+Morph": {
        "tier": 1, "tier_label": "CV Tradicional", "icon": "📱",
        "ram": "~15 MB", "disk": "0 MB", "gpu": False, "offline": True,
        "iou": "0.161", "det_rate": "96.7%", "time_ms": 309,
        "description": "Canny adaptativo + dilatação morfológica. Melhor método tradicional (IoU 0.161).",
    },
    "Gabor Filter": {
        "tier": 1, "tier_label": "CV Tradicional", "icon": "📱",
        "ram": "~30 MB", "disk": "0 MB", "gpu": False, "offline": True,
        "iou": "0.009", "det_rate": "100%", "time_ms": 468,
        "description": "Bank de filtros Gabor em 4 orientações. Alta detecção, baixo IoU (0.009).",
    },
    "Adaptive Thresh": {
        "tier": 1, "tier_label": "CV Tradicional", "icon": "📱",
        "ram": "~15 MB", "disk": "0 MB", "gpu": False, "offline": True,
        "iou": "0.117", "det_rate": "96.7%", "time_ms": 830,
        "description": "Threshold adaptativo gaussiano. Lento em imagens grandes.",
    },
    "Sobel+Otsu": {
        "tier": 1, "tier_label": "CV Tradicional", "icon": "📱",
        "ram": "~15 MB", "disk": "0 MB", "gpu": False, "offline": True,
        "iou": "0.107", "det_rate": "100%", "time_ms": 192,
        "description": "Gradiente Sobel + threshold Otsu. Detecta bordas abruptas, baixa precisão.",
    },
    "YOLOv8n-seg": {
        "tier": 2, "tier_label": "CPU (Notebook)", "icon": "💻",
        "ram": "~300 MB", "disk": "6 MB", "gpu": False, "offline": True,
        "iou": "—", "det_rate": "—", "time_ms": 0,
        "description": "YOLOv8n-seg fine-tuned em 1241 imagens de trincas. Melhor método.",
    },
    "VLM_Claude": {
        "tier": 4, "tier_label": "Cloud API", "icon": "☁️",
        "ram": "0 MB", "disk": "0 MB", "gpu": False, "offline": False,
        "iou": "N/A", "det_rate": "95%*", "time_ms": 1500,
        "description": "Claude via API. Detecção zero-shot, sem bbox preciso.",
    },
}

TIER_RUNNABLE = {1, 2}

_methods_cache: dict | None = None


def _get_methods() -> dict:
    global _methods_cache
    if _methods_cache is not None:
        return _methods_cache

    from benchmark import (
        method_canny_morph,
        method_gabor,
        method_adaptive_thresh,
        method_sobel,
        method_yolov8,
    )
    from solution import _load_yolo

    yolo = _load_yolo()

    def _yolo_wrapper(image):
        return method_yolov8(image, yolo)

    _methods_cache = {
        "Canny+Morph": method_canny_morph,
        "Gabor Filter": method_gabor,
        "Adaptive Thresh": method_adaptive_thresh,
        "Sobel+Otsu": method_sobel,
        "YOLOv8n-seg": _yolo_wrapper,
    }
    return _methods_cache


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(_HTML)


@app.get("/api/hardware")
async def hardware_endpoint():
    return JSONResponse(HARDWARE)


@app.get("/api/model_status")
async def model_status():
    weights = BASE / "weights" / "best.pt"
    return JSONResponse({
        "trained": weights.exists(),
        "path": str(weights) if weights.exists() else None,
    })


@app.post("/detect")
async def detect_endpoint(
    file: UploadFile = File(...),
    method: str = Form("YOLOv8n-seg"),
):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        return JSONResponse({"error": "Imagem inválida"}, status_code=400)

    methods = _get_methods()
    fn = methods.get(method)
    if fn is None:
        return JSONResponse({"error": f"Método '{method}' não disponível"}, status_code=400)

    t0 = time.perf_counter()
    try:
        pred_mask, annotated = fn(image.copy())
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Conta regiões conectadas como "trincas"
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(pred_mask, connectivity=8)
    h, w = image.shape[:2]
    min_area = h * w * 0.0001
    n_cracks = sum(1 for i in range(1, n_labels) if stats[i, cv2.CC_STAT_AREA] >= min_area)

    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
    img_b64 = base64.b64encode(buf.tobytes()).decode()

    return JSONResponse({
        "cracks_found": n_cracks,
        "method": method,
        "elapsed_ms": round(elapsed_ms, 1),
        "annotated_image": f"data:image/jpeg;base64,{img_b64}",
        "hardware": HARDWARE.get(method, {}),
    })


@app.post("/benchmark")
async def benchmark_endpoint(file: UploadFile = File(...)):
    """Roda todos os métodos Tier 1 e 2 simultaneamente."""
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        return JSONResponse({"error": "Imagem inválida"}, status_code=400)

    methods = _get_methods()
    runnable = {k for k, v in HARDWARE.items() if v["tier"] in TIER_RUNNABLE}
    results = []

    for name in runnable:
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

        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_b64 = base64.b64encode(buf.tobytes()).decode()

        results.append({
            "method": name,
            "cracks_found": n_cracks,
            "elapsed_ms": round(elapsed_ms, 1),
            "annotated_image": f"data:image/jpeg;base64,{img_b64}",
            "hardware": HARDWARE[name],
        })

    return JSONResponse({"results": results})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8081, reload=True)
