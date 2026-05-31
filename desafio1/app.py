"""
Desafio 1 — Screw Counter Web App
FastAPI backend servindo a aplicação de contagem de parafusos.
"""

from __future__ import annotations

import base64
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

BASE = Path(__file__).parent

app = FastAPI(title="Screw Counter — CDIA")
app.mount("/static", StaticFiles(directory=str(BASE / "static")), name="static")
_HTML = (BASE / "templates" / "index.html").read_text()

# ── Hardware profiles ──────────────────────────────────────────────────────────
HARDWARE = {
    "WS_RegionalMax": {
        "tier": 1, "tier_label": "Smartphone", "icon": "📱",
        "ram": "~50 MB", "disk": "0 MB", "gpu": False, "offline": True,
        "accuracy": "5/5", "time_ms": 8, "stars": 5,
        "description": "Watershed com Regional Maxima (scipy). Melhor método geral.",
    },
    "ConvexDefects": {
        "tier": 1, "tier_label": "Smartphone", "icon": "📱",
        "ram": "~10 MB", "disk": "0 MB", "gpu": False, "offline": True,
        "accuracy": "3/5", "time_ms": 3, "stars": 3,
        "description": "Concavidades do hull convexo. Ultraleve, sem dependências.",
    },
    "Watershed": {
        "tier": 1, "tier_label": "Smartphone", "icon": "📱",
        "ram": "~50 MB", "disk": "0 MB", "gpu": False, "offline": True,
        "accuracy": "4/5", "time_ms": 12, "stars": 4,
        "description": "Watershed clássico com threshold global. Robusto e rápido.",
    },
    "ColorBG_Sub": {
        "tier": 1, "tier_label": "Smartphone", "icon": "📱",
        "ram": "~50 MB", "disk": "0 MB", "gpu": False, "offline": True,
        "accuracy": "2/5", "time_ms": 12, "stars": 2,
        "description": "Subtração de background por cor. Funciona melhor em fundos uniformes.",
    },
    "HoughCircles": {
        "tier": 1, "tier_label": "Smartphone", "icon": "📱",
        "ram": "~20 MB", "disk": "0 MB", "gpu": False, "offline": True,
        "accuracy": "1/5", "time_ms": 8, "stars": 1,
        "description": "Transformada de Hough para círculos. Falha em parafusos não-circulares.",
    },
    "SimpleBlobDet": {
        "tier": 1, "tier_label": "Smartphone", "icon": "📱",
        "ram": "~20 MB", "disk": "0 MB", "gpu": False, "offline": True,
        "accuracy": "1/5", "time_ms": 5, "stars": 1,
        "description": "Detector de blobs por área/convexidade. Sensível a sobreposição.",
    },
    "NCC_Template": {
        "tier": 2, "tier_label": "CPU (Notebook)", "icon": "💻",
        "ram": "~80 MB", "disk": "0 MB", "gpu": False, "offline": True,
        "accuracy": "1/5", "time_ms": 135, "stars": 1,
        "description": "NCC multi-escala com img2 como template. Não-invariante à rotação.",
    },
    "LoG_Blobs": {
        "tier": 2, "tier_label": "CPU (Notebook)", "icon": "💻",
        "ram": "~100 MB", "disk": "0 MB", "gpu": False, "offline": True,
        "accuracy": "0/5", "time_ms": 1600, "stars": 0,
        "description": "LoG scale-space (scipy). Alto recall, baixa precisão sem treino.",
    },
    "MobileNet_CNN": {
        "tier": 2, "tier_label": "CPU (Notebook)", "icon": "💻",
        "ram": "~200 MB", "disk": "14 MB", "gpu": False, "offline": True,
        "accuracy": "0/5", "time_ms": 1080, "stars": 0,
        "description": "MobileNetV2 saliency (ImageNet). Features genéricas demais p/ parafusos.",
    },
    "OwlViT": {
        "tier": 3, "tier_label": "GPU Recomendada", "icon": "🖥️",
        "ram": "~2 GB", "disk": "340 MB", "gpu": True, "offline": True,
        "accuracy": "?/5", "time_ms": 5000, "stars": None,
        "description": "Open-vocabulary detection (Google). Texto→bounding boxes. Zero-shot.",
    },
    "CLIP_Count": {
        "tier": 3, "tier_label": "GPU Recomendada", "icon": "🖥️",
        "ram": "~1 GB", "disk": "300 MB", "gpu": True, "offline": True,
        "accuracy": "?/5", "time_ms": 2000, "stars": None,
        "description": "CLIP text-similarity counting. Escolhe contagem mais provável por texto.",
    },
    "VLM_Claude": {
        "tier": 4, "tier_label": "Cloud API", "icon": "☁️",
        "ram": "0 MB", "disk": "0 MB", "gpu": False, "offline": False,
        "accuracy": "5/5", "time_ms": 1000, "stars": 5,
        "description": "Claude Sonnet via API. Melhor resultado mas requer internet e chave.",
    },
}

TIER_RUNNABLE = {1, 2}   # tiers que rodam localmente no backend

# ── Lazy method loader ─────────────────────────────────────────────────────────
_methods_cache: dict | None = None


def get_method(name: str):
    global _methods_cache
    if _methods_cache is None:
        from benchmark import (
            method_blob_detector,
            method_color_bg,
            method_convex_defects,
            method_hough_circles,
            method_log_blobs,
            method_mobilenet_salience,
            method_ncc_template,
            method_vlm_claude,
            method_watershed,
            method_ws_regional_max,
        )
        _methods_cache = {
            "WS_RegionalMax": method_ws_regional_max,
            "Watershed": method_watershed,
            "ConvexDefects": method_convex_defects,
            "ColorBG_Sub": method_color_bg,
            "HoughCircles": method_hough_circles,
            "SimpleBlobDet": method_blob_detector,
            "NCC_Template": method_ncc_template,
            "LoG_Blobs": method_log_blobs,
            "MobileNet_CNN": method_mobilenet_salience,
            "VLM_Claude": method_vlm_claude,
        }
        # OwlViT / CLIP — optional
        try:
            from advanced_models import method_owlvit, method_clip_count
            _methods_cache["OwlViT"] = method_owlvit
            _methods_cache["CLIP_Count"] = method_clip_count
        except ImportError:
            pass

    return _methods_cache.get(name)


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(_HTML)


@app.get("/api/hardware")
async def hardware_endpoint():
    return JSONResponse(HARDWARE)


@app.post("/count")
async def count_endpoint(
    file: UploadFile = File(...),
    method: str = Form("WS_RegionalMax"),
):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        return JSONResponse({"error": "Imagem inválida"}, status_code=400)

    fn = get_method(method)
    if fn is None:
        return JSONResponse({"error": f"Método '{method}' não disponível"}, status_code=400)

    t0 = time.perf_counter()
    try:
        count, annotated = fn(image.copy())
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
    img_b64 = base64.b64encode(buf.tobytes()).decode()

    profile = HARDWARE.get(method, {})

    return JSONResponse({
        "count": int(count) if count >= 0 else None,
        "method": method,
        "elapsed_ms": round(elapsed_ms, 1),
        "annotated_image": f"data:image/jpeg;base64,{img_b64}",
        "hardware": profile,
    })


@app.post("/benchmark")
async def benchmark_endpoint(file: UploadFile = File(...)):
    """Roda todos os métodos Tier 1 e 2 simultaneamente."""
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        return JSONResponse({"error": "Imagem inválida"}, status_code=400)

    runnable = {k: v for k, v in HARDWARE.items() if v["tier"] in TIER_RUNNABLE}
    results = []

    for method_name in runnable:
        fn = get_method(method_name)
        if fn is None:
            continue
        t0 = time.perf_counter()
        try:
            count, annotated = fn(image.copy())
        except Exception:
            continue
        elapsed_ms = (time.perf_counter() - t0) * 1000

        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_b64 = base64.b64encode(buf.tobytes()).decode()

        results.append({
            "method": method_name,
            "count": int(count) if count >= 0 else None,
            "elapsed_ms": round(elapsed_ms, 1),
            "annotated_image": f"data:image/jpeg;base64,{img_b64}",
            "hardware": HARDWARE[method_name],
        })

    return JSONResponse({"results": results})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
