"""
Modelos avançados — Tier 3 (GPU recomendada)
OwlViT: zero-shot object detection via texto ("a screw")
CLIP:   contagem por similaridade texto-imagem
"""

from __future__ import annotations
import cv2
import numpy as np

# ── OwlViT ─────────────────────────────────────────────────────────────────────
_owlvit_model = None
_owlvit_proc = None


def _load_owlvit():
    global _owlvit_model, _owlvit_proc
    if _owlvit_model is None:
        from transformers import OwlViTForObjectDetection, OwlViTProcessor
        _owlvit_proc = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        _owlvit_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
        _owlvit_model.eval()
    return _owlvit_proc, _owlvit_model


def method_owlvit(image: np.ndarray) -> tuple[int, np.ndarray]:
    """
    OwlViT (Google, 2022) — Open-vocabulary object detection.
    Detecta objetos via prompts textuais: 'a screw', 'a bolt', 'a metal fastener'.
    Threshold adaptativo: começa em 0.05 (zero-shot é conservador).
    NMS final com IoU=0.5.
    Requer ~340 MB disco e ~2 GB RAM; GPU acelera 10×.
    """
    import torch
    from PIL import Image as PILImage

    proc, model = _load_owlvit()

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = PILImage.fromarray(img_rgb)

    queries = [["a screw", "a bolt", "a metal fastener", "a metallic cylinder"]]
    inputs = proc(text=queries, images=pil_img, return_tensors="pt")

    h, w = image.shape[:2]
    with torch.no_grad():
        outputs = model(**inputs)

    # Decode raw outputs: logits [1, n_patches, n_queries], pred_boxes [1, n_patches, 4]
    logits = outputs.logits[0]          # [n_patches, n_queries]
    pred_boxes = outputs.pred_boxes[0]  # [n_patches, 4] cx,cy,w,h normalized

    scores = logits.sigmoid().max(dim=-1).values   # [n_patches]
    threshold = 0.08
    mask = scores > threshold
    filt_scores = scores[mask].numpy()
    filt_boxes = pred_boxes[mask].numpy()

    if len(filt_boxes) == 0:
        return 0, image.copy()

    # cx,cy,w,h → x1,y1,x2,y2 absolute
    cx, cy, bw, bh = filt_boxes[:, 0], filt_boxes[:, 1], filt_boxes[:, 2], filt_boxes[:, 3]
    boxes = np.stack([
        (cx - bw / 2) * w, (cy - bh / 2) * h,
        (cx + bw / 2) * w, (cy + bh / 2) * h,
    ], axis=1)

    # NMS greedy por score
    order = np.argsort(-filt_scores)
    kept_boxes = []
    used = np.zeros(len(order), dtype=bool)
    for i in order:
        if used[i]:
            continue
        kept_boxes.append(boxes[i])
        for j in order:
            if not used[j] and j != i:
                if _iou(boxes[i], boxes[j]) > 0.4:
                    used[j] = True

    ann = image.copy()
    for idx, (x1, y1, x2, y2) in enumerate(kept_boxes):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(ann, (x1, y1), (x2, y2), (80, 200, 255), 2)
        cv2.putText(ann, str(idx + 1), (x1 + 4, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 200, 255), 2)

    return len(kept_boxes), ann


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / (union + 1e-6)


# ── CLIP Counting ───────────────────────────────────────────────────────────────
_clip_model = None
_clip_proc = None


def _load_clip():
    global _clip_model, _clip_proc
    if _clip_model is None:
        from transformers import CLIPModel, CLIPProcessor
        _clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model.eval()
    return _clip_proc, _clip_model


def method_clip_count(image: np.ndarray) -> tuple[int, np.ndarray]:
    """
    CLIP (OpenAI, 2021) — Contagem por similaridade texto-imagem.
    Gera 25 candidatos: 'a photo with exactly N screws' (N=1..25).
    Escolhe N com maior softmax similarity entre imagem e texto.
    Abordagem criativa: sem bounding boxes, puramente semântica.
    Funciona melhor com imagens de baixa contagem (1–5).
    Requer ~300 MB disco e ~1 GB RAM.
    """
    import torch
    from PIL import Image as PILImage

    proc, model = _load_clip()

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = PILImage.fromarray(img_rgb)

    max_count = 25
    # Prompt mais curto = embedding mais limpo
    words = ["one","two","three","four","five","six","seven","eight","nine","ten",
             "eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen",
             "eighteen","nineteen","twenty","twenty-one","twenty-two","twenty-three",
             "twenty-four","twenty-five"]
    texts = [f"{words[n-1]} screw{'s' if n > 1 else ''}" for n in range(1, max_count + 1)]

    inputs = proc(text=texts, images=pil_img, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits_per_image.softmax(dim=1)
    count = int(logits.argmax().item()) + 1  # range começa em 1

    ann = image.copy()
    h, w = image.shape[:2]
    # Top-3 candidates como gráfico de barras
    top3_idx = logits[0].topk(3).indices.numpy()
    for rank, idx in enumerate(top3_idx):
        n = idx + 1
        prob = float(logits[0][idx])
        bar_w = int(w * 0.3 * prob / float(logits[0][top3_idx[0]]))
        y = 30 + rank * 28
        color = (80, 200, 255) if rank == 0 else (100, 100, 120)
        cv2.rectangle(ann, (10, y - 16), (10 + bar_w, y), color, -1)
        cv2.putText(ann, f"{n} screws ({prob:.1%})", (14, y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return count, ann
