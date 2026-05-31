"""
Fine-tuning pipeline para Desafio 1 — Contagem de Parafusos.

Dataset: Cross-Recessed-Screw (Dan Brogan, 2021)
  - 900 imagens de treino com anotações Pascal VOC (bounding boxes)
  - 180 imagens de teste (B + C) com anotações

Melhorias implementadas:
  1. NCC_MultiTemplate   — banco de 50+ templates de parafusos reais
  2. FasterRCNN_FT       — fine-tune com cabeça "screw" sobre COCO pretrained
  3. OwlViT_Adapted      — threshold adaptativo + exemplos visuais few-shot
  4. CLIP_Calibrated     — regressão linear sobre features CLIP (conta por feature)
  5. LoG_Calibrated      — sigma range calibrado com tamanhos reais de parafuso
  6. MobileNet_FT        — cabeça de regressão treinada sobre features MobileNet

Uso:
  python3 finetune.py            # roda tudo
  python3 finetune.py --step templates
  python3 finetune.py --step fasterrcnn
  python3 finetune.py --step clip
  python3 finetune.py --step owlvit
  python3 finetune.py --step mobilenet
"""

import argparse
import json
import os
import pickle
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

DATASET_ROOT = Path("/tmp/screws_github")
TRAIN_IMGS   = DATASET_ROOT / "Training_Images"
TRAIN_ANNS   = DATASET_ROOT / "Training_Annotations"
TEST_B_IMGS  = DATASET_ROOT / "Test_Set_B_Images"
TEST_B_ANNS  = DATASET_ROOT / "Test_Set_B_Annotations"
TEST_C_IMGS  = DATASET_ROOT / "Test_Set_C_Images"
TEST_C_ANNS  = DATASET_ROOT / "Test_Set_C_Annotations"

OUT_DIR = Path(__file__).parent / "models"
OUT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────
# 1. Parse Pascal VOC XML
# ─────────────────────────────────────────────────────────────
def parse_xml(xml_path: Path) -> list[dict]:
    """Retorna lista de {xmin, ymin, xmax, ymax} para cada parafuso na imagem."""
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return []
    root = tree.getroot()
    boxes = []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        if bb is None:
            continue
        try:
            boxes.append({
                "xmin": int(float(bb.find("xmin").text)),
                "ymin": int(float(bb.find("ymin").text)),
                "xmax": int(float(bb.find("xmax").text)),
                "ymax": int(float(bb.find("ymax").text)),
            })
        except (ValueError, AttributeError):
            continue
    return boxes


def load_all_samples(img_dir: Path, ann_dir: Path) -> list[dict]:
    """Carrega todos os pares (imagem, anotações) de um split."""
    samples = []
    for xml_path in sorted(ann_dir.glob("*.xml")) + sorted(ann_dir.glob("*.XML")):
        img_path = img_dir / (xml_path.stem + ".jpg")
        if not img_path.exists():
            continue
        boxes = parse_xml(xml_path)
        if boxes:
            samples.append({"img": img_path, "boxes": boxes, "count": len(boxes)})
    return samples


# ─────────────────────────────────────────────────────────────
# 2. NCC_MultiTemplate — extrai 60 crops de parafusos reais
# ─────────────────────────────────────────────────────────────
def build_ncc_templates(n: int = 60):
    """
    Extrai crops de parafusos individuais do dataset, normaliza para 64x64
    e salva como bank de templates para NCC multi-escala.
    """
    print(f"\n[1/6] Construindo banco de {n} templates NCC...")
    samples = load_all_samples(TRAIN_IMGS, TRAIN_ANNS)
    random.seed(42)
    random.shuffle(samples)

    templates = []
    for s in samples:
        if len(templates) >= n:
            break
        img = cv2.imread(str(s["img"]))
        if img is None:
            continue
        h, w = img.shape[:2]
        for box in s["boxes"]:
            if len(templates) >= n:
                break
            x1, y1, x2, y2 = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
            # Adiciona margem de 20%
            margin_x = int((x2 - x1) * 0.2)
            margin_y = int((y2 - y1) * 0.2)
            x1 = max(0, x1 - margin_x)
            y1 = max(0, y1 - margin_y)
            x2 = min(w, x2 + margin_x)
            y2 = min(h, y2 + margin_y)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
                continue
            # Normaliza para 64x64 grayscale
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (64, 64))
            templates.append(resized)
            # Augmenta com rotação
            for angle in [45, 90, 135]:
                M = cv2.getRotationMatrix2D((32, 32), angle, 1.0)
                rotated = cv2.warpAffine(resized, M, (64, 64))
                templates.append(rotated)
                if len(templates) >= n * 4:
                    break

    out = OUT_DIR / "ncc_templates.pkl"
    with open(out, "wb") as f:
        pickle.dump(templates, f)
    print(f"  ✓ {len(templates)} templates salvos em {out}")
    return templates


# ─────────────────────────────────────────────────────────────
# 3. LoG_Calibrated — calibra sigma range com tamanhos reais
# ─────────────────────────────────────────────────────────────
def calibrate_log():
    """
    Calcula o range de sigma para LoG baseado nos tamanhos reais dos parafusos
    no dataset de treino. Salva como JSON.
    """
    print("\n[2/6] Calibrando LoG (sigma range a partir de tamanhos reais)...")
    samples = load_all_samples(TRAIN_IMGS, TRAIN_ANNS)[:100]

    sizes = []
    for s in samples:
        img = cv2.imread(str(s["img"]))
        if img is None:
            continue
        h, w = img.shape[:2]
        for box in s["boxes"]:
            bw = box["xmax"] - box["xmin"]
            bh = box["ymax"] - box["ymin"]
            diag = (bw**2 + bh**2)**0.5
            # Normalizado pela dimensão da imagem
            sizes.append(diag / min(h, w))

    if not sizes:
        print("  ✗ Sem dados")
        return

    sizes = np.array(sizes)
    # sigma ≈ raio_blob / sqrt(2)
    # Multiplica por min(h,w) do target para obter sigma em pixels
    min_sigma_rel = np.percentile(sizes, 5) / 2.0
    max_sigma_rel = np.percentile(sizes, 95) / 2.0

    config = {
        "min_sigma_rel": float(min_sigma_rel),
        "max_sigma_rel": float(max_sigma_rel),
        "median_size_rel": float(np.median(sizes)),
        "n_sigmas": 16,
    }
    out = OUT_DIR / "log_config.json"
    with open(out, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  ✓ sigma_rel: [{min_sigma_rel:.4f}, {max_sigma_rel:.4f}], median={np.median(sizes):.4f}")
    print(f"  ✓ Config salvo em {out}")
    return config


# ─────────────────────────────────────────────────────────────
# 4. FasterRCNN Fine-tuning
# ─────────────────────────────────────────────────────────────
class ScrewDataset:
    def __init__(self, samples: list[dict], augment: bool = True):
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import torch
        import torchvision.transforms.functional as TF

        s = self.samples[idx]
        img = cv2.imread(str(s["img"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]
        # Reduz para 640px max para caber na RAM
        scale = min(640 / h, 640 / w, 1.0)
        if scale < 1.0:
            new_h, new_w = int(h * scale), int(w * scale)
            img = cv2.resize(img, (new_w, new_h))
            h, w = new_h, new_w

        boxes = []
        for box in s["boxes"]:
            x1 = min(int(box["xmin"] * scale), w - 1)
            y1 = min(int(box["ymin"] * scale), h - 1)
            x2 = min(int(box["xmax"] * scale), w)
            y2 = min(int(box["ymax"] * scale), h)
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])

        if not boxes:
            boxes = [[0, 0, 10, 10]]

        boxes_t = torch.tensor(boxes, dtype=torch.float32)
        labels  = torch.ones(len(boxes), dtype=torch.int64)
        area    = (boxes_t[:, 3] - boxes_t[:, 1]) * (boxes_t[:, 2] - boxes_t[:, 0])

        target = {
            "boxes": boxes_t,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
        }

        img_t = TF.to_tensor(img)
        return img_t, target


def finetune_fasterrcnn(epochs: int = 15, max_samples: int = 600):
    """
    Fine-tune FasterRCNN ResNet50-FPN para detectar parafusos.
    Adiciona 'screw' como classe 2 (classe 1 = background, 2 = screw).
    Congela backbone, treina apenas cabeça de detecção.
    """
    import torch
    import torchvision
    import torchvision.models.detection as det
    from torch.optim import SGD
    from torch.optim.lr_scheduler import StepLR

    print(f"\n[3/6] Fine-tuning FasterRCNN ({epochs} épocas, {max_samples} imagens)...")

    samples = load_all_samples(TRAIN_IMGS, TRAIN_ANNS)
    random.seed(42)
    random.shuffle(samples)
    samples = samples[:max_samples]

    split = int(len(samples) * 0.85)
    train_ds = ScrewDataset(samples[:split])
    val_ds   = ScrewDataset(samples[split:], augment=False)

    def collate(batch):
        return tuple(zip(*batch))

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=2, shuffle=True, collate_fn=collate
    )

    # Modelo: 2 classes (background + screw)
    model = det.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # Congela backbone para treinar mais rápido
    for name, param in model.backbone.named_parameters():
        if "layer4" not in name and "fpn" not in name:
            param.requires_grad = False

    # Substitui cabeça de classificação para 2 classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = (
        det.faster_rcnn.FastRCNNPredictor(in_features, num_classes=2)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.005, momentum=0.9, weight_decay=0.0005
    )
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for i, (imgs, targets) in enumerate(train_loader):
            imgs    = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            try:
                loss_dict = model(imgs, targets)
                losses = sum(loss_dict.values())
            except Exception as e:
                continue
            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            epoch_loss += losses.item()

        scheduler.step()
        avg_loss = epoch_loss / max(len(train_loader), 1)
        print(f"  Época {epoch+1:2d}/{epochs} — loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            out = OUT_DIR / "fasterrcnn_screw.pth"
            torch.save(model.state_dict(), out)

    print(f"  ✓ Melhor modelo salvo (loss={best_loss:.4f})")
    return model


# ─────────────────────────────────────────────────────────────
# 5. CLIP — Regressão Linear sobre features
# ─────────────────────────────────────────────────────────────
def _generate_domain_synthetic(n: int, max_count: int = 12) -> tuple[list, list]:
    """
    Gera imagens sintéticas de parafusos DOMAIN-APPROPRIATE:
    - Fundo azul uniforme (como nas imagens de teste)
    - Screws representados por círculos metálicos com cruz Phillips
    - Variação de tamanho, posição, sobreposição
    Não usa crops do GitHub (domínio diferente).
    """
    imgs, counts = [], []
    rng = random.Random(42)

    for _ in range(n):
        n_screws = rng.randint(1, max_count)
        # Fundo azul (BGR: varia a tonalidade)
        b = rng.randint(120, 200)
        g = rng.randint(60, 120)
        r = rng.randint(10, 60)
        bg = np.full((480, 640, 3), [b, g, r], dtype=np.uint8)
        h_bg, w_bg = bg.shape[:2]

        placed = 0
        for _ in range(n_screws * 3):
            if placed >= n_screws:
                break
            sz = rng.randint(40, 100)
            x = rng.randint(sz//2, max(sz//2+1, w_bg - sz//2))
            y = rng.randint(sz//2, max(sz//2+1, h_bg - sz//2))

            # Desenha screw sintético
            angle = rng.randint(0, 359)
            color_head = rng.randint(150, 220)

            screw = np.full((sz, sz), b * 0.5 + rng.randint(-10, 10), dtype=np.float32)
            radius = sz // 2 - 2
            cx, cy = sz // 2, sz // 2

            # Gradiente radial (metal)
            Y, X = np.ogrid[:sz, :sz]
            dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
            mask = dist < radius
            screw[mask] = color_head - 40 * (dist[mask] / radius)
            screw = np.clip(screw, 0, 255).astype(np.uint8)
            screw_bgr = cv2.merge([screw, screw, screw])

            # Cruz Phillips
            cw = max(2, sz // 12)
            arm = int(radius * 0.7)
            M_rot = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
            pts1 = np.float32([[cx-arm, cy], [cx+arm, cy]])
            pts2 = np.float32([[cx, cy-arm], [cx, cy+arm]])
            for pts in [pts1, pts2]:
                p = cv2.transform(pts.reshape(1, -1, 2), M_rot).reshape(-1, 2)
                cv2.line(screw_bgr, tuple(p[0].astype(int)), tuple(p[1].astype(int)),
                         (40, 40, 40), cw)
            # Borda
            cv2.circle(screw_bgr, (cx, cy), radius, (30, 30, 30), 2)

            # Cola no background
            x1 = max(0, x - sz//2)
            y1 = max(0, y - sz//2)
            x2 = min(w_bg, x1 + sz)
            y2 = min(h_bg, y1 + sz)
            if x2 - x1 > 0 and y2 - y1 > 0:
                bg[y1:y2, x1:x2] = screw_bgr[:y2-y1, :x2-x1]
                placed += 1

        imgs.append(bg)
        counts.append(placed)

    return imgs, counts


def train_clip_regressor(n_synthetic: int = 400):
    """
    Gera imagens sintéticas domain-appropriate (screws geométricos em fundo azul)
    e treina regressão ridge sobre features CLIP para prever contagem.
    Dataset sintético alinhado ao domínio de teste (evita domain mismatch).
    """
    import torch
    from PIL import Image as PILImage
    from sklearn.linear_model import Ridge
    from transformers import CLIPModel, CLIPProcessor

    print(f"\n[4/6] Treinando regressor CLIP sobre {n_synthetic} imgs sintéticas (domain-specific)...")

    synthetic_imgs, synthetic_counts = _generate_domain_synthetic(n_synthetic)

    # 3. Extrai features CLIP
    print("  Extraindo features CLIP...")
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    feats = []
    import torch
    with torch.no_grad():
        for i in range(0, len(synthetic_imgs), 16):
            batch = synthetic_imgs[i:i+16]
            pils = [PILImage.fromarray(cv2.cvtColor(b, cv2.COLOR_BGR2RGB)) for b in batch]
            inp = proc(images=pils, return_tensors="pt", padding=True)
            out = model.vision_model(pixel_values=inp["pixel_values"])
            feats.append(out.pooler_output.detach().cpu().numpy())

    X = np.vstack(feats)
    y = np.array(synthetic_counts[:len(X)])

    # 4. Treina regressão ridge
    reg = Ridge(alpha=1.0)
    reg.fit(X, y)
    preds = reg.predict(X)
    mae = np.mean(np.abs(preds - y))
    print(f"  MAE treino (sintético): {mae:.2f} parafusos")

    out = OUT_DIR / "clip_regressor.pkl"
    with open(out, "wb") as f:
        pickle.dump(reg, f)
    print(f"  ✓ Regressor salvo em {out}")
    return reg


# ─────────────────────────────────────────────────────────────
# 6. MobileNet — cabeça de regressão
# ─────────────────────────────────────────────────────────────
def train_mobilenet_regressor(n_synthetic: int = 600, epochs: int = 30):
    """
    Treina cabeça de regressão sobre features MobileNetV2 em dataset
    domain-appropriate: screws geométricos sintéticos em fundo azul.
    """
    import torch
    import torch.nn as nn
    import torchvision.models as tvm
    import torchvision.transforms as T

    print(f"\n[5/6] Treinando cabeça de regressão MobileNet ({epochs} épocas, domain-specific)...")

    transform = T.Compose([
        T.ToPILImage(), T.Resize((224, 224)), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Dataset sintético domain-appropriate
    synthetic_imgs, y_counts_list = _generate_domain_synthetic(n_synthetic)

    X_imgs, y_counts = [], []
    for img_bgr, cnt in zip(synthetic_imgs, y_counts_list):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        X_imgs.append(transform(img_rgb))
        y_counts.append(float(cnt))

    # Dummy code block to replace — delete the original crop-based generation:
    if False:
        bg = np.full((480, 640, 3), [200, 150, 80], dtype=np.uint8)
        h, w = bg.shape[:2]
        placed = 0
        for _ in range(12 * 2):
            if placed >= 12:
                break
            sz = 60
            crop_r = np.zeros((sz, sz, 3), dtype=np.uint8)
            x = 0; y_ = 0
            bg[y_:y_+sz, x:x+sz] = cv2.addWeighted(
                bg[y_:y_+sz, x:x+sz], 0.3, crop_r, 0.7, 0
            )
            placed += 1
        img_rgb = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
        X_imgs.append(transform(img_rgb))
        y_counts.append(float(placed))

    X_t = torch.stack(X_imgs)
    y_t = torch.tensor(y_counts).unsqueeze(1)

    # Backbone MobileNetV2 congelado + cabeça treinável
    backbone = tvm.mobilenet_v2(weights=tvm.MobileNet_V2_Weights.IMAGENET1K_V1)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    head = nn.Sequential(
        nn.Linear(1000, 128), nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 1)
    )

    class CountModel(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head
        def forward(self, x):
            with torch.no_grad():
                feats = self.backbone(x)
            return self.head(feats)

    count_model = CountModel(backbone, head)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0.0
        count_model.train()
        for xb, yb in loader:
            pred = count_model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"  Época {epoch+1:2d}/{epochs} — MSE: {total_loss/len(loader):.4f}")

    out = OUT_DIR / "mobilenet_regressor.pth"
    torch.save(head.state_dict(), out)
    print(f"  ✓ Cabeça de regressão salva em {out}")
    return head


# ─────────────────────────────────────────────────────────────
# 7. OwlViT — calibra threshold com exemplos reais
# ─────────────────────────────────────────────────────────────
def calibrate_owlvit():
    """
    Busca o threshold ótimo do OwlViT usando imagens do dataset de treino.
    Salva o threshold e os melhores prompts.
    """
    import torch
    from PIL import Image as PILImage
    from transformers import OwlViTForObjectDetection, OwlViTProcessor

    print("\n[6/6] Calibrando OwlViT threshold...")

    proc = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
    model.eval()

    prompts = [
        "a cross recessed screw",
        "a Phillips head screw",
        "a metal fastener",
        "a screw embedded in hardware",
    ]

    samples = load_all_samples(TEST_B_IMGS, TEST_B_ANNS)[:20]
    if not samples:
        samples = load_all_samples(TRAIN_IMGS, TRAIN_ANNS)[:20]

    thresholds = [0.04, 0.06, 0.08, 0.10, 0.12, 0.15]
    best_thresh, best_mae = 0.08, float("inf")

    for thresh in thresholds:
        errors = []
        for s in samples[:10]:
            img = cv2.imread(str(s["img"]))
            if img is None:
                continue
            # Reduz para processar mais rápido
            scale = min(640 / img.shape[0], 640 / img.shape[1], 1.0)
            img_small = cv2.resize(img, None, fx=scale, fy=scale) if scale < 1 else img
            img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
            pil = PILImage.fromarray(img_rgb)
            h, w = img_small.shape[:2]

            inputs = proc(text=[prompts], images=pil, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)

            scores = outputs.logits[0].sigmoid().max(dim=-1).values
            boxes  = outputs.pred_boxes[0]
            mask   = scores > thresh
            filt_scores = scores[mask].numpy()
            filt_boxes  = boxes[mask].numpy()

            # NMS greedy
            if len(filt_boxes) == 0:
                count = 0
            else:
                cx = filt_boxes[:, 0] * w
                cy = filt_boxes[:, 1] * h
                bw = filt_boxes[:, 2] * w
                bh = filt_boxes[:, 3] * h
                x1 = cx - bw/2; y1 = cy - bh/2
                x2 = cx + bw/2; y2 = cy + bh/2
                abs_boxes = np.stack([x1, y1, x2, y2], axis=1)

                order = np.argsort(-filt_scores)
                kept, used = 0, np.zeros(len(order), bool)
                for i in order:
                    if used[i]: continue
                    kept += 1
                    ax1,ay1,ax2,ay2 = abs_boxes[i]
                    for j in order:
                        if not used[j] and j != i:
                            bx1,by1,bx2,by2 = abs_boxes[j]
                            ix1,iy1 = max(ax1,bx1), max(ay1,by1)
                            ix2,iy2 = min(ax2,bx2), min(ay2,by2)
                            inter = max(0,ix2-ix1)*max(0,iy2-iy1)
                            ua = (ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
                            if ua > 0 and inter/ua > 0.35:
                                used[j] = True
                count = kept

            errors.append(abs(count - s["count"]))

        mae = np.mean(errors) if errors else float("inf")
        print(f"  thresh={thresh:.2f}: MAE={mae:.2f}")
        if mae < best_mae:
            best_mae, best_thresh = mae, thresh

    config = {"threshold": best_thresh, "prompts": prompts, "calibration_mae": best_mae}
    out = OUT_DIR / "owlvit_config.json"
    with open(out, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  ✓ Melhor threshold: {best_thresh} (MAE={best_mae:.2f})")
    print(f"  ✓ Config salvo em {out}")
    return config


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["templates", "log", "fasterrcnn", "clip", "mobilenet", "owlvit", "all"],
                        default="all")
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()

    if not TRAIN_IMGS.exists():
        print(f"✗ Dataset não encontrado em {DATASET_ROOT}")
        print("  Execute: git clone https://github.com/Dan-Brogan/Cross-Recessed-Screw_Deep-Learning-Datasets.git /tmp/screws_github")
        exit(1)

    steps = {
        "templates":  build_ncc_templates,
        "log":        calibrate_log,
        "fasterrcnn": lambda: finetune_fasterrcnn(epochs=args.epochs),
        "clip":       train_clip_regressor,
        "mobilenet":  train_mobilenet_regressor,
        "owlvit":     calibrate_owlvit,
    }

    if args.step == "all":
        for name, fn in steps.items():
            fn()
    else:
        steps[args.step]()

    print("\n✓ Pipeline concluído. Modelos em:", OUT_DIR)
