"""
Retreina CLIP regressor e MobileNet FT usando o dataset sintético de alta qualidade
gerado por generate_dataset.py (25 base + 400 augmented = 425 imagens).
"""

import json, pickle, time
from pathlib import Path
import cv2, numpy as np

DATASET_DIR = Path(__file__).parent / "synthetic_dataset"
MODELS_DIR  = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def load_dataset(max_samples: int = 425):
    gt = json.loads((DATASET_DIR / "ground_truth.json").read_text())
    imgs, counts = [], []
    for rec in gt[:max_samples]:
        p = DATASET_DIR / rec["file"]
        img = cv2.imread(str(p))
        if img is None:
            continue
        imgs.append(img)
        counts.append(rec["count"])
    print(f"  Carregadas {len(imgs)} imagens, contagens: {min(counts)}–{max(counts)}")
    return imgs, counts


# ─── CLIP Regressor ──────────────────────────────────────────────────────────
def train_clip(imgs, counts):
    import torch
    from PIL import Image as PILImage
    from sklearn.linear_model import Ridge
    from transformers import CLIPModel, CLIPProcessor

    print("\n[1/2] Treinando CLIP Regressor no dataset sintético...")
    proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    feats = []
    with torch.no_grad():
        for i in range(0, len(imgs), 16):
            batch_imgs = imgs[i:i+16]
            pils = [PILImage.fromarray(cv2.cvtColor(b, cv2.COLOR_BGR2RGB)) for b in batch_imgs]
            inp  = proc(images=pils, return_tensors="pt", padding=True)
            out  = model.vision_model(pixel_values=inp["pixel_values"])
            feats.append(out.pooler_output.detach().cpu().numpy())
            if (i // 16) % 5 == 0:
                print(f"  Extraindo features: {i+len(batch_imgs)}/{len(imgs)}")

    X = np.vstack(feats)
    y = np.array(counts[:len(X)])

    from sklearn.model_selection import cross_val_score
    reg = Ridge(alpha=0.5)
    scores = cross_val_score(reg, X, y, cv=5, scoring='neg_mean_absolute_error')
    print(f"  CV MAE: {-scores.mean():.2f} ± {scores.std():.2f}")

    reg.fit(X, y)
    preds = reg.predict(X)
    print(f"  MAE treino: {np.mean(np.abs(preds - y)):.2f}")

    out = MODELS_DIR / "clip_regressor.pkl"
    with open(out, "wb") as f:
        pickle.dump(reg, f)
    print(f"  ✓ Salvo: {out}")
    return reg


# ─── MobileNet FT ────────────────────────────────────────────────────────────
def train_mobilenet(imgs, counts, epochs: int = 40):
    import torch
    import torch.nn as nn
    import torchvision.models as tvm
    import torchvision.transforms as T

    print(f"\n[2/2] Treinando MobileNet Regressor ({epochs} épocas, {len(imgs)} imgs)...")

    transform = T.Compose([
        T.ToPILImage(), T.Resize((224, 224)), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    X_imgs, y_t = [], []
    for img_bgr, cnt in zip(imgs, counts):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        X_imgs.append(transform(img_rgb))
        y_t.append(float(cnt))

    X_t = torch.stack(X_imgs)
    y_t = torch.tensor(y_t).unsqueeze(1)

    backbone = tvm.mobilenet_v2(weights=tvm.MobileNet_V2_Weights.IMAGENET1K_V1)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    # Congela menos para melhor adaptação ao domínio
    for name, p in backbone.named_parameters():
        if "features.17" in name or "features.18" in name or "classifier" in name:
            p.requires_grad = True

    head = nn.Sequential(
        nn.Linear(1000, 256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, 64),  nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(64, 1)
    )

    class CountModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.head = head
        def forward(self, x):
            feats = self.backbone(x)
            return self.head(feats)

    model = CountModel()
    params = [p for p in model.backbone.parameters() if p.requires_grad]
    params += list(model.head.parameters())
    optimizer = torch.optim.Adam(params, lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.HuberLoss(delta=1.0)

    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=24, shuffle=True)

    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        scheduler.step()
        avg = total / len(loader)
        if (epoch + 1) % 8 == 0:
            print(f"  Época {epoch+1:3d}/{epochs} — HuberLoss: {avg:.4f}")
        if avg < best_loss:
            best_loss = avg
            torch.save(head.state_dict(), MODELS_DIR / "mobilenet_regressor.pth")

    print(f"  ✓ Melhor loss: {best_loss:.4f}")
    print(f"  ✓ Salvo: {MODELS_DIR}/mobilenet_regressor.pth")
    return head


if __name__ == "__main__":
    imgs, counts = load_dataset()
    train_clip(imgs, counts)
    train_mobilenet(imgs, counts)
    print("\n✓ Retreino completo!")
