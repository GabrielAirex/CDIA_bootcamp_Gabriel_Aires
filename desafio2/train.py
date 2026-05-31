"""
Desafio 2 — Treinamento YOLOv8n-seg para Detecção de Trincas
Fluxo: split dataset → data.yaml → treinar → reportar métricas
"""

import random
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO

BASE = Path(__file__).parent
DATASET_DIR = BASE / "dataset"
TRAIN_DIR = BASE / "dataset" / "train"
VAL_DIR = BASE / "dataset" / "val"
WEIGHTS_DIR = BASE / "weights"
RUNS_DIR = BASE / "runs"

TRAIN_RATIO = 0.8
EPOCHS = 50
IMG_SIZE = 640
BATCH = 8
MODEL_BASE = "yolov8n-seg.pt"


def split_dataset(seed: int = 42) -> tuple[int, int]:
    """Divide flat dataset em train/val mantendo labels pareados."""
    images = sorted((DATASET_DIR / "images").glob("*.jpg"))
    random.seed(seed)
    random.shuffle(images)

    n_train = int(len(images) * TRAIN_RATIO)
    splits = {"train": images[:n_train], "val": images[n_train:]}

    for split, imgs in splits.items():
        (DATASET_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / split / "labels").mkdir(parents=True, exist_ok=True)
        for img_path in imgs:
            lbl_path = DATASET_DIR / "labels" / img_path.with_suffix(".txt").name
            dst_img = DATASET_DIR / split / "images" / img_path.name
            dst_lbl = DATASET_DIR / split / "labels" / lbl_path.name
            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)
            if lbl_path.exists() and not dst_lbl.exists():
                shutil.copy2(lbl_path, dst_lbl)

    n_val = len(images) - n_train
    print(f"Dataset split: {n_train} train / {n_val} val ({len(images)} total)")
    return n_train, n_val


def create_data_yaml() -> Path:
    """Gera data.yaml para treino YOLO."""
    config = {
        "path": str(DATASET_DIR.resolve()),
        "train": "train/images",
        "val": "val/images",
        "nc": 1,
        "names": ["crack"],
    }
    out = BASE / "data.yaml"
    with open(out, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"data.yaml gerado em: {out}")
    return out


def train(data_yaml: Path) -> Path:
    """Fine-tuna YOLOv8n-seg e retorna path do melhor modelo."""
    model = YOLO(MODEL_BASE)
    results = model.train(
        data=str(data_yaml.resolve()),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        project=str(RUNS_DIR.resolve()),
        name="crack_seg",
        exist_ok=True,
        verbose=True,
        patience=15,
        save=True,
        plots=True,
    )
    save_dir = Path(results.save_dir)
    best = save_dir / "weights" / "best.pt"
    WEIGHTS_DIR.mkdir(exist_ok=True)
    dst = WEIGHTS_DIR / "best.pt"
    shutil.copy2(best, dst)
    print(f"\nMelhor modelo salvo em: {dst}")
    return dst


def validate(model_path: Path, data_yaml: Path):
    """Valida o modelo treinado e exibe métricas."""
    model = YOLO(str(model_path))
    metrics = model.val(data=str(data_yaml), imgsz=IMG_SIZE, verbose=False)
    print("\n═══ RESULTADOS DE VALIDAÇÃO ═══")
    print(f"  mAP50 (box):    {metrics.box.map50:.4f}")
    print(f"  mAP50-95 (box): {metrics.box.map:.4f}")
    if hasattr(metrics, "seg"):
        print(f"  mAP50 (seg):    {metrics.seg.map50:.4f}")
        print(f"  mAP50-95 (seg): {metrics.seg.map:.4f}")
    print("════════════════════════════════")


if __name__ == "__main__":
    print("=== Desafio 2 — Treino YOLOv8n-seg ===\n")

    print("1. Dividindo dataset…")
    split_dataset()

    print("\n2. Gerando data.yaml…")
    data_yaml = create_data_yaml()

    print("\n3. Treinando YOLOv8n-seg…")
    best_model = train(data_yaml)

    print("\n4. Validando modelo…")
    validate(best_model, data_yaml)

    print("\nTreino concluído. Rode `python3 benchmark.py` para benchmark completo.")
