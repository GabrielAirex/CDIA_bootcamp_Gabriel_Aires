"""
Retreino rápido: 10 épocas partindo do best.pt já treinado.
"""
import shutil
from pathlib import Path
from ultralytics import YOLO

BASE = Path(__file__).parent
WEIGHTS_DIR = BASE / "weights"
RUNS_DIR = BASE / "runs"
DATA_YAML = BASE / "data.yaml"
CURRENT_BEST = BASE / "runs" / "segment" / "runs" / "crack_seg" / "weights" / "best.pt"


def main():
    if not CURRENT_BEST.exists():
        print(f"ERRO: best.pt não encontrado em {CURRENT_BEST}")
        return

    print(f"Carregando modelo de: {CURRENT_BEST}")
    model = YOLO(str(CURRENT_BEST))

    print("Iniciando retreino — 10 épocas...\n")
    results = model.train(
        data=str(DATA_YAML.resolve()),
        epochs=10,
        imgsz=640,
        batch=8,
        project=str(RUNS_DIR.resolve()),
        name="crack_seg",
        exist_ok=True,
        verbose=True,
        patience=10,
        save=True,
        plots=True,
        resume=False,
    )

    save_dir = Path(results.save_dir)
    best = save_dir / "weights" / "best.pt"
    WEIGHTS_DIR.mkdir(exist_ok=True)
    dst = WEIGHTS_DIR / "best.pt"
    shutil.copy2(best, dst)
    print(f"\nMelhor modelo salvo em: {dst}")

    print("\n═══ Rodando validação final ═══")
    val_metrics = model.val(data=str(DATA_YAML), imgsz=640, verbose=False)
    print(f"  mAP50 (box):    {val_metrics.box.map50:.4f}")
    print(f"  mAP50-95 (box): {val_metrics.box.map:.4f}")
    if hasattr(val_metrics, "seg") and val_metrics.seg is not None:
        print(f"  mAP50 (seg):    {val_metrics.seg.map50:.4f}")
        print(f"  mAP50-95 (seg): {val_metrics.seg.map:.4f}")
    print("═══════════════════════════════")
    print("\nDone. Pode rodar post_train.py para benchmark completo.")


if __name__ == "__main__":
    main()
