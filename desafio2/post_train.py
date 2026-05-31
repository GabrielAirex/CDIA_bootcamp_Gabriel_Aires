"""
Roda após train.py terminar:
1. Copia best.pt para weights/ (caso o script inline não tenha conseguido)
2. Valida o modelo final
3. Roda benchmark completo com YOLOv8
"""

import shutil
from pathlib import Path

BASE = Path(__file__).parent
RUNS_BASE = BASE / "runs" / "segment"
WEIGHTS_DIR = BASE / "weights"


def find_best_weights() -> Path | None:
    """Procura best.pt em qualquer subpasta de runs."""
    for pt in RUNS_BASE.rglob("weights/best.pt"):
        return pt
    return None


def copy_weights():
    best = find_best_weights()
    if best is None:
        print("best.pt não encontrado em runs/")
        return None
    WEIGHTS_DIR.mkdir(exist_ok=True)
    dst = WEIGHTS_DIR / "best.pt"
    shutil.copy2(best, dst)
    print(f"Copiado: {best} → {dst}")
    return dst


def validate(weights: Path):
    from ultralytics import YOLO
    model = YOLO(str(weights))
    data_yaml = BASE / "data.yaml"
    metrics = model.val(data=str(data_yaml), imgsz=640, verbose=False)
    print("\n═══ MÉTRICAS FINAIS ═══")
    print(f"  mAP50 (box):    {metrics.box.map50:.4f}")
    print(f"  mAP50-95 (box): {metrics.box.map:.4f}")
    if hasattr(metrics, "seg") and metrics.seg is not None:
        print(f"  mAP50 (seg):    {metrics.seg.map50:.4f}")
        print(f"  mAP50-95 (seg): {metrics.seg.map:.4f}")
    print("═══════════════════════")
    return metrics


if __name__ == "__main__":
    print("=== Pós-treino D2 ===\n")
    weights = copy_weights()
    if weights:
        validate(weights)
        print("\nRodando benchmark completo...")
        from benchmark import run_benchmark
        run_benchmark(n_images=30)
