"""Entrena el detector YOLO y copia los mejores pesos a outputs/detector/ para la app.

Uso:
    pip install ultralytics
    python scripts/train_detector.py --data data/yolo_fracture/data.yaml --epochs 60
"""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _finish(model, a):
    from ultralytics import YOLO

    run = ROOT / "outputs" / "detector_runs" / a.name
    best = run / "weights" / "best.pt"
    dest = ROOT / "outputs" / "detector"
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, dest / f"{a.name}_best.pt")
    metrics = YOLO(str(best)).val(data=a.data, split="test", device=a.device)
    line = (f"mAP50: {metrics.box.map50:.3f} | mAP50-95: {metrics.box.map:.3f} "
            f"| precisión: {metrics.box.mp:.3f} | recall: {metrics.box.mr:.3f}")
    print(line)
    (run / "test_metrics.txt").write_text(line + "\n")
    print(f"Pesos copiados a {dest / (a.name + '_best.pt')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolov8n.pt", help="Pesos base (yolov8n/s/m.pt)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="fracture")
    ap.add_argument("--device", default="mps", help="mps (Mac), cpu o 0 (CUDA)")
    ap.add_argument("--resume", action="store_true", help="Continuar desde outputs/detector_runs/<name>/weights/last.pt")
    a = ap.parse_args()

    from ultralytics import YOLO

    if a.resume:
        last = ROOT / "outputs" / "detector_runs" / a.name / "weights" / "last.pt"
        model = YOLO(str(last))
        model.train(resume=True)
        _finish(model, a)
        return

    model = YOLO(a.model)
    model.train(
        data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, seed=42, device=a.device, workers=4,
        project=str(ROOT / "outputs" / "detector_runs"), name=a.name, exist_ok=True,
        fliplr=0.5, flipud=0.0, hsv_h=0.0, hsv_s=0.0, mosaic=0.5,
    )
    _finish(model, a)


if __name__ == "__main__":
    main()
