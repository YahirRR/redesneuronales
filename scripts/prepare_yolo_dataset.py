"""Prepara un dataset en formato YOLO (carpetas images/ y labels/) para entrenar el detector.

Uso:
    python scripts/prepare_yolo_dataset.py --src RUTA/AL/DATASET --out data/yolo_fracture \
        --classes fracture [--val 0.15 --test 0.10]

Espera imágenes y archivos .txt con el mismo nombre base (en cualquier subcarpeta de --src).
La partición se hace con semilla fija. Si el dataset tiene varias imágenes por paciente,
agrupa antes por paciente para evitar fuga de datos entre train/val/test.
"""
import argparse
import random
import shutil
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="data/yolo_fracture")
    ap.add_argument("--classes", nargs="+", required=True, help="Nombres de clase en orden de índice")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    src, out = Path(a.src), Path(a.out)
    labels = {p.stem: p for p in src.rglob("*.txt") if p.name != "classes.txt"}
    pairs = [(p, labels[p.stem]) for p in src.rglob("*") if p.suffix.lower() in IMG_EXT and p.stem in labels]
    if not pairs:
        raise SystemExit("No se encontraron pares imagen/etiqueta .txt en --src.")

    random.Random(a.seed).shuffle(pairs)
    n = len(pairs)
    n_test, n_val = int(n * a.test), int(n * a.val)
    splits = {"test": pairs[:n_test], "val": pairs[n_test:n_test + n_val], "train": pairs[n_test + n_val:]}

    for name, items in splits.items():
        (out / "images" / name).mkdir(parents=True, exist_ok=True)
        (out / "labels" / name).mkdir(parents=True, exist_ok=True)
        for img, lab in items:
            shutil.copy2(img, out / "images" / name / img.name)
            shutil.copy2(lab, out / "labels" / name / f"{img.stem}.txt")
        print(f"{name}: {len(items)} imágenes")

    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(a.classes))
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{names}\n"
    )
    print(f"Listo: {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
