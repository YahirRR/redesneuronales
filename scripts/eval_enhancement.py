"""Evalúa la mejora de imagen con degradaciones sintéticas (PSNR/SSIM contra la imagen limpia).

Uso:
    python scripts/eval_enhancement.py --images CARPETA --n 50
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.enhance import EnhanceParams, degrade, enhance, reference_metrics  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--n", type=int, default=50)
    a = ap.parse_args()

    files = sorted(p for p in Path(a.images).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})[: a.n]
    if not files:
        raise SystemExit("Sin imágenes.")

    configs = {
        "degradada (sin procesar)": None,
        "solo denoising": EnhanceParams(5, 0, 0),
        "denoising + nitidez": EnhanceParams(5, 0, 0.6),
        "cadena completa": EnhanceParams(5, 2.0, 0.6),
    }
    rows = {k: [] for k in configs}
    for f in files:
        clean = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if clean is None:
            continue
        bad = degrade(clean)
        for k, p in configs.items():
            rows[k].append(reference_metrics(clean, bad if p is None else enhance(bad, p)))

    print(f"{'método':28s} {'PSNR (dB)':>10s} {'SSIM':>8s}")
    for k, v in rows.items():
        print(f"{k:28s} {np.mean([m['psnr'] for m in v]):10.2f} {np.mean([m['ssim'] for m in v]):8.3f}")
    print("\nNota: CLAHE cambia el contraste a propósito, por lo que puede bajar PSNR/SSIM aunque mejore la legibilidad.")


if __name__ == "__main__":
    main()
