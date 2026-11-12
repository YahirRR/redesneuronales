from dataclasses import dataclass
from typing import Dict

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


@dataclass
class EnhanceParams:
    denoise: float = 5.0  # fuerza del denoising (0 = desactivado)
    clahe_clip: float = 1.0  # límite de contraste CLAHE (0 = desactivado)
    sharpen: float = 0.3  # cantidad de unsharp mask (0 = desactivado)


def to_gray_u8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img


def enhance(img: np.ndarray, params: EnhanceParams = EnhanceParams()) -> np.ndarray:
    """Cadena conservadora: denoising -> CLAHE -> unsharp mask. No genera estructuras nuevas."""
    out = to_gray_u8(img)

    if params.denoise > 0:
        out = cv2.fastNlMeansDenoising(out, None, h=params.denoise, templateWindowSize=7, searchWindowSize=21)

    if params.clahe_clip > 0:
        clahe = cv2.createCLAHE(clipLimit=params.clahe_clip, tileGridSize=(8, 8))
        out = clahe.apply(out)

    if params.sharpen > 0:
        blur = cv2.GaussianBlur(out, (0, 0), sigmaX=2.0)
        out = cv2.addWeighted(out, 1 + params.sharpen, blur, -params.sharpen, 0)

    return out


def quality_metrics(img: np.ndarray) -> Dict[str, float]:
    """Métricas sin referencia: nitidez, contraste y entropía."""
    g = to_gray_u8(img)
    hist = np.bincount(g.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    p = p[p > 0]
    return {
        "nitidez": float(cv2.Laplacian(g, cv2.CV_64F).var()),
        "contraste": float(g.std()),
        "entropia": float(-(p * np.log2(p)).sum()),
    }


def degrade(img: np.ndarray, noise_sigma: float = 8.0, blur_sigma: float = 1.5, scale: float = 0.5, seed: int = 0) -> np.ndarray:
    """Degradación sintética para evaluar restauración con referencia."""
    rng = np.random.default_rng(seed)
    g = to_gray_u8(img)
    h, w = g.shape
    out = cv2.GaussianBlur(g, (0, 0), blur_sigma)
    out = cv2.resize(out, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
    out = out.astype(np.float32) + rng.normal(0, noise_sigma, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def reference_metrics(reference: np.ndarray, test: np.ndarray) -> Dict[str, float]:
    ref, tst = to_gray_u8(reference), to_gray_u8(test)
    return {
        "psnr": float(peak_signal_noise_ratio(ref, tst, data_range=255)),
        "ssim": float(structural_similarity(ref, tst, data_range=255)),
    }
