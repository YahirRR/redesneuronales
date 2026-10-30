from typing import Tuple

import cv2
import numpy as np
import torch
from torchvision import transforms

from .config import INPUT_SIZE


class CLAHETransform:
    """Apply CLAHE to a single-channel or 3-channel image (numpy HxWxC or HxW).

    This is designed to be used before converting to tensor.
    """

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            # single channel
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            return clahe.apply(img)
        elif img.ndim == 3:
            # apply CLAHE on L channel in LAB space
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        else:
            return img


def get_train_transforms(apply_clahe: bool = True) -> transforms.Compose:
    """Transforms para entrenamiento.

    Augmentaciones pensadas para radiografías:
    - Resize a INPUT_SIZE
    - CLAHE opcional (mejora contraste en rayos X)
    - RandomRotation suave (hasta 20°)
    - RandomHorizontalFlip
    - RandomVerticalFlip con baja probabilidad (algunas rx pueden estar invertidas)
    - ColorJitter leve (solo brillo/contraste, sin saturación/hue en grises convertidas)
    - RandomAffine: pequeñas translaciones y escalados
    - GaussianBlur: simula pequeñas variaciones de nitidez
    - Normalización ImageNet
    """

    transform_list = [
        transforms.ToPILImage(),
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    ]

    if apply_clahe:
        transform_list.append(transforms.Lambda(lambda im: _apply_clahe_pil(im)))

    transform_list.extend(
        [
            transforms.RandomRotation(degrees=20),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.9, 1.1),
            ),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            # Borrar aleatoriamente pequeñas regiones (simula artefactos o zonas ocluidas)
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.08)),
        ]
    )

    return transforms.Compose(transform_list)


def get_eval_transforms(apply_clahe: bool = True) -> transforms.Compose:
    """Transforms para validación/test/inferencia.

    Sin augmentaciones aleatorias, solo preprocesado determinista.
    """

    transform_list = [
        transforms.ToPILImage(),
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    ]

    if apply_clahe:
        transform_list.append(transforms.Lambda(lambda im: _apply_clahe_pil(im)))

    transform_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return transforms.Compose(transform_list)


def _apply_clahe_pil(pil_img):
    """Helper: aplica CLAHE a una PIL image y devuelve PIL image."""

    img_np = np.array(pil_img)

    if img_np.ndim == 2:
        # grayscale
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_np = clahe.apply(img_np)
        return transforms.functional.to_pil_image(img_np)

    if img_np.ndim == 3:
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        img_np = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        return transforms.functional.to_pil_image(img_np)

    return pil_img


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Desnormaliza un tensor imagen usando estadísticas ImageNet (útil para visualización)."""

    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(3, 1, 1)
    return tensor * std + mean
