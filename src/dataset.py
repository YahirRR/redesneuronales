import os
import random
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .config import (
    BINARY_DATASET_ROOT,
    EXTRA_DATASET_ROOT,
    CLASS_NAMES,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
)


def seed_worker(worker_id: int) -> None:
    """Función para inicializar la semilla en cada worker del DataLoader."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class FractureDataset(Dataset):
    """Dataset genérico para clasificación binaria: fractured vs not_fractured.

    Mejoras:
    - Ordena las muestras para garantizar reproducibilidad entre ejecuciones.
    - Manejo de errores mejorado: imagen no leída devuelve tensor en negro en lugar
      de lanzar excepción cruda (útil cuando el dataset tiene archivos corruptos).
    """

    def __init__(
        self,
        root_dir: str,
        subset: str,
        transform=None,
        use_extra: bool = False,
    ):
        """
        Args:
            root_dir: BINARY_DATASET_ROOT
            subset: one of {"train", "val", "test"}
            transform: torchvision transforms
            use_extra: si True y subset=="train", incluye imágenes del dataset extra
        """

        assert subset in {"train", "val", "test"}, f"subset debe ser train/val/test, recibido: {subset}"
        self.root_dir = root_dir
        self.subset = subset
        self.transform = transform

        self.samples: list[Tuple[str, int]] = []  # (image_path, label_idx)

        # Dataset principal curado
        subset_dir = os.path.join(root_dir, subset)
        for class_name in sorted(os.listdir(subset_dir)):  # sorted → orden determinista
            class_path = os.path.join(subset_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            label = CLASS_NAMES.index(
                "fractured" if "fractured" in class_name.lower() else "not_fractured"
            )

            for fname in sorted(os.listdir(class_path)):  # sorted → orden determinista
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append((os.path.join(class_path, fname), label))

        # Dataset extra opcional: solo para entrenamiento
        if use_extra and subset == "train" and os.path.isdir(EXTRA_DATASET_ROOT):
            for class_name in ["fracture", "normal"]:
                class_path = os.path.join(EXTRA_DATASET_ROOT, class_name)
                if not os.path.isdir(class_path):
                    continue

                label = CLASS_NAMES.index(
                    "fractured" if class_name == "fracture" else "not_fractured"
                )
                for fname in sorted(os.listdir(class_path)):
                    if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                        self.samples.append((os.path.join(class_path, fname), label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            # Imagen corrupta o faltante: devolvemos tensor negro del tamaño correcto
            # para evitar que falle el DataLoader en batches completos
            import warnings
            warnings.warn(f"No se pudo leer la imagen: {img_path}. Se usará imagen en negro.")
            from .config import INPUT_SIZE
            img = np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=np.uint8)

        # Convertir escala de grises a RGB-like 3 canales
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        if self.transform is not None:
            img_rgb = self.transform(img_rgb)

        return img_rgb, label


def create_dataloaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    apply_clahe: bool = True,
    use_extra_for_train: bool = True,
    seed: int = SEED,
):
    """Crea train/val/test DataLoaders con reproducibilidad garantizada.

    Se usa un generador de PyTorch con semilla fija para que el shuffle
    sea reproducible entre ejecuciones.
    """

    from .transforms import get_train_transforms, get_eval_transforms

    train_ds = FractureDataset(
        root_dir=BINARY_DATASET_ROOT,
        subset="train",
        transform=get_train_transforms(apply_clahe=apply_clahe),
        use_extra=use_extra_for_train,
    )

    val_ds = FractureDataset(
        root_dir=BINARY_DATASET_ROOT,
        subset="val",
        transform=get_eval_transforms(apply_clahe=apply_clahe),
        use_extra=False,
    )

    test_ds = FractureDataset(
        root_dir=BINARY_DATASET_ROOT,
        subset="test",
        transform=get_eval_transforms(apply_clahe=apply_clahe),
        use_extra=False,
    )

    # Generador con semilla para shuffle reproducible
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
