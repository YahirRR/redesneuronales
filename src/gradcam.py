import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .config import GRADCAM_DIR, CLASS_NAMES
from .models import load_model
from .transforms import get_eval_transforms, denormalize


class GradCAM:
    """Implementación de Grad-CAM para clasificadores basados en ResNet/EfficientNet.

    Mejoras respecto a la versión anterior:
    - Los hooks se eliminan correctamente al hacer .remove() para no acumularlos
      en caso de múltiples instancias (evita memory leaks en la app Streamlit).
    - Se expone remove_hooks() para limpieza explícita.
    """

    def __init__(self, model: torch.nn.Module, target_layer_name: str = "layer4"):
        self.model = model
        self.model.eval()

        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        # Para EfficientNet el último bloque se llama "features"
        children = dict([*self.model.backbone.named_children()])
        if target_layer_name not in children:
            # Fallback: intentar con "features" (EfficientNet) o el último bloque disponible
            fallback = "features" if "features" in children else list(children.keys())[-1]
            target_layer_name = fallback

        target_layer = children[target_layer_name]
        self._fwd_handle = target_layer.register_forward_hook(self._forward_hook)
        self._bwd_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        """Elimina los hooks registrados. Llamar cuando ya no se necesite el GradCAM."""
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """Genera el mapa de calor Grad-CAM para un único tensor de entrada (1xCxHxW)."""

        self.model.zero_grad()

        logits = self.model(input_tensor)
        probs = torch.sigmoid(logits)

        if target_class is None:
            # Clasificador binario: gradiente respecto a la probabilidad de fractura
            target = probs
        else:
            target = probs if target_class == 1 else 1 - probs

        target.backward()

        gradients = self.gradients   # (N, C, H, W)
        activations = self.activations  # (N, C, H, W)

        if gradients is None or activations is None:
            # Situación inesperada: devolvemos mapa vacío
            h, w = input_tensor.size(-2), input_tensor.size(-1)
            return np.zeros((h, w), dtype=np.float32)

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam[0, 0].cpu().numpy()
        cam = cv2.resize(cam, (input_tensor.size(-1), input_tensor.size(-2)))
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam

    def __del__(self):
        """Limpieza automática al destruir el objeto."""
        try:
            self.remove_hooks()
        except Exception:
            pass


def overlay_heatmap_on_image(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Superpone un heatmap (HxW, 0-1) sobre una imagen RGB (HxWx3, 0-255)."""

    heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)
    return overlay


def run_gradcam_on_folder(
    checkpoint_path: str,
    input_dir: str,
    output_dir: Optional[str] = None,
    backbone: str = "resnet18",
    apply_clahe: bool = True,
    device: str | torch.device = "cpu",
):
    """Genera imágenes Grad-CAM para todas las imágenes de una carpeta."""

    if output_dir is None:
        run_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
        output_dir = os.path.join(GRADCAM_DIR, run_name)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(device)
    model = load_model(backbone=backbone, checkpoint_path=checkpoint_path, map_location=device)
    model.to(device)
    model.eval()

    gradcam = GradCAM(model)
    transform = get_eval_transforms(apply_clahe=apply_clahe)

    for fname in sorted(os.listdir(input_dir)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        img_path = os.path.join(input_dir, fname)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Copia redimensionada antes de normalización (para visualizar)
        resize_size = transform.transforms[1].size
        if isinstance(resize_size, int):
            resize_size = (resize_size, resize_size)
        img_resized = cv2.resize(img_rgb, (resize_size[0], resize_size[1]))

        inp = transform(img_rgb).unsqueeze(0).to(device)
        heatmap = gradcam.generate(inp)

        # Predicción
        with torch.inference_mode():
            logits = model(inp)
            prob = torch.sigmoid(logits).item()

        overlay = overlay_heatmap_on_image(img_resized, heatmap)

        # Añadir texto de probabilidad
        label_text = f"P(fracture)={prob:.2f}"
        cv2.putText(
            overlay,
            label_text,
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        out_path = os.path.join(output_dir, fname)
        cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"  Guardado: {out_path}")

    gradcam.remove_hooks()


__all__ = ["GradCAM", "overlay_heatmap_on_image", "run_gradcam_on_folder"]
