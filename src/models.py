from typing import Optional

import torch
import torch.nn as nn
from torchvision import models

from .config import CLASS_NAMES


class FractureClassifier(nn.Module):
    """Binary classifier using a ResNet or EfficientNet backbone.

    Mejoras respecto a la versión anterior:
    - pretrained=True por defecto: tranfer learning desde ImageNet
    - Dropout configurable antes de la capa final para regularización
    - Soporte para backbone "efficientnet_b0" además de resnet18/resnet50
    """

    SUPPORTED_BACKBONES = ("resnet18", "resnet50", "efficientnet_b0")

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()

        if backbone == "resnet18":
            self.backbone = models.resnet18(
                weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            )
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, 1),
            )

        elif backbone == "resnet50":
            self.backbone = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            )
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, 1),
            )

        elif backbone == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            )
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, 1),
            )

        else:
            raise ValueError(
                f"Backbone '{backbone}' no soportado. "
                f"Opciones: {self.SUPPORTED_BACKBONES}"
            )

        self._backbone_name = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).squeeze(1)


def load_model(
    backbone: str = "resnet18",
    checkpoint_path: Optional[str] = None,
    map_location: str | torch.device = "cpu",
    pretrained: bool = False,  # False al cargar checkpoint (ya tiene pesos)
) -> FractureClassifier:
    model = FractureClassifier(backbone=backbone, pretrained=pretrained)
    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
        sd = state["model_state"] if "model_state" in state else state
        # Checkpoints antiguos guardaron la capa final sin Dropout delante (fc.weight -> fc.1.weight).
        for head in ("backbone.fc", "backbone.classifier"):
            if f"{head}.weight" in sd:
                sd[f"{head}.1.weight"] = sd.pop(f"{head}.weight")
                sd[f"{head}.1.bias"] = sd.pop(f"{head}.bias")
        model.load_state_dict(sd)
    return model
