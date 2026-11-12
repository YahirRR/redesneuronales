import os
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float
    label: str


def detector_available() -> bool:
    try:
        import ultralytics  # noqa: F401
        return True
    except ImportError:
        return False


class FractureDetector:
    """Envoltorio de YOLO (ultralytics) entrenado con scripts/train_detector.py."""

    def __init__(self, weights_path: str):
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(weights_path)
        from ultralytics import YOLO

        self.model = YOLO(weights_path)

    def predict(self, img_rgb: np.ndarray, conf: float = 0.25) -> List[Detection]:
        res = self.model.predict(img_rgb, conf=conf, verbose=False)[0]
        names = res.names
        out = []
        for b in res.boxes:
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
            out.append(Detection(x1, y1, x2, y2, float(b.conf[0]), names[int(b.cls[0])]))
        return out


def draw_detections(img_rgb: np.ndarray, dets: List[Detection], color=(220, 38, 38)) -> np.ndarray:
    out = img_rgb.copy()
    thick = max(2, out.shape[1] // 300)
    for d in dets:
        cv2.rectangle(out, (d.x1, d.y1), (d.x2, d.y2), color, thick)
        text = f"{d.label} {d.conf:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y0 = max(d.y1, th + 8)
        cv2.rectangle(out, (d.x1, y0 - th - 8), (d.x1 + tw + 6, y0), color, -1)
        cv2.putText(out, text, (d.x1 + 3, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out
