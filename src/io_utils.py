import io
from typing import Tuple

import cv2
import numpy as np

# Etiquetas DICOM que identifican al paciente; nunca se muestran ni se guardan.
_PHI_TAGS = ("PatientName", "PatientID", "PatientBirthDate", "PatientAddress", "OtherPatientIDs")


def load_image(data: bytes, filename: str) -> Tuple[np.ndarray, dict]:
    """Lee PNG/JPG o DICOM y devuelve (imagen RGB uint8, metadatos no identificables)."""
    if filename.lower().endswith((".dcm", ".dicom")):
        return _load_dicom(data)

    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise ValueError("No se pudo leer la imagen.")
    return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB), {}


def _load_dicom(data: bytes) -> Tuple[np.ndarray, dict]:
    import pydicom
    from pydicom.pixels import apply_voi_lut  # pydicom >= 3

    ds = pydicom.dcmread(io.BytesIO(data))
    px = ds.pixel_array.astype(np.float32)
    if px.ndim == 3:
        px = px[0]

    try:
        px = apply_voi_lut(px, ds).astype(np.float32)
    except Exception:
        pass

    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        px = px.max() - px

    lo, hi = np.percentile(px, (0.5, 99.5))
    px = np.clip((px - lo) / max(hi - lo, 1e-6), 0, 1)
    gray = (px * 255).astype(np.uint8)

    meta = {
        "Modalidad": str(getattr(ds, "Modality", "—")),
        "Parte del cuerpo": str(getattr(ds, "BodyPartExamined", "—")),
        "Vista": str(getattr(ds, "ViewPosition", "—")),
    }
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB), meta
