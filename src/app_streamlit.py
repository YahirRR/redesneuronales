import os

import cv2
import numpy as np
import streamlit as st
import torch

from .config import CHECKPOINT_DIR, OUTPUT_DIR
from .detector import FractureDetector, detector_available, draw_detections
from .enhance import EnhanceParams, enhance, quality_metrics
from .gradcam import GradCAM, overlay_heatmap_on_image
from .io_utils import load_image
from .models import load_model
from .transforms import get_eval_transforms

DETECTOR_DIR = os.path.join(OUTPUT_DIR, "detector")

CSS = """
<style>
  .block-container {padding-top: 1.5rem; max-width: 1200px;}
  .hdr {background: linear-gradient(90deg,#0F766E,#0E7490); color:#fff; padding:18px 24px;
        border-radius:12px; margin-bottom:14px;}
  .hdr h1 {color:#fff; font-size:1.55rem; margin:0; padding:0;}
  .hdr p {color:#D5EEF0; margin:4px 0 0 0; font-size:.9rem;}
  .card {background:#fff; border:1px solid #D7E4E9; border-radius:12px; padding:14px 18px;}
  .res {border-radius:12px; padding:16px 20px; margin:8px 0; border-left:8px solid;}
  .res.hi {background:#FEF2F2; border-color:#DC2626; color:#7F1D1D;}
  .res.lo {background:#F0FDF4; border-color:#16A34A; color:#14532D;}
  .res.mid {background:#FFFBEB; border-color:#D97706; color:#78350F;}
  .res h3 {margin:0 0 4px 0; font-size:1.15rem;}
  .disc {background:#EFF6FF; border:1px solid #BFDBFE; color:#1E3A8A; border-radius:10px;
         padding:10px 14px; font-size:.85rem;}
</style>
"""


@st.cache_resource
def load_classifier(path: str, backbone: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(backbone=backbone, checkpoint_path=path, map_location=device).to(device).eval()
    return model, GradCAM(model), get_eval_transforms(apply_clahe=True), device


@st.cache_resource
def load_detector(path: str):
    return FractureDetector(path)


def list_files(folder: str, ext: str):
    if not os.path.isdir(folder):
        return []
    return sorted(f for f in os.listdir(folder) if f.endswith(ext))


def verdict(prob: float, threshold: float):
    if prob >= threshold:
        return "hi", "Alta sospecha de fractura", "Se recomienda revisión prioritaria por el especialista."
    if prob >= threshold * 0.6:
        return "mid", "Sospecha intermedia", "Zona gris: se sugiere revisión cuidadosa de la imagen."
    return "lo", "Baja sospecha de fractura", "No descarta fractura; interpretar junto con la clínica."


def main():
    st.set_page_config(page_title="Asistente de Fracturas", page_icon="🩻", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="hdr"><h1>Asistente de apoyo para fracturas óseas</h1>'
        "<p>Detección asistida en radiografías · Solo investigación y apoyo a la lectura médica</p></div>",
        unsafe_allow_html=True,
    )

    # ── Barra lateral ────────────────────────────────────────────────────────
    sb = st.sidebar
    sb.header("Configuración")

    sb.subheader("Clasificador")
    ckpts = list_files(CHECKPOINT_DIR, ".pt")
    ckpt = sb.selectbox("Modelo", ckpts, index=len(ckpts) - 1 if ckpts else None) if ckpts else None
    if not ckpts:
        sb.warning("Sin checkpoints en outputs/checkpoints.")
    backbone = sb.selectbox("Arquitectura", ["resnet18", "resnet50", "efficientnet_b0"])
    threshold = sb.slider("Umbral de decisión", 0.1, 0.9, 0.5, 0.05,
                          help="Bajarlo aumenta la sensibilidad a costa de más falsos positivos.")

    sb.subheader("Detector de localización")
    weights = list_files(DETECTOR_DIR, ".pt")
    det_file, det_conf = None, 0.25
    if not detector_available():
        sb.info("Instala `ultralytics` para activar la localización con cajas.")
    elif not weights:
        sb.info("Sin pesos del detector. Entrénalo con scripts/train_detector.py.")
    else:
        det_file = sb.selectbox("Pesos YOLO", weights)
        det_conf = sb.slider("Confianza mínima", 0.05, 0.9, 0.25, 0.05)

    sb.subheader("Mejora de imagen")
    use_enh = sb.checkbox("Aplicar mejora", value=True)
    p = EnhanceParams(
        denoise=sb.slider("Reducción de ruido", 0.0, 15.0, 5.0, 1.0),
        clahe_clip=sb.slider("Contraste local (CLAHE)", 0.0, 4.0, 1.0, 0.5),
        sharpen=sb.slider("Nitidez", 0.0, 1.5, 0.3, 0.1),
    )

    # ── Carga de imagen ──────────────────────────────────────────────────────
    up = st.file_uploader("Radiografía (DICOM, PNG, JPG)", type=["dcm", "dicom", "png", "jpg", "jpeg"])
    if up is None:
        st.info("Sube una radiografía para comenzar.")
        st.markdown(
            '<div class="disc"><b>Aviso:</b> herramienta de investigación. No es un dispositivo médico '
            "autorizado ni sustituye el criterio de un radiólogo u ortopedista.</div>",
            unsafe_allow_html=True,
        )
        return

    try:
        img, meta = load_image(up.read(), up.name)
    except Exception as e:
        st.error(f"No se pudo leer el archivo: {e}")
        return

    enhanced = enhance(img, p) if use_enh else img[..., 0]
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

    # ── Clasificación ────────────────────────────────────────────────────────
    prob, heat_overlay = None, None
    if ckpt:
        with st.spinner("Analizando imagen..."):
            model, gradcam, tfm, device = load_classifier(os.path.join(CHECKPOINT_DIR, ckpt), backbone)
            inp = tfm(img).unsqueeze(0).to(device)  # el clasificador aplica su propio CLAHE; recibe la original
            with torch.inference_mode():
                prob = torch.sigmoid(model(inp)).item()
            heat = gradcam.generate(inp)
            base = cv2.resize(enhanced_rgb, (inp.size(-1), inp.size(-2)))
            heat_overlay = overlay_heatmap_on_image(base, heat)

    # ── Localización ─────────────────────────────────────────────────────────
    dets, boxed = [], None
    if det_file:
        with st.spinner("Localizando..."):
            det = load_detector(os.path.join(DETECTOR_DIR, det_file))
            dets = det.predict(enhanced_rgb, conf=det_conf)
            boxed = draw_detections(enhanced_rgb, dets)

    # ── Resultado ────────────────────────────────────────────────────────────
    c1, c2 = st.columns([1.1, 1])
    with c1:
        if prob is not None:
            cls, title, msg = verdict(prob, threshold)
            st.markdown(
                f'<div class="res {cls}"><h3>{title}</h3>Probabilidad estimada: <b>{prob:.1%}</b> '
                f"(umbral {threshold:.0%})<br>{msg}</div>",
                unsafe_allow_html=True,
            )
            st.progress(float(prob))
        if det_file:
            if dets:
                st.markdown(
                    f'<div class="res hi"><h3>{len(dets)} zona(s) localizada(s)</h3>'
                    + "<br>".join(f"{d.label} — confianza {d.conf:.0%}" for d in dets)
                    + "</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown('<div class="res lo"><h3>Sin zonas localizadas</h3>'
                            "El detector no encontró hallazgos con la confianza mínima elegida.</div>",
                            unsafe_allow_html=True)
    with c2:
        if meta:
            st.markdown("**Datos del estudio (sin información del paciente)**")
            st.table({k: [v] for k, v in meta.items()})

    # ── Visualizaciones ──────────────────────────────────────────────────────
    tabs = ["Comparar original / mejorada"]
    if boxed is not None:
        tabs.append("Localización")
    if heat_overlay is not None:
        tabs.append("Zona de atención")
    tabs.append("Métricas de calidad")
    t = dict(zip(tabs, st.tabs(tabs)))

    with t["Comparar original / mejorada"]:
        a, b = st.columns(2)
        a.image(img, caption="Original", use_container_width=True)
        b.image(enhanced_rgb, caption="Mejorada" if use_enh else "Sin mejora", use_container_width=True)
        st.download_button(
            "Descargar imagen mejorada (PNG)",
            cv2.imencode(".png", enhanced)[1].tobytes(),
            file_name="radiografia_mejorada.png",
            mime="image/png",
        )

    if "Localización" in t:
        with t["Localización"]:
            st.image(boxed, use_container_width=True)
            st.caption("Cajas predichas por el detector YOLO. Requieren confirmación del especialista.")

    if "Zona de atención" in t:
        with t["Zona de atención"]:
            st.image(heat_overlay, use_container_width=True)
            st.caption("Mapa Grad-CAM: regiones que más influyen en la decisión del modelo. "
                       "Es una explicación aproximada, no una localización de la fractura.")

    with t["Métricas de calidad"]:
        m0, m1 = quality_metrics(img), quality_metrics(enhanced)
        cols = st.columns(3)
        for col, k in zip(cols, m0):
            col.metric(k.capitalize(), f"{m1[k]:.1f}", f"{m1[k] - m0[k]:+.1f} vs original")
        st.caption("Métricas sin referencia: valores más altos de nitidez y contraste no siempre "
                   "implican mejor diagnóstico; ajusta los controles con criterio visual.")

    st.markdown(
        '<div class="disc"><b>Aviso:</b> herramienta de investigación, sin autorización sanitaria. '
        "Los resultados son orientativos y deben ser revisados por personal médico calificado.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
