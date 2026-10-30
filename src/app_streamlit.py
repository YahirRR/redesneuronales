import os

import cv2
import numpy as np
import streamlit as st
import torch

from .config import CLASS_NAMES, CHECKPOINT_DIR
from .models import load_model
from .transforms import get_eval_transforms, denormalize
from .gradcam import GradCAM, overlay_heatmap_on_image


# ─── Carga del modelo (cacheada para no recargar en cada interacción) ─────────

@st.cache_resource
def load_inference_components(checkpoint_path: str, backbone: str = "resnet18"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(backbone=backbone, checkpoint_path=checkpoint_path, map_location=device)
    model.to(device)
    model.eval()

    gradcam = GradCAM(model)
    transform = get_eval_transforms(apply_clahe=True)

    return model, gradcam, transform, device


# ─── Página principal ─────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Bone Fracture AI Assistant",
        page_icon="🦴",
        layout="wide",
    )

    st.title("🦴 Asistente de Diagnóstico de Fracturas Óseas")
    st.markdown(
        """
        **Uso clínico asistido** — Esta herramienta está diseñada como sistema de apoyo a la decisión
        para ayudar a visualizar posibles zonas de fractura en radiografías. **No reemplaza** el juicio
        de un especialista en radiología u ortopedia.
        """
    )

    # ── Barra lateral ──────────────────────────────────────────────────────────
    st.sidebar.header("⚙️ Configuración del modelo")

    # Selector de backbone
    backbone = st.sidebar.selectbox(
        "Backbone de la red",
        options=["resnet18", "resnet50", "efficientnet_b0"],
        index=0,
        help="Arquitectura del modelo. Debe coincidir con la usada al entrenar el checkpoint.",
    )

    # Umbral de decisión ajustable
    threshold = st.sidebar.slider(
        "Umbral de decisión",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help=(
            "Probabilidad mínima para clasificar como 'fractura'. "
            "Bajar el umbral aumenta sensibilidad (más verdaderos positivos) "
            "a costa de más falsos positivos."
        ),
    )

    # Selector de checkpoint
    checkpoint_path_input = st.sidebar.text_input(
        "Carpeta o archivo de checkpoint (.pt)",
        value=CHECKPOINT_DIR,
        help="Ruta a la carpeta de checkpoints o a un archivo .pt específico.",
    )

    selected_ckpt = None
    if os.path.isdir(checkpoint_path_input):
        ckpts = sorted([f for f in os.listdir(checkpoint_path_input) if f.endswith(".pt")])
        if ckpts:
            selected_ckpt_name = st.sidebar.selectbox(
                "Checkpoint disponible",
                options=ckpts,
                index=len(ckpts) - 1,
            )
            selected_ckpt = os.path.join(checkpoint_path_input, selected_ckpt_name)
            st.sidebar.success(f"Checkpoint: `{selected_ckpt_name}`")
        else:
            st.sidebar.warning("No se encontraron checkpoints (.pt) en la carpeta.")
    elif checkpoint_path_input.endswith(".pt") and os.path.isfile(checkpoint_path_input):
        selected_ckpt = checkpoint_path_input
        st.sidebar.success(f"Checkpoint: `{os.path.basename(selected_ckpt)}`")
    else:
        st.sidebar.warning("Especifica una carpeta con checkpoints o un archivo .pt válido.")

    # ── Subida de imagen ───────────────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "📂 Sube una radiografía (PNG / JPG / JPEG)",
        type=["png", "jpg", "jpeg"],
    )

    if uploaded_file is None:
        st.info("⬆️ Sube una imagen para comenzar el análisis.")
        return

    if selected_ckpt is None:
        st.error("❌ Primero especifica un checkpoint de modelo válido en la barra lateral.")
        return

    # ── Carga del modelo ───────────────────────────────────────────────────────
    with st.spinner("Cargando modelo..."):
        model, gradcam, transform, device = load_inference_components(selected_ckpt, backbone)

    # ── Preprocesamiento e inferencia ──────────────────────────────────────────
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img_gray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if img_gray is None:
        st.error("❌ No se pudo leer la imagen subida. Asegúrate de que sea una imagen válida.")
        return

    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

    inp = transform(img_rgb).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(inp)
        prob = torch.sigmoid(logits).item()

    heatmap = gradcam.generate(inp)

    # Imagen redimensionada para visualización (misma resolución que el modelo recibe)
    img_display = cv2.resize(img_rgb, (inp.size(-1), inp.size(-2)))
    overlay = overlay_heatmap_on_image(img_display, heatmap)

    # ── Visualizaciones ────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔬 Imagen original (preprocesada)")
        st.image(img_display, channels="RGB", use_container_width=True)

    with col2:
        st.subheader("🌡️ Mapa de calor Grad-CAM")
        st.image(overlay, channels="RGB", use_container_width=True)
        st.caption(
            "Zonas rojas/amarillas = regiones donde la red neuronal pone más atención."
        )

    st.markdown("---")

    # ── Resultado ──────────────────────────────────────────────────────────────
    st.subheader("📊 Resultado de la red neuronal")

    # Barra de progreso visual de la probabilidad
    st.metric(
        label="Probabilidad de fractura",
        value=f"{prob:.1%}",
        delta=f"Umbral: {threshold:.0%}",
    )
    st.progress(prob)

    if prob >= threshold:
        st.error(
            f"⚠️ La red sugiere **alta probabilidad de fractura** ({prob:.1%} ≥ {threshold:.0%}). "
            "Revise cuidadosamente la imagen y el mapa de calor, y correlacione con clínica."
        )
    else:
        st.success(
            f"✅ La red sugiere **baja probabilidad de fractura** ({prob:.1%} < {threshold:.0%}). "
            "Interpretar siempre en conjunto con la evaluación médica completa."
        )

    # Detalle técnico expandible
    with st.expander("🔧 Detalles técnicos"):
        st.markdown(
            f"""
            | Parámetro | Valor |
            |---|---|
            | Backbone | `{backbone}` |
            | Checkpoint | `{os.path.basename(selected_ckpt)}` |
            | Dispositivo | `{device}` |
            | Umbral aplicado | `{threshold}` |
            | Logit raw | `{logits.item():.4f}` |
            | Probabilidad | `{prob:.6f}` |
            """
        )

    st.markdown("---")
    st.markdown(
        """
        > **Aviso importante**: Este modelo ha sido entrenado con datasets públicos de radiografías óseas
        > y no ha pasado por validación regulatoria. Su uso debe limitarse a investigación, docencia o soporte
        > exploratorio bajo supervisión de personal médico calificado.
        """
    )


if __name__ == "__main__":
    main()
