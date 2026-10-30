# Clasificación de fracturas óseas en radiografías

Clasificador binario (`not_fractured` / `fractured`) de radiografías con transfer learning
(ResNet18, ResNet50 o EfficientNet-B0 preentrenados en ImageNet), explicabilidad con Grad-CAM
y una app en Streamlit para inferencia.

## Estructura

```
src/
  config.py          # rutas, semilla e hiperparámetros por defecto
  dataset.py         # datasets y dataloaders (dataset principal + dataset extra opcional)
  transforms.py      # preprocesamiento (CLAHE opcional) y aumentos de datos
  models.py          # FractureClassifier y load_model()
  train.py           # entrenamiento, validación, early stopping y métricas
  gradcam.py         # Grad-CAM
  app_streamlit.py   # app de inferencia
app.py               # punto de entrada de Streamlit
data/                # datasets (no incluidos en el repo)
outputs/             # checkpoints, logs y mapas Grad-CAM (generados, no incluidos)
```

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Datos

Los datasets no se incluyen en el repo por su tamaño. Ver [`data/README.dataset.txt`](data/README.dataset.txt)
para el enlace de descarga. Deben quedar en estas rutas (definidas en `src/config.py`):

- `data/Bone_Fracture_Binary_Classification/Bone_Fracture_Binary_Classification/` (con splits train/val/test)
- `data/Bone fracture dataset/Bone fracture dataset/Dataset/` (carpetas `normal` / `fracture`, opcional)

## Entrenamiento

Ejecutar desde la raíz del repo:

```bash
python -m src.train --backbone resnet18 --epochs 20
```

Opciones principales: `--backbone {resnet18,resnet50,efficientnet_b0}`, `--epochs`, `--batch_size`,
`--lr`, `--weight_decay`, `--dropout`, `--patience`, `--no_clahe`, `--no_extra`, `--no_pretrained`.

Los checkpoints se guardan en `outputs/checkpoints/` y los logs de TensorBoard en `outputs/logs/`:

```bash
tensorboard --logdir outputs/logs
```

## App de inferencia

`src/app_streamlit.py` usa imports relativos, por lo que se lanza desde [`app.py`](app.py) en la raíz:

```bash
streamlit run app.py
```

## Datos y licencia del dataset

Dataset Fracture Multi-Region X-ray bajo licencia ODC-By v1.0 (ver `data/README.dataset.txt`).
