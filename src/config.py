import os


# ─── Rutas ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")

# Dataset binario principal con splits train/val/test explícitos
BINARY_DATASET_ROOT = os.path.join(
    DATA_ROOT, "Bone_Fracture_Binary_Classification", "Bone_Fracture_Binary_Classification"
)

# Dataset extra con carpetas normal/fracture (sin splits explícitos)
EXTRA_DATASET_ROOT = os.path.join(
    DATA_ROOT, "Bone fracture dataset", "Bone fracture dataset", "Dataset"
)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
GRADCAM_DIR = os.path.join(OUTPUT_DIR, "gradcam")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(GRADCAM_DIR, exist_ok=True)


# ─── Reproducibilidad ─────────────────────────────────────────────────────────
SEED = 42


# ─── Hiperparámetros de entrenamiento ─────────────────────────────────────────
BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3

# Scheduler: lr mínimo al final del cosine annealing (lr * LR_MIN_FACTOR)
LR_MIN_FACTOR = 0.01

# Early stopping: detener si no hay mejora durante esta cantidad de épocas
EARLY_STOPPING_PATIENCE = 6

# Para evitar problemas de multiproceso/pickle en algunos entornos (p.ej. Python 3.14)
NUM_WORKERS = 0

INPUT_SIZE = 224  # imágenes redimensionadas a (224, 224)


# ─── Clases ───────────────────────────────────────────────────────────────────
CLASS_NAMES = ["not_fractured", "fractured"]
