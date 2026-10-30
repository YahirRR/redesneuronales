import argparse
import os
from datetime import datetime

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:  # tensorboard no instalado o incompatible
    class SummaryWriter:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def close(self):
            pass

from .config import (
    CHECKPOINT_DIR,
    LOG_DIR,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    CLASS_NAMES,
    EARLY_STOPPING_PATIENCE,
)
from .dataset import create_dataloaders
from .models import FractureClassifier


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, writer=None):
    model.train()
    running_loss = 0.0
    all_labels, all_preds = [], []

    for images, labels in tqdm(loader, desc=f"Epoch {epoch} [train]", ncols=80):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        # Gradient clipping para estabilidad
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend((probs > 0.5).astype(int).tolist())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)

    if writer is not None:
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        writer.add_scalar("Accuracy/train", epoch_acc, epoch)

    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device, epoch, split="val", writer=None):
    model.eval()
    running_loss = 0.0
    all_labels, all_probs, all_preds = [], [], []

    with torch.inference_mode():
        for images, labels in tqdm(loader, desc=f"Epoch {epoch} [{split}]", ncols=80):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()

            logits = model(images)
            loss = criterion(logits, labels)
            running_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.tolist())
            all_preds.extend((probs > 0.5).astype(int).tolist())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)

    try:
        epoch_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        epoch_auc = float("nan")

    # Métricas adicionales para clasificación médica (evitar div/0 si una clase no aparece)
    epoch_f1 = f1_score(all_labels, all_preds, zero_division=0)
    epoch_precision = precision_score(all_labels, all_preds, zero_division=0)
    epoch_recall = recall_score(all_labels, all_preds, zero_division=0)

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])

    if writer is not None:
        writer.add_scalar(f"Loss/{split}", epoch_loss, epoch)
        writer.add_scalar(f"Accuracy/{split}", epoch_acc, epoch)
        writer.add_scalar(f"F1/{split}", epoch_f1, epoch)
        writer.add_scalar(f"Precision/{split}", epoch_precision, epoch)
        writer.add_scalar(f"Recall/{split}", epoch_recall, epoch)
        if not (epoch_auc != epoch_auc):  # check NaN
            writer.add_scalar(f"AUC/{split}", epoch_auc, epoch)

    return epoch_loss, epoch_acc, epoch_auc, epoch_f1, epoch_precision, epoch_recall, cm


class EarlyStopping:
    """Detiene el entrenamiento si la métrica monitoreada no mejora durante `patience` épocas."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        """Devuelve True si se debe detener el entrenamiento."""
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def main():
    parser = argparse.ArgumentParser(description="Train fracture classifier")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet50", "efficientnet_b0"],
    )
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout before final layer")
    parser.add_argument("--no_clahe", action="store_true", help="Disable CLAHE in preprocessing")
    parser.add_argument("--no_extra", action="store_true", help="Do not use extra dataset")
    parser.add_argument(
        "--no_pretrained",
        action="store_true",
        help="Train from scratch (no ImageNet weights)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=EARLY_STOPPING_PATIENCE,
        help="Early stopping patience (épocas sin mejora antes de parar)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    apply_clahe = not args.no_clahe
    use_extra = not args.no_extra
    pretrained = not args.no_pretrained

    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=args.batch_size,
        apply_clahe=apply_clahe,
        use_extra_for_train=use_extra,
    )

    model = FractureClassifier(
        backbone=args.backbone,
        pretrained=pretrained,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Cosine Annealing: lr baja suavemente hasta lr_min al final de las épocas
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    early_stopping = EarlyStopping(patience=args.patience)

    run_name = (
        datetime.now().strftime("%Y%m%d-%H%M%S")
        + f"_bb-{args.backbone}_clahe-{apply_clahe}_extra-{use_extra}"
    )
    log_dir = os.path.join(LOG_DIR, run_name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    best_val_auc = -1.0
    best_ckpt_path = None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer
        )

        val_loss, val_acc, val_auc, val_f1, val_prec, val_rec, val_cm = evaluate(
            model, val_loader, criterion, device, epoch, split="val", writer=writer
        )

        # Actualizar learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        if writer is not None:
            writer.add_scalar("LearningRate", current_lr, epoch)

        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train  -> loss={train_loss:.4f}  acc={train_acc:.4f}")
        print(
            f"  Val    -> loss={val_loss:.4f}  acc={val_acc:.4f}  auc={val_auc:.4f}"
            f"  f1={val_f1:.4f}  prec={val_prec:.4f}  rec={val_rec:.4f}"
        )
        print("  Val confusion matrix (rows=true [not_fractured, fractured]; cols=pred):")
        print(f"  {val_cm}")
        print(f"  LR={current_lr:.6f}")

        # Guardar mejor checkpoint por AUC
        if not (val_auc != val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"best_{run_name}.pt")
            torch.save({"model_state": model.state_dict()}, ckpt_path)
            best_ckpt_path = ckpt_path
            print(f"  ✔ Nuevo mejor checkpoint (AUC={val_auc:.4f}): {ckpt_path}")

        # Early stopping
        monitor_score = val_auc if not (val_auc != val_auc) else val_acc
        if early_stopping.step(monitor_score):
            print(f"\nEarly stopping activado en epoch {epoch} (sin mejora en {args.patience} épocas).")
            break

    # Guardar checkpoint de la última época
    last_ckpt_path = os.path.join(CHECKPOINT_DIR, f"last_{run_name}.pt")
    torch.save({"model_state": model.state_dict()}, last_ckpt_path)
    print(f"\nCheckpoint final guardado: {last_ckpt_path}")

    # Evaluación final en test con el mejor checkpoint
    eval_ckpt_path = best_ckpt_path or last_ckpt_path
    print(f"Cargando checkpoint para evaluación en test: {eval_ckpt_path}")
    state = torch.load(eval_ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state"])

    test_loss, test_acc, test_auc, test_f1, test_prec, test_rec, test_cm = evaluate(
        model, test_loader, criterion, device, epoch=args.epochs, split="test", writer=writer
    )

    print("\n" + "=" * 50)
    print("RESULTADOS FINALES EN TEST")
    print("=" * 50)
    print(f"  loss      = {test_loss:.4f}")
    print(f"  accuracy  = {test_acc:.4f}")
    print(f"  AUC-ROC   = {test_auc:.4f}")
    print(f"  F1-score  = {test_f1:.4f}")
    print(f"  Precision = {test_prec:.4f}")
    print(f"  Recall    = {test_rec:.4f}")
    print("  Confusion matrix (rows=true [not_fractured, fractured]; cols=pred):")
    print(f"  {test_cm}")

    writer.close()


if __name__ == "__main__":
    main()
