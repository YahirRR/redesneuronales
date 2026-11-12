#!/usr/bin/env bash
# Lanza el entrenamiento del detector en tmux. No consume nada de Claude mientras corre.
#   ./scripts/launch_training.sh            -> 40 épocas desde cero (o reanuda si hay last.pt)
#   tmux attach -t fracture                 -> ver el entrenamiento (salir sin matarlo: Ctrl+b, d)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SESSION="fracture"
NAME="fracture"
EPOCHS="${EPOCHS:-40}"
RUN="$ROOT/outputs/detector_runs/$NAME"
LOG="$ROOT/outputs/train_$NAME.log"
PY="$ROOT/.venv/bin/python"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Ya existe la sesión tmux '$SESSION'. Conéctate con: tmux attach -t $SESSION"
  exit 1
fi

EXTRA=""
if [ -f "$RUN/weights/last.pt" ]; then
  EXTRA="--resume"
  echo "Se encontró last.pt: se reanudará la corrida."
fi

CMD="caffeinate -is $PY scripts/train_detector.py --data data/grazpedwri/data.yaml --epochs $EPOCHS --name $NAME --device mps $EXTRA 2>&1 | tee -a '$LOG'; echo; echo '=== ENTRENAMIENTO TERMINADO (Enter para cerrar) ==='; read"

MON='while true; do clear; date; echo; echo "--- Procesos ---"; pgrep -fl train_detector | cut -c1-90; echo; echo "--- results.csv (ultimas epocas: epoca, mAP50, mAP50-95, precision, recall) ---"; if [ -f outputs/detector_runs/'"$NAME"'/results.csv ]; then python3 -c "
import csv
r=list(csv.DictReader(open(\"outputs/detector_runs/'"$NAME"'/results.csv\")))
for x in r[-8:]:
    x={k.strip():v.strip() for k,v in x.items()}
    print(x[\"epoch\"].rjust(3), x[\"metrics/mAP50(B)\"][:6], x[\"metrics/mAP50-95(B)\"][:6], x[\"metrics/precision(B)\"][:6], x[\"metrics/recall(B)\"][:6])
print(\"Epocas completadas:\", len(r), \"/ '"$EPOCHS"'\")
"; else echo "(aun sin epocas completas)"; fi; echo; echo "--- Memoria ---"; vm_stat | head -4; sleep 30; done'

mkdir -p "$ROOT/outputs"
tmux new-session -d -s "$SESSION" -n entrenamiento -c "$ROOT" "$CMD"
tmux new-window -t "$SESSION" -n monitor -c "$ROOT" "$MON"
tmux select-window -t "$SESSION:entrenamiento"

echo "Lanzado en tmux (sesión '$SESSION', $EPOCHS épocas)."
echo "  Ver:        tmux attach -t $SESSION"
echo "  Cambiar:    Ctrl+b n   (entrenamiento <-> monitor)"
echo "  Salir:      Ctrl+b d   (sigue corriendo)"
echo "  Log:        $LOG"
