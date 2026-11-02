#!/usr/bin/env bash
# Monitorea en tiempo real (tail -F) eventos mlx5_core temp_warn / "High temperature"
# en spectre3, spectre4 y spectre5.
#
# Uso:
#   export SSHPASS='<password>'   # si no se define, se pide por prompt
#   ./monitor_mlx5_temp.sh
#
# Variables opcionales:
#   SSH_USER (default: root)
#   PATTERN  (default: mlx5_core.*(temp_warn|High temperature))
#   LOGFILE  (default: /var/log/messages; si no existe usa journalctl -kf)

SSH_USER="${SSH_USER:-root}"
PATTERN="${PATTERN:-mlx5_core.*(temp_warn|High temperature)}"
LOGFILE="${LOGFILE:-/var/log/messages}"

NODES=(
  "spectre3:9.11.137.50"
  "spectre4:9.11.137.55"
  "spectre5:9.11.137.56"
)

command -v sshpass >/dev/null 2>&1 || {
  echo "Falta sshpass (macOS: brew install sshpass | Linux: yum/apt install sshpass)" >&2
  exit 1
}

if [ -z "${SSHPASS:-}" ]; then
  read -r -s -p "Password SSH para ${SSH_USER}: " SSHPASS
  echo
fi
export SSHPASS

COLORS=(31 32 34)   # rojo, verde, azul
pids=()
trap 'kill "${pids[@]}" 2>/dev/null; exit 0' INT TERM

i=0
for entry in "${NODES[@]}"; do
  name="${entry%%:*}"
  ip="${entry##*:}"
  color="${COLORS[$i]}"
  remote="if [ -f '$LOGFILE' ]; then tail -F -n 0 '$LOGFILE'; else journalctl -k -f -n 0; fi | grep --line-buffered -E '$PATTERN'"

  sshpass -e ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30 \
      "${SSH_USER}@${ip}" "$remote" 2>&1 \
    | sed -u "s/^/$(printf '\033[%sm[%s %s]\033[0m ' "$color" "$name" "$ip")/" &
  pids+=($!)
  i=$((i + 1))
done

echo "Monitoreando ${#NODES[@]} nodos (patrón: $PATTERN). Ctrl+C para salir."
wait
