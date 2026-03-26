#!/usr/bin/env bash
set -euo pipefail

# Runs 10 network runs: 2 replicates for each INITIAL_CONDITION_MODE.
# No env vars required; this script temporarily edits main.py, runs, then restores.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_PY="$ROOT_DIR/main.py"
NETWORK_DIR="$ROOT_DIR/network"

if [[ ! -f "$MAIN_PY" ]]; then
  echo "ERROR: main.py not found at $MAIN_PY" >&2
  exit 1
fi

STAMP="$(date +%Y%m%d-%H%M%S)"

# ask for sudo once and keep credentials cached for the duration of the script
sudo -v
( while true; do sudo -n true; sleep 60; done ) &
SUDO_KEEPALIVE_PID=$!

cleanup_sudo() {
  kill "$SUDO_KEEPALIVE_PID" >/dev/null 2>&1 || true
}
trap cleanup_sudo EXIT

restart_network() {
  if [[ ! -d "$NETWORK_DIR" ]]; then
    echo "ERROR: network directory not found at $NETWORK_DIR" >&2
    exit 1
  fi
  (cd "$NETWORK_DIR" && sudo sh -c 'docker compose down -v && docker compose up -d')
}

# activate venv if one exists (optional, but helpful)
if [[ -f "$ROOT_DIR/venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$ROOT_DIR/venv/bin/activate"
fi

modes=(uniform bimodal_polarized moderate_centered skew_pro skew_anti)
replicates=2

# Seeds are deterministic per (mode, replicate) to ensure distinct initial conditions.
base_seed=1000

for mode in "${modes[@]}"; do
  for ((r=0; r<replicates; r++)); do
    seed=$((base_seed + r))

    echo ""
    echo "=== Running mode=$mode replicate=$((r+1))/${replicates} seed=$seed ==="

    echo "[run_initial_conditions] restarting docker compose (fresh network)"
    restart_network

    # Run the network with per-run initial condition override.
    INITIAL_CONDITION_MODE="$mode" \
    INITIAL_CONDITION_SEED="$seed" \
      python "$MAIN_PY"
  done
done

echo ""
echo "All runs completed. main.py was not modified."
