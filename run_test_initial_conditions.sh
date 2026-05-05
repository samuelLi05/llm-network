#!/usr/bin/env bash
set -euo pipefail

# Runs held-out initial-condition modes for test-set generation.
# Each mode is run with exactly 2 repetitions and unique seeds.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_PY="$ROOT_DIR/main.py"
NETWORK_DIR="$ROOT_DIR/network"

if [[ ! -f "$MAIN_PY" ]]; then
  echo "ERROR: main.py not found at $MAIN_PY" >&2
  exit 1
fi

# Ask for sudo once and keep credentials cached while this script runs.
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

# Activate venv if one exists (optional).
if [[ -f "$ROOT_DIR/venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$ROOT_DIR/venv/bin/activate"
fi

# Held-out modes intended for TEST set only.
modes=(
  test_consensus_pro
  test_consensus_anti
  test_center_extremes
  test_hollow_center
)
replicates=2
base_seed=7000

for mode in "${modes[@]}"; do
  for ((r=0; r<replicates; r++)); do
    seed=$((base_seed + r))

    echo ""
    echo "=== TEST mode=$mode repetition=$((r+1))/${replicates} seed=$seed ==="

    echo "[run_test_initial_conditions] restarting docker compose (fresh network)"
    restart_network

    INITIAL_CONDITION_MODE="$mode" \
    INITIAL_CONDITION_SEED="$seed" \
      python "$MAIN_PY"
  done
done

echo ""
echo "All held-out test runs completed."
