#!/usr/bin/env bash
set -euo pipefail

# Run a train/test sweep over initial conditions and graph types.
#
# Train phase: 24 runs by default using the training initial-condition set.
# Test phase: 8 runs by default using the held-out initial-condition set.
# Each run chooses a graph type from {community, chung-lu, erdos-renyi}
# and uses a phase-specific graph seed range so train/test do not overlap.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_PY="$ROOT_DIR/main.py"
NETWORK_DIR="$ROOT_DIR/network"

TRAIN_RUNS=${TRAIN_RUNS:-24}
TEST_RUNS=${TEST_RUNS:-8}
TRAIN_INITIAL_SEED_BASE=${TRAIN_INITIAL_SEED_BASE:-1000}
TEST_INITIAL_SEED_BASE=${TEST_INITIAL_SEED_BASE:-7000}
TRAIN_GRAPH_SEED_BASE=${TRAIN_GRAPH_SEED_BASE:-12000}
TEST_GRAPH_SEED_BASE=${TEST_GRAPH_SEED_BASE:-22000}

TRAIN_MODES=(uniform bimodal_polarized moderate_centered skew_pro skew_anti)
TEST_MODES=(test_consensus_pro test_consensus_anti test_center_extremes test_hollow_center)
GRAPH_TYPES=(community chung-lu erdos-renyi)

if [[ ! -f "$MAIN_PY" ]]; then
  echo "ERROR: main.py not found at $MAIN_PY" >&2
  exit 1
fi

if [[ ! -d "$NETWORK_DIR" ]]; then
  echo "ERROR: network directory not found at $NETWORK_DIR" >&2
  exit 1
fi

sudo -v
( while true; do sudo -n true; sleep 60; done ) &
SUDO_KEEPALIVE_PID=$!

cleanup_sudo() {
  kill "$SUDO_KEEPALIVE_PID" >/dev/null 2>&1 || true
}
trap cleanup_sudo EXIT

restart_network() {
  (cd "$NETWORK_DIR" && sudo sh -c 'docker compose down -v && docker compose up -d')
}

if [[ -f "$ROOT_DIR/venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$ROOT_DIR/venv/bin/activate"
fi

run_phase() {
  local phase_name="$1"
  local run_count="$2"
  local initial_seed_base="$3"
  local graph_seed_base="$4"
  local -n modes_ref=$5

  echo ""
  echo "=== Starting ${phase_name} phase (${run_count} runs) ==="

  for ((i=0; i<run_count; i++)); do
    local mode
    local graph_type
    local initial_seed
    local graph_seed

    mode="${modes_ref[$((RANDOM % ${#modes_ref[@]}))]}"
    graph_type="${GRAPH_TYPES[$((RANDOM % ${#GRAPH_TYPES[@]}))]}"
    initial_seed=$((initial_seed_base + i))
    graph_seed=$((graph_seed_base + i))

    echo ""
    echo "=== ${phase_name} run $((i+1))/${run_count} mode=${mode} graph=${graph_type} initial_seed=${initial_seed} graph_seed=${graph_seed} ==="

    restart_network

    INITIAL_CONDITION_MODE="$mode" \
    INITIAL_CONDITION_SEED="$initial_seed" \
    GRAPH_TYPE="$graph_type" \
    GRAPH_SEED="$graph_seed" \
      python "$MAIN_PY"
  done
}

run_phase "train" "$TRAIN_RUNS" "$TRAIN_INITIAL_SEED_BASE" "$TRAIN_GRAPH_SEED_BASE" TRAIN_MODES
run_phase "test" "$TEST_RUNS" "$TEST_INITIAL_SEED_BASE" "$TEST_GRAPH_SEED_BASE" TEST_MODES

echo ""
echo "All train/test sweep runs completed."