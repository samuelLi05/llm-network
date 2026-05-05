#!/usr/bin/env bash
set -euo pipefail

# Script: run_graph_seeds.sh
# Purpose: Run multiple network runs each with a different GRAPH_SEED to
#          produce different random connection-graph structures. For each
#          run the script brings the docker compose network down and up
#          (fresh network), then runs `main.py` with GRAPH_SEED set.
#
# Features:
# - Accepts base seed / count or explicit comma-separated seeds
# - Optionally runs each `python main.py` invocation in background with logs
# - Restarts docker compose before each run (same behavior as run_initial_conditions.sh)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_PY="$ROOT_DIR/main.py"
NETWORK_DIR="$ROOT_DIR/network"


usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --base-seed N        Base seed (default: 2000)
  --count N            Number of sequential seeds starting at base-seed (default: 10)
  --seeds S1,S2,...    Explicit comma-separated seeds (overrides base-seed/count)
  --restart-only       Only restart docker compose and do not run main.py
  --help               Show this help and exit

Examples:
  $(basename "$0") --base-seed 2000 --count 10
  $(basename "$0") --seeds 2001,2005,2010
EOF
}

# defaults
base_seed=1000
count=4
explicit_seeds=""
 # background flag removed
restart_only=0

# parse args
while [[ ${#} -gt 0 ]]; do
  case "$1" in
    --base-seed)
      base_seed="$2"; shift 2;;
    --count)
      count="$2"; shift 2;;
    --seeds)
      explicit_seeds="$2"; shift 2;;
    # --background flag removed
    --restart-only)
      restart_only=1; shift;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "Unknown arg: $1" >&2; usage; exit 1;;
  esac
done

if [[ ! -f "$MAIN_PY" ]]; then
  echo "ERROR: main.py not found at $MAIN_PY" >&2
  exit 1
fi

# ensure network dir exists
if [[ ! -d "$NETWORK_DIR" ]]; then
  echo "ERROR: network directory not found at $NETWORK_DIR" >&2
  exit 1
fi



# keep sudo alive (used for restarting docker compose)
sudo -v
( while true; do sudo -n true; sleep 60; done ) &
SUDO_KEEPALIVE_PID=$!

cleanup_sudo() {
  kill "$SUDO_KEEPALIVE_PID" >/dev/null 2>&1 || true
}
trap cleanup_sudo EXIT

restart_network() {
  echo "[run_graph_seeds] restarting docker compose (fresh network)"
  (cd "$NETWORK_DIR" && sudo sh -c 'docker compose down -v && docker compose up -d')
}

# activate venv if one exists
if [[ -f "$ROOT_DIR/venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$ROOT_DIR/venv/bin/activate"
fi

# build seed list
if [[ -n "$explicit_seeds" ]]; then
  IFS=',' read -r -a seeds_arr <<<"$explicit_seeds"
else
  seeds_arr=()
  for ((i=0;i<count;i++)); do
    seeds_arr+=( $((base_seed + i)) )
  done
fi

# iterate seeds
for seed in "${seeds_arr[@]}"; do
  echo ""
  echo "=== Running GRAPH_SEED=$seed ==="

  restart_network

  if [[ $restart_only -eq 1 ]]; then
    echo "Restart only mode; skipping main.py invocation for seed $seed"
    continue
  fi

  echo "Running main.py (foreground)"
  GRAPH_SEED="$seed" python "$MAIN_PY"
  # short pause between runs to allow network to stabilize
  sleep 2

done

echo "All seeds processed."
