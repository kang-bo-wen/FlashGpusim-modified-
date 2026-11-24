#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CONFIG_NAME=${CONFIG_NAME:-SM90_H100}
RUN_DIR=${RUN_DIR:-"${HERE}/runs-${CONFIG_NAME}"}
python3 "${HERE}/../analyze_steady_state.py" --level l2 --run-dir "${RUN_DIR}" \
  --theoretical "${RUN_DIR}/theoretical.csv" --csv "${RUN_DIR}/steady-state.csv"
