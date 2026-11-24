#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${HERE}/../common.sh"
benchmark_init
OUT="${HERE}/runs-${CONFIG_NAME}"
mkdir -p "${OUT}"
BLOCKS=${BLOCKS:-1056}
for elements in ${ELEMENTS:-16777216 33554432 67108864 134217728}; do
  benchmark_run_one "${OUT}" "hbm-e${elements}-b${BLOCKS}" \
    --level hbm --elements "${elements}" --blocks "${BLOCKS}"
done
benchmark_finalize "${OUT}" hbm
