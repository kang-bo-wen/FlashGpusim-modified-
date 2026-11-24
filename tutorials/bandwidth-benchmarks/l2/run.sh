#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${HERE}/../common.sh"
benchmark_init
OUT="${HERE}/runs-${CONFIG_NAME}"
mkdir -p "${OUT}"
ELEMENTS=${ELEMENTS:-4194304}
BLOCKS=${BLOCKS:-2112}
for iterations in ${ITERATIONS:-8 16 32 64}; do
  benchmark_run_one "${OUT}" "l2-e${ELEMENTS}-b${BLOCKS}-i${iterations}" \
    --level l2 --elements "${ELEMENTS}" --blocks "${BLOCKS}" --iterations "${iterations}"
done
benchmark_finalize "${OUT}" l2
