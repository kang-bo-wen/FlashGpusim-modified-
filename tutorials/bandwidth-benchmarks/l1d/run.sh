#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${HERE}/../common.sh"
benchmark_init
OUT="${HERE}/runs-${CONFIG_NAME}"
mkdir -p "${OUT}"
BLOCKS=${BLOCKS:-528}
for iterations in ${ITERATIONS:-64 128 256}; do
  benchmark_run_one "${OUT}" "l1-b${BLOCKS}-i${iterations}" \
    --level l1 --blocks "${BLOCKS}" --iterations "${iterations}"
done
benchmark_finalize "${OUT}" l1d
