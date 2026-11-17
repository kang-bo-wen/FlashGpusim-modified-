#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
WORK_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CONFIG_NAME=${CONFIG_NAME:-SM90_H100}
CONFIG_DIR=${CONFIG_DIR:-${ROOT_DIR}/configs/${CONFIG_NAME}}
RUN_PREFIX=${RUN_PREFIX:-${CONFIG_NAME}}
RESULTS_FILE=${RESULTS_FILE:-results.csv}
M=${M:-256}
N=${N:-256}
K=${K:-256}
VERSIONS=${VERSIONS:-"1 2 3 4"}
OMP_THREADS=${OMP_THREADS:-32}

if [[ -z "${GPGPUSIM_ROOT:-}" ]]; then
  echo "GPGPUSIM_ROOT is not set. Run: source ${ROOT_DIR}/setup_environment" >&2
  exit 1
fi

if [[ ! -d "${CONFIG_DIR}" ]]; then
  echo "Unknown simulator config directory: ${CONFIG_DIR}" >&2
  exit 2
fi

make -C "${WORK_DIR}" CONFIG_NAME="${CONFIG_NAME}"

# FlashGPU-Sim keeps several thread-local structures sized by
# FLASH_GPGPU_SIM_OMP_MAX_THREADS (currently 32). Hosts with more cores can
# otherwise create out-of-range OpenMP thread IDs on larger simulations.
export OMP_NUM_THREADS="${OMP_THREADS}"
export OMP_THREAD_LIMIT="${OMP_THREADS}"
for version in ${VERSIONS}; do
  run_dir="${WORK_DIR}/run/${RUN_PREFIX}_v${version}_m${M}_n${N}_k${K}"
  rm -rf "${run_dir}"
  mkdir -p "${run_dir}"
  cp -a "${CONFIG_DIR}/." "${run_dir}/"
  cp "${WORK_DIR}/gemm" "${run_dir}/"
  echo "[simulate] version=${version} shape=${M}x${N}x${K} config=${CONFIG_NAME}"
  (cd "${run_dir}" && ./gemm --m "${M}" --n "${N}" --k "${K}" --version "${version}" 2>&1 | tee simulation.log)
done

python3 "${WORK_DIR}/parse_results.py" "${WORK_DIR}"/run/${RUN_PREFIX}_v*_m${M}_n${N}_k${K}/simulation.log --output "${WORK_DIR}/${RESULTS_FILE}"
echo "Results written to ${WORK_DIR}/${RESULTS_FILE}"
