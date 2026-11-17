#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
WORK_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPORT=${REPORT:-"${WORK_DIR}/validation_report.txt"}
CONFIG_NAME=${CONFIG_NAME:-SM90_H100}
M_NATIVE=${M_NATIVE:-64}
M_SIM=${M_SIM:-32}
SIM_TIMEOUT=${SIM_TIMEOUT:-300}
RUN_NATIVE=${RUN_NATIVE:-0}

pass=0
fail=0

log() { printf '%s\n' "$*" | tee -a "${REPORT}"; }
check_rc() {
  local name=$1 rc=$2
  if [[ ${rc} -eq 0 ]]; then
    log "[PASS] ${name}"
    pass=$((pass + 1))
  else
    log "[FAIL] ${name} (rc=${rc})"
    fail=$((fail + 1))
  fi
}

: > "${REPORT}"
log "FlashGPU-Sim GEMM validation"
log "root=${ROOT_DIR}"
log "work=${WORK_DIR}"
log "date=$(date -Is 2>/dev/null || date)"
log "CUDA_INSTALL_PATH=${CUDA_INSTALL_PATH:-unset}"
log "GPGPUSIM_ROOT=${GPGPUSIM_ROOT:-unset}"
log "CONFIG_NAME=${CONFIG_NAME}"

if [[ -z "${CUDA_INSTALL_PATH:-}" && -x /usr/local/cuda/bin/nvcc ]]; then
  export CUDA_INSTALL_PATH=/usr/local/cuda
fi
export PATH="${CUDA_INSTALL_PATH:-/usr/local/cuda}/bin:${PATH}"

if ! command -v nvcc >/dev/null 2>&1; then
  log "[FAIL] nvcc is not in PATH"
  log "Set CUDA_INSTALL_PATH and add $CUDA_INSTALL_PATH/bin to PATH."
  exit 2
fi
nvcc --version 2>&1 | tee -a "${REPORT}"

log "=== Build custom GEMM ==="
make -C "${WORK_DIR}" clean all CONFIG_NAME="${CONFIG_NAME}" 2>&1 | tee -a "${REPORT}"
build_rc=${PIPESTATUS[0]}
check_rc "CUDA build" "${build_rc}"
if [[ ${build_rc} -ne 0 ]]; then
  exit 3
fi

if [[ "${RUN_NATIVE}" == "1" ]]; then
  log "=== Physical GPU correctness V1-V5 (${M_NATIVE}^3) ==="
  log "Native execution requires a physical GPU compatible with ${CONFIG_NAME}."
  for version in 1 2 3 4 5; do
    "${WORK_DIR}/gemm" --m "${M_NATIVE}" --n "${M_NATIVE}" --k "${M_NATIVE}" --version "${version}" 2>&1 | tee -a "${REPORT}"
    rc=${PIPESTATUS[0]}
    check_rc "V${version} physical correctness" "${rc}"
  done
else
  log "[SKIP] Physical GPU execution; simulator target does not need local GPU compatibility"
fi

log "=== Parser syntax ==="
python3 -m py_compile "${WORK_DIR}/parse_results.py" 2>&1 | tee -a "${REPORT}"
check_rc "parse_results.py syntax" "${PIPESTATUS[0]}"

log "=== FlashGPU-Sim smoke V1 (${M_SIM}^3) ==="
if [[ -z "${GPGPUSIM_ROOT:-}" ]]; then
  # setup_environment is a legacy sourced script and expects unset variables
  # to expand as empty strings. Temporarily disable nounset while sourcing it.
  set +u
  export OPENCL_REMOTE_GPU_HOST=${OPENCL_REMOTE_GPU_HOST:-}
  export GPGPUSIM_POWER_MODEL=${GPGPUSIM_POWER_MODEL:-}
  export PTXAS_CUDA_INSTALL_PATH=${PTXAS_CUDA_INSTALL_PATH:-${CUDA_INSTALL_PATH}}
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/setup_environment" >>"${REPORT}" 2>&1
  setup_rc=$?
  set -u
  if [[ ${setup_rc} -ne 0 ]]; then
    log "[FAIL] setup_environment returned rc=${setup_rc}"
  fi
fi
if [[ -z "${GPGPUSIM_ROOT:-}" ]]; then
  log "[FAIL] setup_environment did not set GPGPUSIM_ROOT"
  fail=$((fail + 1))
else
  (cd "${WORK_DIR}" && CONFIG_NAME="${CONFIG_NAME}" M="${M_SIM}" N="${M_SIM}" K="${M_SIM}" VERSIONS="1" timeout "${SIM_TIMEOUT}" bash ./run.sh) 2>&1 | tee -a "${REPORT}"
  sim_rc=${PIPESTATUS[0]}
  check_rc "FlashGPU-Sim V1 smoke" "${sim_rc}"
fi

log "=== Summary ==="
log "passed=${pass} failed=${fail}"
log "report=${REPORT}"
[[ ${fail} -eq 0 ]]
