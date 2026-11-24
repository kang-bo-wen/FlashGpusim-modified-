#!/usr/bin/env bash
set -euo pipefail

BENCH_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${BENCH_ROOT}/../.." && pwd)
CONFIG_NAME=${CONFIG_NAME:-SM90_H100}
SIZES=${SIZES:-"128 256 512"}
VARIANTS=${VARIANTS:-"naive shared vector"}
OUT_DIR=${OUT_DIR:-"${BENCH_ROOT}/runs-${CONFIG_NAME}"}
RESULT_DIR=${RESULT_DIR:-"${BENCH_ROOT}/results"}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-32}

if [[ ${GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN:-0} != 1 ]]; then
  set +u
  source "${REPO_ROOT}/setup_environment"
  set -u
fi
make -C "${BENCH_ROOT}" CONFIG_NAME="${CONFIG_NAME}"
mkdir -p "${OUT_DIR}" "${RESULT_DIR}"

run_one() {
  local variant=$1
  local size=$2
  local name="${variant}-n${size}"
  local dir="${OUT_DIR}/${name}"
  if [[ -f "${dir}/simulation.log" ]] && grep -q 'Test PASSED' "${dir}/simulation.log"; then
    echo "[skip] ${name}"
    return
  fi
  rm -rf "${dir}"
  mkdir -p "${dir}"
  cp -a "${REPO_ROOT}/configs/${CONFIG_NAME}/." "${dir}/"
  sed -i 's/^-gpgpu_perf_sim_memcpy .*/-gpgpu_perf_sim_memcpy 0/' \
    "${dir}/gpgpusim.config"
  printf '%s\n' \
    '-gpgpu_l2_multi_issue_port_model 1' \
    '-gpgpu_l2_lookup_sectors_per_cycle 1' \
    '-gpgpu_l2_data_port_sectors_per_cycle 1' \
    '-gpgpu_l2_data_port_cycle_period 1' \
    '-gpgpu_l2_fill_port_sectors_per_cycle 1' \
    '-gpgpu_l2_shared_data_array 0' >> "${dir}/gpgpusim.config"
  printf '%q ' "${BENCH_ROOT}/transpose" --kernel "${variant}" \
    --rows "${size}" --cols "${size}" > "${dir}/command.txt"
  printf '\n' >> "${dir}/command.txt"
  echo "[run] ${name}"
  (cd "${dir}" && "${BENCH_ROOT}/transpose" --kernel "${variant}" \
    --rows "${size}" --cols "${size}") 2>&1 | tee "${dir}/simulation.log"
  grep -q 'Test PASSED' "${dir}/simulation.log"
}

for size in ${SIZES}; do
  for variant in ${VARIANTS}; do
    run_one "${variant}" "${size}"
  done
done

python3 "${BENCH_ROOT}/analyze.py" \
  --run-dir "${OUT_DIR}" --output "${RESULT_DIR}/transpose-results.csv" \
  --reference-config "${CONFIG_NAME}"
python3 "${BENCH_ROOT}/plot_results.py" \
  "${RESULT_DIR}/transpose-results.csv" \
  "${RESULT_DIR}/bandwidth-comparison.svg"
