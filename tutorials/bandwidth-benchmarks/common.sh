#!/usr/bin/env bash
set -euo pipefail
BENCH_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${BENCH_ROOT}/../.." && pwd)
CONFIG_NAME=${CONFIG_NAME:-SM90_H100}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-32}

benchmark_init() {
  if [[ ${GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN:-0} != 1 ]]; then
    set +u
    source "${REPO_ROOT}/setup_environment"
    set -u
  fi
  make -C "${BENCH_ROOT}" CONFIG_NAME="${CONFIG_NAME}"
}

benchmark_run_one() {
  local out=$1
  local name=$2
  shift 2
  local dir="${out}/${name}"
  if [[ -f "${dir}/simulation.log" ]] && grep -q 'Test PASSED' "${dir}/simulation.log"; then
    echo "[skip] ${name}"
    return
  fi
  rm -rf "${dir}"
  mkdir -p "${dir}"
  cp -a "${REPO_ROOT}/configs/${CONFIG_NAME}/." "${dir}/"
  sed -i 's/^-gpgpu_perf_sim_memcpy .*/-gpgpu_perf_sim_memcpy 0/' "${dir}/gpgpusim.config"
  printf '%s\n' \
    '-gpgpu_l2_multi_issue_port_model 1' \
    '-gpgpu_l2_lookup_sectors_per_cycle 1' \
    '-gpgpu_l2_data_port_sectors_per_cycle 1' \
    '-gpgpu_l2_data_port_cycle_period 1' \
    '-gpgpu_l2_fill_port_sectors_per_cycle 1' \
    '-gpgpu_l2_shared_data_array 0' >> "${dir}/gpgpusim.config"
  printf '%q ' "${BENCH_ROOT}/hierarchy_bandwidth" "$@" > "${dir}/command.txt"
  printf '\n' >> "${dir}/command.txt"
  echo "[run] ${name}"
  (cd "${dir}" && "${BENCH_ROOT}/hierarchy_bandwidth" "$@") \
    > "${dir}/simulation.log" 2>&1
  grep -q 'Test PASSED' "${dir}/simulation.log"
}

benchmark_finalize() {
  local out=$1
  local level=$2
  local first_config
  first_config=$(find "${out}" -mindepth 2 -name gpgpusim.config -print -quit)
  python3 "${BENCH_ROOT}/theoretical.py" "${first_config}" --csv "${out}/theoretical.csv"
  python3 "${BENCH_ROOT}/analyze_steady_state.py" \
    --level "${level}" --run-dir "${out}" --theoretical "${out}/theoretical.csv" \
    --csv "${out}/steady-state.csv"
}
