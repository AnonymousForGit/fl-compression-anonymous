#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${EXAMPLES_DIR}" || exit 1

PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_CLIENTS="${1:-10}"
CLIENT_CONFIG="${2:-./resources/configs/fashion_mnist/client_1.yaml}"
SERVER_CONFIG="${3:-./resources/configs/fashion_mnist/server_qsgd.yaml}"

QSGD_BITS_STR="${QSGD_BITS:-10 7 5 4 3}"
read -r -a QSGD_BITS <<< "${QSGD_BITS_STR}"

total_runs=0
failed_runs=0

for bit in "${QSGD_BITS[@]}"; do
  [[ "${bit}" =~ ^[0-9]+$ ]] || { echo "[sweep] invalid bit value: ${bit}"; exit 1; }
  level=$(( (1 << bit) - 1 ))
  total_runs=$((total_runs + 1))
  echo "[sweep] (${total_runs}) qsgd_bits=${bit} qsgd_level=${level}"
  "${PYTHON_BIN}" ./experiments/run_federated.py \
    --server_config "${SERVER_CONFIG}" \
    --client_config "${CLIENT_CONFIG}" \
    --num_clients "${NUM_CLIENTS}" \
    --qsgd_level "${level}"
  status=$?
  if [[ ${status} -ne 0 ]]; then
    failed_runs=$((failed_runs + 1))
    echo "[sweep] failed (exit=${status}) for qsgd_level=${level}"
  fi
  echo
done

echo "[sweep] finished: total=${total_runs}, failed=${failed_runs}"
[[ ${failed_runs} -eq 0 ]]
