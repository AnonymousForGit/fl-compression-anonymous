#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${EXAMPLES_DIR}" || exit 1

PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_CLIENTS="${1:-2}"
CLIENT_CONFIG="${2:-./resources/configs/cifar10/client_1.yaml}"

ERROR_BOUNDS=("1e-3" "1e-2" "3e-2" "5e-2" "1e-1")

declare -a SERVER_CONFIGS=(
  "./resources/configs/cifar10/server_predictor.yaml"
  "./resources/configs/cifar10/server_sz3_resnet18.yaml"
)

total_runs=0
failed_runs=0

for server_cfg in "${SERVER_CONFIGS[@]}"; do
  [[ -f "${server_cfg}" ]] || { echo "[sweep] skip missing config: ${server_cfg}"; continue; }
  for eb in "${ERROR_BOUNDS[@]}"; do
    total_runs=$((total_runs + 1))
    echo "[sweep] (${total_runs}) server_config=${server_cfg} error_bound=${eb}"
    "${PYTHON_BIN}" ./experiments/run_federated.py \
      --server_config "${server_cfg}" \
      --client_config "${CLIENT_CONFIG}" \
      --num_clients "${NUM_CLIENTS}" \
      --error_bound "${eb}"
    status=$?
    if [[ ${status} -ne 0 ]]; then
      failed_runs=$((failed_runs + 1))
      echo "[sweep] failed (exit=${status}) for ${server_cfg} @ error_bound=${eb}"
    fi
    echo
  done
done

echo "[sweep] finished: total=${total_runs}, failed=${failed_runs}"
[[ ${failed_runs} -eq 0 ]]
